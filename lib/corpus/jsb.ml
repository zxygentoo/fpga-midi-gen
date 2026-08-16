open Core
module Json = Yojson.Safe

type chorale =
  { steps : int list array
  ; legal_shifts : int list
  }

type t =
  { train : chorale list
  ; valid : chorale list
  ; test : chorale list
  }

type stream =
  { codes : int array
  ; positions : int array
  ; anchors : int array
  }

let bar_steps = 16

(* the bars of the memory window: the rows of the frame table of the model. It pairs with
   [Transformer.progress_buckets] the way [bar_steps] pairs with the bar-phase table, and
   the two counts must agree. *)
let frame_bars = 16

(* the period of the rolling coordinate: the phase is the low four bits and the frame is
   the high four *)
let window_steps = bar_steps * frame_bars
let default_path = "corpus/JSB-Chorales-dataset/Jsb16thSeparated.json"

(* The observed range of each voice over the corpus, from the corpus study of 2026-08-06
   and the design document: soprano, alto, tenor, bass. The transposition policy reads
   these bounds. *)
let voice_ranges = [| 60, 81; 52, 74; 46, 69; 36, 66 |]

(* the reserved-code rules of the design document; the JSB corpus never takes these paths.
   A rest cell (-1) passes through. *)
let escape_reserved pitch = if pitch = 0 then 1 else if pitch = 127 then 126 else pitch

(* One step of the separated file: four cells indexed by voice, the soprano first. A cell
   holds the pitch that its voice sings, or -1 for a rest. *)
let cells_of_json json =
  let cell_of_json = function
    | `Int pitch -> if pitch < 0 then pitch else escape_reserved pitch
    | `Float pitch ->
      let pitch = Int.of_float pitch in
      if pitch < 0 then pitch else escape_reserved pitch
    | json -> invalid_argf "a cell is not a number: %s" (Json.to_string json) ()
  in
  let cells = json |> Json.Util.to_list |> List.map ~f:cell_of_json in
  if List.length cells <> Array.length voice_ranges
  then invalid_argf "a step holds %d cells, not four" (List.length cells) ();
  cells
;;

(* The legal transpositions of one piece: every shift that keeps each voice inside the
   observed range of its voice. The bounds intersect over the voices in closed form; a
   piece inside the ranges always keeps shift zero, and a silent piece takes it alone. *)
let legal_shifts_of_cells cells =
  let widen ranges step =
    List.iteri step ~f:(fun voice pitch ->
      if pitch > 0
      then (
        let range =
          match ranges.(voice) with
          | None -> pitch, pitch
          | Some (low, high) -> min low pitch, max high pitch
        in
        ranges.(voice) <- Some range))
  in
  let ranges = Array.create ~len:(Array.length voice_ranges) None in
  List.iter cells ~f:(widen ranges);
  if Array.for_all ranges ~f:Option.is_none
  then [ 0 ]
  else (
    let low, high =
      Array.foldi ranges ~init:(-127, 127) ~f:(fun voice (low, high) range ->
        match range with
        | None -> low, high
        | Some (lowest, highest) ->
          let corpus_low, corpus_high = voice_ranges.(voice) in
          max low (corpus_low - lowest), min high (corpus_high - highest))
    in
    if low > high then [ 0 ] else List.range ~stop:`inclusive low high)
;;

(* the flat view of one step: the sounding set, ascending, unisons merged *)
let flatten_cells cells =
  cells
  |> List.filter ~f:(fun pitch -> pitch > 0)
  |> List.dedup_and_sort ~compare:Int.compare
;;

let chorale_of_json json =
  let cells = json |> Json.Util.to_list |> List.map ~f:cells_of_json in
  { steps = Array.of_list_map cells ~f:flatten_cells
  ; legal_shifts = legal_shifts_of_cells cells
  }
;;

let load ~path =
  let json = Json.from_file path in
  let chorales name =
    json |> Json.Util.member name |> Json.Util.to_list |> List.map ~f:chorale_of_json
  in
  { train = chorales "train"; valid = chorales "valid"; test = chorales "test" }
;;

let transpose ~by { steps; legal_shifts } =
  let transpose_pitch pitch =
    match pitch + by with
    | moved when moved < 1 || moved > 126 ->
      invalid_argf "pitch %d moved by %d leaves 1 to 126" pitch by ()
    | transposed -> transposed
  in
  let transpose_step step = List.map step ~f:transpose_pitch in
  { steps = Array.map steps ~f:transpose_step
  ; legal_shifts = List.map legal_shifts ~f:(fun shift -> shift - by)
  }
;;

(* One step, one sentence: the OFFs of the pitches that stop, the ONs of the pitches that
   start, then [End]. A pitch in both neighbour steps is a held note and takes no token.

   The OFFs climb and the ONs fall, thus one chord has one sentence and not a permutation
   family. [Sounding_state] holds both directions, thus they are rules of the instrument
   and this tokenizer only obeys them. The fall is the melody leading: the top voice is
   chosen first and conditions on no voice below it. The climb then makes the two runs
   meet in the middle, so the release of the top moving voice sits beside its attack.

   The sentences stay apart, because the packer counts the tokens of each step. *)
let sentences steps =
  let sentence ~previous ~current =
    let on pitch = Token.On pitch in
    let off pitch = Token.Off pitch in
    let to_set data = Set.of_list (module Int) data in
    let ascending from ~minus = Set.diff (to_set from) (to_set minus) |> Set.to_list in
    let descending from ~minus = List.rev (ascending from ~minus) in
    List.map (ascending previous ~minus:current) ~f:off
    @ List.map (descending current ~minus:previous) ~f:on
    @ [ Token.End ]
  in
  let aux previous current = current, sentence ~previous ~current in
  List.folding_map steps ~init:[] ~f:aux
;;

(* The cadential holds of one chorale, from the corpus study of 2026-08-06: a sonority
   that rings six steps or more marks a cadence, and the cadences sit on the downbeats.
   The result is the start step of each hold, ascending. *)
let cadential_holds steps =
  let sonorities = Array.to_list (Array.map steps ~f:(Set.of_list (module Int))) in
  let runs = List.group sonorities ~break:(fun a b -> not (Set.equal a b)) in
  let (_ : int), holds =
    List.fold_map runs ~init:0 ~f:(fun start run ->
      let cadential = (not (Set.is_empty (List.hd_exn run))) && List.length run >= 6 in
      start + List.length run, if cadential then Some start else None)
  in
  List.filter_opt holds
;;

(* the vote of one rotation: the count of holds it puts on a downbeat *)
let vote holds rotation =
  List.count holds ~f:(fun hold -> (hold - rotation) % bar_steps = 0)
;;

(* The rotation of one piece: the placement on the rolling clock that puts the most of its
   cadences on a downbeat, and the smallest rotation on a tie. Fewer than three holds
   gives no signal, and the piece opens on the downbeat.

   The clock has one length, thus a piece has no bar length of its own. The corpus study
   of 2026-08-14 in docs/improviser.md settles this: 27 of the 382 pieces vote for a
   12-step bar, they hold 8.1 percent of the steps, and a rolling clock of 16 steps still
   puts 44.6 percent of their cadences on a downbeat. To remove them buys 1.9 points of
   alignment over the whole corpus and costs those steps. *)
let rotation steps =
  let holds = cadential_holds steps in
  if List.length holds < 3
  then 0
  else
    List.range 0 bar_steps
    |> List.max_elt ~compare:(fun a b -> Int.compare (vote holds a) (vote holds b))
    |> Option.value_exn
;;

(* The state of the placement: the count of steps laid down, those steps with the newest
   first, and the first step of each piece with the newest first. The steps do not take
   the name [steps]: the field of a chorale holds that name, and one of the two would then
   need a type to read it. *)
type placement =
  { at : int
  ; behind : int list list
  ; starts : int list
  }

let empty_steps count steps = List.init count ~f:(fun (_ : int) -> []) @ steps

(* One piece and the seam before it. The gap is the smallest count of empty steps that
   puts the downbeats of the piece on the clock. It is never zero after another piece,
   because the release needs one step; the stream itself may open with no silence.

   Every piece is a whole number of quarter notes and every rotation is one, thus the gap
   is 4, 8, 12 or 16 steps. The quiet of a seam is never shorter than a quarter note and
   never longer than a bar, and no rule states this. *)
let place state chorale =
  let lead = (-rotation chorale.steps - state.at) % bar_steps in
  let gap = if state.at = 0 then lead else if lead = 0 then bar_steps else lead in
  let start = state.at + gap in
  { at = start + Array.length chorale.steps
  ; behind =
      Array.fold chorale.steps ~init:(empty_steps gap state.behind) ~f:(fun behind step ->
        step :: behind)
  ; starts = start :: state.starts
  }
;;

let pack chorales =
  let placed = List.fold chorales ~init:{ at = 0; behind = []; starts = [] } ~f:place in
  (* the stream closes as a seam does: it leaves no chord sounding, and it ends on a bar
     boundary *)
  let tail =
    match -placed.at % bar_steps with
    | 0 -> bar_steps
    | gap -> gap
  in
  let sentences = sentences (List.rev (empty_steps tail placed.behind)) in
  let of_each ~f = Array.of_list (List.concat_mapi sentences ~f) in
  let first_token =
    List.folding_map sentences ~init:0 ~f:(fun at tokens -> at + List.length tokens, at)
    |> Array.of_list
  in
  { codes = of_each ~f:(fun (_ : int) tokens -> List.map tokens ~f:Token.to_code)
  ; positions =
      of_each ~f:(fun step tokens ->
        List.map tokens ~f:(fun (_ : Token.t) -> step % window_steps))
  ; anchors = Array.of_list_map (List.rev placed.starts) ~f:(Array.get first_token)
  }
;;

(* One stream is one draw of the transpositions, thus a split needs more than one. The
   first is the canonical stream, and both trainers make it the same way: the referee
   reads it alone, thus Gate A and Gate B stay deterministic. *)
let streams chorales ~count ~random_state =
  let drawn_shift chorale =
    let shifts = Array.of_list chorale.legal_shifts in
    transpose chorale ~by:shifts.(Random.State.int random_state (Array.length shifts))
  in
  let drawn (_ : int) =
    List.permute chorales ~random_state |> List.map ~f:drawn_shift |> pack
  in
  (* [List.init] walks its range from the end; the range keeps the draws in order *)
  pack chorales :: List.map (List.range 0 (count - 1)) ~f:drawn
;;

let%expect_test "the cells of one step" =
  let cells json = cells_of_json (Json.from_string json) in
  (* a full chord, the soprano first *)
  print_s ([%sexp_of: int list] (cells "[74, 70, 65, 58]"));
  [%expect {| (74 70 65 58) |}];
  (* a rest in the alto; a float parses as its pitch *)
  print_s ([%sexp_of: int list] (cells "[74.0, -1, 65, 58]"));
  [%expect {| (74 -1 65 58) |}];
  (* the reserved codes escape: pitch 127 falls, pitch 0 rises *)
  print_s ([%sexp_of: int list] (cells "[127, 64, 55, 0]"));
  [%expect {| (126 64 55 1) |}];
  (* a step without its four voices refuses *)
  (match cells "[74, 70, 65]" with
   | (_ : int list) -> ()
   | exception Invalid_argument message -> print_endline message);
  [%expect {| a step holds 3 cells, not four |}]
;;

let%expect_test "the shifts of the range-limited policy" =
  (* soprano 72..76, alto 64, tenor 55, bass 48: the soprano allows -12..+5, the tenor
     -9..+14 — the intersection is -9..+5 *)
  let cells =
    [ [ 72; 64; 55; 48 ]; [ 76; 64; 55; 48 ]; [ 74; -1; 55; 48 ]; [ 72; 64; 55; 48 ] ]
  in
  print_s ([%sexp_of: int list] (legal_shifts_of_cells cells));
  [%expect {| (-9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5) |}];
  (* a silent piece takes the identity alone *)
  print_s ([%sexp_of: int list] (legal_shifts_of_cells [ [ -1; -1; -1; -1 ] ]));
  [%expect {| (0) |}]
;;

let%expect_test "the chorale of the separated json" =
  (* two chords with a hold; a bass rest and a tenor-alto unison in the second. The flat
     steps drop the rests, merge the unisons and sort ascending; the shifts come from the
     voices — the tenor at 69 touches its ceiling, thus no shift up *)
  let json =
    {|[[74, 70, 65, 58], [74, 70, 65, 58], [76, 70, 69, -1], [76, 70, 69, -1]]|}
  in
  let { steps; legal_shifts } = chorale_of_json (Json.from_string json) in
  print_s ([%sexp_of: int list array] steps);
  print_s ([%sexp_of: int list] legal_shifts);
  [%expect
    {|
    ((58 65 70 74) (58 65 70 74) (69 70 76) (69 70 76))
    (-14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0)
    |}]
;;

let%expect_test "the rotation from the cadential holds" =
  (* a bar = a six-step hold on the downbeat, then moving single notes *)
  let hold = Array.create ~len:6 [ 60; 64; 67 ] in
  let motion count = Array.init count ~f:(fun i -> [ 40 + i ]) in
  let bar_44 = Array.append hold (motion 10) in
  let bar_34 = Array.append hold (motion 6) in
  (* three 16-step bars: common time, downbeats at 0, 16, 32 *)
  let common = Array.concat [ bar_44; bar_44; bar_44 ] in
  print_s ([%sexp_of: int] (rotation common));
  [%expect {| 0 |}];
  (* a four-step pickup moves every downbeat: the rotation follows *)
  let pickup = Array.append (motion 4) common in
  print_s ([%sexp_of: int] (rotation pickup));
  [%expect {| 4 |}];
  (* Three 12-step bars: a waltz. The clock cannot hold its downbeats — 0, 12 and 24 sit
     in three classes of the 16 grid — thus one hold is the most that any rotation takes,
     and the smallest rotation of the tie wins. *)
  let waltz = Array.concat [ bar_34; bar_34; bar_34 ] in
  print_s ([%sexp_of: int] (rotation waltz));
  [%expect {| 0 |}];
  (* two holds are no signal: the piece opens on the downbeat *)
  let short = Array.concat [ bar_44; bar_44 ] in
  print_s ([%sexp_of: int] (rotation short));
  [%expect {| 0 |}]
;;

let%expect_test "the walk of a small chorale" =
  (* a chord, a hold, a move of two voices, a rest, then a unison *)
  let steps = [ [ 67; 64; 60 ]; [ 67; 64; 60 ]; [ 67; 65; 62 ]; []; [ 60; 60 ] ] in
  print_s ([%sexp_of: Token.t list] (List.concat (sentences steps)));
  [%expect
    {|
    ((On 67) (On 64) (On 60) End End (Off 60) (Off 64) (On 65) (On 62) End
     (Off 62) (Off 65) (Off 67) End (On 60) End)
    |}]
;;

let%expect_test "the packed stream of two pieces" =
  (* Two pieces of four steps, each one chord held. Neither holds three cadences, thus
     both take rotation zero and both must open on a downbeat of the clock. *)
  let chorale pitches = { steps = Array.create ~len:4 pitches; legal_shifts = [ 0 ] } in
  let { codes; positions; anchors } =
    pack [ chorale [ 60; 64; 67 ]; chorale [ 62; 65; 69 ] ]
  in
  (* The first piece opens the stream and takes no seam. The seam between the pieces is
     twelve steps: the release at step 4, then eleven silent steps, and the second piece
     at step 16. *)
  print_s ([%sexp_of: int array] anchors);
  [%expect {| (0 22) |}];
  print_s ([%sexp_of: int array] codes);
  [%expect
    {|
    (195 192 188 0 0 0 0 60 64 67 0 0 0 0 0 0 0 0 0 0 0 0 197 193 190 0 0 0 0 62
     65 69 0 0 0 0 0 0 0 0 0 0 0 0)
    |}];
  (* the coordinate rolls with the stream and never restarts at a piece *)
  print_s ([%sexp_of: int array] positions);
  [%expect
    {|
    (0 0 0 0 1 2 3 4 4 4 4 5 6 7 8 9 10 11 12 13 14 15 16 16 16 16 17 18 19 20 20
     20 20 21 22 23 24 25 26 27 28 29 30 31)
    |}]
;;
