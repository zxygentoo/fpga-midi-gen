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

let bar_steps = 16
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
   meet in the middle, so the release of the top moving voice sits beside its attack. *)
let tokenize steps =
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
  steps |> Array.to_list |> List.folding_map ~init:[] ~f:aux |> List.concat
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

(* The vote of one bar length: the rotation that puts the most holds on a downbeat, with
   the smallest rotation on a tie. The lift — hits times bar length — normalizes the two
   bar lengths against their chance rates, thus the votes of 12 and 16 compare. *)
let vote holds bar =
  List.map (List.range 0 bar) ~f:(fun rotation ->
    let hits = List.count holds ~f:(fun hold -> (hold - rotation) mod bar = 0) in
    hits * bar, rotation)
  |> List.max_elt ~compare:(fun (lift, _) (lift', _) -> Int.compare lift lift')
  |> Option.value_exn
;;

(* Fewer than three holds gives no signal: the piece keeps the plain sixteen-step grid. *)
let metre steps =
  let holds = cadential_holds steps in
  if List.length holds < 3
  then bar_steps, 0
  else (
    let lift12, rotation12 = vote holds 12 in
    let lift16, rotation16 = vote holds 16 in
    if lift16 >= lift12 then 16, rotation16 else 12, rotation12)
;;

(* The walk opens with START, per the design document: the boot writes START and the music
   follows at once. START takes phase zero; the entry draw does not see a bar position.
   The phases of the piece align to the estimated downbeats. *)
let encode { steps; legal_shifts = _ } =
  let bar, rotation = metre steps in
  let tokens = tokenize steps in
  let codes =
    Array.of_list (Token.to_code Token.Start :: List.map tokens ~f:Token.to_code)
  in
  let (_ : int), piece_phases =
    List.fold_map tokens ~init:0 ~f:(fun step token ->
      let next =
        match token with
        | Token.Start | On _ | Off _ -> step
        | End -> step + 1
      in
      next, (((step - rotation) mod bar) + bar) mod bar)
  in
  ~codes, ~phases:(Array.of_list (0 :: piece_phases))
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

let%expect_test "the metre from the cadential holds" =
  (* a bar = a six-step hold on the downbeat, then moving single notes *)
  let hold = Array.create ~len:6 [ 60; 64; 67 ] in
  let motion count = Array.init count ~f:(fun i -> [ 40 + i ]) in
  let bar_44 = Array.append hold (motion 10) in
  let bar_34 = Array.append hold (motion 6) in
  (* three 16-step bars: common time, downbeats at 0, 16, 32 *)
  let common = Array.concat [ bar_44; bar_44; bar_44 ] in
  print_s ([%sexp_of: int * int] (metre common));
  [%expect {| (16 0) |}];
  (* three 12-step bars: a waltz, downbeats at 0, 12, 24 *)
  let waltz = Array.concat [ bar_34; bar_34; bar_34 ] in
  print_s ([%sexp_of: int * int] (metre waltz));
  [%expect {| (12 0) |}];
  (* a four-step pickup moves every downbeat: the rotation follows *)
  let pickup = Array.append (motion 4) common in
  print_s ([%sexp_of: int * int] (metre pickup));
  [%expect {| (16 4) |}];
  (* two holds are no signal: the plain grid *)
  let short = Array.concat [ bar_44; bar_44 ] in
  print_s ([%sexp_of: int * int] (metre short));
  [%expect {| (16 0) |}]
;;

let%expect_test "the walk of a small chorale" =
  (* a chord, a hold, a move of two voices, a rest, then a unison *)
  let steps = [| [ 67; 64; 60 ]; [ 67; 64; 60 ]; [ 67; 65; 62 ]; []; [ 60; 60 ] |] in
  print_s ([%sexp_of: Token.t list] (tokenize steps));
  [%expect
    {|
    ((On 67) (On 64) (On 60) End End (Off 60) (Off 64) (On 65) (On 62) End
     (Off 62) (Off 65) (Off 67) End (On 60) End)
    |}];
  let ~codes, ~phases = encode { steps; legal_shifts = [ 0 ] } in
  print_s ([%sexp_of: int array] codes);
  print_s ([%sexp_of: int array] phases);
  [%expect
    {|
    (255 195 192 188 0 0 60 64 193 190 0 62 65 67 0 188 0)
    (0 0 0 0 0 1 2 2 2 2 2 3 3 3 3 4 4)
    |}]
;;
