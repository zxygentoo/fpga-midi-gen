open Core
module Json = Yojson.Safe

type chorale =
  { cells : int list array
  ; legal_shifts : int list
  }

type t =
  { train : chorale list
  ; valid : chorale list
  ; test : chorale list
  }

type stream =
  { frames : int array
  ; positions : int array
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

(* the table holds one row for each voice, thus it counts them *)
let voices = Array.length voice_ranges

(* One step of the separated file: four cells indexed by voice, the soprano first. A cell
   holds the pitch that its voice sings, or -1 for a rest.

   No cell is escaped. The voice code of the step frame holds all of 0 to 127 in its pitch
   field and reserves nothing, thus the reader moves no note and the MIDI range is the
   MIDI range. *)
let cells_of_json json =
  let cell_of_json = function
    | `Int pitch -> pitch
    | `Float pitch -> Int.of_float pitch
    | json -> invalid_argf "a cell is not a number: %s" (Json.to_string json) ()
  in
  let cells = json |> Json.Util.to_list |> List.map ~f:cell_of_json in
  if List.length cells <> voices
  then invalid_argf "a step holds %d cells, not four" (List.length cells) ();
  cells
;;

(* The legal transpositions of one piece: every shift that keeps each voice inside the
   observed range of its voice. The bounds intersect over the voices in closed form; a
   piece inside the ranges always keeps shift zero, and a silent piece takes it alone. *)
let legal_shifts_of_cells cells =
  let widen ranges step =
    List.iteri step ~f:(fun voice pitch ->
      if pitch >= 0
      then (
        let range =
          match ranges.(voice) with
          | None -> pitch, pitch
          | Some (low, high) -> min low pitch, max high pitch
        in
        ranges.(voice) <- Some range))
  in
  let ranges = Array.create ~len:voices None in
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

let chorale_of_json json =
  let cells = json |> Json.Util.to_list |> List.map ~f:cells_of_json in
  { cells = Array.of_list cells; legal_shifts = legal_shifts_of_cells cells }
;;

let load ~path =
  let json = Json.from_file path in
  let chorales name =
    json |> Json.Util.member name |> Json.Util.to_list |> List.map ~f:chorale_of_json
  in
  { train = chorales "train"; valid = chorales "valid"; test = chorales "test" }
;;

let transpose ~by { cells; legal_shifts } =
  let transpose_cell pitch =
    if pitch < 0
    then pitch
    else (
      match pitch + by with
      | moved when moved < 0 || moved > 127 ->
        invalid_argf "pitch %d moved by %d leaves 0 to 127" pitch by ()
      | moved -> moved)
  in
  { cells = Array.map cells ~f:(List.map ~f:transpose_cell)
  ; legal_shifts = List.map legal_shifts ~f:(fun shift -> shift - by)
  }
;;

(* The frame of one step. [Frame] owns the word — the voice code, the seat order and the
   decode — and this reader only turns the order around: the file gives the soprano first
   and seat 0 is the lowest voice, thus the reversed cells land the bass in the low byte
   and the soprano in seat 3. *)
let frame_of_cells cells =
  List.rev cells |> List.map ~f:Frame.code_of_pitch |> Frame.of_codes
;;

let silent_frame = Frame.silent

(* the pitches that sound in one step, as a set: a unison is one pitch, as it is on the
   wire, and a rest is no pitch. The metre reads this and nothing else does — the frame
   keeps the voices apart. *)
let sonority cells = Set.of_list (module Int) (List.filter cells ~f:(fun p -> p >= 0))

(* The cadential holds of one chorale, from the corpus study of 2026-08-06: a sonority
   that rings six steps or more marks a cadence, and the cadences sit on the downbeats.
   The result is the start step of each hold, ascending.

   A hold is a run of one sonority and not a run of one frame. An exchange of two voices
   keeps the sonority and breaks the frame, thus the frame would cut a hold that the ear
   hears whole and move the vote. *)
let cadential_holds cells =
  let sonorities = Array.to_list (Array.map cells ~f:sonority) in
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
   of 2026-08-14 in docs/transformer_model.md settles this: 27 of the 382 pieces vote for
   a 12-step bar, they hold 8.1 percent of the steps, and a rolling clock of 16 steps
   still puts 44.6 percent of their cadences on a downbeat. To remove them buys 1.9 points
   of alignment over the whole corpus and costs those steps. *)
let rotation cells =
  let holds = cadential_holds cells in
  if List.length holds < 3
  then 0
  else
    List.range 0 bar_steps
    |> List.max_elt ~compare:(fun a b -> Int.compare (vote holds a) (vote holds b))
    |> Option.value_exn
;;

(* The state of the placement: the count of steps laid down, and those steps with the
   newest first. *)
type placement =
  { at : int
  ; behind : int list
  }

let empty_steps count frames = List.init count ~f:(fun (_ : int) -> silent_frame) @ frames

(* One piece and the seam before it. The gap is the smallest count of empty steps that
   puts the downbeats of the piece on the clock. It is never zero after another piece,
   because the release of the piece before it needs one step; the stream itself may open
   with no silence.

   Every piece is a whole number of quarter notes and every rotation is one, thus the gap
   is 4, 8, 12 or 16 steps. The quiet of a seam is never shorter than a quarter note and
   never longer than a bar, and no rule states this. *)
let place state chorale =
  let lead = (-rotation chorale.cells - state.at) % bar_steps in
  let gap = if state.at = 0 then lead else if lead = 0 then bar_steps else lead in
  { at = state.at + gap + Array.length chorale.cells
  ; behind =
      Array.fold
        chorale.cells
        ~init:(empty_steps gap state.behind)
        ~f:(fun behind cells -> frame_of_cells cells :: behind)
  }
;;

let pack chorales =
  let placed = List.fold chorales ~init:{ at = 0; behind = [] } ~f:place in
  (* the stream closes as a seam does: it leaves no chord sounding, and it ends on a bar
     boundary *)
  let tail =
    match -placed.at % bar_steps with
    | 0 -> bar_steps
    | gap -> gap
  in
  let frames = Array.of_list (List.rev (empty_steps tail placed.behind)) in
  { frames
  ; positions = Array.init (Array.length frames) ~f:(fun step -> step % window_steps)
  }
;;

(* One stream is one draw of the transpositions, thus a split needs more than one. The
   first is the canonical stream, and every referee reads it alone, thus a measurement
   over it stays deterministic. *)
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
  (* no code is reserved: the ends of the MIDI range pass through unmoved *)
  print_s ([%sexp_of: int list] (cells "[127, 64, 55, 0]"));
  [%expect {| (127 64 55 0) |}];
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

let%expect_test "the frame of one step" =
  let frame json =
    printf "%08x\n" (frame_of_cells (cells_of_json (Json.from_string json)))
  in
  (* the soprano takes the high byte and the bass the low one: the file order turns around *)
  frame "[74, 70, 65, 58]";
  [%expect {| cac6c1ba |}];
  (* a rest is 0x00, thus the silent voice carries no pitch *)
  frame "[74, -1, 65, 58]";
  [%expect {| ca00c1ba |}];
  (* a silent step is the word zero, thus a cleared context reads as silence *)
  frame "[-1, -1, -1, -1]";
  [%expect {| 00000000 |}]
;;

let%expect_test "the rotation from the cadential holds" =
  (* a bar = a six-step hold on the downbeat, then moving single notes *)
  let chord = [ 67; 64; 60; -1 ] in
  let hold = Array.create ~len:6 chord in
  let motion count = Array.init count ~f:(fun i -> [ 40 + i; -1; -1; -1 ]) in
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

let%expect_test "the packed stream of two pieces" =
  (* Two pieces of four steps, each one chord held. Neither holds three cadences, thus
     both take rotation zero and both must open on a downbeat of the clock. *)
  let chorale cells = { cells = Array.create ~len:4 cells; legal_shifts = [ 0 ] } in
  let { frames; positions } =
    pack [ chorale [ 67; 64; 60; -1 ]; chorale [ 69; 65; 62; -1 ] ]
  in
  (* The first piece opens the stream and takes no seam. The seam between the pieces is
     twelve steps of silence: the release needs no step of its own, because the decode
     makes it from the first silent frame. *)
  print_endline
    (String.concat ~sep:" " (List.map (Array.to_list frames) ~f:(sprintf "%08x")));
  [%expect
    {| c3c0bc00 c3c0bc00 c3c0bc00 c3c0bc00 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 c5c1be00 c5c1be00 c5c1be00 c5c1be00 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 |}];
  (* the coordinate rolls with the stream and never restarts at a piece *)
  print_s ([%sexp_of: int array] positions);
  [%expect
    {|
    (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
     29 30 31)
    |}]
;;
