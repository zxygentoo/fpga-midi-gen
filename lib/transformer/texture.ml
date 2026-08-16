open Core

type window =
  { onsets : float
  ; single_on : float
  ; median_duration : float
  ; under_a_quarter : float
  }

(* a quarter note on the sixteenth grid: the length the decayed walk of the era never went
   below *)
let quarter_note = Jsb.bar_steps / 4

let ons_of_step step =
  List.count step ~f:(function
    | Token.On (_ : int) -> true
    | Token.Start | Token.Off _ | Token.End -> false)
;;

(* The notes of a walk: the step each one starts in, and the steps it sounds. A note that
   still sounds at the end has no duration and drops — four voices at the most, against
   the thousands of notes of a long walk. A pitch that starts again before it stops cannot
   happen: the grammar refuses an ON of a sounding pitch. *)
let notes music =
  let opened = Hashtbl.create (module Int) in
  let closed = Queue.create () in
  List.iteri music ~f:(fun step sentence ->
    List.iter sentence ~f:(function
      | Token.On pitch -> Hashtbl.set opened ~key:pitch ~data:step
      | Token.Off pitch ->
        Option.iter (Hashtbl.find_and_remove opened pitch) ~f:(fun start ->
          Queue.enqueue closed (start, step - start))
      | Token.Start | Token.End -> ()));
  Queue.to_list closed
;;

let median values =
  let sorted = Array.of_list values in
  Array.sort sorted ~compare:Float.compare;
  match Array.length sorted with
  | 0 -> Float.nan
  | count when count % 2 = 1 -> sorted.(count / 2)
  | count ->
    let high = count / 2 in
    let low = high - 1 in
    Float.((sorted.(low) + sorted.(high)) / 2.)
;;

let windows music ~span =
  let count = (List.length music + span - 1) / span in
  (* a note belongs to the window its onset falls in, thus each one counts one time *)
  let durations = Array.create ~len:(max 1 count) [] in
  List.iter (notes music) ~f:(fun (start, duration) ->
    let index = start / span in
    durations.(index) <- Float.of_int duration :: durations.(index));
  List.mapi (List.chunks_of music ~length:span) ~f:(fun index block ->
    let steps = Float.of_int (max 1 (List.length block)) in
    let share count = Float.of_int count /. steps in
    let sounded = durations.(index) in
    let short =
      List.count sounded ~f:(fun steps -> Float.(steps < of_int quarter_note))
    in
    { onsets = share (List.sum (module Int) block ~f:ons_of_step)
    ; single_on = share (List.count block ~f:(fun step -> ons_of_step step = 1))
    ; median_duration = median sounded
    ; under_a_quarter = Float.of_int short /. Float.of_int (max 1 (List.length sounded))
    })
;;

let steps_of_codes codes =
  let (_ : Token.t list), steps =
    Array.fold codes ~init:([], []) ~f:(fun (current, steps) code ->
      match Token.of_code code with
      | Token.End -> [], List.rev current :: steps
      | token -> token :: current, steps)
  in
  List.rev steps
;;

let%expect_test "the texture of a walk it knows" =
  (* Four steps of one chord, then four of silence, twice over: the seam of a packed
     stream in miniature. Each window holds one of the two. *)
  let chord = [ Token.On 67; Token.On 64; Token.On 60 ] in
  let release = [ Token.Off 60; Token.Off 64; Token.Off 67 ] in
  let held = List.init 3 ~f:(fun (_ : int) -> []) in
  let quiet = List.init 3 ~f:(fun (_ : int) -> []) in
  let piece = ((chord :: held) @ (release :: quiet) : Token.t list list) in
  let music = piece @ piece in
  let show index { onsets; single_on; median_duration; under_a_quarter } =
    printf
      "window %d  onsets/step %.3f  single-ON %.0f%%  median duration %.1f  under a \
       quarter %.2f\n"
      index
      onsets
      (100. *. single_on)
      median_duration
      under_a_quarter
  in
  List.iteri (windows music ~span:8) ~f:show;
  (* three ONs over eight steps; no step holds exactly one ON; each note sounds four
     steps, from its ON to the OFF of the release step *)
  [%expect
    {|
    window 0  onsets/step 0.375  single-ON 0%  median duration 4.0  under a quarter 0.00
    window 1  onsets/step 0.375  single-ON 0%  median duration 4.0  under a quarter 0.00
    |}];
  (* one window over the whole walk holds both chords, and the count of notes doubles *)
  List.iteri (windows music ~span:16) ~f:show;
  [%expect
    {| window 0  onsets/step 0.375  single-ON 0%  median duration 4.0  under a quarter 0.00 |}];
  (* a walk that never releases: the open notes drop, and no duration is left to take *)
  List.iteri (windows (chord :: held) ~span:8) ~f:show;
  [%expect
    {| window 0  onsets/step 0.750  single-ON 0%  median duration nan  under a quarter 0.00 |}]
;;

let%expect_test "the codes of a stream become its steps" =
  let codes =
    Array.of_list_map
      [ Token.On 60; Token.End; Token.End; Token.Off 60; Token.End ]
      ~f:Token.to_code
  in
  print_s ([%sexp_of: Token.t list list] (steps_of_codes codes));
  [%expect {| (((On 60)) () ((Off 60))) |}]
;;
