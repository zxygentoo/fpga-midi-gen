open Base

module Params = struct
  type t =
    { rows : int
    ; root : int
    ; degrees : int
    ; stretch : int
    }
end

(* the C major pentatonic scale, as the semitone of each degree above the octave start *)
let pentatonic = [ 0; 2; 4; 7; 9 ]

(* re-rolls the first [count] rows, with one draw for each row, in ascending order *)
let rec reroll prng rows ~count =
  match rows with
  | _ :: rest when count > 0 ->
    let prng, value = Prng.next prng in
    let prng, rest = reroll prng rest ~count:(count - 1) in
    prng, value :: rest
  | rows -> prng, rows
;;

(* The scale of the model rotated to start on the root of the voice. Therefore a voice
   holds the pitch classes of the one scale, whatever its root, and the rotation is a
   result and not a value that a person writes. *)
let rotate ~scale ~root =
  let pitch_class = root % 12 in
  match List.findi scale ~f:(fun _ degree -> degree = pitch_class) with
  | None -> invalid_arg "Pink: the root is not a pitch class of the scale"
  | Some (start, _) ->
    let scale = Array.of_list scale in
    let length = Array.length scale in
    List.init length ~f:(fun i -> (scale.((start + i) % length) - pitch_class + 12) % 12)
;;

(* one definition serves the reference and the RTL elaboration *)
let degree_offsets ~scale (params : Params.t) =
  if List.is_empty scale then invalid_arg "Pink: the scale is empty";
  let offsets = Array.of_list (rotate ~scale ~root:params.root) in
  let length = Array.length offsets in
  List.init params.degrees ~f:(fun degree ->
    (12 * (degree / length)) + offsets.(degree % length))
;;

let mapper ~scale (params : Params.t) =
  let { Params.rows; root; degrees; stretch } = params in
  let offsets = Array.of_list (degree_offsets ~scale params) in
  let full = rows * 256 in
  let window = full / stretch in
  if window < 1 then invalid_arg "Pink: the stretch window is empty";
  let low = (full - window) / 2 in
  fun sum ->
    let x = Int.clamp_exn (sum - low) ~min:0 ~max:(window - 1) in
    root + offsets.(x * degrees / window)
;;

module Voice = struct
  type t =
    { params : Params.t
    ; restrike : bool
    }
end

type t =
  { scale : int list
  ; voices : Voice.t list
  }

(* The voices from row 0 upward: the soprano takes the fastest rows, the bass the slowest,
   and the partition is the rhythm — a group that starts at row [r] re-articulates every
   [2**r] steps, thus 2+2+2+2 gives the periods 1, 4, 16 and 64. The registers are
   disjoint, and each root is a pitch class of the one scale, thus every voice holds the
   pitch classes of C major pentatonic. *)
let default_voices : Voice.t list =
  [ (* the soprano: every step, A4 to A6, gated by the player *)
    { params = { rows = 2; root = 69; degrees = 11; stretch = 2 }; restrike = true }
  ; (* the alto: every 4th step, C4 to G4 *)
    { params = { rows = 2; root = 60; degrees = 4; stretch = 2 }; restrike = true }
  ; (* the tenor: every 16th step, C3 to A3 *)
    { params = { rows = 2; root = 48; degrees = 5; stretch = 2 }; restrike = false }
  ; (* the bass: every 64th step, A1 to A2; it speaks only when it moves *)
    { params = { rows = 2; root = 33; degrees = 6; stretch = 2 }; restrike = false }
  ]
;;

let default = { scale = pentatonic; voices = default_voices }

type state =
  { note : int
  ; due : bool
  }

(* the state of one run: the model, the PRNG and the row values *)
type walk =
  { model : t
  ; prng : Prng.t
  ; rows : int list (* the row values; the head is row 0 *)
  ; step : int (* the number of the steps taken *)
  }

let validate ~scale (params : Params.t) =
  let { Params.rows; root; degrees; stretch } = params in
  if rows < 1 || degrees < 1 || stretch < 1
  then invalid_arg "Pink: rows, degrees and stretch must be at least 1";
  if rows * 256 / stretch < 1 then invalid_arg "Pink: the stretch window is empty";
  List.iter (degree_offsets ~scale params) ~f:(fun offset ->
    if root + offset < 0 || root + offset > 127
    then invalid_arg "Pink: a degree gives a note outside 0 to 127")
;;

let total_rows voices = List.sum (module Int) voices ~f:(fun v -> v.Voice.params.rows)

let create ~model ~seed =
  if List.is_empty model.scale then invalid_arg "Pink: the scale is empty";
  if List.is_empty model.voices then invalid_arg "Pink: the model needs a voice";
  List.iter model.voices ~f:(fun v -> validate ~scale:model.scale v.Voice.params);
  let total = total_rows model.voices in
  let prng = Prng.create ~seed in
  let prng, rows = reroll prng (List.init total ~f:(fun _ -> 0)) ~count:total in
  { model; prng; rows; step = 0 }
;;

let next_step w =
  let step = w.step + 1 in
  let count = Int.min (total_rows w.model.voices) (Int.ctz step + 1) in
  let prng, rows = reroll w.prng w.rows ~count in
  (* walk the groups from row 0: a voice is due when the re-roll count reaches into its
     group *)
  let states =
    List.folding_map w.model.voices ~init:(0, rows) ~f:(fun (start, remaining) v ->
      let params = v.Voice.params in
      let mine = List.take remaining params.rows in
      let rest = List.drop remaining params.rows in
      let sum = List.sum (module Int) mine ~f:Fn.id in
      ( (start + params.rows, rest)
      , { note = mapper ~scale:w.model.scale params sum; due = step = 1 || count > start }
      ))
  in
  { w with prng; rows; step }, List.rev states
;;

(* the note stream of a one-voice model: the tests below use it to show the shape of the
   mapping of one voice *)
let notes ~model ~seed =
  Sequence.unfold ~init:(create ~model ~seed) ~f:(fun w ->
    let w, states = next_step w in
    Some ((List.hd_exn states).note, w))
;;

(* the histogram makes the shape of the mapping visible: the sum of the rows of a voice
   concentrates in the middle of its range, thus a map of the full range uses the outer
   degrees rarely, and the stretch widens their use. The soprano has the largest register
   of the four voices, thus it shows the shape best. *)
let%expect_test "the note histogram of the soprano, stretch 1 against stretch 2" =
  let soprano = (List.hd_exn default_voices).Voice.params in
  List.iter [ 1; 2 ] ~f:(fun stretch ->
    Stdio.printf "stretch %d:\n" stretch;
    let params = { soprano with Params.stretch } in
    let model = { default with voices = [ { Voice.params; restrike = true } ] } in
    notes ~model ~seed:Control.Default.seed
    |> (fun sequence -> Sequence.take sequence 4096)
    |> Sequence.to_list
    |> List.sort_and_group ~compare:Int.compare
    |> List.iter ~f:(fun group ->
      let note = List.hd_exn group in
      let count = List.length group in
      Stdio.printf "  %3d %s\n" note (String.make (count / 32) '#')));
  [%expect
    {|
    stretch 1:
       69 ##
       72 ######
       74 ###########
       76 ###############
       79 #################
       81 #######################
       84 ###################
       86 ##############
       88 ##########
       91 ######
       93 #
    stretch 2:
       69 #######################
       72 #######
       74 ########
       76 #########
       79 ##########
       81 ###########
       84 ##########
       86 #########
       88 ########
       91 #######
       93 #####################
    |}]
;;

let%expect_test "the articulation grid is the ctz schedule" =
  let t = ref (create ~model:default ~seed:7) in
  let due_steps = Array.create ~len:(List.length default_voices) [] in
  for step = 1 to 128 do
    let t', states = next_step !t in
    t := t';
    List.iteri states ~f:(fun k s -> if s.due then due_steps.(k) <- step :: due_steps.(k))
  done;
  let count k = List.length due_steps.(k) in
  Stdio.printf
    "bass %d (every 64), tenor %d (every 16), alto %d (every 4), soprano %d (every step)\n"
    (count 0)
    (count 1)
    (count 2)
    (count 3);
  Stdio.printf
    "bass at steps %s\n"
    (String.concat ~sep:" " (List.rev_map due_steps.(0) ~f:Int.to_string));
  [%expect
    {|
    bass 3 (every 64), tenor 9 (every 16), alto 33 (every 4), soprano 128 (every step)
    bass at steps 1 64 128
    |}]
;;
