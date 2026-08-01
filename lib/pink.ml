open Base

module Prng = struct
  type t = int

  let mask = 0xFFFF_FFFF

  let create ~seed =
    if seed = 0 || seed land mask <> seed
    then invalid_arg "Pink.Prng.create: the seed must fit 32 bits and must not be 0";
    seed
  ;;

  let next t =
    let t = t lxor (t lsl 13) land mask in
    let t = t lxor (t lsr 17) in
    let t = t lxor (t lsl 5) land mask in
    t, t land 0xff
  ;;

  let state t = t
end

module Params = struct
  type t =
    { rows : int
    ; root : int
    ; degrees : int
    ; scale : int list
    ; stretch : int
    }

  let pentatonic = [ 0; 2; 4; 7; 9 ]
  let a_pentatonic = [ 0; 3; 5; 7; 10 ]
  let default = { rows = 8; root = 60; degrees = 15; scale = pentatonic; stretch = 2 }
end

(* re-rolls the first [count] rows, with one draw for each row, in ascending order *)
let rec reroll prng rows ~count =
  match rows with
  | _ :: rest when count > 0 ->
    let prng, value = Prng.next prng in
    let prng, rest = reroll prng rest ~count:(count - 1) in
    prng, value :: rest
  | rows -> prng, rows
;;

(* one definition serves the reference and the RTL elaboration *)
let degree_offsets (params : Params.t) =
  if List.is_empty params.scale then invalid_arg "Pink.degree_offsets: the scale is empty";
  let scale = Array.of_list params.scale in
  let length = Array.length scale in
  List.init params.degrees ~f:(fun degree ->
    (12 * (degree / length)) + scale.(degree % length))
;;

let mapper (params : Params.t) =
  let { Params.rows; root; degrees; stretch; scale = _ } = params in
  let offsets = Array.of_list (degree_offsets params) in
  let full = rows * 256 in
  let window = full / stretch in
  if window < 1 then invalid_arg "Pink.mapper: the stretch window is empty";
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

(* The voices from row 0 upward: the soprano takes the fastest rows, the bass the slowest,
   and the partition is the rhythm — a group that starts at row [r] re-articulates every
   [2**r] steps, thus 2+2+2+2 gives the periods 1, 4, 16 and 64. The registers are
   disjoint, and the A-rooted ones take the rotation of the pentatonic that starts on A,
   thus every voice stays on the pitch classes of C major pentatonic. *)
let default_voices : Voice.t list =
  [ (* the soprano: every step, A4 to A6, gated by the player *)
    { params =
        { rows = 2; root = 69; degrees = 11; scale = Params.a_pentatonic; stretch = 2 }
    ; restrike = true
    }
  ; (* the alto: every 4th step, C4 to G4 *)
    { params =
        { rows = 2; root = 60; degrees = 4; scale = Params.pentatonic; stretch = 2 }
    ; restrike = true
    }
  ; (* the tenor: every 16th step, C3 to A3 *)
    { params =
        { rows = 2; root = 48; degrees = 5; scale = Params.pentatonic; stretch = 2 }
    ; restrike = false
    }
  ; (* the bass: every 64th step, A1 to A2; it speaks only when it moves *)
    { params =
        { rows = 2; root = 33; degrees = 6; scale = Params.a_pentatonic; stretch = 2 }
    ; restrike = false
    }
  ]
;;

type state =
  { note : int
  ; due : bool
  }

type t =
  { voices : Voice.t list
  ; prng : Prng.t
  ; rows : int list (* the row values; the head is row 0 *)
  ; step : int (* the number of the steps taken *)
  }

let validate (params : Params.t) =
  let { Params.rows; root; degrees; scale; stretch } = params in
  if List.is_empty scale then invalid_arg "Pink.create: the scale is empty";
  if rows < 1 || degrees < 1 || stretch < 1
  then invalid_arg "Pink.create: rows, degrees and stretch must be at least 1";
  if rows * 256 / stretch < 1 then invalid_arg "Pink.create: the stretch window is empty";
  List.iter (degree_offsets params) ~f:(fun offset ->
    if root + offset < 0 || root + offset > 127
    then invalid_arg "Pink.create: a degree gives a note outside 0 to 127")
;;

let total_rows voices = List.sum (module Int) voices ~f:(fun v -> v.Voice.params.rows)

let create ~voices ~seed =
  if List.is_empty voices then invalid_arg "Pink.create: the model needs a voice";
  List.iter voices ~f:(fun v -> validate v.Voice.params);
  let total = total_rows voices in
  let prng = Prng.create ~seed in
  let prng, rows = reroll prng (List.init total ~f:(fun _ -> 0)) ~count:total in
  { voices; prng; rows; step = 0 }
;;

let next_step t =
  let step = t.step + 1 in
  let count = Int.min (total_rows t.voices) (Int.ctz step + 1) in
  let prng, rows = reroll t.prng t.rows ~count in
  (* walk the groups from row 0: a voice is due when the re-roll count reaches into its
     group *)
  let states =
    List.folding_map t.voices ~init:(0, rows) ~f:(fun (start, remaining) v ->
      let params = v.Voice.params in
      let mine = List.take remaining params.rows in
      let rest = List.drop remaining params.rows in
      let sum = List.sum (module Int) mine ~f:Fn.id in
      ( (start + params.rows, rest)
      , { note = mapper params sum; due = step = 1 || count > start } ))
  in
  { t with prng; rows; step }, List.rev states
;;

(* the one-voice model: the whole row set in one group, the shape of the shipped circuit *)
let notes params ~seed =
  let voices = [ { Voice.params; restrike = true } ] in
  Sequence.unfold ~init:(create ~voices ~seed) ~f:(fun t ->
    let t, states = next_step t in
    Some ((List.hd_exn states).note, t))
;;

let%expect_test "the prng walk from seed 1" =
  let rec walk prng count =
    if count > 0
    then (
      let prng, value = Prng.next prng in
      Stdio.printf "%08x %02x\n" (Prng.state prng) value;
      walk prng (count - 1))
  in
  walk (Prng.create ~seed:1) 4;
  [%expect {|
    00042021 21
    04080601 01
    9dcca8c5 c5
    1255994f 4f
    |}]
;;

let%expect_test "the first notes of the power-on state" =
  let first =
    notes Params.default ~seed:Control.Default.seed
    |> (fun sequence -> Sequence.take sequence 16)
    |> Sequence.to_list
  in
  Stdio.print_s ([%sexp_of: int list] first);
  [%expect {| (86 84 84 91 93 93 93 81 84 86 91 88 91 93 93 79) |}]
;;

(* the histogram makes the shape of the mapping visible: the sum of 8 uniform bytes
   concentrates in the middle, and the stretch widens the use of the outer degrees *)
let%expect_test "the note histogram, stretch 1 against stretch 2" =
  List.iter [ 1; 2 ] ~f:(fun stretch ->
    let params = { Params.default with stretch } in
    Stdio.printf "stretch %d:\n" stretch;
    notes params ~seed:Control.Default.seed
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
       64
       67 #
       69 #####
       72 ##############
       74 #########################
       76 ###############################
       79 ##########################
       81 ###############
       84 #####
       86 #
       88
    stretch 2:
       60 ##
       62 ##
       64 ####
       67 #######
       69 #########
       72 #############
       74 ##############
       76 ################
       79 ##############
       81 ############
       84 ###########
       86 #######
       88 ####
       91 ##
       93 ##
    |}]
;;

let%expect_test "the first steps of the power-on seed" =
  let t = ref (create ~voices:default_voices ~seed:Control.Default.seed) in
  Stdio.printf "step   bass   tenor   alto  soprano\n";
  for step = 1 to 10 do
    let t', states = next_step !t in
    t := t';
    Stdio.printf
      "%4d %s\n"
      step
      (String.concat
         (List.map states ~f:(fun s ->
            Printf.sprintf "  %3d%s" s.note (if s.due then "*" else " "))))
  done;
  [%expect
    {|
    step   bass   tenor   alto  soprano
       1    45*   57*   62*   88*
       2    45    57    62    84*
       3    45    57    62    84*
       4    45    57    67*   88*
       5    45    57    67    93*
       6    45    57    67    91*
       7    45    57    67    93*
       8    45    57    62*   74*
       9    45    57    62    76*
      10    45    57    62    86*
    |}]
;;

let%expect_test "the articulation grid is the ctz schedule" =
  let t = ref (create ~voices:default_voices ~seed:7) in
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
