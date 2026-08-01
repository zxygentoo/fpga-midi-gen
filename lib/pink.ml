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
  let default = { rows = 8; root = 60; degrees = 15; scale = pentatonic; stretch = 2 }
end

type t =
  { params : Params.t
  ; map : int -> int (* the sum-to-note map, from [mapper] *)
  ; prng : Prng.t
  ; rows : int list (* the row values; the head is row 0 *)
  ; step : int (* the number of the steps taken *)
  }

(* re-rolls the first [count] rows, with one draw for each row, in ascending order *)
let rec reroll prng rows ~count =
  match rows with
  | _ :: rest when count > 0 ->
    let prng, value = Prng.next prng in
    let prng, rest = reroll prng rest ~count:(count - 1) in
    prng, value :: rest
  | rows -> prng, rows
;;

(* one definition serves [create] and the RTL elaboration *)
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

let create (params : Params.t) ~seed =
  let { Params.rows; root; degrees; scale; stretch } = params in
  if List.is_empty scale then invalid_arg "Pink.create: the scale is empty";
  if rows < 1 || degrees < 1 || stretch < 1
  then invalid_arg "Pink.create: rows, degrees and stretch must be at least 1";
  if rows * 256 / stretch < 1 then invalid_arg "Pink.create: the stretch window is empty";
  List.iter (degree_offsets params) ~f:(fun offset ->
    if root + offset < 0 || root + offset > 127
    then invalid_arg "Pink.create: a degree gives a note outside 0 to 127");
  let prng = Prng.create ~seed in
  let prng, rows = reroll prng (List.init rows ~f:(fun _ -> 0)) ~count:rows in
  { params; map = mapper params; prng; rows; step = 0 }
;;

let next_note t =
  let step = t.step + 1 in
  let count = Int.min (List.length t.rows) (Int.ctz step + 1) in
  let prng, rows = reroll t.prng t.rows ~count in
  let sum = List.sum (module Int) rows ~f:Fn.id in
  { t with prng; rows; step }, t.map sum
;;

let notes params ~seed =
  Sequence.unfold ~init:(create params ~seed) ~f:(fun t ->
    let t, note = next_note t in
    Some (note, t))
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
