(* The facts of the walk — see diffusion.mli for the contract and docs/diffusion_rtl.md
   for the design. The model itself is in JAX; what stands here is what the CIRCUIT reads,
   and every rule of it is one the RTL must equal rather than restate. *)

open Core

let rows = Vocab.classes
let voices = Frame.voices

(* the 24-bit grid of the generator: a uniform is [k * 2 ** -24], thus [u * grid] is the
   integer [k] and a threshold compare over it is exact in a double *)
let grid = Float.of_int (1 lsl Prng.uniform_bits)

(* The register of each seat: the lowest and the highest pitch it sings anywhere in this
   corpus, seat 0 the bass — [Jsb.voice_ranges] turned around, thus the corpus library's
   own test pins it. The corpus table gives the soprano first, as the file does.

   [opening_canvas] draws inside these, thus a cell the first Bernoulli leaves standing
   states a note a chorale could hold. They are the RANGES of [jax/measure.py], stated as
   pitches. *)
let seat_ranges = Array.of_list (List.rev (Array.to_list Jsb.voice_ranges))

(* the top of the corpus: the classes 1 to 46 cover [Vocab.pitch_low] to here, and the
   spare class stands one above it *)
let pitch_high = Vocab.pitch_low + rows - 3

(* The pitch of one corpus cell becomes its class, and the reader is DELIBERATELY NARROWER
   THAN [Vocab]: the window holds one spare class above the corpus — pitch 82, the draw's
   to state and never the corpus's — thus a file that names it is corrupt and refuses
   loudly rather than filing the fault under the spare. The twin reader of [jax/data.py]
   refuses the same range with the same reason. *)
let class_of_cell pitch =
  if pitch < 0
  then Vocab.silence
  else if pitch < Vocab.pitch_low || pitch > pitch_high
  then
    invalid_argf
      "the pitch %d is outside the corpus's %d to %d"
      pitch
      Vocab.pitch_low
      pitch_high
      ()
  else pitch - Vocab.pitch_low + 1
;;

type opening =
  { low : int
  ; width : int
  }

(* The register of each seat as classes: [seat_ranges] read through [class_of_cell]
   itself, thus the map has ONE home and a circuit that restated [low - pitch_low + 1]
   could not part from it. [opening_canvas] draws inside these and the elaboration of the
   circuit carries them, thus the opening of the walk and the opening of the board are one
   rule. *)
let seat_openings =
  Array.map seat_ranges ~f:(fun (low, high) ->
    { low = class_of_cell low; width = high - low + 1 })
;;

module Config = struct
  type t =
    { layers : int
    ; width : int
    }
end

let tensor_column x ~step ~channel ~channels =
  Array.init rows ~f:(fun row -> x.((((step * rows) + row) * channels) + channel))
;;

(* ==================================================================== *)
(* The walk *)
(* ==================================================================== *)

let anneal_threshold ~step ~walk =
  let low = 0.1
  and high = 0.9 in
  (* the paper's schedule, in the trainer's expression: both language sides evaluate the
     same doubles, thus the thresholds agree bit for bit *)
  let alpha =
    Float.max
      low
      (high -. ((high -. low) *. Float.of_int step /. (0.7 *. Float.of_int walk)))
  in
  Int.of_float (Float.round_down (alpha *. grid))
;;

(* a threshold compare on the 24-bit grid: exact, because [u * grid] is an integer *)
let under ~threshold u = Float.(u *. grid < of_int threshold)

(* The cell order of the walk: one step at a time, and the seats inside a step. Every
   uniform of the walk is drawn in this order and every hidden cell is drawn in it, thus
   the integer twin takes the same walk by taking the same order. *)
let cell_order ~steps = List.cartesian_product (List.range 0 steps) (List.range 0 voices)

(* one uniform for each cell in the cell order, folded into [f] *)
let over_cells state ~steps ~f =
  List.fold (cell_order ~steps) ~init:state ~f:(fun state (step, voice) ->
    let next, value = Prng.run Prng.uniform state in
    f ~step ~voice value;
    next)
;;

let opening_canvas state ~steps =
  let canvas = Array.make_matrix ~dimx:steps ~dimy:voices 0 in
  (* the product [u * width] is exact on the grid, thus the twin and the circuit state the
     same class from the same uniform *)
  let state =
    over_cells state ~steps ~f:(fun ~step ~voice u ->
      let { low; width } = seat_openings.(voice) in
      canvas.(step).(voice)
      <- low + Int.of_float (Float.round_down (u *. Float.of_int width)))
  in
  state, canvas
;;

let hidden_cells state ~steps ~threshold =
  let hidden = Array.make_matrix ~dimx:steps ~dimy:voices false in
  let state =
    over_cells state ~steps ~f:(fun ~step ~voice u ->
      hidden.(step).(voice) <- under ~threshold u)
  in
  state, hidden
;;

let frames_of_canvas canvas =
  Array.map canvas ~f:(fun step -> Vocab.frame_of_classes (Array.to_list step))
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the opening puts every voice inside the register of its seat" =
  (* the draw is over the seat's range and never the whole roll, thus a cell the first
     masks leave standing states a note a chorale could hold *)
  let (_ : Prng.state), canvas = opening_canvas (Prng.create_folded ~seed:9) ~steps:64 in
  let inside =
    Array.for_all canvas ~f:(fun step ->
      Array.for_alli step ~f:(fun voice index ->
        let low, high = seat_ranges.(voice) in
        let pitch = index + Vocab.pitch_low - 1 in
        index <> Vocab.silence && pitch >= low && pitch <= high))
  in
  printf "64 steps of 4 seats, every cell in register: %b\n" inside;
  [%expect {| 64 steps of 4 seats, every cell in register: true |}]
;;

let%expect_test "the anneal thresholds: the paper's schedule on the 24-bit grid" =
  let walk = 512 in
  let at step = anneal_threshold ~step ~walk in
  printf "step 0: %d = floor of 0.9 * 2^24\n" (at 0);
  printf "the floor: %d = floor of 0.1 * 2^24\n" (at (walk - 1));
  let thresholds = Array.init walk ~f:(fun step -> at step) in
  let falling =
    Array.for_alli thresholds ~f:(fun step value ->
      value <= if step = 0 then value else thresholds.(step - 1))
  in
  printf "the schedule only falls: %b\n" falling;
  (* it reaches the floor after the span share of the walk, not at its end *)
  let settles = Int.of_float (Float.round_up (0.7 *. Float.of_int walk)) in
  printf "settled at step %d: %b\n" settles (at settles = at (walk - 1));
  [%expect
    {|
    step 0: 15099494 = floor of 0.9 * 2^24
    the floor: 1677721 = floor of 0.1 * 2^24
    the schedule only falls: true
    settled at step 359: true
    |}]
;;

let%expect_test "a canvas becomes the frames of the wire" =
  (* pitch 60 at the bass and pitch 81 at the soprano: the bass lands in the low byte, the
     two silent seats carry no pitch, and the events follow the rule of the frame *)
  let canvas =
    [| [| class_of_cell 60; Vocab.silence; Vocab.silence; class_of_cell 81 |] |]
  in
  let frames = frames_of_canvas canvas in
  printf "%08x\n" frames.(0);
  [%expect {| d10000bc |}]
;;
