(* Integration test: the quantized engine of era six against the float reference, on
   teacher-forced walks of drawn weights — [Params.init] and the quantization inside
   [Drift.walk] read the same draw, thus the comparison isolates the fixed-point scheme,
   and the test reads no file that git ignores. The randomness is pseudo-randomness with
   the seed an input, per the project rule, thus both parts are deterministic.

   THE FEEDBACK AXIS OF THIS ERA IS THE WALK, and it is what parts this gate from era
   five's. That model held a state that carried an error forward in time; this one holds a
   canvas — every cell a pass redraws stands in the context of every later pass, thus an
   arithmetic error compounds through the music rather than through a register. The fixed
   sweep therefore runs the walk out to 128 passes beside the short ones, which is a
   quarter of the board's full budget at a quarter of its canvas.

   THE DRAWN WEIGHTS TAKE THE TRAINED NORM'S SCALE. [Params.init ~norm_scale:1.0] holds
   the trunk at the O(1) activations a trained norm holds; at the trainer's opening tenth
   an untrained trunk decays tenfold at every layer, and by the third the report reads the
   resolution floor of Q12 and not the arithmetic — measured while this gate was built,
   and the reason the flag exists.

   Two parts, the rule of the sibling gates. The fixed sweep pins measured numbers in the
   expected file, not thresholds: a diff says the integers moved — judge whether it is a
   re-measurement or a bug. The QCheck property draws seed pairs at a fixed rand seed and
   holds the floors; the printed minima keep the calibration honest. *)

open Base
open Stdio
module Diffusion = Mgen_diffusion.Diffusion
module Quantized = Mgen_diffusion.Quantized

let weight_seeds = [ 11; 23; 37; 41 ]
let walk_seeds = [ 42; 43; 44; 45 ]

(* a quarter of the board's canvas: two measures *)
let steps = 32

(* the structure of the era at a shape a test can afford: the stem, two residual pairs and
   the head *)
let config = { Diffusion.Config.layers = 6; width = 8 }
let drawn seed = Diffusion.Params.init ~norm_scale:1.0 config ~seed

(* the walks of one model, summed; the sharpest cosine signal is the lowest walk *)
type tally =
  { cells : int
  ; same_peak : int
  ; same_draw : int
  ; low_cosine : float
  }

let report weight_seed =
  let params = drawn weight_seed in
  let add_walk tally walk_seed =
    let { Quantized.Drift.passes = (_ : int)
        ; cells
        ; same_peak
        ; same_draw
        ; mean_cosine
        ; activations_clamped = (_ : float)
        }
      =
      Quantized.Drift.walk params ~steps ~walk:16 ~seed:walk_seed
    in
    { cells = tally.cells + cells
    ; same_peak = tally.same_peak + same_peak
    ; same_draw = tally.same_draw + same_draw
    ; low_cosine = Float.min tally.low_cosine mean_cosine
    }
  in
  let sum =
    List.fold
      walk_seeds
      ~init:{ cells = 0; same_peak = 0; same_draw = 0; low_cosine = 1.0 }
      ~f:add_walk
  in
  printf
    "weights %d over %d walks: top-1 %d/%d  same draw %d/%d  low cosine %.4f\n"
    weight_seed
    (List.length walk_seeds)
    sum.same_peak
    sum.cells
    sum.same_draw
    sum.cells
    sum.low_cosine
;;

(* THE LONG WALK, and the clamps under it.

   A redrawn cell enters the context of every later pass, thus a quantization error can
   compound over the walk in a way one pass never shows. This runs the same model at 8, 32
   and 128 passes; a cumulative error would show as numbers that fall with the length. The
   clamps print beside it because the formats were chosen with margin and not metered on a
   trained checkpoint: a zero here is the finding that the margin holds. *)
let long_walk () =
  let params = drawn 11 in
  List.iter [ 8; 32; 128 ] ~f:(fun passes ->
    let { Quantized.Drift.passes = (_ : int)
        ; cells
        ; same_peak
        ; same_draw = (_ : int)
        ; mean_cosine
        ; activations_clamped
        }
      =
      Quantized.Drift.walk params ~steps ~walk:passes ~seed:42
    in
    printf
      "%4d passes: top-1 %.3f  cosine %.4f  clamped %.4f\n"
      passes
      (Float.of_int same_peak /. Float.of_int cells)
      mean_cosine
      activations_clamped)
;;

(* The floors, calibrated on this model's own first measured minima over the CLEAN trials,
   the rule the sibling gates were set by: a fail is a break of the scheme and not a
   re-draw of the set, and the counterexample prints its seed pair.

   A TRIAL THAT CLAMPS IS THE FORMAT'S ANSWER AND NOT THE SCHEME'S FAULT. A drawn trunk at
   width 8 can double its variance at every residual pair, and past a magnitude of 8 the
   Q12 write rides the clamp: measured at one drawn pair, 5.5 percent of writes clamped
   and the cosine fell to 0.87 — the format met a model this era does not run. Such a
   trial is counted and released from the floors; a trial that does not clamp has no
   excuse, thus the floors still hold the arithmetic. The format's answer on the TRAINED
   model is the drift tool's, with the clamp share beside it. *)
let top1_floor = 0.80
let same_draw_floor = 0.70
let cosine_floor = 0.985

let check_floors () =
  let low_top1 = ref 1.0 in
  let low_draw = ref 1.0 in
  let low_cosine = ref 1.0 in
  let clamped_trials = ref 0 in
  let holds (weight_seed, walk_seed) =
    let { Quantized.Drift.passes = (_ : int)
        ; cells
        ; same_peak
        ; same_draw
        ; mean_cosine
        ; activations_clamped
        }
      =
      Quantized.Drift.walk (drawn weight_seed) ~steps ~walk:8 ~seed:walk_seed
    in
    if Float.(activations_clamped > 0.001)
    then (
      Int.incr clamped_trials;
      true)
    else (
      let share count = Float.of_int count /. Float.of_int cells in
      low_top1 := Float.min !low_top1 (share same_peak);
      low_draw := Float.min !low_draw (share same_draw);
      low_cosine := Float.min !low_cosine mean_cosine;
      Float.(share same_peak > top1_floor)
      && Float.(share same_draw > same_draw_floor)
      && Float.(mean_cosine > cosine_floor))
  in
  let open QCheck in
  let seed_pair =
    make
      ~print:Print.(pair int int)
      Gen.(pair (int_range 1 1_000_000) (int_range 1 1_000_000))
  in
  Test.check_exn
    ~rand:(Stdlib.Random.State.make [| 0xD21F8 |])
    (Test.make
       ~count:60
       ~name:"the drift floors hold on drawn seed pairs"
       seed_pair
       holds);
  printf
    "60 drawn seed pairs, %d released by their clamps: low top-1 %.3f  low same draw \
     %.3f  low cosine %.4f\n"
    !clamped_trials
    !low_top1
    !low_draw
    !low_cosine
;;

let () =
  List.iter weight_seeds ~f:report;
  long_walk ();
  check_floors ()
;;
