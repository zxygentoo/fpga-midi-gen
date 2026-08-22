(* Integration test: the quantized engine against the float model, on teacher-forced walks
   of drawn weights — [Params.init] and the quantization inside [Drift.walk] read the same
   draw, thus the comparison isolates the fixed-point scheme, and the test reads no file
   that git ignores. The randomness is pseudo-randomness with the seed an input, per the
   project rule, thus both parts are deterministic.

   THE WALK LENGTH IS THE POINT HERE, and it is what parts this gate from era four's. That
   circuit held a window: an arithmetic error entered a ring row, weighed on the steps
   that could still see it, and died. This one holds a state that carries forward, thus an
   error that a short walk hides is exactly the error this era was warned about. The fixed
   sweep therefore runs a walk of 1 024 steps beside the short ones — past many decay
   lifetimes — and both models take one step for one step, which is what makes that
   affordable at all.

   Two parts. The fixed sweep pins measured numbers in the expected file, not thresholds:
   a diff says the integers moved — judge whether it is a re-measurement or a bug. The
   QCheck property draws seed pairs in the idiom of [Cobs] — a fixed rand seed keeps the
   run deterministic — and holds the floors; the printed minima keep the calibration
   honest. *)

open Base
open Stdio
module Mamba = Mgen_mamba.Mamba
module Quantized = Mgen_mamba.Quantized

let weight_seeds = [ 11; 23; 37; 41 ]
let walk_seeds = [ 42; 43; 44; 45 ]

(* the walk runs well past the lead-in of one bar, thus every trial draws *)
let steps = 64

(* The whole plan of the era at a shape a test can afford: two blocks, the Zamba head and
   the feed-forward. The head brings a SECOND source of drift that era five's trunk did
   not have — a coarse ring, a softmax and a division — thus the report answers for the
   whole model and not for the recurrence alone. *)
let config =
  { Mamba.Config.d = 16
  ; d_in = 32
  ; heads = 2
  ; state = 8
  ; taps = 4
  ; plan = [| Block; Block; Attention; Feed_forward |]
  ; span = Mamba.elected_span
  ; ring = 16
  }
;;

(* the walks of one model, summed; the sharpest cosine signal is the lowest walk *)
type tally =
  { draws : int
  ; same_peak : int
  ; same_draw : int
  ; low_cosine : float
  }

let report weight_seed =
  let params = Mamba.Params.init config ~seed:weight_seed in
  let add_walk tally walk_seed =
    let { Quantized.Drift.steps = (_ : int)
        ; draws
        ; same_peak
        ; same_draw
        ; mean_cosine
        ; dt_clamped = (_ : float)
        ; beta_clamped = (_ : float)
        ; state_clamped = (_ : float)
        }
      =
      Quantized.Drift.walk config params ~steps ~seed:walk_seed
    in
    { draws = tally.draws + draws
    ; same_peak = tally.same_peak + same_peak
    ; same_draw = tally.same_draw + same_draw
    ; low_cosine = Float.min tally.low_cosine mean_cosine
    }
  in
  let sum =
    List.fold
      walk_seeds
      ~init:{ draws = 0; same_peak = 0; same_draw = 0; low_cosine = 1.0 }
      ~f:add_walk
  in
  printf
    "weights %d over %d walks: top-1 %d/%d  same draw %d/%d  low cosine %.4f\n"
    weight_seed
    (List.length walk_seeds)
    sum.same_peak
    sum.draws
    sum.same_draw
    sum.draws
    sum.low_cosine
;;

(* THE LONG WALK, and the clamps under it.

   The trap of the era, stated in docs/mamba.md: an error in the state carries forward
   where an attention error died with its step, thus a walk of one window's length proves
   less here than it proved there. This runs the same model out to 1 024 steps — the decay
   of a trained head empties its state in tens of steps, thus this is many lifetimes — and
   prints the drift at each length. A cumulative error would show as a number that falls
   with the length. It does not: measured 2026-08-20 and again with the Zamba head in the
   plan, the cosine holds to three decimals from 64 steps to 1 024.

   The clamps are printed beside it because the formats of this era were chosen with
   margin and not metered on a trained checkpoint. Every one of them reads zero here, and
   that is the finding: the margins hold, and the format round the design document defers
   is deferred on evidence and not on hope. *)
let long_walk () =
  let params = Mamba.Params.init config ~seed:11 in
  List.iter [ 64; 256; 1024 ] ~f:(fun steps ->
    let { Quantized.Drift.steps = (_ : int)
        ; draws
        ; same_peak
        ; same_draw = (_ : int)
        ; mean_cosine
        ; dt_clamped
        ; beta_clamped
        ; state_clamped
        }
      =
      Quantized.Drift.walk config params ~steps ~seed:42
    in
    printf
      "%4d steps: top-1 %.3f  cosine %.4f  clamped dt %.4f beta %.4f state %.4f\n"
      steps
      (Float.of_int same_peak /. Float.of_int draws)
      mean_cosine
      dt_clamped
      beta_clamped
      state_clamped)
;;

(* The floors, calibrated 2026-08-20 on this model's own first measured minima, which is
   the rule era four's floors were set by. The trunk alone read 0.938, 0.974 and 0.9987
   over the trial set; with the Zamba head in the plan it reads 0.875, 0.979 and 0.9972.
   The floors sit far under both, thus a fail is a break of the scheme and not a re-draw
   of the set — and the counterexample prints its seed pair.

   THE HEAD IS NOW THE LARGER SOURCE OF DRIFT, and it is one format: the key and value
   ring keeps the top byte of a Q12 row, which is era four's ring carried over whole. The
   state of the recurrence keeps its int16 because a state error accumulates; a ring error
   dies with its window, and era four shipped six such rings. The block RAM is there to
   widen it if a later round wants the top-1 share back — 3.5 tiles of 135 at the elected
   depth — and this number is what that round would be buying.

   The floors are still much tighter than era four's 0.55, 0.8 and 0.98, and the reason is
   a format and not a virtue: this datapath keeps the gate product whole into the norm
   that reads it, where a truncation back to the working class cost 0.10 of the cosine on
   its own. A scheme that measures this well must be held to it. *)
let top1_floor = 0.80
let same_draw_floor = 0.90
let cosine_floor = 0.99

let check_floors () =
  let low_top1 = ref 1.0 in
  let low_draw = ref 1.0 in
  let low_cosine = ref 1.0 in
  let holds (weight_seed, walk_seed) =
    let params = Mamba.Params.init config ~seed:weight_seed in
    let { Quantized.Drift.steps = (_ : int)
        ; draws
        ; same_peak
        ; same_draw
        ; mean_cosine
        ; dt_clamped = (_ : float)
        ; beta_clamped = (_ : float)
        ; state_clamped = (_ : float)
        }
      =
      Quantized.Drift.walk config params ~steps ~seed:walk_seed
    in
    let share count = Float.of_int count /. Float.of_int draws in
    low_top1 := Float.min !low_top1 (share same_peak);
    low_draw := Float.min !low_draw (share same_draw);
    low_cosine := Float.min !low_cosine mean_cosine;
    Float.(share same_peak > top1_floor)
    && Float.(share same_draw > same_draw_floor)
    && Float.(mean_cosine > cosine_floor)
  in
  let open QCheck in
  let seed_pair =
    make
      ~print:Print.(pair int int)
      Gen.(pair (int_range 1 1_000_000) (int_range 1 1_000_000))
  in
  Test.check_exn
    ~rand:(Stdlib.Random.State.make [| 0xD21F7 |])
    (Test.make
       ~count:60
       ~name:"the drift floors hold on drawn seed pairs"
       seed_pair
       holds);
  printf
    "60 drawn seed pairs: low top-1 %.3f  low same draw %.3f  low cosine %.4f\n"
    !low_top1
    !low_draw
    !low_cosine
;;

let () =
  List.iter weight_seeds ~f:report;
  long_walk ();
  check_floors ()
;;
