(* Integration test: the quantized engine against the float model, on teacher-forced walks
   of drawn weights — [Params.init] and the quantization inside [Drift.walk] read the same
   draw, thus the comparison isolates the fixed-point scheme, and the test reads no file
   that git ignores. The randomness is pseudo-randomness with the seed an input, per the
   project rule, thus both parts are deterministic. The small shape makes the window the
   seam: a walk of [steps] steps draws at least [steps] tokens, more than [context], thus
   the KV ring always wraps and must agree with the float sampler's window truncation.

   Two parts. The fixed sweep pins measured numbers in the expected file, not thresholds:
   a diff says the integers moved — judge whether it is a re-measurement or a bug. The
   QCheck property draws seed pairs in the idiom of [Cobs] — a fixed rand seed keeps the
   run deterministic — and holds the floors; the printed minima keep the calibration
   honest. *)

open Base
open Stdio
module Quantized = Mgen_transformer.Quantized
module Transformer = Mgen_transformer.Transformer

let weight_seeds = [ 11; 23; 37; 41 ]
let walk_seeds = [ 42; 43; 44; 45 ]
let steps = 24
let context = 16
let config = { Transformer.Config.d = 16; layers = 2; heads = 4; context; slope_span = 8 }

(* the walks of one model, summed; the sharpest cosine signal is the lowest walk *)
type tally =
  { draws : int
  ; same_peak : int
  ; same_draw : int
  ; low_cosine : float
  }

let report weight_seed =
  let params = Transformer.Params.init config ~seed:weight_seed in
  let add_walk tally walk_seed =
    let { Quantized.Drift.draws; events = (_ : int); same_peak; same_draw; mean_cosine } =
      Quantized.Drift.walk config params ~steps ~seed:walk_seed
    in
    (* the seam under test is the window: every walk must wrap the ring *)
    assert (draws > context);
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

(* The floors of the property, with the calibration of 2026-08-13 — the int8 KV ring moved
   the minima down, most at this small shape: a head averages four lanes, thus the ring's
   coarse byte weighs about twice what the board's shape feels. The minima over this trial
   set are the printed line of the expected file, and the floors sit far under them. Thus
   a fail is a break of the scheme, not a re-draw of the set, and the counterexample
   prints its seed pair. *)
let top1_floor = 0.55
let same_draw_floor = 0.8
let cosine_floor = 0.98

let check_floors () =
  let low_top1 = ref 1.0 in
  let low_draw = ref 1.0 in
  let low_cosine = ref 1.0 in
  let holds (weight_seed, walk_seed) =
    let params = Transformer.Params.init config ~seed:weight_seed in
    let { Quantized.Drift.draws; events = (_ : int); same_peak; same_draw; mean_cosine } =
      Quantized.Drift.walk config params ~steps ~seed:walk_seed
    in
    let share count = Float.of_int count /. Float.of_int draws in
    low_top1 := Float.min !low_top1 (share same_peak);
    low_draw := Float.min !low_draw (share same_draw);
    low_cosine := Float.min !low_cosine mean_cosine;
    draws > context
    && Float.(share same_peak > top1_floor)
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
       ~count:100
       ~name:"the drift floors hold on drawn seed pairs"
       seed_pair
       holds);
  printf
    "100 drawn seed pairs: low top-1 %.3f  low same draw %.3f  low cosine %.4f\n"
    !low_top1
    !low_draw
    !low_cosine
;;

let () =
  List.iter weight_seeds ~f:report;
  check_floors ()
;;
