(* The era's face — see source.mli, and docs/diffusion_rtl.md, "Phase II: the locked
   design", for what the two units under it do. *)

open Hardcaml

(* THE SILENCE BETWEEN TWO SHEETS, in steps of the grid: two bars at STEP_MS 200. It is
   the software default of jax/diffusion/infer.py's --gap, which the ear elected on
   2026-08-25, and it stands here and NOT in a host cell: a knob is runtime state every
   capture must then pin, for a number nobody turns. The floor of 1 and the reason for it
   are the scheduler's, because the drain is. *)
let gap = 32
let create ~e ~seed i = Scheduler.create ~e ~gap ~seed ~generator:(Generator.create ~e) i
