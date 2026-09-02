(** The scheduler: the succession of sheets, and the behaviour of era six's [Source].

    It drives the [Generator] and answers the socket: the sequencer sees one source that
    never stops giving sheets. [Source] is what the board seats — this unit at the elected
    gap over a generator — thus nothing outside the era names this one. The design is
    [docs/diffusion_rtl.md], "Phase II: the locked design — the generator and the
    scheduler".

    The walk: Rest, then OPEN, then DRAW, then COPY, then PLAY, and PLAY loops through the
    boundary.

    - **Rest** is the power-on state alone. [idle] is 1 and a [step] answers the silent
      frame, so that the behaviour is total; no walk of the sequencer reaches it, because
      a run stop leaves the unit in PLAY.
    - **OPEN** latches the seed view and waits any stale walk out, then strobes the
      generator's [start]. **DRAW** waits that walk out in turn. **COPY** takes the
      finished sheet through the transfer face into the frame store, [T] strobes, and
      opens the next draw over the generator's own sheet. [idle] is 0 through all three.
    - **PLAY** answers from the store: steps 0 to [T] - 1 are the sheet and the [gap]
      steps behind them are the silent frame, the first of which is THE DRAIN. [idle]
      holds at 1 through PLAY, and PLAY answers one cycle behind the strobe as the
      transfer face does.
    - **The boundary** is the step after the gap. The strobe is taken, its answer waits on
      a fresh copy, and the step stretches by the microseconds that costs — the socket is
      latency-insensitive, thus the bytes stay exact.

    What a caller must know:

    - **THE PING-PONG IS TWO MEMORIES THAT ALREADY EXIST.** Gibbs rewrites the sheet in
      place, thus the generator's own sheet registers hold the draw and the frame store
      here holds what plays. The store carries FRAMES and not classes, because the
      transfer face decodes: this unit names no vocabulary and no class width.
    - **A [rewind] inside PLAY re-anchors the run.** [idle] falls at once, and the unit
      then waits out whatever the generator is walking — about 45 s to the first note at
      the worst rung. The generator gains no abort port for a gesture this rare: the reset
      button clears everything and redraws in 22 s.
    - **The probes carry pinned names** — [scheduler_state], [seed_reg], [play_step],
      [copy_step], [gen_step], [store_write] — because the unit's own gates read them.
      [scheduler_state] and not [state]: the generator seats a machine of its own and the
      engine under it seats a third, thus one simulation holds three and the bare name
      would probe whichever the name mangler reached first. *)

open Hardcaml
module I = Source_intf.I
module O = Source_intf.O

(** [create ~e ~gap ~seed ~generator i] is the block.

    - [e] is the era's elaboration. The scheduler reads [T] from it and nothing else; the
      frame widths are [Frame]'s.
    - [gap] is the silent steps between two sheets, 1 or more — the first is the drain. It
      raises [Invalid_argument] at 0, because a gap with no drain step would carry held
      pitches across the boundary where the software drains whole and strikes again, and
      the two streams would part.
    - [seed] is the SEED view, latched at [rewind]; sheet k of the run plays
      [(seed + k) mod 2^32].
    - [generator] seats the draw machine, and THE SCHEDULER OWNS ITS SEED WIRE: [Source]
      passes [Generator.create ~e] and applies nothing. *)
val create
  :  e:Elaboration.t
  -> gap:int
  -> seed:Signal.t
  -> generator:(seed:Signal.t -> Signal.t Generator.I.t -> Signal.t Generator.O.t)
  -> Signal.t I.t
  -> Signal.t O.t
