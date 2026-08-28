(** The placement family: what the eras state to Vivado about WHERE a thing goes.

    Every value here is a rule about placement and none of them is about arithmetic: they
    say which primitive holds a product, how many copies a broadcast stands as, and which
    memory a store maps to. They are facts of the device and of an era's budget for it,
    thus they stand one time for the whole repository and not once in each unit that obeys
    them. The design is [docs/diffusion_rtl.md]. *)

open Hardcaml

(** [no_dsp product] pins a product into LUTs. THE COLUMN ARRAY OWNS THE DSPS — the fused
    rung is 48 by 5, the device's whole 240 — thus every other unit pins its products away
    from them. *)
val no_dsp : Signal.t -> Signal.t

(** [replica copy] states one copy of a broadcast and keeps it apart from its siblings. No
    net of this scale keeps a single driver, thus a broadcast stands as one register for
    each slice of its consumers and every copy is [dont_touch] — so the tools neither
    merge the copies back into one net nor absorb them into what they feed. *)
val replica : Signal.t -> Signal.t

(** the rows one replica slice covers: 8. It is a placement fact of the device and not of
    a model, thus a caller that banks a column of its own slices on this value. It stands
    beside [replica] because it is the replica's grain and means nothing without it. *)
val slice_rows : int

(** [slices_for ~rows] is the replica slices a column of [rows] takes:
    [ceil (rows / slice_rows)]. *)
val slices_for : rows:int -> int

(** the attribute a store or a ROM of this width carries: block RAM and not distributed. A
    memory of thousands of words in LUTs would take the fabric the datapath needs, thus
    the attribute STATES the intent rather than leaving it to the inference heuristic. IT
    IS NOT A GUARANTEE: a memory the tile budget cannot hold is demoted to fabric anyway
    and silently — the ROM round's trap in [docs/diffusion_rtl.md] — and what makes a
    memory fit is the banking, not this attribute. *)
val block_ram : Rtl_attribute.t
