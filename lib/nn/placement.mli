(** The placement family: what the eras state to Vivado about WHERE a thing goes.

    Every value here is a rule about placement and none of them is about arithmetic: they
    say which primitive holds a product, how many copies a broadcast stands as, and which
    memory a store maps to. They stand one time here and not once in each unit that obeys
    them.

    THEY ARE OF TWO KINDS, and a reader should not take the second for the first. [rom],
    [block_rom], [block_ram] and [slice_rows] are FACTS OF THE DEVICE: any era that stores
    a weight reads them, and all three do. [no_dsp], [replica] and [slices_for] are ERA
    SIX'S BUDGET for that device as it stands today — the column array took the DSPs, thus
    era six pins its other products away from them — and the frozen eras, whose [Mac]
    deliberately takes a DSP, read none of the three. The design is
    [docs/diffusion_rtl.md]. *)

open Hardcaml

(** [no_dsp product] pins a product into LUTs. THE COLUMN ARRAY OWNS THE DSPS OF ERA SIX —
    the fused rung is 48 by 5, the device's whole 240 — thus every other unit OF THAT ERA
    pins its products away from them. *)
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

(** the attribute a store of this width carries: block RAM and not distributed. A memory
    of thousands of words in LUTs would take the fabric the datapath needs, thus the
    attribute STATES the intent rather than leaving it to the inference heuristic. IT IS
    NOT A GUARANTEE: a memory the tile budget cannot hold is demoted to fabric anyway and
    silently — the ROM round's trap in [docs/diffusion_rtl.md] — and what makes a memory
    fit is the banking, not this attribute. *)
val block_ram : Rtl_attribute.t

(** [block_ram]'s counterpart for a memory that is never written. Hardcaml states
    RAM_STYLE and SRL_STYLE and no ROM_STYLE, thus this one is made here. *)
val block_rom : Rtl_attribute.t

(** [rom ?attributes ~read_addresses image] is a read-only memory of [image], one output
    for each address.

    IT IS [Signal.rom] THAT TAKES AN ATTRIBUTE. Hardcaml's own [rom] carries none, and a
    weight ROM that wants [block_rom] had to be built as a [multiport_memory] with a DEAD
    WRITE PORT instead — a write enable tied low, which puts an [always] block that can
    never fire into the Verilog and asks Vivado to read a written memory. This builds the
    primitive with no write port at all: an [initial] block, one [assign] for each read,
    and the attribute the caller states. The gate beside it holds that Verilog.

    It raises when the image is empty, when the image states more than one width, when
    there is no read address, or when an address does not size on the image — the checks
    [Signal.rom] takes from Hardcaml's validation and this path must state itself. *)
val rom
  :  ?attributes:Rtl_attribute.t list
  -> read_addresses:Signal.t array
  -> Bits.t array
  -> Signal.t array
