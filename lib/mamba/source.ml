(* The state-space note source. [source.mli] states the contract, and [docs/mamba_rtl.md]
   states the design of the whole — the five layers, the memories and the cost. This
   header holds what neither says: the reasons tied to this code.

   The file runs L2, the schedule, then L3, the compiler, then L4, the outer FSM. L0 and
   L1 are the units it drives.

   The rules that hold the shape are era four's, restated because the op vocabulary is
   different and the abstraction of that era is an open question by standing rule:

   - A unit of L1 is built once and the ops mux its ports. No op owns a resource, and the
     per-op facts stay in the ops: the address formulas, the operand sources, the landing
     writes.
   - The ops dispatch as one [switch] on the pc, not as a chain of guards.
   - An op's finish runs the next op's entry actions in the same cycle. This one
     convention replaces a hand-kept register reset for each op.
   - The op vocabulary is closed and concrete. The rule: when a field's meaning would
     depend on another field, stop extending and write a new op.
   - A repeated op is an inlined program step and takes no return register.

   What this machine holds that era four's did not, and what it drops:

   - THE STATE RAM is read and written in place, and it is the only memory here that
     survives a step. The update walk reads element i and writes it back about four cycles
     later, while the reads are at i + 4; a linear walk never revisits an address, thus no
     read of the walk sees a stale row. The readout op runs after the update retires, on
     the op boundary, thus it reads only finished state.
   - THE ORIGIN IS A MUX AND NOT A CLEARING WALK. At step 0 the state reads as zero and
     the conv taps read zero by their age rule, thus [rewind] stays what era four made it:
     load the PRNG, clear the counters, run nothing. [idle] never falls for it. The key
     and value rings need no clearing either: the fill count masks the slots the walk has
     not written, exactly as era four's did.
   - THE PLAN, and it is the one thing a program of this machine reads that era four's did
     not: a layer is a block, a Zamba head or a feed-forward, and the schedule builds the
     ops of the kind the checkpoint states. A LAYER'S PLACE IN THE PLAN IS NOT ITS PLACE
     IN A MEMORY — the state RAM and the tap ring hold a region for each block and the
     rings one for each head, thus the ops carry the ordinal of their own kind and never
     the index of the layer.
   - THE NORMED EMBEDDING STANDS FOR THE WHOLE STEP, in a memory of its own. Every other
     vector here dies inside its layer; this one is written once, after the embed, and the
     head of the last layer still reads it — which is what the Zamba query and key are.
   - NO WALK STALLS, and the head is where that rule was in danger. Era four merged the
     lanes of a head in one walk and froze it while the divide of a finished row ran, thus
     every read register and every tag of that machine carried an enable. [Attend] merges
     ONE LANE A WALK and waits on the divide with the walk already retired, thus the
     enables are not built here and [Mac] takes its hold at ground.

   The timing rules of era four are inherited as rules and not re-derived: every read two
   cycles from address to data, the ROM's first cycle on the address side (the register is
   load-bearing — the retiming trap is measured and recorded in era four), the DSP a
   two-register multiply with the fabric accumulator behind it, the banked ROM under
   RAM_STYLE with a gated-off write port.

   Two debts, both inherited:

   - The tick positions of the bespoke chains are hand-encoded against the two-cycle
     product latency and the two-cycle table reads. If the pipe deepens, replace the ticks
     with a wait on a valid bit; do not renumber.
   - FOUR OPS HERE CHOOSE THEIR OPERAND BY THE POSITION INSIDE THE ROW, which era four
     never did: its ops each read one memory. A choice like that must follow the DATA and
     not the address, thus [ii_at_data] and [oo_at_data] carry the counters forward by the
     read latency. Using [mac.ii] there would select with the address and read the wrong
     memory two cycles early. The fourth is the joined query: the top bit of its inner
     counter says whether the term comes from the stream or from the embedding. *)

open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

(* The two units era five takes from era four as they stand. They are model-free — their
   interfaces speak widths and strobes, not transformers — thus this library imports them
   and moves nothing. [Mac] could not come the same way and [Mac.walk_bits] says why;
   [Divider] could not either, and its interface file says why. *)
module Isqrt = Mgen_transformer.Isqrt
module Exp2 = Mgen_transformer.Exp2

let clamp16 v =
  let v = sresize v ~width:32 in
  mux2
    (v >+ of_signed_int ~width:32 32767)
    (of_signed_int ~width:16 32767)
    (mux2
       (v <+ of_signed_int ~width:32 (-32768))
       (of_signed_int ~width:16 (-32768))
       (sel_bottom v ~width:16))
;;

(* value * 2^-from as value * 2^-target; the shift count is an elaboration constant *)
let rescale ~from ~target v =
  if target >= from then sll v ~by:(target - from) else sra v ~by:(from - target)
;;

(* ==================================================================== *)
(* L2 — the schedule: the walk as a value *)
(* ==================================================================== *)

module Op = struct
  (* Where a tensor the walk reads begins. A block of the seat tensor is not a constant,
     because the chain walks the four seats with one program: the seat register names the
     block, and the four addresses are constants that one mux chooses. *)
  type where =
    | Fixed of int
    | Seat_block of int
  [@@deriving sexp_of]

  (* one weight tensor as the circuit sees it: where it starts, and its exponent *)
  type tensor =
    { base : where
    ; e : int
    }
  [@@deriving sexp_of]

  (* Which vector a norm reads and where it lands, and it carries four facts at once: the
     memory, the format it arrives in, the width it divides by, and the memory the normed
     vector goes to. The four move together — the stream is Q16 over [d] into the y RAM
     and the gate product is Q24 over [d_in] into the same one — thus one field names them
     and the op stays closed.

     [Embedding] reads the same stream as [Stream] at the same format and the same width,
     and it lands in a memory of its own: the head of the step norms the EMBEDDING once,
     and that vector must stand while every layer writes over the stream. *)
  type over =
    | Stream
    | Embedding
    | Gated
  [@@deriving sexp_of]

  (* Which vector a matvec walks against the weight.

     [Joined] is the one of the three that is not one memory but a PAIR: the query and the
     key of the Zamba head read [2 d] terms — the normed stream, then the normed embedding
     — thus the operand follows the top bit of the inner counter, and it must follow it AT
     THE DATA. *)
  type source =
    | Y
    | Joined
    | Hidden
  [@@deriving sexp_of]

  (* where a finished matvec sum lands *)
  type landing =
    | To_v (* clamp16 (rescale to v_q), the shared RAM from its base *)
    | To_q (* clamp16 (rescale to v_q), the query RAM *)
    | To_ring of
        { k : bool
        ; ring : int
        }
      (* the top byte of the same value, into the ring row of the newest slot *)
    | To_hidden (* rescale to hid_q, relu, clamp16 — the feed-forward hidden *)
    | To_logits (* one shift by [e]; the peak tracked for the temper *)
    | Add_to_h (* the residual join: the whole sum, rescaled onto the stream *)
  [@@deriving sexp_of]

  type matvec =
    { src : source
    ; w : tensor
    ; outer_major : bool (* the weight address order; true only for a seat readout *)
    ; inner : int
    ; outer : int
    ; landing : landing
    }
  [@@deriving sexp_of]

  (* One step of the walk: the facts that one case of the pc switch needs. The bespoke ops
     close over the model at elaboration and carry no fields here beyond the layer they
     read, which names the memories and the per-head constants. *)
  type t =
    (* The five rows of the input add row for row — the seat row of each of the four
       seats, and the bar phase — thus one exponent covers them all and the walk reads two
       bases. *)
    | Embed of
        { seats : int
        ; phase : int
        ; e : int
        }
    | Rms_norm of { over : over }
    | Matvec of matvec
    (* the depthwise causal convolution of one block: the taps take the step's input, then
       one row of [conv_taps] terms for each channel *)
    | Conv of
        { block : int
        ; w : tensor
        }
    (* SiLU in place over a range of the shared RAM: the conv output, and the gate *)
    | Silu_over of
        { from : int
        ; count : int
        }
    (* dt and the decay of each head: softplus, then one exp2 *)
    | Decay of { block : int }
    (* the inject operands, then the state read-modify-write *)
    | State_update of { block : int }
    (* one row of [state + 1] terms for each lane, the skip folded in as the last term *)
    | Readout of { block : int }
    (* the gated norm's input: the readout against the SiLU of the gate, kept wide *)
    | Gate
    (* the attention of one head after another over the ages of its ring: the scores, the
       exp2 weight of each age, then one merged lane at a time *)
    | Attend of { ring : int }
    | Temper
    | Draw
    | Threshold
    (* the drawn class lands in the register of the seat the chain is at *)
    | Pick
    (* The chain writes the row the seat drew onto the stream, in place: the stream is
       dead after the chain, because the forward pass starts from the embedding. *)
    | Accumulate of
        { base : where
        ; e : int
        }
  [@@deriving sexp_of]

  (* The cycles the builders below cost, each derived from the unit that spends them. The
     bespoke chains are the one exception: a chain's cost is the length of its tick list,
     which lives inside [create] where the bodies close over the datapath, thus the counts
     stand here as numbers. The cycle bench holds every one of these against the measured
     circuit. *)
  module Cost = struct
    (* a walk retires its last term this long after its last issue *)
    let drain = Mac.read_latency + 2

    (* the start cycle, the walk, and the cycle the wait releases *)
    let divide = Divider.busy_cycles + 2

    (* the same, less the start cycle: a divide that a retiring walk starts in its own
       last cycle costs the caller only the wait *)
    let divide_behind = Divider.busy_cycles + 1
    let root = Isqrt.busy_cycles + 1

    (* the ticks of the bespoke chains *)
    let silu = 6
    let decay = 9
    let exp_weight = 7
    let draw = 4
    let threshold = 5
    let pick = 2
  end

  (* The analytic cost of one op, in cycles from its go to its finish, both the cycle a
     predecessor's finish runs.

     [n] is the ages the ring holds at this step, and [Attend] is the ONE op that reads
     it. The trunk of era five dropped the fill count era four's model carried; the Zamba
     head brings it back for its own op alone, and every other number here stays a
     constant of the shape. *)
  let cycles (config : Mamba.Config.t) ~n (op : t) =
    let { Mamba.Config.d; d_in; heads; state; _ } = config in
    let channels = Mamba.Config.channels config in
    let head_d = Mamba.Config.head_d config in
    let classes = Vocab.classes in
    let norm ~width = width + Cost.drain + Cost.root + (width * (1 + Cost.divide)) in
    match op with
    (* four seat rows and the phase row *)
    | Embed _ -> ((Frame.voices + 1) * d) + Cost.drain
    | Rms_norm { over = Stream | Embedding } -> norm ~width:d
    | Rms_norm { over = Gated } -> norm ~width:d_in
    | Matvec { inner; outer; landing; _ } ->
      (inner * outer)
      + Cost.drain
      +
        (match landing with
        | Add_to_h -> 1
        | To_v | To_q | To_ring _ | To_hidden | To_logits -> 0)
    (* the taps take the input, then one row of taps for each channel *)
    | Conv _ -> channels + Cost.drain + (channels * config.taps) + Cost.drain
    | Silu_over { count; _ } -> Cost.silu * count
    | Decay _ -> Cost.decay * heads
    (* the inject walk, then two terms for each element of the state *)
    | State_update _ -> (heads * state) + Cost.drain + (2 * d_in * state) + Cost.drain
    | Readout _ -> (d_in * (state + 1)) + Cost.drain
    | Gate -> d_in + Cost.drain
    (* For each head: one row of lanes an age, one weight chain an age, then ONE LANE A
       WALK with its divide behind it. Era four merged every lane in one walk and stalled
       that walk while the divide of a finished row ran; a lane a walk pays the drain of
       each and keeps the machine's rule that no walk ever stalls. *)
    | Attend _ ->
      heads
      * ((n * head_d)
         + Cost.drain
         + (Cost.exp_weight * n)
         + (head_d * (n + Cost.drain + Cost.divide_behind)))
    | Temper -> Cost.exp_weight * classes
    | Draw -> Cost.draw
    | Threshold -> Cost.threshold
    | Pick -> Cost.pick * classes
    (* one row of one term for each element of the stream, and the join lands one cycle
       behind the walk, as every residual join does *)
    | Accumulate _ -> d + Cost.drain + 1
  ;;
end

(* The two programs of the source. It runs [forward] over the frame it is about to state,
   and then [chain] to draw the frame after it.

   The chain runs from the soprano down and it stands FIRST in the step of the reference —
   [Quantized.Engine.next_step] draws and then forwards. The circuit takes the two in the
   other order for one reason: it answers [step] from a frame it drew already, thus the
   forward of the stated frame and the chain of the next one both fall inside one step
   period, and the wire never waits for the network.

   **[chain] is ONE seat and the machine runs it [Frame.voices] times**, counting the seat
   register down from the soprano. Era four measured what inlining the four seats costs —
   47 percent more fabric, because every case of the program counter that writes a
   register widens that register's parallel case — and the seat register is the price of
   the room. *)
type program =
  { chain : Op.t list
  ; forward : Op.t list
  }

let schedule (model : Quantized.Model.t) : program =
  let { Mamba.Config.d; d_in; plan; _ } = model.config in
  let classes = Vocab.classes in
  let bases = Quantized.Model.rom_bases model in
  let tensor_at (q : Quantized.Model.quantized) base = { Op.base; e = q.e } in
  let matvec ?(src = Op.Y) ?(outer_major = false) ~inner ~outer w base landing =
    Op.Matvec { src; w = tensor_at w (Op.Fixed base); outer_major; inner; outer; landing }
  in
  (* A layer's place in the plan is not its place in a memory: [ordinals] gives each block
     the region of the state RAM and the tap ring it owns, and each head the ring it owns. *)
  let ordinals = Mamba.Config.ordinals model.config in
  let block ~at (w : Quantized.Model.block) (b : int Quantized.Model.Rom_data.block) =
    [ Op.Rms_norm { over = Stream }
      (* the image stores W_in transposed, thus the outer counter walks [d]: see the note
         at [Quantized.Model.transpose] *)
    ; matvec
        ~outer_major:true
        ~inner:d
        ~outer:(Mamba.Config.projection model.config)
        w.w_in
        b.w_in
        To_v
    ; Op.Conv { block = at; w = tensor_at w.conv (Op.Fixed b.conv) }
    ; Op.Silu_over { from = d_in; count = Mamba.Config.channels model.config }
    ; Op.Decay { block = at }
    ; Op.State_update { block = at }
    ; Op.Readout { block = at }
    ; Op.Silu_over { from = 0; count = d_in }
    ; Op.Gate
    ; Op.Rms_norm { over = Gated }
    ; matvec ~inner:d_in ~outer:d w.w_out b.w_out Add_to_h
    ]
  in
  (* The Zamba head. The query and the key walk [2 d] terms over the JOINED vector — the
     normed stream, then the normed embedding — where the value walks [d] over the stream
     alone. [Attend] leaves its merged context in the y RAM, thus the output projection is
     an ordinary matvec and it needs no landing of its own. *)
  let attention
    ~at
    (w : Quantized.Model.attention)
    (b : int Quantized.Model.Rom_data.attention)
    =
    [ Op.Rms_norm { over = Stream }
    ; matvec ~src:Joined ~inner:(2 * d) ~outer:d w.wq b.wq To_q
    ; matvec
        ~src:Joined
        ~inner:(2 * d)
        ~outer:d
        w.wk
        b.wk
        (To_ring { k = true; ring = at })
    ; matvec ~inner:d ~outer:d w.wv b.wv (To_ring { k = false; ring = at })
    ; Op.Attend { ring = at }
    ; matvec ~inner:d ~outer:d w.wo b.wo Add_to_h
    ]
  in
  let feed_forward
    (w : Quantized.Model.feed_forward)
    (b : int Quantized.Model.Rom_data.feed_forward)
    =
    [ Op.Rms_norm { over = Stream }
    ; matvec ~inner:d ~outer:(4 * d) w.w1 b.w1 To_hidden
    ; matvec ~src:Hidden ~inner:(4 * d) ~outer:d w.w2 b.w2 Add_to_h
    ]
  in
  (* The weights and the image bases are two structures over ONE plan, thus a layer takes
     them as a pair. The mismatch cannot happen — [check_shape] holds the layers against
     the plan and [rom_bases] reads the same plan — and it is stated rather than assumed
     away because both records are open. *)
  let layer index =
    let at = ordinals.(index) in
    match model.layers.(index), bases.Quantized.Model.Rom_data.layers.(index) with
    | Quantized.Model.Block w, Quantized.Model.Rom_data.Block b -> block ~at w b
    | Attention w, Attention b -> attention ~at w b
    | Feed_forward w, Feed_forward b -> feed_forward w b
    | (_ : Quantized.Model.layer), (_ : int Quantized.Model.Rom_data.layer) ->
      invalid_arg "the ROM image and the weights do not agree about the plan"
  in
  (* One seat of the chain, and the machine runs it once for each seat. The seat register
     names the block of the seat tensor, thus the four readouts are one program and one
     mux over four constant addresses. *)
  let seat_block = Op.Seat_block bases.seats in
  (* The normed embedding stands for the whole step and only a Zamba head reads it, thus a
     plan without one norms nothing twice. *)
  let embedding =
    if Mamba.Config.attentions model.config > 0
    then [ Op.Rms_norm { over = Embedding } ]
    else []
  in
  { chain =
      [ Op.Rms_norm { over = Stream }
      ; Op.Matvec
          { src = Y
          ; w = tensor_at model.seats seat_block
          ; outer_major = true
          ; inner = d
          ; outer = classes
          ; landing = To_logits
          }
      ; Temper
      ; Draw
      ; Threshold
      ; Pick
      ; Accumulate { base = seat_block; e = model.seats.e }
      ]
  ; forward =
      (Op.Embed { seats = bases.seats; phase = bases.phase; e = model.seats.e }
       :: embedding)
      @ List.concat (List.init (Array.length plan) ~f:layer)
  }
;;

(* ==================================================================== *)
(* L4 — the outer FSM; L1 and L3 live inside [create] *)
(* ==================================================================== *)

module State = struct
  type t =
    | Idle
    | Run
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(model : Quantized.Model.t) ~seed (i : _ I.t) : _ O.t =
  let { Quantized.Model.config; temper; min_weight; _ } = model in
  let { Mamba.Config.d; d_in; heads; state = n_state; span; ring = slots; _ } = config in
  let head = Mamba.Config.head config in
  let head_d = Mamba.Config.head_d config in
  let channels = Mamba.Config.channels config in
  let projection = Mamba.Config.projection config in
  let taps = config.taps in
  let blocks = Mamba.Config.blocks config in
  let rings = Mamba.Config.attentions config in
  let dff = 4 * d in
  let classes = Vocab.classes in
  (* the shift and address rules of the reference; the packing below derives every width *)
  Quantized.Model.check_shape model;
  assert (Int.is_pow2 Jsb.bar_steps);
  let dbits = Int.floor_log2 d in
  let inbits = Int.floor_log2 d_in in
  let head_bits = Int.floor_log2 heads in
  let lane_bits = Int.floor_log2 head in
  let lane_d_bits = Int.floor_log2 head_d in
  let nbits = Int.floor_log2 n_state in
  let tapbits = Int.floor_log2 taps in
  let slot_bits = Int.floor_log2 slots in
  let phase_bits = Int.floor_log2 Jsb.bar_steps in
  let class_bits = address_bits_for classes in
  let seat_bits = address_bits_for Frame.voices in
  let chan_bits = address_bits_for channels in
  let state_bits = address_bits_for (blocks * d_in * n_state) in
  let tap_bits = address_bits_for (blocks * channels * taps) in
  (* the rings are sized at the attention count and not at the width of a rounded-up
     field, thus a plan with one head pays for one; a model with none elaborates none *)
  let ring_bits = address_bits_for (Int.max 1 rings * slots * d) in
  let score_shift = Quantized.Constants.score_shift ~head_d in
  (* vram serves zxbcdt, then the SiLU outputs and the gate product, then the scores and
     the age weights of the head, the feed-forward hidden, and the logits and the sampler
     weights of the chain *)
  let vram_size = List.reduce_exn ~f:Int.max [ classes; projection; dff; slots ] in
  let vbits = address_bits_for vram_size in
  (* yram takes both norms: [d] entries for the stream and [d_in] for the gated one *)
  let yram_size = Int.max d d_in in
  let ybits = address_bits_for yram_size in
  (* The blocks of the plan in their own order. A block op carries the ordinal that
     indexes the state RAM and the tap ring, thus it names its per-head constants through
     the same index and no op has to know where its layer stands in the plan. *)
  let block_weights =
    Array.filter_map model.layers ~f:(function
      | Quantized.Model.Block w -> Some w
      | Attention (_ : Quantized.Model.attention)
      | Feed_forward (_ : Quantized.Model.feed_forward) -> None)
  in
  let prog = schedule model in
  let forward_length = List.length prog.forward in
  let pc_bits = address_bits_for (forward_length + List.length prog.chain) in
  let rom_bits = Quantized.Model.rom_bits model in
  let rom_addr_bits = address_bits_for (Array.length rom_bits) in
  let rom_const at = of_unsigned_int ~width:rom_addr_bits at in
  let min32 = of_signed_int ~width:32 (-(1 lsl 31)) in
  let eps48 = of_unsigned_int ~width:48 Quantized.Constants.eps_q in
  let walk = Mac.walk_bits in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  (* L1 — the walk registers *)
  let pc = Variable.reg spec ~width:pc_bits in
  (* The position inside a bespoke chain. It is a COUNTER and not a target of every case
     that runs one: it steps by itself, and a body states only where it holds and where it
     returns to the head. Era four measured what the other shape costs — a three-bit value
     over 48 cases stood five levels deep in the critical path — and this one is four
     bits, because the decay chain waits on two table reads and a product. *)
  let tick = Variable.reg spec ~width:4 in
  let stage = Variable.reg spec ~width:2 in
  let ii = Variable.reg spec ~width:walk in
  let oo = Variable.reg spec ~width:walk in
  let hd = Variable.reg spec ~width:head_bits in
  let thi = Variable.reg spec ~width:43 in
  (* the landing helpers: the residual read-modify-write *)
  let rmw = Variable.reg spec ~width:1 in
  let rmw_row = Variable.reg spec ~width:dbits in
  let rmw_sum = Variable.reg spec ~width:48 in
  let done_p = Variable.reg spec ~width:1 in
  let peak = Variable.reg spec ~width:32 in
  let diff = Variable.reg spec ~width:32 in
  let nn = Variable.reg spec ~width:22 in
  (* the softmax denominator of one head: it accumulates over the age weights and divides
     each merged lane *)
  let den = Variable.reg spec ~width:24 in
  (* the bespoke chains hold one value each: the SiLU input, and the biased dt draw *)
  let sv = Variable.reg spec ~width:16 in
  let dtv = Variable.reg spec ~width:16 in
  (* dt and the decay of each head: [heads] values a layer, written by Decay and read by
     the inject walk. They are registers and not a RAM because there are four of them. *)
  let dt = Array.init heads ~f:(fun (_ : int) -> Variable.reg spec ~width:16) in
  let alpha = Array.init heads ~f:(fun (_ : int) -> Variable.reg spec ~width:16) in
  (* the sampler *)
  let u24 = Variable.reg spec ~width:24 in
  let total = Variable.reg spec ~width:24 in
  let thr = Variable.reg spec ~width:24 in
  let cum = Variable.reg spec ~width:25 in
  let found = Variable.reg spec ~width:1 in
  (* The walk: the frame the source states, and the classes the chain drew for it. *)
  let held = Variable.reg spec ~width:(Frame.code_bits * Frame.voices) in
  let drawn =
    Array.init Frame.voices ~f:(fun (_ : int) -> Variable.reg spec ~width:class_bits)
  in
  let seat = Variable.reg spec ~width:seat_bits in
  let valid = Variable.reg spec ~width:1 in
  (* 32 bits: the lead-in test below reads the step counter and the conv taps read its low
     bits, thus a wrap would put the walk back inside the lead-in. At 8 ms a step — the
     floor of the wire — 16 bits wrap in under nine minutes and 32 bits in a thousand
     years. *)
  let s = Variable.reg spec ~width:32 in
  let _ = sm.current -- "state" in
  let _ = pc.value -- "pc" in
  let _ = held.value -- "frame" in
  let _ = s.value -- "step" in
  (* L0 — the units, and the wires that drive them *)
  let prng_step = Variable.wire ~default:gnd () in
  let prng =
    Prng.Rtl.create
      { Prng.Rtl.I.clock = i.clock
      ; clear = i.clear
      ; load = i.rewind &: sm.is Idle
      ; seed
      ; step = prng_step.value
      }
  in
  let prng_byte = sel_bottom prng.value ~width:8 in
  let div_start = Variable.wire ~default:gnd () in
  let div_num = Variable.wire ~default:(zero 40) () in
  let div_den = Variable.wire ~default:(zero 24) () in
  let { Divider.O.quotient = div_quotient; busy = div_busy } =
    Divider.create
      { Divider.I.clock = i.clock
      ; clear = i.clear
      ; start = div_start.value
      ; numerator = div_num.value
      ; denominator = div_den.value
      }
  in
  let sq_start = Variable.wire ~default:gnd () in
  let sq_value = Variable.wire ~default:(zero 42) () in
  let { Isqrt.O.root = sq_root; busy = sq_busy } =
    Isqrt.create
      { Isqrt.I.clock = i.clock
      ; clear = i.clear
      ; start = sq_start.value
      ; value = sq_value.value
      }
  in
  let { Exp2.O.e = exp2_e } = Exp2.create { Exp2.I.clock = i.clock; nn = nn.value } in
  (* the two tables of the era: a registered read, no start and no busy, the caller holds
     the input two cycles *)
  let { Sigmoid.O.s = sigmoid_s } =
    Sigmoid.create { Sigmoid.I.clock = i.clock; v = sv.value }
  in
  let { Softplus.O.c = softplus_c } =
    Softplus.create { Softplus.I.clock = i.clock; v = dtv.value }
  in
  (* the walk engine's commands, and the freeze that keeps its tags with its data *)
  let mac_go = Variable.wire ~default:gnd () in
  let mac_inner = Variable.wire ~default:(zero walk) () in
  let mac_outer = Variable.wire ~default:(zero walk) () in
  (* L1 — the memories; every read lands in a register, thus block RAM is inferred *)
  let rom_addr = Variable.wire ~default:(zero rom_addr_bits) () in
  let st_raddr = Variable.wire ~default:(zero state_bits) () in
  let st_wen = Variable.wire ~default:gnd () in
  let st_waddr = Variable.wire ~default:(zero state_bits) () in
  let st_wdata = Variable.wire ~default:(zero 16) () in
  let tap_raddr = Variable.wire ~default:(zero tap_bits) () in
  let tap_wen = Variable.wire ~default:gnd () in
  let tap_waddr = Variable.wire ~default:(zero tap_bits) () in
  let tap_wdata = Variable.wire ~default:(zero 16) () in
  let vram_raddr = Variable.wire ~default:(zero vbits) () in
  let vram_wen = Variable.wire ~default:gnd () in
  let vram_waddr = Variable.wire ~default:(zero vbits) () in
  let vram_wdata = Variable.wire ~default:(zero 32) () in
  let _ = vram_wen.value -- "vram_wen" in
  let _ = vram_waddr.value -- "vram_waddr" in
  let _ = vram_wdata.value -- "vram_wdata" in
  let hram_raddr = Variable.wire ~default:(zero dbits) () in
  let hram_wen = Variable.wire ~default:gnd () in
  let hram_waddr = Variable.wire ~default:(zero dbits) () in
  let hram_wdata = Variable.wire ~default:(zero 32) () in
  (* The write port of the residual stream, named so that a simulation can watch it. A
     frame gate says only THAT the circuit and the reference parted; the stream after each
     layer says WHERE, and these three names are how a probe reads it. Two address faults
     and one operand that followed its address instead of its data were found through
     them. *)
  let _ = hram_wen.value -- "hram_wen" in
  let _ = hram_waddr.value -- "hram_waddr" in
  let _ = hram_wdata.value -- "hram_wdata" in
  let yram_raddr = Variable.wire ~default:(zero ybits) () in
  let yram_wen = Variable.wire ~default:gnd () in
  let yram_waddr = Variable.wire ~default:(zero ybits) () in
  let yram_wdata = Variable.wire ~default:(zero 16) () in
  let _ = yram_wen.value -- "yram_wen" in
  let _ = yram_waddr.value -- "yram_waddr" in
  let _ = yram_wdata.value -- "yram_wdata" in
  let oram_raddr = Variable.wire ~default:(zero inbits) () in
  let oram_wen = Variable.wire ~default:gnd () in
  let oram_waddr = Variable.wire ~default:(zero inbits) () in
  let oram_wdata = Variable.wire ~default:(zero 16) () in
  (* the two small vectors of the head: the query of the step, and the NORMED EMBEDDING,
     which the head of the step writes once and every layer after it leaves alone *)
  let qram_raddr = Variable.wire ~default:(zero dbits) () in
  let qram_wen = Variable.wire ~default:gnd () in
  let qram_waddr = Variable.wire ~default:(zero dbits) () in
  let qram_wdata = Variable.wire ~default:(zero 16) () in
  let eram_raddr = Variable.wire ~default:(zero dbits) () in
  let eram_wen = Variable.wire ~default:gnd () in
  let eram_waddr = Variable.wire ~default:(zero dbits) () in
  let eram_wdata = Variable.wire ~default:(zero 16) () in
  (* the key and value rings; the two share a write address, because the two matvecs that
     fill them write the same slot of the same step *)
  let kc_raddr = Variable.wire ~default:(zero ring_bits) () in
  let vc_raddr = Variable.wire ~default:(zero ring_bits) () in
  let kc_wen = Variable.wire ~default:gnd () in
  let vc_wen = Variable.wire ~default:gnd () in
  let ring_waddr = Variable.wire ~default:(zero ring_bits) () in
  let ring_wdata = Variable.wire ~default:(zero 8) () in
  let bram_raddr = Variable.wire ~default:(zero (head_bits + nbits)) () in
  let bram_wen = Variable.wire ~default:gnd () in
  let bram_waddr = Variable.wire ~default:(zero (head_bits + nbits)) () in
  let bram_wdata = Variable.wire ~default:(zero 16) () in
  (* The banking rules are in [docs/mamba_rtl.md]; the measurements behind them are era
     four's and they are inherited as rules. A bank is an initialized memory with a
     gated-off write port, and RAM_STYLE pins it. The address registers once before the
     tree, and each bank registers its data once behind it: two cycles from address to
     data, as one ROM, because [reg (reg rom.(addr))] equals [reg (rom.(reg addr))] when
     the contents never change. The address register is load-bearing, not style: with a
     combinational address the tools retime the data register onto the address pins of
     every block RAM primitive and rebuild the whole op-dispatch address cone inside each
     one. *)
  let rec rom_banked bits addr =
    let count = Array.length bits in
    if Int.is_pow2 count && count <= 1 lsl 15
    then (
      let data =
        (multiport_memory
           ~attributes:[ Rtl_attribute.Vivado.Ram_style.block ]
           ~initialize_to:bits
           count
           ~write_ports:
             [| { Write_port.write_clock = i.clock
                ; write_address = zero (width addr)
                ; write_enable = gnd
                ; write_data = zero 8
                }
             |]
           ~read_addresses:[| addr |]).(0)
      in
      reg spec data)
    else (
      let split = if Int.is_pow2 count then count / 2 else 1 lsl Int.floor_log2 count in
      let low = rom_banked (Array.subo bits ~len:split) (lsbs addr) in
      let high =
        rom_banked
          (Array.subo bits ~pos:split)
          (sel_bottom (lsbs addr) ~width:(address_bits_for (count - split)))
      in
      mux2 (reg spec (msb addr)) high low)
  in
  let romd = rom_banked rom_bits (reg spec rom_addr.value) in
  let write_port waddr wen wdata =
    { Write_port.write_clock = i.clock
    ; write_address = waddr
    ; write_enable = wen
    ; write_data = wdata
    }
  in
  let ram ~size ~waddr ~wen ~wdata ~raddr =
    (multiport_memory
       size
       ~write_ports:[| write_port waddr wen wdata |]
       ~read_addresses:[| raddr |]).(0)
  in
  (* Every memory the walk reads stands two registers deep, and the small RAMs keep the
     one-register tap for the bespoke chains.

     THERE IS NO HOLD, and that is a simplification the recurrence buys. Era four's
     attention stalled its merge walk while a pending divide finished, thus every read
     register and every tag carried a freeze enable; no walk of this machine ever stalls —
     the two divides live in a bespoke chain that waits on its own tick, not inside a walk
     — thus the enables are gone and the [Mac] is held at [gnd]. *)
  let two_deep x = reg spec (reg spec x) in
  let one_deep x = reg spec x in
  let std =
    ram
      ~size:(blocks * d_in * n_state)
      ~waddr:st_waddr.value
      ~wen:st_wen.value
      ~wdata:st_wdata.value
      ~raddr:st_raddr.value
  in
  let std2 = two_deep std in
  let tapd =
    two_deep
      (ram
         ~size:(blocks * channels * taps)
         ~waddr:tap_waddr.value
         ~wen:tap_wen.value
         ~wdata:tap_wdata.value
         ~raddr:tap_raddr.value)
  in
  (* The read of a ring restores the eight zero low bits that [Quantized.coarse_to_ring]
     dropped at the write, thus the format stays Q12 at a granularity of 2^-4.

     The ring WRITE stands one register behind its landing, and it is era four's travel
     stage: the sum-to-write route across the die wants it. The register is safe by the
     schedule and not by luck — a ring row's nearest read is a whole op away, because the
     head scores its ages only after the value matvec has retired. *)
  let ring_waddr_r = reg spec ring_waddr.value in
  let ring_wdata_r = reg spec ring_wdata.value in
  let kc_wen_r = reg spec kc_wen.value in
  let vc_wen_r = reg spec vc_wen.value in
  let ring_of ~wen ~raddr =
    two_deep
      (ram
         ~size:(Int.max 1 rings * slots * d)
         ~waddr:ring_waddr_r
         ~wen
         ~wdata:ring_wdata_r
         ~raddr
       @: zero 8)
  in
  let kcd = ring_of ~wen:kc_wen_r ~raddr:kc_raddr.value in
  let vcd = ring_of ~wen:vc_wen_r ~raddr:vc_raddr.value in
  let vramd =
    one_deep
      (ram
         ~size:vram_size
         ~waddr:vram_waddr.value
         ~wen:vram_wen.value
         ~wdata:vram_wdata.value
         ~raddr:vram_raddr.value)
  in
  let vramd2 = one_deep vramd in
  let hramd =
    one_deep
      (ram
         ~size:d
         ~waddr:hram_waddr.value
         ~wen:hram_wen.value
         ~wdata:hram_wdata.value
         ~raddr:hram_raddr.value)
  in
  let hramd2 = one_deep hramd in
  let yd2 =
    two_deep
      (ram
         ~size:yram_size
         ~waddr:yram_waddr.value
         ~wen:yram_wen.value
         ~wdata:yram_wdata.value
         ~raddr:yram_raddr.value)
  in
  let oramd2 =
    two_deep
      (ram
         ~size:d_in
         ~waddr:oram_waddr.value
         ~wen:oram_wen.value
         ~wdata:oram_wdata.value
         ~raddr:oram_raddr.value)
  in
  let qd2 =
    two_deep
      (ram
         ~size:d
         ~waddr:qram_waddr.value
         ~wen:qram_wen.value
         ~wdata:qram_wdata.value
         ~raddr:qram_raddr.value)
  in
  let ed2 =
    two_deep
      (ram
         ~size:d
         ~waddr:eram_waddr.value
         ~wen:eram_wen.value
         ~wdata:eram_wdata.value
         ~raddr:eram_raddr.value)
  in
  let bramd2 =
    two_deep
      (ram
         ~size:(heads * n_state)
         ~waddr:bram_waddr.value
         ~wen:bram_wen.value
         ~wdata:bram_wdata.value
         ~raddr:bram_raddr.value)
  in
  (* L1 — [Mac], the walk behind the one 25 by 18 multiplier *)
  let mul_a = Variable.wire ~default:(zero 25) () in
  let mul_b = Variable.wire ~default:(zero 18) () in
  let mac =
    Mac.create
      { Mac.I.clock = i.clock
      ; clear = i.clear
      ; go = mac_go.value
      ; inner = mac_inner.value
      ; outer = mac_outer.value
      ; hold = gnd
      ; a = mul_a.value
      ; b = mul_b.value
      }
  in
  (* THE COUNTERS AT THE DATA, and the note in the module comment states why they exist:
     three ops choose which memory feeds the multiplier by the position inside the row,
     and that choice must follow the data. The pipeline freezes with the walk. *)
  let ii_at_data = pipeline spec ~n:Mac.read_latency mac.ii in
  let oo_at_data = pipeline spec ~n:Mac.read_latency mac.oo in
  (* the walk slices *)
  let oo_v = sel_bottom oo.value ~width:vbits in
  let mac_ii_tap = sel_bottom mac.ii ~width:tapbits in
  let mac_ii_n = sel_bottom mac.ii ~width:nbits in
  let mac_oo_d = sel_bottom mac.oo ~width:dbits in
  let mac_oo_chan = sel_bottom mac.oo ~width:chan_bits in
  let mac_oo_in = sel_bottom mac.oo ~width:inbits in
  let mac_row_d = sel_bottom mac.row ~width:dbits in
  let mac_row_chan = sel_bottom mac.row ~width:chan_bits in
  let mac_row_in = sel_bottom mac.row ~width:inbits in
  let oo_class = sel_bottom oo.value ~width:class_bits in
  let phase = sel_bottom s.value ~width:phase_bits in
  (* The ring, read off the step counter and not off registers of its own. Era four kept a
     slot register and a filled flag; here the step counter already states both — the
     newest slot is its low bits, and the ring is full once the walk has run [slots]
     steps. *)
  let cur = sel_bottom s.value ~width:slot_bits in
  let filled = s.value >=: of_unsigned_int ~width:32 slots in
  let nfill =
    uresize
      (mux2
         filled
         (of_unsigned_int ~width:(slot_bits + 1) slots)
         (uresize cur ~width:(slot_bits + 1) +:. 1))
      ~width:walk
  in
  let mac_ii_slot = sel_bottom mac.ii ~width:slot_bits in
  let mac_oo_slot = sel_bottom mac.oo ~width:slot_bits in
  let mac_row_slot = sel_bottom mac.row ~width:slot_bits in
  let mac_ii_lane = sel_bottom mac.ii ~width:lane_d_bits in
  (* the ring slot an age names: the walk counts the age, and [cur] is the newest slot *)
  let slot_of_oo = cur -: mac_oo_slot in
  let slot_of_ii = cur -: mac_ii_slot in
  (* One row of a ring: the rows of one head stand together, thus the head's ring stands
     above the slot and the dimension and the address is a concatenation — the slots and
     [d] are powers of two, thus the ring's offset has nothing but zeros below it. *)
  let ring_layer_bits = ring_bits - slot_bits - dbits in
  let ring_row ~ring ~slot ~dim =
    if ring_layer_bits = 0
    then slot @: dim
    else of_unsigned_int ~width:ring_layer_bits ring @: slot @: dim
  in
  (* the ALiBi penalty of the age the score walk has retired: the slope of head k is
     2^-(span (k+1) / heads), thus the penalty is a shift of the age, in Q12 *)
  let alibi =
    mux
      hd.value
      (List.init heads ~f:(fun head ->
         let exponent = Quantized.Constants.slope_exponent ~span ~heads ~head in
         sll (uresize mac.row ~width:32) ~by:(Quantized.Constants.y_q - exponent)))
  in
  let score = sel_bottom (sra mac.sum ~by:score_shift) ~width:32 -: alibi in
  let den_next = den.value +: uresize exp2_e ~width:24 in
  (* The state address of an element: the layer stands above the lane and the lane above
     the state index, and every field is a power of two, thus the whole address is a
     concatenation and no adder pays for it. The update walk counts elements, thus its
     [oo] IS the (lane, n) pair already. *)
  let state_layer_bits = state_bits - inbits - nbits in
  let state_row ~layer ~lane_n =
    if state_layer_bits = 0
    then lane_n
    else of_unsigned_int ~width:state_layer_bits layer @: lane_n
  in
  (* The tap ring of a layer, and it is the one address here that an adder pays for.

     A channel and a slot concatenate — the taps are a power of two — but the CHANNEL
     COUNT is not: [d_in + 2 N] is 160 at the baseline and 48 at the gate shape. A third
     field concatenated above them would therefore stride by the rounded-up power of two,
     thus every layer above the first would sit at the wrong base and the top layer's
     region would run off the end of the memory. The layer base is one constant add
     instead, which puts the circuit on the reference's own address and wastes no row. The
     state address needs none: [d_in] and [state] are both powers of two, thus the layer
     field packs. *)
  let tap_row ~layer ~channel ~slot =
    let inside = uresize (channel @: slot) ~width:tap_bits in
    if layer = 0
    then inside
    else of_unsigned_int ~width:tap_bits (layer * channels * taps) +: inside
  in
  (* the derived values of L1, named once: a builder runs for each op of the program, and
     an expression written inside one is elaborated once for each of them *)
  let quotient16 = clamp16 div_quotient in
  let below_peak = vramd -: peak.value in
  let drawn_at_seat =
    mux seat.value (List.map (Array.to_list drawn) ~f:(fun c -> c.Always.Variable.value))
  in
  let write_drawn value =
    switch
      seat.value
      (List.init Frame.voices ~f:(fun s ->
         of_unsigned_int ~width:seat_bits s, [ drawn.(s) <-- value ]))
  in
  let base_of (where : Op.where) =
    match where with
    | Fixed at -> rom_const at
    | Seat_block at ->
      mux
        seat.value
        (List.init Frame.voices ~f:(fun s ->
           of_unsigned_int ~width:rom_addr_bits (at + (s * classes * d))))
  in
  (* the frame word of the classes the chain drew: [Vocab.Rtl] states the map, and seat 0
     takes the low byte *)
  let frame_word =
    concat_msb
      (List.rev_map (Array.to_list drawn) ~f:(fun c ->
         Vocab.Rtl.code_of_class c.Always.Variable.value))
  in
  (* ================================================================== *)
  (* L3 — the compiler: one builder per op kind, then the chain *)
  (* ================================================================== *)
  let by_tick bodies =
    let last = List.length bodies - 1 in
    switch
      tick.value
      (List.mapi bodies ~f:(fun k body ->
         of_unsigned_int ~width:4 k, if k = last then body else (tick <--. k + 1) :: body))
  in
  let by_stage bodies =
    switch
      stage.value
      (List.mapi bodies ~f:(fun k body -> of_unsigned_int ~width:2 k, body))
  in
  (* [exp_weight_chain] is era four's: one vram value becomes its exp2 weight over the
     same address. Only [Temper] runs it here — the softmax that was the other caller is
     gone with the attention. *)
  let exp_weight_chain ~addr ~(scale : Quantized.Constants.scale) ~land_ ~advance =
    let at_addr = uresize addr ~width:vbits in
    [ vram_raddr <-- at_addr
    ; mul_a <-- sel_bottom diff.value ~width:25
    ; mul_b <-- of_signed_int ~width:18 scale.q_value
    ; by_tick
        [ []
        ; [ diff <-- below_peak ]
        ; []
        ; []
        ; [ nn <-- sel_bottom (negate (sra mac.product ~by:scale.q)) ~width:22 ]
        ; []
        ; [ vram_wen <-- vdd; vram_waddr <-- at_addr ] @ land_ @ [ tick <--. 0 ] @ advance
        ]
    ]
  in
  (* [join_to_h ~from] is the residual read-modify-write: a finished row's sum lands on
     the stream one cycle behind its retirement, thus the walk and the join never contend
     for the h RAM. [rmw] is the retirement delayed one cycle and nothing else. *)
  let join_to_h ~from ~finish =
    [ rmw <-- mac.row_done
    ; when_
        mac.row_done
        [ rmw_row <-- mac_row_d; rmw_sum <-- mac.sum; hram_raddr <-- mac_row_d ]
    ; when_ mac.done_ [ done_p <-- vdd ]
    ; when_
        rmw.value
        [ hram_wen <-- vdd
        ; hram_waddr <-- rmw_row.value
        ; hram_wdata
          <-- sel_bottom
                (sresize hramd ~width:48
                 +: rescale ~from ~target:Quantized.Constants.h_q rmw_sum.value)
                ~width:32
        ; when_ done_p.value ([ done_p <-- gnd ] @ finish)
        ]
    ]
  in
  let join_entry = [ rmw <-- gnd; done_p <-- gnd; mac_go <-- vdd ] in
  (* [build op ~finish] gives the entry actions and the body of one program step. [finish]
     runs in the op's last cycle: the next op's entry, and the pc move. *)
  let build (op : Op.t) ~(finish : Always.t list) =
    match op with
    | Op.Embed { seats; phase = ph; e } ->
      let entry = [ mac_go <-- vdd ] in
      let table base index =
        rom_const base +: uresize (index @: mac_oo_d) ~width:rom_addr_bits
      in
      let rows =
        List.init Frame.voices ~f:(fun seat ->
          table (seats + (seat * classes * d)) drawn.(seat).value)
        @ [ table ph phase ]
      in
      let body =
        [ mac_inner <--. Frame.voices + 1
        ; mac_outer <--. d
        ; rom_addr
          <-- mux (sel_bottom mac.ii ~width:(address_bits_for (Frame.voices + 1))) rows
        ; mul_a <-- sresize romd ~width:25
        ; mul_b <-- of_signed_int ~width:18 1
        ; when_
            mac.row_done
            [ hram_wen <-- vdd
            ; hram_waddr <-- mac_row_d
            ; hram_wdata
              <-- sel_bottom
                    (rescale ~from:e ~target:Quantized.Constants.h_q mac.sum)
                    ~width:32
            ]
        ; when_ mac.done_ finish
        ]
      in
      entry, body
    | Accumulate { base; e } ->
      let body =
        [ mac_inner <--. 1
        ; mac_outer <--. d
        ; rom_addr
          <-- base_of base +: uresize (drawn_at_seat @: mac_oo_d) ~width:rom_addr_bits
        ; mul_a <-- sresize romd ~width:25
        ; mul_b <-- of_signed_int ~width:18 1
        ]
        @ join_to_h ~from:e ~finish
      in
      join_entry, body
    | Rms_norm { over } ->
      (* stage 0 sums the squares of a Q12 copy on the walk; stage 1 waits on the isqrt;
         stage 2 divides each element. The vector, its format, its width and the memory it
         lands in move together, thus [over] names all four. *)
      let width, raddr, tap1, tap2, square_shift, num_shift =
        match over with
        | Op.Stream | Embedding ->
          ( d
          , (fun a -> hram_raddr <-- sel_bottom a ~width:dbits)
          , hramd
          , hramd2
          , Quantized.Constants.h_q - Quantized.Constants.y_q
          , (2 * Quantized.Constants.y_q) - Quantized.Constants.h_q )
        | Gated ->
          ( d_in
          , (fun a -> vram_raddr <-- uresize a ~width:vbits)
          , vramd
          , vramd2
          , Quantized.Constants.gate_q - Quantized.Constants.y_q
          , (2 * Quantized.Constants.y_q) - Quantized.Constants.gate_q )
      in
      (* the normed embedding takes a memory of its own, because every layer after it
         writes over the y RAM and the head of the last one still reads it *)
      let land_normed at value =
        match over with
        | Op.Embedding ->
          [ eram_wen <-- vdd
          ; eram_waddr <-- sel_bottom at ~width:dbits
          ; eram_wdata <-- value
          ]
        | Stream | Gated ->
          [ yram_wen <-- vdd
          ; yram_waddr <-- uresize at ~width:ybits
          ; yram_wdata <-- value
          ]
      in
      let wbits = address_bits_for width in
      let square = sel_bottom (sra tap2 ~by:square_shift) ~width:25 in
      let sum_squares =
        [ mac_inner <--. width
        ; mac_outer <--. 1
        ; raddr (sel_bottom mac.ii ~width:wbits)
        ; mul_a <-- square
        ; mul_b <-- sel_bottom square ~width:18
        ; when_
            mac.done_
            [ sq_start <-- vdd
            ; sq_value
              <-- sel_bottom (srl mac.sum ~by:(Int.floor_log2 width) +: eps48) ~width:42
            ; ii <--. 0
            ; tick <--. 0
            ; stage <--. 1
            ]
        ]
      in
      let await_root = [ when_ ~:sq_busy [ stage <--. 2; ii <--. 0; tick <--. 0 ] ] in
      let at = sel_bottom ii.value ~width:wbits in
      let divide_elements =
        [ raddr at
        ; by_tick
            [ []
            ; [ div_start <-- vdd
              ; div_num <-- sll (sresize tap1 ~width:40) ~by:num_shift
              ; div_den <-- uresize sq_root ~width:24
              ]
            ; [ when_
                  ~:div_busy
                  (land_normed at quotient16
                   @ [ tick <--. 0
                     ; if_ (ii.value ==:. width - 1) finish [ ii <-- ii.value +:. 1 ]
                     ])
              ]
            ]
        ]
      in
      ( [ stage <--. 0; mac_go <-- vdd ]
      , [ by_stage [ sum_squares; await_root; divide_elements ] ] )
    | Matvec { src; w; outer_major; inner; outer; landing } ->
      let ibits = address_bits_for inner in
      let obits = address_bits_for outer in
      let ii_i = sel_bottom mac.ii ~width:ibits in
      let oo_o = sel_bottom mac.oo ~width:obits in
      let row_o = sel_bottom mac.row ~width:obits in
      let addr = if outer_major then oo_o @: ii_i else ii_i @: oo_o in
      let read_src, srcd =
        match src with
        | Op.Y -> [ yram_raddr <-- uresize ii_i ~width:ybits ], yd2
        (* The joined vector is [2 d] terms: the normed stream, then the normed embedding.
           Both memories read at the low bits of the counter and THE MUX FOLLOWS THE DATA
           — the top bit carried forward by the read latency — because a selection on the
           address side would take the wrong memory two cycles early. *)
        | Joined ->
          ( [ yram_raddr <-- uresize (sel_bottom ii_i ~width:dbits) ~width:ybits
            ; eram_raddr <-- sel_bottom ii_i ~width:dbits
            ]
          , mux2 (msb (sel_bottom ii_at_data ~width:ibits)) ed2 yd2 )
        | Hidden ->
          [ vram_raddr <-- uresize ii_i ~width:vbits ], sel_bottom vramd2 ~width:16
      in
      let common =
        read_src
        @ [ mac_inner <--. inner
          ; mac_outer <--. outer
          ; rom_addr <-- base_of w.base +: uresize addr ~width:rom_addr_bits
          ; mul_a <-- sresize srcd ~width:25
          ; mul_b <-- sresize romd ~width:18
          ]
      in
      let simple ?(entry_extra = []) writes =
        ( entry_extra @ [ mac_go <-- vdd ]
        , common @ [ when_ mac.row_done writes; when_ mac.done_ finish ] )
      in
      (* the working-class landing of a projection, which four of the six landings take *)
      let to_v v =
        clamp16
          (rescale
             ~from:(Quantized.Constants.y_q + w.e)
             ~target:Quantized.Constants.v_q
             v)
      in
      (match landing with
       | To_v ->
         simple
           [ vram_wen <-- vdd
           ; vram_waddr <-- uresize row_o ~width:vbits
           ; vram_wdata <-- sresize (to_v mac.sum) ~width:32
           ]
       | To_q ->
         simple
           [ qram_wen <-- vdd
           ; qram_waddr <-- sel_bottom row_o ~width:dbits
           ; qram_wdata <-- to_v mac.sum
           ]
       | To_ring { k; ring } ->
         simple
           [ (if k then kc_wen else vc_wen) <-- vdd
           ; ring_waddr <-- ring_row ~ring ~slot:cur ~dim:(sel_bottom row_o ~width:dbits)
           ; ring_wdata <-- sel_top (to_v mac.sum) ~width:8
           ]
       | To_hidden ->
         let shifted =
           rescale
             ~from:(Quantized.Constants.y_q + w.e)
             ~target:Quantized.Constants.hid_q
             mac.sum
         in
         simple
           [ vram_wen <-- vdd
           ; vram_waddr <-- uresize row_o ~width:vbits
           ; vram_wdata
             <-- sresize (clamp16 (mux2 (shifted <+ zero 48) (zero 48) shifted)) ~width:32
           ]
       | To_logits ->
         (* no mask stands before the draw, thus the peak is the peak *)
         let logit = sel_bottom (sra mac.sum ~by:w.e) ~width:32 in
         simple
           ~entry_extra:[ peak <-- min32 ]
           [ vram_wen <-- vdd
           ; vram_waddr <-- uresize row_o ~width:vbits
           ; vram_wdata <-- logit
           ; when_ (logit >+ peak.value) [ peak <-- logit ]
           ]
       | Add_to_h ->
         let from =
           match src with
           | Op.Y | Joined -> Quantized.Constants.v_q
           | Hidden -> Quantized.Constants.hid_q
         in
         join_entry, common @ join_to_h ~from:(from + w.e) ~finish)
    | Conv { block = layer; w } ->
      (* Stage 0 walks the step's input into the tap ring, one channel a row, the DSP a
         wire. Stage 1 walks the taps: tap k of a channel reads the step k back, and it
         reads ZERO while the walk has not run k steps — one compare against the step
         counter, which is why the origin needs no clearing walk. *)
      let slot = sel_bottom s.value ~width:tapbits -: mac_ii_tap in
      let young = s.value <: uresize ii_at_data ~width:32 in
      let load_taps =
        [ mac_inner <--. 1
        ; mac_outer <--. channels
        ; vram_raddr
          <-- of_unsigned_int ~width:vbits d_in +: uresize mac_oo_chan ~width:vbits
        ; mul_a <-- sresize (sel_bottom vramd2 ~width:16) ~width:25
        ; mul_b <-- of_signed_int ~width:18 1
        ; when_
            mac.row_done
            [ tap_wen <-- vdd
            ; tap_waddr
              <-- tap_row
                    ~layer
                    ~channel:mac_row_chan
                    ~slot:(sel_bottom s.value ~width:tapbits)
            ; tap_wdata <-- clamp16 mac.sum
            ]
        ; when_ mac.done_ [ stage <--. 1; mac_go <-- vdd ]
        ]
      in
      let convolve =
        [ mac_inner <--. taps
        ; mac_outer <--. channels
        ; tap_raddr <-- tap_row ~layer ~channel:mac_oo_chan ~slot
        ; rom_addr
          <-- base_of w.base +: uresize (mac_oo_chan @: mac_ii_tap) ~width:rom_addr_bits
        ; mul_a <-- mux2 young (zero 25) (sresize tapd ~width:25)
        ; mul_b <-- sresize romd ~width:18
        ; when_
            mac.row_done
            [ vram_wen <-- vdd
            ; vram_waddr
              <-- of_unsigned_int ~width:vbits d_in +: uresize mac_row_chan ~width:vbits
            ; vram_wdata
              <-- sresize
                    (clamp16
                       (rescale
                          ~from:(Quantized.Constants.v_q + w.e)
                          ~target:Quantized.Constants.v_q
                          mac.sum))
                    ~width:32
            ]
        ; when_ mac.done_ finish
        ]
      in
      [ stage <--. 0; mac_go <-- vdd ], [ by_stage [ load_taps; convolve ] ]
    | Silu_over { from; count } ->
      (* one element a chain: read it, take its sigmoid, multiply, and land the product
         back over it. [sv] holds the value for the table, which reads in two cycles. *)
      let at = of_unsigned_int ~width:vbits from +: uresize oo_v ~width:vbits in
      let entry = [ oo <--. 0; tick <--. 0 ] in
      let body =
        [ vram_raddr <-- at
        ; mul_a <-- sresize sv.value ~width:25
        ; mul_b <-- uresize sigmoid_s ~width:18
        ; by_tick
            [ []
            ; [ sv <-- sel_bottom vramd ~width:16 ]
            ; []
            ; []
            ; []
            ; [ vram_wen <-- vdd
              ; vram_waddr <-- at
              ; vram_wdata
                <-- sresize
                      (clamp16 (sra mac.product ~by:Quantized.Constants.alpha_q))
                      ~width:32
              ; tick <--. 0
              ; if_ (oo.value ==:. count - 1) finish [ oo <-- oo.value +:. 1 ]
              ]
            ]
        ]
      in
      entry, body
    | Decay { block = layer } ->
      (* one head a chain: the biased draw, the softplus of it, then one exp2 of dt times
         the head's decay constant. The three per-head numbers are elaboration constants,
         thus a mux over [heads] carries each. *)
      let lay = block_weights.(layer) in
      let bias =
        mux hd.value (Array.to_list (Array.map lay.dt_bias ~f:(of_signed_int ~width:16)))
      in
      let rate =
        mux
          hd.value
          (Array.to_list
             (Array.map lay.decay ~f:(fun (c : Quantized.Constants.scale) ->
                of_signed_int ~width:25 c.q_value)))
      in
      let decay_q = lay.decay.(0).q in
      let raw = sel_bottom vramd ~width:16 in
      let ramp = mux2 (msb dtv.value) (zero 16) dtv.value in
      let entry = [ hd <--. 0; tick <--. 0 ] in
      let body =
        [ vram_raddr
          <-- of_unsigned_int ~width:vbits (d_in + channels)
              +: uresize hd.value ~width:vbits
        ; mul_a <-- rate
        ; mul_b
          <-- sresize
                (mux hd.value (List.map (Array.to_list dt) ~f:(fun r -> r.value)))
                ~width:18
        ; by_tick
            [ []
            ; [ dtv <-- clamp16 (sresize raw ~width:32 +: sresize bias ~width:32) ]
            ; []
            ; [ switch
                  hd.value
                  (List.init heads ~f:(fun k ->
                     ( of_unsigned_int ~width:head_bits k
                     , [ dt.(k)
                         <-- clamp16
                               (uresize ramp ~width:32 +: uresize softplus_c ~width:32)
                       ] )))
              ]
            ; []
            ; []
            ; [ nn <-- sel_bottom (sra mac.product ~by:decay_q) ~width:22 ]
            ; []
            ; [ switch
                  hd.value
                  (List.init heads ~f:(fun k ->
                     of_unsigned_int ~width:head_bits k, [ alpha.(k) <-- exp2_e ]))
              ; tick <--. 0
              ; if_ (hd.value ==:. heads - 1) finish [ hd <-- hd.value +:. 1 ]
              ]
            ]
        ]
      in
      entry, body
    | State_update { block = layer } ->
      (* Stage 0 writes the inject operands: [heads * state] products of a head's dt
         against

         B. Stage 1 is the read-modify-write of the state itself, two terms an element:

         S' = (alpha * S + x * beta) >> 15, clamped

         The walk counts ELEMENTS, thus its [oo] is the (lane, n) pair already and the
         state address is that pair under the layer — a concatenation and no adder. The
         operand of a term follows the DATA, not the address: see the module comment. *)
      let b_at ~n =
        of_unsigned_int ~width:vbits (d_in + d_in) +: uresize n ~width:vbits
      in
      let dt_of h = mux h (List.map (Array.to_list dt) ~f:(fun r -> r.value)) in
      let alpha_of h = mux h (List.map (Array.to_list alpha) ~f:(fun r -> r.value)) in
      let inject =
        [ mac_inner <--. 1
        ; mac_outer <--. heads * n_state
        ; vram_raddr <-- b_at ~n:(sel_bottom mac.oo ~width:nbits)
        ; mul_a <-- sresize (sel_bottom vramd2 ~width:16) ~width:25
          (* the head follows the DATA: [dt] is a register and B comes from a memory two
             cycles behind it, and a head selected on the address side would pair the two
             rows at each head boundary with the wrong step size *)
        ; mul_b
          <-- sresize
                (dt_of (select oo_at_data ~high:(nbits + head_bits - 1) ~low:nbits))
                ~width:18
        ; when_
            mac.row_done
            [ bram_wen <-- vdd
            ; bram_waddr <-- sel_bottom mac.row ~width:(head_bits + nbits)
            ; bram_wdata
              <-- clamp16
                    (sra
                       mac.sum
                       ~by:
                         (Quantized.Constants.v_q
                          + Quantized.Constants.v_q
                          - Quantized.Constants.beta_q))
            ]
        ; when_ mac.done_ [ mac_go <-- vdd; stage <--. 1 ]
        ]
      in
      (* the element the walk is at, on the address side and at the data side *)
      let lane_of o = select o ~high:(nbits + inbits - 1) ~low:nbits in
      let head_of o = select o ~high:(nbits + inbits - 1) ~low:(nbits + lane_bits) in
      let update =
        [ mac_inner <--. 2
        ; mac_outer <--. d_in * n_state
        ; st_raddr
          <-- state_row ~layer ~lane_n:(sel_bottom mac.oo ~width:(inbits + nbits))
        ; vram_raddr
          <-- of_unsigned_int ~width:vbits d_in +: uresize (lane_of mac.oo) ~width:vbits
        ; bram_raddr <-- head_of mac.oo @: sel_bottom mac.oo ~width:nbits
          (* term 0 is alpha times the state, term 1 is x times the inject operand; the
             state reads as ZERO at the origin, which is the mux the design document names
             in place of a clearing walk *)
        ; mul_a
          <-- mux2
                (lsb ii_at_data)
                (sresize (sel_bottom vramd2 ~width:16) ~width:25)
                (mux2 (s.value ==:. 0) (zero 25) (sresize std2 ~width:25))
        ; mul_b
          <-- mux2
                (lsb ii_at_data)
                (sresize bramd2 ~width:18)
                (uresize (alpha_of (head_of oo_at_data)) ~width:18)
        ; when_
            mac.row_done
            [ st_wen <-- vdd
            ; st_waddr
              <-- state_row ~layer ~lane_n:(sel_bottom mac.row ~width:(inbits + nbits))
            ; st_wdata <-- clamp16 (sra mac.sum ~by:Quantized.Constants.alpha_q)
            ]
        ; when_ mac.done_ finish
        ]
      in
      [ stage <--. 0; mac_go <-- vdd ], [ by_stage [ inject; update ] ]
    | Readout { block = layer } ->
      (* one row of [state + 1] terms for each lane: the state against C, and the skip
         folded in as the last term with [d_skip] in Q12, thus every product of the row
         lands Q24. It reads the state the update just wrote. *)
      let lay = block_weights.(layer) in
      let is_skip o = o ==:. n_state in
      let skip_at_data = is_skip ii_at_data in
      let lane = mac_oo_in in
      let head_at_data = select oo_at_data ~high:(inbits - 1) ~low:lane_bits in
      (* the convolved channels run x, then B, then C — thus C stands a state width above
         the block the inject walk reads *)
      let c_at =
        of_unsigned_int ~width:vbits (d_in + d_in + n_state)
        +: uresize mac_ii_n ~width:vbits
      in
      let body =
        [ mac_inner <--. n_state + 1
        ; mac_outer <--. d_in
        ; st_raddr <-- state_row ~layer ~lane_n:(lane @: mac_ii_n)
        ; vram_raddr
          <-- mux2
                (is_skip mac.ii)
                (of_unsigned_int ~width:vbits d_in +: uresize lane ~width:vbits)
                c_at
          (* NO ORIGIN MUX HERE. The update runs before this op and has written the whole
             state of the layer, thus the readout reads finished state at every step —
             including the first, where the update read zero. The mux belongs to the
             reader of the PREVIOUS step's state and to that one alone. *)
        ; mul_a
          <-- mux2
                skip_at_data
                (sresize (sel_bottom vramd2 ~width:16) ~width:25)
                (sresize std2 ~width:25)
        ; mul_b
          <-- mux2
                skip_at_data
                (mux
                   head_at_data
                   (Array.to_list (Array.map lay.d_skip ~f:(of_signed_int ~width:18))))
                (sresize (sel_bottom vramd2 ~width:16) ~width:18)
        ; when_
            mac.row_done
            [ oram_wen <-- vdd
            ; oram_waddr <-- mac_row_in
            ; oram_wdata <-- clamp16 (sra mac.sum ~by:Quantized.Constants.s_q)
            ]
        ; when_ mac.done_ finish
        ]
      in
      [ mac_go <-- vdd ], body
    | Gate ->
      (* the readout against the SiLU of the gate, one term a row, and the product lands
         WIDE: the norm that reads it divides by the size of the vector and does not care
         what scale it arrives in, thus a truncation here would throw away bits for
         nothing *)
      let body =
        [ mac_inner <--. 1
        ; mac_outer <--. d_in
        ; oram_raddr <-- mac_oo_in
        ; vram_raddr <-- uresize mac_oo_in ~width:vbits
        ; mul_a <-- sresize oramd2 ~width:25
        ; mul_b <-- sresize (sel_bottom vramd2 ~width:16) ~width:18
        ; when_
            mac.row_done
            [ vram_wen <-- vdd
            ; vram_waddr <-- uresize mac_row_in ~width:vbits
            ; vram_wdata <-- sel_bottom mac.sum ~width:32
            ]
        ; when_ mac.done_ finish
        ]
      in
      [ mac_go <-- vdd ], body
    | Attend { ring } ->
      (* The attention of one head after another, in four stages. Stage 0 scores the ages
         — one row of lanes an age — and tracks the peak. Stage 1 turns each score into
         its exp2 weight over the score's own vram row and sums the denominator. Stages 2
         and 3 merge ONE LANE: a walk of one row over the ages, weight row against value
         ring, then a wait on the divide that lands it.

         Era four merged every lane in one walk and froze that walk while a row's divide
         ran, thus every read register and every tag of its machine carried an enable. A
         lane a walk needs no freeze: the walk has retired before the divide starts. That
         is what keeps the rule of this machine — NO WALK EVER STALLS — with an attention
         layer inside it, and it is why [Mac] still takes its hold at ground.

         Age [a] reads slot [(cur - a) & (slots - 1)], thus the ALiBi distance is the age
         and the causal wall is the walk: [nfill] states how many ages the ring holds. *)
      let head_entry =
        [ ii <--. 0
        ; tick <--. 0
        ; den <--. 0
        ; peak <-- min32
        ; stage <--. 0
        ; mac_go <-- vdd
        ]
      in
      let entry = [ hd <--. 0 ] @ head_entry in
      let lane = sel_bottom ii.value ~width:lane_d_bits in
      let score_ages =
        [ mac_inner <--. head_d
        ; mac_outer <-- nfill
        ; qram_raddr <-- hd.value @: mac_ii_lane
        ; kc_raddr <-- ring_row ~ring ~slot:slot_of_oo ~dim:(hd.value @: mac_ii_lane)
        ; mul_a <-- sresize qd2 ~width:25
        ; mul_b <-- sresize kcd ~width:18
        ; when_
            mac.row_done
            [ vram_wen <-- vdd
            ; vram_waddr <-- uresize mac_row_slot ~width:vbits
            ; vram_wdata <-- score
            ; when_ (score >+ peak.value) [ peak <-- score ]
            ]
        ; when_ mac.done_ [ ii <--. 0; tick <--. 0; stage <--. 1 ]
        ]
      in
      let weigh_ages =
        exp_weight_chain
          ~addr:(sel_bottom ii.value ~width:slot_bits)
          ~scale:Quantized.Constants.log2e
          ~land_:[ vram_wdata <-- uresize exp2_e ~width:32; den <-- den_next ]
          ~advance:
            [ if_
                (ii.value ==: nfill -:. 1)
                [ ii <--. 0; stage <--. 2; mac_go <-- vdd ]
                [ ii <-- ii.value +:. 1 ]
            ]
      in
      let merge_lane =
        [ mac_inner <-- nfill
        ; mac_outer <--. 1
        ; vram_raddr <-- uresize mac_ii_slot ~width:vbits
        ; vc_raddr <-- ring_row ~ring ~slot:slot_of_ii ~dim:(hd.value @: lane)
        ; mul_a <-- uresize (sel_bottom vramd2 ~width:16) ~width:25
        ; mul_b <-- sresize vcd ~width:18
        ; when_
            mac.done_
            [ div_start <-- vdd
            ; div_num <-- sel_bottom mac.sum ~width:40
            ; div_den <-- den.value
            ; stage <--. 3
            ]
        ]
      in
      let land_lane =
        [ when_
            ~:div_busy
            [ yram_wen <-- vdd
            ; yram_waddr <-- uresize (hd.value @: lane) ~width:ybits
            ; yram_wdata <-- quotient16
            ; if_
                (ii.value ==:. head_d - 1)
                [ if_
                    (hd.value ==:. heads - 1)
                    finish
                    ([ hd <-- hd.value +:. 1 ] @ head_entry)
                ]
                [ ii <-- ii.value +:. 1; stage <--. 2; mac_go <-- vdd ]
            ]
        ]
      in
      entry, [ by_stage [ score_ages; weigh_ages; merge_lane; land_lane ] ]
    | Temper ->
      let entry = [ oo <--. 0; tick <--. 0; total <--. 0 ] in
      let keep = exp2_e >=: of_unsigned_int ~width:16 min_weight in
      let w = mux2 keep exp2_e (zero 16) in
      let body =
        exp_weight_chain
          ~addr:oo_class
          ~scale:temper
          ~land_:
            [ vram_wdata <-- uresize w ~width:32
            ; total <-- total.value +: uresize w ~width:24
            ]
          ~advance:[ if_ (oo.value ==:. classes - 1) finish [ oo <-- oo.value +:. 1 ] ]
      in
      entry, body
    | Draw ->
      let entry = [ tick <--. 0 ] in
      let body =
        [ by_tick
            [ [ prng_step <-- vdd ]
            ; [ prng_step <-- vdd; u24 <-- sel_bottom u24.value ~width:16 @: prng_byte ]
            ; [ prng_step <-- vdd; u24 <-- sel_bottom u24.value ~width:16 @: prng_byte ]
            ; [ u24 <-- sel_bottom u24.value ~width:16 @: prng_byte ] @ finish
            ]
        ]
      in
      entry, body
    | Threshold ->
      (* (u24 * total) >> 24 in two DSP passes: the high twelve bits of the total, then
         the low twelve — the same integer as one wide multiply *)
      let entry = [ tick <--. 0 ] in
      let body =
        [ mul_a <-- uresize u24.value ~width:25
        ; mul_b
          <-- uresize
                (mux2
                   (tick.value <:. 2)
                   (select total.value ~high:23 ~low:12)
                   (sel_bottom total.value ~width:12))
                ~width:18
        ; by_tick
            [ []
            ; []
            ; [ thi <-- mac.product ]
            ; []
            ; [ thr
                <-- sel_bottom
                      (srl
                         (sll (uresize thi.value ~width:56) ~by:12
                          +: uresize mac.product ~width:56)
                         ~by:24)
                      ~width:24
              ]
              @ finish
            ]
        ]
      in
      entry, body
    | Pick ->
      (* The first class whose running total passes the threshold, and the last class
         catches a walk that no weight stopped — which is the rule of the reference and
         not a fallback: the threshold is below the total by construction. *)
      let entry = [ oo <--. 0; tick <--. 0; cum <--. 0; found <--. 0 ] in
      let body =
        [ vram_raddr <-- uresize oo_class ~width:vbits
        ; by_tick
            [ []
            ; [ (let w = sel_bottom vramd ~width:24 in
                 let cum_next = cum.value +: uresize w ~width:25 in
                 let passes = cum_next >: uresize thr.value ~width:25 in
                 proc
                   [ cum <-- cum_next
                   ; when_
                       ~:(found.value)
                       [ when_ passes [ found <-- vdd; write_drawn oo_class ]
                       ; when_
                           (oo.value ==:. classes - 1)
                           [ write_drawn (of_unsigned_int ~width:class_bits (classes - 1))
                           ]
                       ]
                   ])
              ; tick <--. 0
              ; if_ (oo.value ==:. classes - 1) finish [ oo <-- oo.value +:. 1 ]
              ]
            ]
        ]
      in
      entry, body
  in
  (* the link: op [k]'s finish is op [k+1]'s entry and the pc move; the last op of a
     program takes the final actions instead *)
  let rec link index final = function
    | [] -> final, []
    | op :: rest ->
      let next_entry, tail = link (index + 1) final rest in
      let entry, body = build op ~finish:next_entry in
      entry @ [ pc <--. index ], (index, body) :: tail
  in
  (* The chain is one seat and it runs once for each of them, counting down from the
     soprano. Its last op returns to its first until the bass has drawn, thus the loop
     closes on the op boundary and costs no cycle of its own.

     The head's entry is taken from a build of its own: an op's ENTRY is the actions its
     predecessor runs and does not depend on the op's finish, thus building the head twice
     states the loop back without a circular definition. *)
  let chain_head = List.hd_exn prog.chain in
  let chain_head_entry = fst (build chain_head ~finish:[]) @ [ pc <--. forward_length ] in
  let chain_done =
    [ if_
        (seat.value ==:. 0)
        [ sm.set_next Idle ]
        ([ seat <-- seat.value -:. 1 ] @ chain_head_entry)
    ]
  in
  let chain_entry, chain_bodies = link forward_length chain_done prog.chain in
  let enter_chain = [ seat <--. Frame.voices - 1 ] @ chain_entry in
  (* The forward has stated step [s], thus the chain would draw the step after it. Through
     the lead-in the chain does not run: the frame stays silence, the drawn classes stand
     at [Vocab.silence], and the PRNG does not move, because [Draw] is the only thing that
     steps it. *)
  let next_index = s.value +:. 1 in
  let forward_done =
    [ s <-- next_index
    ; if_ (next_index >=:. Jsb.bar_steps) enter_chain [ sm.set_next Idle ]
    ]
  in
  let forward_entry, forward_bodies = link 0 forward_done prog.forward in
  let run_body =
    [ switch
        pc.value
        (List.map (forward_bodies @ chain_bodies) ~f:(fun (index, body) ->
           of_unsigned_int ~width:pc_bits index, body))
    ]
  in
  compile
    [ valid <-- gnd
    ; sm.switch
        [ ( Idle
          , (* The one reset is the rewind, and it runs nothing: the state reads as zero
               at step 0 and the taps read zero by their age rule, thus neither memory
               needs a clearing walk. *)
            [ when_
                i.rewind
                ([ s <--. 0; held <--. Frame.silent ]
                 @ List.init Frame.voices ~f:(fun seat -> drawn.(seat) <--. Vocab.silence)
                )
            ; when_
                (i.step &: ~:(i.rewind))
                ([ held <-- frame_word; valid <-- vdd ]
                 @ forward_entry
                 @ [ sm.set_next Run ])
            ] )
        ; Run, run_body
        ]
    ];
  { O.frame = held.value; valid = valid.value; idle = sm.is Idle }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the program is data: the state table prints" =
  let model = Quantized.Model.For_test.init Quantized.Model.For_test.config ~seed:11 in
  let { chain; forward } = schedule model in
  let show tag ops =
    List.iteri ops ~f:(fun index op ->
      Stdio.printf "%s%-2d %s\n" tag index (Sexp.to_string (Op.sexp_of_t op)))
  in
  show "f" forward;
  show "c" chain;
  (* The elected shape, which no simulation can afford and every cost of the board is: the
     op count and the cycles of a drawn step at a full ring, out of the cost model. The
     bench above holds that model to the measured circuit at a shape a test can run. *)
  let elected = Mamba.Config.baseline in
  let baseline = schedule (Quantized.Model.For_test.init elected ~seed:11) in
  let step ops = List.sum (module Int) ops ~f:(Op.cycles elected ~n:elected.ring) in
  Stdio.printf
    "%s: %d forward ops, %d chain ops, %d cycles a drawn step\n"
    (Mamba.Kind.spell elected.plan)
    (List.length baseline.forward)
    (List.length baseline.chain)
    (step baseline.forward + (Frame.voices * step baseline.chain));
  [%expect
    {|
    f0  (Embed(seats 0)(phase 3072)(e 10))
    f1  (Rms_norm(over Embedding))
    f2  (Rms_norm(over Stream))
    f3  (Matvec((src Y)(w((base(Fixed 3328))(e 10)))(outer_major true)(inner 16)(outer 82)(landing To_v)))
    f4  (Conv(block 0)(w((base(Fixed 4640))(e 11))))
    f5  (Silu_over(from 32)(count 48))
    f6  (Decay(block 0))
    f7  (State_update(block 0))
    f8  (Readout(block 0))
    f9  (Silu_over(from 0)(count 32))
    f10 Gate
    f11 (Rms_norm(over Gated))
    f12 (Matvec((src Y)(w((base(Fixed 4832))(e 11)))(outer_major false)(inner 32)(outer 16)(landing Add_to_h)))
    f13 (Rms_norm(over Stream))
    f14 (Matvec((src Joined)(w((base(Fixed 5344))(e 11)))(outer_major false)(inner 32)(outer 16)(landing To_q)))
    f15 (Matvec((src Joined)(w((base(Fixed 5856))(e 10)))(outer_major false)(inner 32)(outer 16)(landing(To_ring(k true)(ring 0)))))
    f16 (Matvec((src Y)(w((base(Fixed 6368))(e 11)))(outer_major false)(inner 16)(outer 16)(landing(To_ring(k false)(ring 0)))))
    f17 (Attend(ring 0))
    f18 (Matvec((src Y)(w((base(Fixed 6624))(e 11)))(outer_major false)(inner 16)(outer 16)(landing Add_to_h)))
    f19 (Rms_norm(over Stream))
    f20 (Matvec((src Y)(w((base(Fixed 6880))(e 10)))(outer_major false)(inner 16)(outer 64)(landing To_hidden)))
    f21 (Matvec((src Hidden)(w((base(Fixed 7904))(e 10)))(outer_major false)(inner 64)(outer 16)(landing Add_to_h)))
    c0  (Rms_norm(over Stream))
    c1  (Matvec((src Y)(w((base(Seat_block 0))(e 10)))(outer_major true)(inner 16)(outer 48)(landing To_logits)))
    c2  Temper
    c3  Draw
    c4  Threshold
    c5  Pick
    c6  (Accumulate(base(Seat_block 0))(e 10))
    MMMMMMZF: 77 forward ops, 7 chain ops, 403074 cycles a drawn step
    |}]
;;

(* the first step where the two walks part, if they part: a mismatch names the step to
   read, and a walk of tens of steps hides that index inside two long lists *)
let first_divergence circuit reference =
  List.findi (List.zip_exn circuit reference) ~f:(fun (_ : int) (c, r) -> c <> r)
  |> Option.map ~f:fst
;;

(* The frame comparison: the circuit against the reference, step for step, on drawn
   weights. This is the gate that holds the circuit to [Quantized], and the walk crosses
   the lead-in — the first drawn step is the one that reads a state the lead-in filled. *)
let frames_agree ~model ~seed ~steps =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 seed)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let budget = ref 20_000_000 in
  let cycle () =
    Cyclesim.cycle sim;
    Int.decr budget;
    assert (!budget > 0)
  in
  inp.rewind := Bits.vdd;
  cycle ();
  inp.rewind := Bits.gnd;
  cycle ();
  (* One step: the command, then the cycle [valid] answers it — the frame stands there,
     because the chain that moves the drawn classes runs behind the recurrence. *)
  let step () =
    inp.step := Bits.vdd;
    cycle ();
    inp.step := Bits.gnd;
    cycle ();
    assert (Bits.to_bool !(out.valid));
    let frame = Bits.to_int_trunc !(out.frame) in
    while not (Bits.to_bool !(out.idle)) do
      cycle ()
    done;
    frame
  in
  let circuit =
    List.rev
      (List.fold (List.range 0 steps) ~init:[] ~f:(fun acc (_ : int) -> step () :: acc))
  in
  let (_ : Quantized.Engine.t), reference =
    List.fold_map
      (List.range 0 steps)
      ~init:(Quantized.Engine.init model ~seed)
      ~f:(fun engine (_ : int) ->
        let engine, step = Quantized.Engine.next_step engine in
        engine, step.Quantized.Engine.frame)
  in
  List.iteri circuit ~f:(fun index frame ->
    if index < 2 || index >= Jsb.bar_steps - 1
    then Stdio.printf "step %2d  %08x\n" index frame);
  let divergence = first_divergence circuit reference in
  Stdio.printf "%d steps, the frames agree: %b\n" steps (Option.is_none divergence);
  Option.iter divergence ~f:(fun index ->
    Stdio.printf "the first step that parts is %d\n" index;
    Stdio.print_s ([%sexp_of: int list] circuit);
    Stdio.print_s ([%sexp_of: int list] reference))
;;

(* The three memories that carry a layer field, and the two rules they take.

   The state address and the ring address PACK: [d_in], [state], [d] and the ring depth
   are powers of two, thus the region stands above the row and the whole address is a
   concatenation. The tap address ADDS: the channel count is not a power of two, thus a
   concatenated layer field would stride by the rounded-up power and the top block would
   run off the end of the memory.

   The region field is EMPTY at one block and at one head, thus the plan of a gate decides
   which half of each rule the simulation ever elaborates — which is why the gates below
   run at two blocks and at two heads as well. *)
let memory_geometry (config : Mamba.Config.t) =
  let { Mamba.Config.d; d_in; state; ring; _ } = config in
  let channels = Mamba.Config.channels config in
  let blocks = Mamba.Config.blocks config in
  let rings = Mamba.Config.attentions config in
  let inbits = Int.floor_log2 d_in in
  let nbits = Int.floor_log2 state in
  let state_bits = address_bits_for (blocks * d_in * state) in
  let chan_bits = address_bits_for channels in
  let ring_bits = address_bits_for (Int.max 1 rings * ring * d) in
  Stdio.printf
    "%d blocks, %d rings: state %d rows, %d bits = block %d + lane %d + n %d (packed); \
     taps %d rows, %d channels in %d bits, block stride %d (added); ring %d rows, %d \
     bits = head %d + slot %d + dim %d (packed)\n"
    blocks
    rings
    (blocks * d_in * state)
    state_bits
    (state_bits - inbits - nbits)
    inbits
    nbits
    (blocks * channels * config.taps)
    channels
    chan_bits
    (channels * config.taps)
    (Int.max 1 rings * ring * d)
    ring_bits
    (ring_bits - Int.floor_log2 ring - Int.floor_log2 d)
    (Int.floor_log2 ring)
    (Int.floor_log2 d)
;;

(* THE RESIDUAL STREAM, WRITE FOR WRITE, and it is the sharp instrument of this era.

   The frame gate below is blunt at the shape a test can afford: weights of scale 0.02 put
   the classes so near each other that a pick is almost the quantile of its uniform alone,
   thus a datapath can be wrong by tens of percent and still draw the same frames for a
   dozen steps. This walks the circuit's h RAM instead — the embed and each layer's join
   write the whole stream, in that order — and holds every element of it against
   [Quantized.Engine.For_test.layer_streams].

   Four faults were found through it and none of them moved a frame at first: a weight
   addressed by a concatenation whose stride was not the tensor's width, a convolution
   channel block read at the gate's offset, an operand selected on the address side of a
   two-cycle read, and a tap ring whose layer stride ran the top layer off the end of its
   memory. A gate that only compares frames would have shipped all four. *)
let streams_agree ~model ~seed ~steps =
  let config = model.Quantized.Model.config in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create
      ~config:Cyclesim.Config.trace_all
      (create ~model ~seed:(of_unsigned_int ~width:32 seed))
  in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let node name = Option.value_exn (Cyclesim.lookup_node_by_name sim name) in
  let wen = node "hram_wen" in
  let waddr = node "hram_waddr" in
  let wdata = node "hram_wdata" in
  let h = Array.create ~len:config.d 0 in
  let writes = ref 0 in
  let snapshots = ref [] in
  let signed v = if v >= 1 lsl 31 then v - (1 lsl 32) else v in
  let cycle () =
    Cyclesim.cycle sim;
    if Cyclesim.Node.to_int wen = 1
    then (
      h.(Cyclesim.Node.to_int waddr) <- signed (Cyclesim.Node.to_int wdata);
      Int.incr writes;
      (* one snapshot for each time the whole stream is written: the embed, then the join
         of each layer, then the accumulates of the chain *)
      if !writes % config.d = 0 then snapshots := Array.copy h :: !snapshots)
  in
  inp.rewind := Bits.vdd;
  cycle ();
  inp.rewind := Bits.gnd;
  cycle ();
  let engine = ref (Quantized.Engine.init model ~seed) in
  let checked = ref 0 in
  let parted = ref 0 in
  for step = 0 to steps - 1 do
    writes := 0;
    snapshots := [];
    inp.step := Bits.vdd;
    cycle ();
    inp.step := Bits.gnd;
    cycle ();
    (* the frame is whole only after the strobe: the command is what latches it *)
    let frame = Bits.to_int_trunc !(out.frame) in
    while not (Bits.to_bool !(out.idle)) do
      cycle ()
    done;
    let taken = List.rev !snapshots in
    let want =
      Quantized.Engine.For_test.layer_streams !engine ~frame ~phase:(step % Jsb.bar_steps)
    in
    List.iteri want ~f:(fun index reference ->
      Int.incr checked;
      match List.nth taken index with
      | None -> Stdio.printf "step %d wrote no stream at %d\n" step index
      | Some got ->
        if not (Array.equal Int.equal reference got)
        then (
          Int.incr parted;
          Stdio.printf
            "step %d, stream write %d: %d of %d elements part\n"
            step
            index
            (Array.counti reference ~f:(fun i v -> v <> got.(i)))
            config.d));
    let next, (_ : Quantized.Engine.step) = Quantized.Engine.next_step !engine in
    engine := next
  done;
  Stdio.printf "%d stream writes over %d steps, %d part\n" !checked steps !parted
;;

let%expect_test "the source agrees with the reference, frame for frame" =
  let config = Quantized.Model.For_test.config in
  memory_geometry config;
  frames_agree ~model:(Quantized.Model.For_test.init config ~seed:11) ~seed:42 ~steps:20;
  [%expect
    {|
    1 blocks, 1 rings: state 256 rows, 8 bits = block 0 + lane 5 + n 3 (packed); taps 192 rows, 48 channels in 6 bits, block stride 192 (added); ring 128 rows, 7 bits = head 0 + slot 3 + dim 4 (packed)
    step  0  00000000
    step  1  00000000
    step 15  00000000
    step 16  aac7cbad
    step 17  b5cbcd00
    step 18  b6cad0cf
    step 19  a7d1adbd
    20 steps, the frames agree: true
    |}]
;;

let%expect_test "the source agrees with the reference, stream write for stream write" =
  let config = Quantized.Model.For_test.config in
  streams_agree ~model:(Quantized.Model.For_test.init config ~seed:11) ~seed:42 ~steps:22;
  [%expect {| 88 stream writes over 22 steps, 0 part |}]
;;

(* The seed 0 on the circuit. It is the fixed point of xorshift32 and the panel can state
   it — all the slide switches down is the rest position of the board — thus the walk
   stands still: every threshold is 0 and each seat takes the first class that min-p left
   standing. [Quantized] states that walk, and this holds the circuit to it, where a PRNG
   that reset to another state or a threshold that rounded the other way would show. *)
let%expect_test "the source agrees with the reference at the seed 0" =
  let config = Quantized.Model.For_test.config in
  frames_agree ~model:(Quantized.Model.For_test.init config ~seed:11) ~seed:0 ~steps:20;
  [%expect
    {|
    step  0  00000000
    step  1  00000000
    step 15  00000000
    step 16  00000000
    step 17  00000000
    step 18  00000000
    step 19  00000000
    20 steps, the frames agree: true
    |}]
;;

(* The same gate at TWO LAYERS, and it is not a wider shape for its own sake. At one layer
   the layer field of the state address and of the tap address is empty and the per-layer
   ROM bases all read the first layer's, thus the [else] branch of [state_row] and of
   [tap_row] and every address that carries a layer are dead in a one-layer simulation and
   live only on the board, which runs six. Era four's test review found exactly that gap
   late. This gate elaborates them.

   The walk also runs past the tap ring — four slots — many times over, thus the ring
   wraps and the age rule is exercised where it is a wrap and not only where it is the
   origin. *)
let%expect_test "the source agrees with the reference at two layers" =
  let config =
    { Quantized.Model.For_test.config with
      plan = [| Block; Block; Attention; Attention |]
    }
  in
  memory_geometry config;
  frames_agree ~model:(Quantized.Model.For_test.init config ~seed:23) ~seed:42 ~steps:24;
  [%expect
    {|
    2 blocks, 2 rings: state 512 rows, 9 bits = block 1 + lane 5 + n 3 (packed); taps 384 rows, 48 channels in 6 bits, block stride 192 (added); ring 256 rows, 8 bits = head 1 + slot 3 + dim 4 (packed)
    step  0  00000000
    step  1  00000000
    step 15  00000000
    step 16  aac7cbad
    step 17  b5cbcd00
    step 18  b6c9d0cf
    step 19  a6d1adbc
    step 20  c9a7b0ae
    step 21  bfaeaab9
    step 22  adc0cfa5
    step 23  bac1b5b6
    24 steps, the frames agree: true
    |}]
;;

(* The stream gate at THREE layers, which is where the tap ring's layer stride showed: at
   one layer the field is absent, at two the top layer's region still fits the memory by
   accident of rounding, and at three it runs off the end. A shape that only ever ran two
   would have passed a circuit the board could not run at six. *)
let%expect_test "the source agrees with the reference at three layers" =
  let config =
    { Quantized.Model.For_test.config with
      d = 32
    ; d_in = 64
    ; heads = 4
    ; state = 16
    ; plan = [| Block; Block; Attention; Block; Attention; Feed_forward |]
    }
  in
  memory_geometry config;
  streams_agree ~model:(Quantized.Model.For_test.init config ~seed:37) ~seed:7 ~steps:20;
  [%expect
    {|
    3 blocks, 2 rings: state 3072 rows, 12 bits = block 2 + lane 6 + n 4 (packed); taps 1152 rows, 96 channels in 7 bits, block stride 384 (added); ring 512 rows, 9 bits = head 1 + slot 3 + dim 5 (packed)
    140 stream writes over 20 steps, 0 part
    |}]
;;

(* The same stream gate at a WIDE STATE AND A WIDE KERNEL, and it is the net under the two
   fields the sweep of the quality round moves. K was a constant of this library until
   that round, thus every width it sizes — the tap ring, the age mux over it and the layer
   stride of the ring — was proven at four alone; N was a field, but no gate ran it above
   16 and the state address never had to carry a wider n. Here K is 16 and N is 32, over
   three layers so that the layer field of both addresses is live, and everything else is
   as small as the gates above.

   A stride written for one K and an address field written for one N both land here. *)
let%expect_test "the source agrees with the reference at a wide state and kernel" =
  let config =
    { Quantized.Model.For_test.config with
      state = 32
    ; taps = 16
    ; plan = [| Block; Block; Block |]
    }
  in
  memory_geometry config;
  streams_agree ~model:(Quantized.Model.For_test.init config ~seed:53) ~seed:9 ~steps:20;
  [%expect
    {|
    3 blocks, 0 rings: state 3072 rows, 12 bits = block 2 + lane 5 + n 5 (packed); taps 4608 rows, 96 channels in 7 bits, block stride 1536 (added); ring 128 rows, 7 bits = head 0 + slot 3 + dim 4 (packed)
    80 stream writes over 20 steps, 0 part
    |}]
;;

(* The cycle bench: the circuit's measured cost against [Op.cycles], step by step. A step
   costs its recurrence, the chain behind it once the lead-in is past, and two cycles of
   command — the cycle that takes [step] and runs the first op's entry, and the cycle the
   bench spends dropping the strobe.

   THE FILL COUNT IS BACK, and it reaches one op. The trunk's cost is a constant of the
   shape, thus a plan of blocks alone costs the same at every step; the Zamba head reads a
   ring, thus a step of a hybrid grows until the ring is full and is constant after it.
   The bench states the fill of each step and the model takes it. *)
let bench ~steps () =
  let config = Quantized.Model.For_test.config in
  let model = Quantized.Model.For_test.init config ~seed:11 in
  let prog = schedule model in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 42)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  (* the ages the ring holds at step [index], which is what [Quantized.Engine] walks *)
  let fill index = Int.min (index + 1) config.ring in
  let sum_ops ~n ops =
    List.fold ops ~init:0 ~f:(fun total op -> total + Op.cycles config ~n op)
  in
  let count_until_idle () =
    let cycles = ref 0 in
    while not (Bits.to_bool !(out.idle)) do
      Cyclesim.cycle sim;
      Int.incr cycles;
      assert (!cycles < 20_000_000)
    done;
    !cycles
  in
  inp.rewind := Bits.vdd;
  Cyclesim.cycle sim;
  inp.rewind := Bits.gnd;
  Cyclesim.cycle sim;
  (* the one reset runs no pass: it loads the PRNG and clears the counters *)
  Stdio.printf "rewind: measured %d\n" (2 + count_until_idle ());
  let step index =
    inp.step := Bits.vdd;
    Cyclesim.cycle sim;
    inp.step := Bits.gnd;
    Cyclesim.cycle sim;
    let measured = 2 + count_until_idle () in
    let draws = index + 1 >= Jsb.bar_steps in
    let n = fill index in
    let modeled =
      2
      + sum_ops ~n prog.forward
      + if draws then Frame.voices * sum_ops ~n prog.chain else 0
    in
    if index < 1 || index >= Jsb.bar_steps - 1
    then
      Stdio.printf
        "step %2d: %s measured %d, model %d, delta %d\n"
        index
        (if draws then "draws, " else "silent,")
        measured
        modeled
        (measured - modeled);
    measured, modeled
  in
  let rounds =
    List.rev
      (List.fold (List.range 0 steps) ~init:[] ~f:(fun acc index -> step index :: acc))
  in
  Stdio.printf
    "%d steps, %d disagree, total %d\n"
    steps
    (List.count rounds ~f:(fun (measured, modeled) -> measured <> modeled))
    (List.fold rounds ~init:0 ~f:(fun total (measured, (_ : int)) -> total + measured))
;;

let%expect_test "the cycle bench: the measured walk against the cost model" =
  bench ~steps:18 ();
  [%expect
    {|
    rewind: measured 2
    step  0: silent, measured 12379, model 12379, delta 0
    step 15: draws,  measured 20621, model 20621, delta 0
    step 16: draws,  measured 20621, model 20621, delta 0
    step 17: draws,  measured 20621, model 20621, delta 0
    18 steps, 0 disagree, total 251090
    |}]
;;
