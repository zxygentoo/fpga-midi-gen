(* The transformer note source. [source.mli] states the contract, and
   [docs/transformer_rtl.md] states the design of the whole — the five layers, the
   memories and the cost. This header holds what neither says: the reasons tied to this
   code.

   The file runs L2, the schedule, then L3, the compiler, then L4, the outer FSM. L0 and
   L1 are the units it drives.

   The rules that hold the shape:

   - A unit of L1 is built once and the ops mux its ports. No op owns a resource, and the
     per-op facts stay in the ops: the address formulas, the operand sources, the landing
     writes.
   - The ops dispatch as one [switch] on the pc, not as a chain of guards. [Always]
     compiles a statement list into a linear chain of muxes, one level for each statement
     that writes a target, and that chain over every op was the thinnest timing path
     measured in this era; a switch with constant cases compiles into one parallel case.
   - An op's finish runs the next op's entry actions in the same cycle. This one
     convention replaces a hand-kept register reset for each op.
   - The op vocabulary is closed and concrete. The rule: when a field's meaning would
     depend on another field, stop extending and write a new op.
   - A repeated op is an inlined program step and takes no return register. [Rms_norm]
     appears twice in a layer and once in each seat of the chain; control is cheap, and
     the units it drives exist once.
   - [Attend] walks lane-major. The age-major order — one exp2, then one MAC over the
     lanes, age by age — was declined: it needs a register file of lane sums, and
     lane-major needs none.

   The timing design, decided against the measured paths (2026-08-13):

   - Every read is two cycles from address to data. The rings and the small RAMs spend
     both on the data side: the read register, then an output register that packs into the
     block RAM (DO_REG; the clock-to-out falls 2.46 -> 0.89 ns at speed grade -1). The ROM
     spends the first cycle on its ADDRESS instead — the note at [rom_banked] states why
     that register is load-bearing. The six-layer build failed on the route from a far ROM
     bank into the DSP at 94 percent occupancy; the travel stage pays for that route, and
     at one term a cycle it costs only fill latency. A ROM bank registers its own data
     before the select mux — a register after the mux stays in the fabric and removes
     nothing. The bespoke chains (Exp, Temper) read the small RAMs at the one-register
     tap; they touch neither the ROM nor the rings.
   - The DSP stays a two-register multiply — the operand registers and [preg] — rated 257
     MHz at -1 against the 100 MHz clock. The accumulator is a fabric adder behind it,
     loaded on a row's first term by the tag. The full in-DSP accumulate was declined: the
     M register cannot drive the fabric, thus every reader of the product moves one
     register later, for headroom this clock does not spend. The bespoke readers of [preg]
     anchor it as the DSP's final register, thus the tools cannot fold the accumulator in.
   - The dormant debt, recorded: the product latency — two cycles from the operands to
     [preg] — is hand-encoded as tick positions, in [exp_weight_chain] for the Exp and
     Temper chains and in the Threshold chain. If the pipe ever deepens, replace the tick
     counts with a wait on a valid bit; do not renumber. *)

open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

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
     block, and the four addresses are constants that one mux chooses. The address is
     stated here and not implied by another field, thus a reader of the schedule sees
     which ops the seat register reaches. *)
  type where =
    | Fixed of int (* the tensor begins at this address *)
    | Seat_block of
        int (* the seat tensor begins here; the seat names the block inside it *)
  [@@deriving sexp_of]

  (* one weight tensor as the circuit sees it: where it starts, and its exponent *)
  type tensor =
    { base : where
    ; e : int
    }
  [@@deriving sexp_of]

  (* where a finished matvec sum lands *)
  type landing =
    | To_q (* clamp16 (rescale to kv_q), the query RAM *)
    | To_ring of
        { k : bool
        ; layer : int
        }
      (* clamp16 (rescale to kv_q), the ring row of slot [cur] *)
    | To_hidden (* rescale to hid_q, relu, clamp16 — the FFN hidden *)
    | To_logits (* one shift by [e]; the peak tracked for the temper *)
    | Add_to_h (* the residual join: the whole sum, rescaled onto the stream *)
  [@@deriving sexp_of]

  type matvec =
    { src : [ `Y | `Hidden ] (* the normed vector, or the FFN hidden *)
    ; w : tensor
    ; outer_major : bool (* the weight address order; true only for a seat readout *)
    ; inner : int
    ; outer : int
    ; landing : landing
    }
  [@@deriving sexp_of]

  (* One step of the walk: the facts that one case of the pc switch needs. The bespoke ops
     close over the model at elaboration and carry no fields here. *)
  type t =
    (* The five rows of the input add row for row — the seat row of each of the four
       seats, and the bar phase — thus one exponent covers them all and the walk reads two
       bases. [Quantized.Model.check_shape] holds the rule, and [create] calls it before
       it reads a base. The seat rows are addressed by the classes the chain drew. *)
    | Embed of
        { seats : int
        ; phase : int
        ; e : int
        }
    | Rms_norm
    | Matvec of matvec
    | Attend of { layer : int }
    | Temper
    | Draw
    | Threshold
    (* the drawn class lands in the register of the seat the chain is at *)
    | Pick
    (* The chain writes the row the seat drew onto the stream, in place: the stream is
       dead after the chain, because the forward pass starts from the embedding. The last
       seat writes a row that nothing reads — the reference does not add it, and the walk
       does not test for it, because the test would cost a case of the program and the row
       costs [d] cycles of a step that has hundreds of thousands. *)
    | Accumulate of
        { base : where
        ; e : int
        }
  [@@deriving sexp_of]

  (* The cycles the builders below cost, each derived from the unit that spends them. The
     bespoke chains are the one exception: a chain's cost is the length of its tick list,
     which lives inside [create] where the bodies close over the datapath, thus the four
     counts stand here as numbers. The cycle bench holds every one of these against the
     measured circuit. *)
  module Cost = struct
    (* a walk retires its last term this long after its last issue *)
    let drain = Mac.read_latency + 2

    (* the start cycle, the walk, and the cycle the wait releases *)
    let divide = Divider.busy_cycles + 2

    (* the same, less the start cycle the caller's walk already counts *)
    let pending_hold = Divider.busy_cycles + 1
    let root = Isqrt.busy_cycles + 1

    (* the ticks of [exp_weight_chain], [Draw], [Threshold] and [Pick] *)
    let exp_weight = 7
    let draw = 4
    let threshold = 5
    let pick = 2
  end

  (* The analytic cost of one op, in cycles from its go to its finish, both the cycle a
     predecessor's finish runs. [n] is the filled slot count of the ring at this step. *)
  let cycles (config : Transformer.Config.t) ~n (op : t) =
    let { Transformer.Config.d; heads; _ } = config in
    let head_d = d / heads in
    let classes = Vocab.classes in
    match op with
    (* four seat rows and the phase row *)
    | Embed _ -> ((Frame.voices + 1) * d) + Cost.drain
    | Rms_norm ->
      (* the sum-squares walk, the isqrt, then one read tick and one divide an element *)
      d + Cost.drain + Cost.root + (d * (1 + Cost.divide))
    | Matvec { inner; outer; landing; _ } ->
      (inner * outer)
      + Cost.drain
      +
        (match landing with
        | Add_to_h -> 1
        | To_q | To_ring _ | To_hidden | To_logits -> 0)
    | Attend _ ->
      heads
      * ((n * head_d)
         + Cost.drain (* the score walk: one row of lanes an age *)
         + (Cost.exp_weight * n) (* the weight of each age *)
         + (n * head_d)
         + Cost.drain (* the merge walk: one row of ages a lane *)
         + (Cost.pending_hold * head_d) (* its one pending divide a lane *))
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
   register down from the soprano. The four seats were inlined at first, which made the
   step one straight-line program and gave the machine no counter at all; that shape cost
   47 percent more fabric than the circuit of era three — 4 406 slice LUTs against 2 999
   at the same six layers and the same 93 percent of the block RAM — and the six-layer
   build missed the period by 0.4 ns where era three met it by 0.110. The cost is not the
   ops themselves but the muxes they share: every case of the program counter that writes
   a register widens that register's parallel case, and inlining put four writers where
   era three had one. The seat register is the price of the room, and it buys back three
   quarters of the chain's control. *)
type program =
  { chain : Op.t list
  ; forward : Op.t list
  }

let schedule (model : Quantized.Model.t) : program =
  let { Transformer.Config.d
      ; layers
      ; heads = (_ : int)
      ; context = (_ : int)
      ; slope_span = (_ : int)
      }
    =
    model.config
  in
  let dff = 4 * d in
  let classes = Vocab.classes in
  let bases = Quantized.Model.rom_bases model in
  let tensor_at (q : Quantized.Model.quantized) base = { Op.base; e = q.e } in
  (* every matvec reads [`Y] into [d] rows of [d] terms, inner-major; the defaults hold
     for all but the two FFN weights and the seat readout *)
  let matvec ?(src = `Y) ?(outer_major = false) ?(inner = d) ?(outer = d) w base landing =
    Op.Matvec { src; w = tensor_at w (Op.Fixed base); outer_major; inner; outer; landing }
  in
  let layer l =
    let w = model.params.layers.(l) in
    let b = bases.Transformer.Params_data.layers.(l) in
    [ Op.Rms_norm
    ; matvec w.wq b.wq To_q
    ; matvec w.wk b.wk (To_ring { k = true; layer = l })
    ; matvec w.wv b.wv (To_ring { k = false; layer = l })
    ; Attend { layer = l }
    ; matvec w.wo b.wo Add_to_h
    ; Rms_norm
    ; matvec ~outer:dff w.w1 b.w1 To_hidden
    ; matvec ~src:`Hidden ~inner:dff w.w2 b.w2 Add_to_h
    ]
  in
  (* One seat of the chain, and the machine runs it once for each seat: the stream the
     seats above wrote, that seat's table read backward, the draw, and the row the draw
     names. The seat register names the block of the seat tensor, thus the four readouts
     are one program and one mux over four constant addresses. *)
  let seat_block = Op.Seat_block bases.seats in
  { chain =
      [ Op.Rms_norm
      ; Op.Matvec
          { src = `Y
          ; w = tensor_at model.params.seats seat_block
          ; outer_major = true
          ; inner = d
          ; outer = classes
          ; landing = To_logits
          }
      ; Temper
      ; Draw
      ; Threshold
      ; Pick
      ; Accumulate { base = seat_block; e = model.params.seats.e }
      ]
  ; forward =
      Embed { seats = bases.seats; phase = bases.phase; e = model.params.seats.e }
      :: List.concat (List.init layers ~f:layer)
  }
;;

(* ==================================================================== *)
(* L4 — the outer FSM; L1 and L3 live inside [create] *)
(* ==================================================================== *)

(* the step: [forward] runs under [Run], and [chain] behind it; [Idle] holds the frame the
   chain drew and takes the commands *)
module State = struct
  type t =
    | Idle
    | Run
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(model : Quantized.Model.t) ~seed (i : _ I.t) : _ O.t =
  let { Quantized.Model.config; params; temper; min_weight } = model in
  let { Transformer.Config.d; heads; context = slots; slope_span = span; layers } =
    config
  in
  let head_d = d / heads in
  let dff = 4 * d in
  let classes = Vocab.classes in
  (* the shift rules of the reference; the packing below derives every width *)
  Quantized.Model.check_shape model;
  (* the bar phase is a slice of the step counter, as every period of this design is *)
  assert (Int.is_pow2 Jsb.bar_steps);
  let dbits = Int.floor_log2 d in
  let lane_bits = Int.floor_log2 head_d in
  let head_bits = Int.floor_log2 heads in
  let slot_bits = Int.floor_log2 slots in
  let phase_bits = Int.floor_log2 Jsb.bar_steps in
  let class_bits = address_bits_for classes in
  let seat_bits = address_bits_for Frame.voices in
  let ring_bits = address_bits_for (layers * slots * d) in
  (* vram serves the scores, the FFN hidden, the logits and the sampler weights *)
  let vram_size = Int.max classes (Int.max dff slots) in
  let vbits = address_bits_for vram_size in
  let score_shift = Quantized.Constants.score_shift ~head_d in
  let prog = schedule model in
  let forward_length = List.length prog.forward in
  let pc_bits = address_bits_for (forward_length + List.length prog.chain) in
  let rom_bits = Quantized.Model.rom_bits model in
  let rom_addr_bits = address_bits_for (Array.length rom_bits) in
  let rom_const at = of_unsigned_int ~width:rom_addr_bits at in
  let min32 = of_signed_int ~width:32 (-(1 lsl 31)) in
  let eps48 = of_unsigned_int ~width:48 Quantized.Constants.eps_q in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  (* L1 — the walk registers *)
  let pc = Variable.reg spec ~width:pc_bits in
  (* The position inside a bespoke chain. It is a COUNTER and not a target of every case
     that runs one: it steps by itself, and a body states only where it holds and where it
     returns to the head. The register therefore takes one assignment, and the cases state
     two one-bit conditions that the tools reduce to an OR.

     The first six-layer build of era four put 48 cases of the program counter on this
     register, and the parallel case of a three-bit value over those cases stood five
     levels deep in the critical path. A hold and a reset carry the same control in one
     bit each. *)
  let tick = Variable.reg spec ~width:3 in
  let stage = Variable.reg spec ~width:2 in
  let ii = Variable.reg spec ~width:9 in
  let oo = Variable.reg spec ~width:9 in
  let hd = Variable.reg spec ~width:head_bits in
  let thi = Variable.reg spec ~width:43 in
  let den = Variable.reg spec ~width:24 in
  (* the landing helpers: the residual read-modify-write, and the one pending divide *)
  let rmw = Variable.reg spec ~width:1 in
  let rmw_row = Variable.reg spec ~width:dbits in
  let rmw_sum = Variable.reg spec ~width:48 in
  let pending = Variable.reg spec ~width:1 in
  let pend_lane = Variable.reg spec ~width:lane_bits in
  let done_p = Variable.reg spec ~width:1 in
  let peak = Variable.reg spec ~width:32 in
  let diff = Variable.reg spec ~width:32 in
  let nn = Variable.reg spec ~width:22 in
  (* the sampler *)
  let u24 = Variable.reg spec ~width:24 in
  let total = Variable.reg spec ~width:24 in
  let thr = Variable.reg spec ~width:24 in
  let cum = Variable.reg spec ~width:25 in
  let found = Variable.reg spec ~width:1 in
  (* The walk: the frame the source states, and the classes the chain drew for it. [Embed]
     and [Accumulate] read the classes and [Pick] writes them, thus the drawn frame lives
     in the form the tables address; [held] is the same frame as the word of the socket,
     latched at the command so that the chain of the next step cannot move it. *)
  let held = Variable.reg spec ~width:(Frame.code_bits * Frame.voices) in
  let drawn =
    Array.init Frame.voices ~f:(fun (_ : int) -> Variable.reg spec ~width:class_bits)
  in
  (* the seat the chain is at: it opens at the soprano and counts down, thus one program
     draws the four seats and the ops of the chain exist one time each *)
  let seat = Variable.reg spec ~width:seat_bits in
  let valid = Variable.reg spec ~width:1 in
  (* 32 bits: the lead-in test below reads the step counter, thus a wrap would put the
     walk back inside the lead-in and stop the music for a bar. At 8 ms a step — the floor
     of the wire — 16 bits wrap in under nine minutes and 32 bits in a thousand years. *)
  let s = Variable.reg spec ~width:32 in
  let cur = Variable.reg spec ~width:slot_bits in
  let filled = Variable.reg spec ~width:1 in
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
  (* the walk engine's commands, and the freeze that keeps its tags with its data *)
  let mac_go = Variable.wire ~default:gnd () in
  let mac_inner = Variable.wire ~default:(zero 9) () in
  let mac_outer = Variable.wire ~default:(zero 9) () in
  let hold = Variable.wire ~default:gnd () in
  let nohold = ~:(hold.value) in
  (* L1 — the memories; every read lands in a register, thus block RAM is inferred *)
  let rom_addr = Variable.wire ~default:(zero rom_addr_bits) () in
  let kc_raddr = Variable.wire ~default:(zero ring_bits) () in
  let vc_raddr = Variable.wire ~default:(zero ring_bits) () in
  let kc_wen = Variable.wire ~default:gnd () in
  let vc_wen = Variable.wire ~default:gnd () in
  let ring_waddr = Variable.wire ~default:(zero ring_bits) () in
  let ring_wdata = Variable.wire ~default:(zero 8) () in
  let vram_raddr = Variable.wire ~default:(zero vbits) () in
  let vram_wen = Variable.wire ~default:gnd () in
  let vram_waddr = Variable.wire ~default:(zero vbits) () in
  let vram_wdata = Variable.wire ~default:(zero 32) () in
  let hram_raddr = Variable.wire ~default:(zero dbits) () in
  let hram_wen = Variable.wire ~default:gnd () in
  let hram_waddr = Variable.wire ~default:(zero dbits) () in
  let hram_wdata = Variable.wire ~default:(zero 32) () in
  let yram_raddr = Variable.wire ~default:(zero dbits) () in
  let yram_wen = Variable.wire ~default:gnd () in
  let yram_waddr = Variable.wire ~default:(zero dbits) () in
  let yram_wdata = Variable.wire ~default:(zero 16) () in
  let qram_raddr = Variable.wire ~default:(zero dbits) () in
  let qram_wen = Variable.wire ~default:gnd () in
  let qram_waddr = Variable.wire ~default:(zero dbits) () in
  let qram_wdata = Variable.wire ~default:(zero 16) () in
  (* The banking rules are in [docs/transformer_rtl.md]; the measurements behind them are
     here. The tools demoted deep write-portless arrays to slice logic under two different
     select shapes — the six-layer image in slice logic is 69 percent of the device — thus
     a bank is an initialized memory with a gated-off write port, and RAM_STYLE pins it.
     The address registers once before the tree, and each bank registers its data once
     behind it: two cycles from address to data, as one ROM, because
     [reg (reg rom.(addr))] equals [reg (rom.(reg addr))] when the contents never change.
     The address register is load-bearing, not style: with a combinational address, the
     tools retime the data register onto the address pins of every block RAM primitive and
     rebuild the whole op-dispatch address cone inside each one — 27 LUTs a primitive, 12
     primitives a layer, which was the entire layer scaling of this block (3 466 -> 2 352
     LUTs at six layers, measured out of context). A read past the image selects a bank at
     a dead offset; no op makes one. *)
  let rec rom_banked bits addr =
    let n = Array.length bits in
    if Int.is_pow2 n && n <= 1 lsl 15
    then (
      let data =
        (multiport_memory
           ~attributes:[ Rtl_attribute.Vivado.Ram_style.block ]
           ~initialize_to:bits
           n
           ~write_ports:
             [| { Write_port.write_clock = i.clock
                ; write_address = zero (width addr)
                ; write_enable = gnd
                ; write_data = zero 8
                }
             |]
           ~read_addresses:[| addr |]).(0)
      in
      reg spec ~enable:nohold data)
    else (
      let split = if Int.is_pow2 n then n / 2 else 1 lsl Int.floor_log2 n in
      let low = rom_banked (Array.subo bits ~len:split) (lsbs addr) in
      let high =
        rom_banked
          (Array.subo bits ~pos:split)
          (sel_bottom (lsbs addr) ~width:(address_bits_for (n - split)))
      in
      mux2 (reg spec ~enable:nohold (msb addr)) high low)
  in
  let romd = rom_banked rom_bits (reg spec ~enable:nohold rom_addr.value) in
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
  (* The read of a ring restores the eight zero low bits that [Quantized.coarse_to_ring]
     dropped at the write. Every memory the walk reads stands two registers deep, and
     [nohold] freezes each stage with the walk's tags; the small RAMs keep the
     one-register tap for the bespoke chains.

     The ring WRITE stands one register behind its landing, for the same reason the reads
     stand two: the rings sit far across the die at high occupancy, and the sum-to-write
     route wants the travel stage. The register is safe by the schedule: a ring row's
     nearest read is an op away — attention reads what the k and v walks wrote — and a
     hold never overlaps a ring write, thus no enable. *)
  let ring_waddr_r = reg spec ring_waddr.value in
  let ring_wdata_r = reg spec ring_wdata.value in
  let kc_wen_r = reg spec kc_wen.value in
  let vc_wen_r = reg spec vc_wen.value in
  let kcd =
    reg
      spec
      ~enable:nohold
      (reg
         spec
         ~enable:nohold
         (ram
            ~size:(layers * slots * d)
            ~waddr:ring_waddr_r
            ~wen:kc_wen_r
            ~wdata:ring_wdata_r
            ~raddr:kc_raddr.value))
    @: zero 8
  in
  let vcd =
    reg
      spec
      ~enable:nohold
      (reg
         spec
         ~enable:nohold
         (ram
            ~size:(layers * slots * d)
            ~waddr:ring_waddr_r
            ~wen:vc_wen_r
            ~wdata:ring_wdata_r
            ~raddr:vc_raddr.value))
    @: zero 8
  in
  let vramd =
    reg
      spec
      ~enable:nohold
      (ram
         ~size:vram_size
         ~waddr:vram_waddr.value
         ~wen:vram_wen.value
         ~wdata:vram_wdata.value
         ~raddr:vram_raddr.value)
  in
  let vramd2 = reg spec ~enable:nohold vramd in
  let hramd =
    reg
      spec
      ~enable:nohold
      (ram
         ~size:d
         ~waddr:hram_waddr.value
         ~wen:hram_wen.value
         ~wdata:hram_wdata.value
         ~raddr:hram_raddr.value)
  in
  let hramd2 = reg spec ~enable:nohold hramd in
  let yd =
    reg
      spec
      ~enable:nohold
      (ram
         ~size:d
         ~waddr:yram_waddr.value
         ~wen:yram_wen.value
         ~wdata:yram_wdata.value
         ~raddr:yram_raddr.value)
  in
  let yd2 = reg spec ~enable:nohold yd in
  let qd =
    reg
      spec
      ~enable:nohold
      (ram
         ~size:d
         ~waddr:qram_waddr.value
         ~wen:qram_wen.value
         ~wdata:qram_wdata.value
         ~raddr:qram_raddr.value)
  in
  let qd2 = reg spec ~enable:nohold qd in
  (* L1 — [Mac], the walk behind the one 25 by 18 multiplier; the ops feed its operand
     wires and read its counters *)
  let mul_a = Variable.wire ~default:(zero 25) () in
  let mul_b = Variable.wire ~default:(zero 18) () in
  let mac =
    Mac.create
      { Mac.I.clock = i.clock
      ; clear = i.clear
      ; go = mac_go.value
      ; inner = mac_inner.value
      ; outer = mac_outer.value
      ; hold = hold.value
      ; a = mul_a.value
      ; b = mul_b.value
      }
  in
  (* the walk slices *)
  let ii_d = sel_bottom ii.value ~width:dbits in
  let oo8 = sel_bottom oo.value ~width:8 in
  let ii_slot = sel_bottom ii.value ~width:slot_bits in
  let mac_ii_d = sel_bottom mac.ii ~width:dbits in
  let mac_oo_d = sel_bottom mac.oo ~width:dbits in
  let mac_row_d = sel_bottom mac.row ~width:dbits in
  let mac_ii_lane = sel_bottom mac.ii ~width:lane_bits in
  let mac_oo_lane = sel_bottom mac.oo ~width:lane_bits in
  let mac_ii_slot = sel_bottom mac.ii ~width:slot_bits in
  let mac_oo_slot = sel_bottom mac.oo ~width:slot_bits in
  let mac_row_slot = sel_bottom mac.row ~width:slot_bits in
  let mac_row_lane = sel_bottom mac.row ~width:lane_bits in
  let oo_class = sel_bottom oo.value ~width:class_bits in
  let phase = sel_bottom s.value ~width:phase_bits in
  (* the filled slot count, at the width the walk counters take *)
  let n9 =
    uresize
      (mux2
         filled.value
         (of_unsigned_int ~width:(slot_bits + 1) slots)
         (uresize cur.value ~width:(slot_bits + 1) +:. 1))
      ~width:9
  in
  let alibi =
    (* the slope of head k is 2^-(span (k+1) / heads): a shift of the age, in Q12; the age
       is the retired row of the score walk *)
    mux
      hd.value
      (List.init heads ~f:(fun head ->
         let exponent = Quantized.Constants.slope_exponent ~span ~heads ~head in
         sll (uresize mac.row ~width:32) ~by:(Quantized.Constants.y_q - exponent)))
  in
  (* The derived values of L1, named once. A builder runs for each op of the program —
     [2 * layers + 1] [Rms_norm] and [layers] [Attend], thus five and two at two layers
     and thirteen and six at six — and an expression written inside one is elaborated once
     for each of them, because Hardcaml shares nothing by itself. These read only wires
     that exist once, thus one instance serves every op: the pc case already muxes the
     destination. *)
  let quotient16 = clamp16 div_quotient in
  let mean_square = sel_bottom (srl mac.sum ~by:(Int.floor_log2 d) +: eps48) ~width:42 in
  let below_peak = vramd -: peak.value in
  let den_next = den.value +: uresize exp2_e ~width:24 in
  let score = sel_bottom (sra mac.sum ~by:score_shift) ~width:32 -: alibi in
  let score_above_peak = score >+ peak.value in
  (* the ring slot an age names: the walk counts the age, and [cur] is the newest slot *)
  let slot_of_oo = cur.value -: mac_oo_slot in
  let slot_of_ii = cur.value -: mac_ii_slot in
  (* One row of a ring: the rows of a layer stand together, thus the layer stands above
     the slot and the dimension and the address is a concatenation — no adder, because the
     slots and the width are powers of two and the layer's offset therefore has nothing
     but zeros below it. The field is [ring_bits] less the two below it, which is zero at
     one layer, where the address is the slot and the dimension alone.

     The rings are sized at the layer count and not at the width of the field, thus six
     layers cost six and not the eight a rounded-up field would take — 64 KB of block RAM
     on a design that stands at 126 tiles of 135. A read above the image names a row no op
     writes, as a read past the weight image names a dead bank. *)
  let layer_bits = ring_bits - slot_bits - dbits in
  let ring_row ~layer ~slot ~dim =
    if layer_bits = 0
    then slot @: dim
    else of_unsigned_int ~width:layer_bits layer @: slot @: dim
  in
  (* Where a tensor of an op begins, as an address. A block of the seat tensor is one mux
     over four constants — the seat register names it — and the mux stands before the
     adder that every ROM address already pays, thus a seat costs one level and no
     arithmetic. *)
  (* the class the seat the chain is at has drawn, and the write of that register: the two
     halves of the seat's port into [drawn], each one parallel case *)
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
  (* [by_tick bodies] is one bespoke chain: body [k] runs at tick [k]. The tick steps by
     itself, thus a body states only its work and the position in the list states its
     time. The last body owns what follows it, because a chain ends either on [finish] or
     on a return to tick 0. A parallel case and not a chain of guards: see the L3 note of
     the module comment. *)
  let by_tick bodies =
    let last = List.length bodies - 1 in
    switch
      tick.value
      (List.mapi bodies ~f:(fun k body ->
         of_unsigned_int ~width:3 k, if k = last then body else (tick <--. k + 1) :: body))
  in
  (* [by_stage bodies] is the stages of a multi-stage op, as a parallel case. A stage
     moves [stage] itself, because the stages wait on different things — a walk, a unit, a
     chain — thus the list states only the work of each. *)
  let by_stage bodies =
    switch
      stage.value
      (List.mapi bodies ~f:(fun k body -> of_unsigned_int ~width:2 k, body))
  in
  (* [exp_weight_chain] is the bespoke chain that turns one vram value into its exp2
     weight, over the same address: read the value, take its distance below the peak,
     scale that into the exp2 argument, and land the weight where the value stood.
     [Attend] runs it over the ages of a head and [Temper] over the codes; they differ in
     the address, in the scale, in what the landing makes of the weight, and in how the
     walk advances. The scale carries its own Q, thus the port and the shift under it
     cannot disagree. The tick numbers of the multiply and of the exp2 read live here
     alone — the dormant debt of the module comment has one home. *)
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
     for the h RAM — a walk reads y, the hidden or the ROM, and never h. The join of a
     layer and the accumulate of the chain differ only in [from], the format the sum
     arrives in. The entry of both is [rmw <-- gnd; done_p <-- gnd; mac_go <-- vdd].

     [rmw] is the retirement delayed one cycle and nothing else, thus it is one plain
     assignment and not a set and a clear. A clear inside the write would stand after the
     set in the statement order and win over it, and the accumulate of the chain retires a
     row every cycle — one term to a row — thus every second row would land and the walk
     would then wait for a write that nothing commands. *)
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
     runs in the op's last cycle: the next op's entry, and the pc move — an op initializes
     its own counters, and its predecessor runs that entry. *)
  let build (op : Op.t) ~(finish : Always.t list) =
    match op with
    | Op.Embed { seats; phase = ph; e } ->
      (* h[row] = the five table rows summed on the walk: a term is a table, a row is an
         element. The four seat tables stand in one tensor, thus a seat begins a constant
         distance above the base and the class it drew names the row. *)
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
      (* the row the seat drew adds onto the stream: one row of one term for each element,
         thus the walk states the address and the retired sum is the weight itself *)
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
    | Rms_norm ->
      (* stage 0 sums the squares of the Q12 copy on the walk, one row of [d] terms; stage
         1 waits on the isqrt; stage 2 divides each element — y = (h << 8) / g, toward
         zero *)
      let sum_squares =
        [ mac_inner <--. d
        ; mac_outer <--. 1
        ; hram_raddr <-- mac_ii_d
        ; mul_a <-- sel_bottom (sra hramd2 ~by:4) ~width:25
        ; mul_b <-- sel_bottom (sra hramd2 ~by:4) ~width:18
        ; when_
            mac.done_
            [ sq_start <-- vdd
            ; sq_value <-- mean_square
            ; ii <--. 0
            ; tick <--. 0
            ; stage <--. 1
            ]
        ]
      in
      let await_root = [ when_ ~:sq_busy [ stage <--. 2; ii <--. 0; tick <--. 0 ] ] in
      let divide_elements =
        [ hram_raddr <-- ii_d
        ; by_tick
            [ []
            ; [ div_start <-- vdd
              ; div_num <-- sll (sresize hramd ~width:40) ~by:8
              ; div_den <-- uresize sq_root ~width:24
              ]
            ; [ when_
                  ~:div_busy
                  [ yram_wen <-- vdd
                  ; yram_waddr <-- ii_d
                  ; yram_wdata <-- quotient16
                  ; tick <--. 0
                  ; if_ (ii.value ==:. d - 1) finish [ ii <-- ii.value +:. 1 ]
                  ]
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
        | `Y -> [ yram_raddr <-- mac_ii_d ], yd2
        | `Hidden ->
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
      (* the four simple landings share one machine: the walk runs, and each finished row
         writes where the landing says. Only the residual join needs one of its own. *)
      let simple ?(entry_extra = []) writes =
        ( entry_extra @ [ mac_go <-- vdd ]
        , common @ [ when_ mac.row_done writes; when_ mac.done_ finish ] )
      in
      let to_kv v =
        clamp16
          (rescale
             ~from:(Quantized.Constants.y_q + w.e)
             ~target:Quantized.Constants.kv_q
             v)
      in
      (match landing with
       | To_q ->
         simple
           [ qram_wen <-- vdd; qram_waddr <-- mac_row_d; qram_wdata <-- to_kv mac.sum ]
       | To_ring { k; layer } ->
         simple
           [ (if k then kc_wen else vc_wen) <-- vdd
           ; ring_waddr <-- ring_row ~layer ~slot:cur.value ~dim:mac_row_d
           ; ring_wdata <-- sel_top ~width:8 (to_kv mac.sum)
           ]
       | To_hidden ->
         let shifted =
           rescale
             ~from:(Quantized.Constants.y_q + w.e)
             ~target:Quantized.Constants.hid_q
             mac.sum
         in
         let relu = mux2 (shifted <+ zero 48) (zero 48) shifted in
         simple
           [ vram_wen <-- vdd
           ; vram_waddr <-- uresize row_o ~width:vbits
           ; vram_wdata <-- sresize (clamp16 relu) ~width:32
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
         let from_q =
           match src with
           | `Y -> Quantized.Constants.kv_q
           | `Hidden -> Quantized.Constants.hid_q
         in
         join_entry, common @ join_to_h ~from:(from_q + w.e) ~finish)
    | Attend { layer } ->
      (* the attention of one layer, head by head, in three walks: stage 0 scores the
         ages; stage 1 turns each score into its exp2 weight, over the score's own vram
         row, and accumulates the total; stage 2 sums the values lane-major — one dot
         product over the ages for each lane, weight row against value ring — and each
         lane's sum divides by the total as it lands. Age [a] reads slot
         [(cur - a) & (slots - 1)], thus the ALiBi distance is the age and the causal wall
         is the walk. *)
      let entry =
        [ hd <--. 0
        ; ii <--. 0
        ; tick <--. 0
        ; den <--. 0
        ; peak <-- min32
        ; stage <--. 0
        ; pending <-- gnd
        ; done_p <-- gnd
        ; mac_go <-- vdd
        ]
      in
      (* one row of [head_d] terms per age; the retired row is the age's score *)
      let score_ages =
        [ mac_inner <--. head_d
        ; mac_outer <-- n9
        ; qram_raddr <-- hd.value @: mac_ii_lane
        ; kc_raddr <-- ring_row ~layer ~slot:slot_of_oo ~dim:(hd.value @: mac_ii_lane)
        ; mul_a <-- sresize qd2 ~width:25
        ; mul_b <-- sresize kcd ~width:18
        ; when_
            mac.row_done
            [ vram_wen <-- vdd
            ; vram_waddr <-- uresize mac_row_slot ~width:vbits
            ; vram_wdata <-- score
            ; when_ score_above_peak [ peak <-- score ]
            ]
        ; when_ mac.done_ [ ii <--. 0; tick <--. 0; stage <--. 1 ]
        ]
      in
      (* the weight of age [ii] lands over its score; den accumulates *)
      let weigh_ages =
        exp_weight_chain
          ~addr:ii_slot
          ~scale:Quantized.Constants.log2e
          ~land_:[ vram_wdata <-- uresize exp2_e ~width:32; den <-- den_next ]
          ~advance:
            [ if_
                (ii.value ==: n9 -:. 1)
                [ ii <--. 0; stage <--. 2; mac_go <-- vdd ]
                [ ii <-- ii.value +:. 1 ]
            ]
      in
      (* lane [row] = (sum over the ages of weight * value) / den; the one pending divide
         holds the walk until its lane lands *)
      let merge_lanes =
        [ mac_inner <-- n9
        ; mac_outer <--. head_d
        ; hold <-- pending.value
        ; vram_raddr <-- uresize mac_ii_slot ~width:vbits
        ; vc_raddr <-- ring_row ~layer ~slot:slot_of_ii ~dim:(hd.value @: mac_oo_lane)
        ; mul_a <-- uresize (sel_bottom vramd2 ~width:16) ~width:25
        ; mul_b <-- sresize vcd ~width:18
        ; when_
            mac.row_done
            [ div_start <-- vdd
            ; div_num <-- sel_bottom mac.sum ~width:40
            ; div_den <-- den.value
            ; pending <-- vdd
            ; pend_lane <-- mac_row_lane
            ]
        ; when_ mac.done_ [ done_p <-- vdd ]
        ; when_
            (pending.value &: ~:div_busy)
            [ yram_wen <-- vdd
            ; yram_waddr <-- hd.value @: pend_lane.value
            ; yram_wdata <-- quotient16
            ; pending <-- gnd
            ; when_
                done_p.value
                [ if_
                    (hd.value ==:. heads - 1)
                    finish
                    [ hd <-- hd.value +:. 1
                    ; stage <--. 0
                    ; den <--. 0
                    ; peak <-- min32
                    ; done_p <-- gnd
                    ; mac_go <-- vdd
                    ]
                ]
            ]
        ]
      in
      entry, [ by_stage [ score_ages; weigh_ages; merge_lanes ] ]
    | Temper ->
      (* the tempered weight of each class: exp2, and refused under min-p *)
      let entry = [ oo <--. 0; tick <--. 0; total <--. 0 ] in
      (* the min-p refusal: a weight under the share of the peak weighs nothing *)
      let keep = exp2_e >=: of_unsigned_int ~width:16 min_weight in
      let w = mux2 keep exp2_e (zero 16) in
      let body =
        exp_weight_chain
          ~addr:oo8
          ~scale:temper
          ~land_:
            [ vram_wdata <-- uresize w ~width:32
            ; total <-- total.value +: uresize w ~width:24
            ]
          ~advance:[ if_ (oo.value ==:. classes - 1) finish [ oo <-- oo.value +:. 1 ] ]
      in
      entry, body
    | Draw ->
      (* three PRNG bytes, high first: the walk of [Prng.uniform] *)
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
         not a fallback: the threshold is below the total by construction, thus the class
         the walk names always holds weight. The class lands in the register of its seat. *)
      let entry = [ oo <--. 0; tick <--. 0; cum <--. 0; found <--. 0 ] in
      let body =
        [ vram_raddr <-- uresize oo8 ~width:vbits
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
     states the loop back without a circular definition. Only the entry of that second
     build is kept, and the bodies of it reach no output and leave the circuit. *)
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
  (* the chain opens at the soprano; the reference draws in the same order *)
  let enter_chain = [ seat <--. Frame.voices - 1 ] @ chain_entry in
  (* The forward has stated step [s], thus the chain would draw the step after it. Through
     the lead-in the chain does not run: the frame stays silence, the drawn classes stand
     at [Vocab.silence], and the PRNG does not move, because [Draw] is the only thing that
     steps it. *)
  let next_index = s.value +:. 1 in
  let forward_done =
    [ s <-- next_index
    ; cur <-- cur.value +:. 1
    ; filled <-- (filled.value |: (cur.value ==:. slots - 1))
    ; if_ (next_index >=:. Jsb.bar_steps) enter_chain [ sm.set_next Idle ]
    ]
  in
  let forward_entry, forward_bodies = link 0 forward_done prog.forward in
  (* one parallel case, not a chain of guards: see the L3 note of the module comment *)
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
          , (* The one reset is the rewind, and it runs nothing: the walk begins at an
               empty ring and no residual, and [Embed] writes the whole stream at the head
               of every forward, thus neither memory needs a clearing walk. *)
            [ when_
                i.rewind
                ([ s <--. 0; cur <--. 0; filled <--. 0; held <--. Frame.silent ]
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
  let config =
    { Transformer.Config.d = 16; layers = 1; heads = 4; context = 16; slope_span = 8 }
  in
  let model = Quantized.Model.For_test.init config ~seed:11 in
  let { chain; forward } = schedule model in
  let show tag ops =
    List.iteri ops ~f:(fun index op ->
      Stdio.printf "%s%-2d %s\n" tag index (Sexp.to_string (Op.sexp_of_t op)))
  in
  show "f" forward;
  show "c" chain;
  let baseline =
    schedule (Quantized.Model.For_test.init Transformer.Config.baseline ~seed:11)
  in
  Stdio.printf
    "baseline: %d forward ops, %d chain ops\n"
    (List.length baseline.forward)
    (List.length baseline.chain);
  [%expect
    {|
    f0  (Embed(seats 0)(phase 3072)(e 10))
    f1  Rms_norm
    f2  (Matvec((src Y)(w((base(Fixed 3328))(e 10)))(outer_major false)(inner 16)(outer 16)(landing To_q)))
    f3  (Matvec((src Y)(w((base(Fixed 3584))(e 11)))(outer_major false)(inner 16)(outer 16)(landing(To_ring(k true)(layer 0)))))
    f4  (Matvec((src Y)(w((base(Fixed 3840))(e 10)))(outer_major false)(inner 16)(outer 16)(landing(To_ring(k false)(layer 0)))))
    f5  (Attend(layer 0))
    f6  (Matvec((src Y)(w((base(Fixed 4096))(e 11)))(outer_major false)(inner 16)(outer 16)(landing Add_to_h)))
    f7  Rms_norm
    f8  (Matvec((src Y)(w((base(Fixed 4352))(e 10)))(outer_major false)(inner 16)(outer 64)(landing To_hidden)))
    f9  (Matvec((src Hidden)(w((base(Fixed 5376))(e 10)))(outer_major false)(inner 64)(outer 16)(landing Add_to_h)))
    c0  Rms_norm
    c1  (Matvec((src Y)(w((base(Seat_block 0))(e 10)))(outer_major true)(inner 16)(outer 48)(landing To_logits)))
    c2  Temper
    c3  Draw
    c4  Threshold
    c5  Pick
    c6  (Accumulate(base(Seat_block 0))(e 10))
    baseline: 55 forward ops, 7 chain ops
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
   the lead-in — the first drawn step is the one that reads a context of silence. *)
let frames_agree ~model ~seed ~steps =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 seed)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let budget = ref 5_000_000 in
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
     because the chain that moves the drawn classes runs behind the forward pass. *)
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

(* The ring address of one shape, by the packing rule the note at [ring_row] states: the
   layer stands above the slot and the dimension. The layer field is empty at one layer,
   thus the gate below prints these widths — the shape of a gate decides which half of
   [ring_row] the simulation ever elaborates, and that fact belongs in the gate. *)
let ring_geometry (config : Transformer.Config.t) =
  let rows = config.layers * config.context * config.d in
  let dbits = Int.floor_log2 config.d in
  let slot_bits = Int.floor_log2 config.context in
  let ring_bits = address_bits_for rows in
  Stdio.printf
    "layers %d: %d ring rows, ring_bits %d = layer_bits %d + slot_bits %d + dbits %d\n"
    config.layers
    rows
    ring_bits
    (ring_bits - slot_bits - dbits)
    slot_bits
    dbits
;;

let%expect_test "the source agrees with the reference, frame for frame" =
  let config = Quantized.Model.For_test.config in
  ring_geometry config;
  frames_agree ~model:(Quantized.Model.For_test.init config ~seed:11) ~seed:42 ~steps:20;
  [%expect
    {|
    layers 1: 512 ring rows, ring_bits 9 = layer_bits 0 + slot_bits 4 + dbits 5
    step  0  00000000
    step  1  00000000
    step 15  00000000
    step 16  aac7cbad
    step 17  b6cbcd00
    step 18  b6cad0d0
    step 19  a6d1adbd
    20 steps, the frames agree: true
    |}]
;;

(* The seed 0 on the circuit. It is the fixed point of xorshift32 and the panel can state
   it — all the slide switches down is the rest position of the board — thus the walk
   stands still: every threshold is 0 and each seat takes the first class that min-p left
   standing. [Quantized] states that walk, and this holds the circuit to it, where a PRNG
   that reset to another state or a threshold that rounded the other way would show.

   The drawn weights of this model leave class 0 standing at every seat, thus the frame is
   silence and the walk plays nothing after the lead-in. The board answers the same with
   the trained model in flash, measured 2026-08-19: all the slide switches down is a
   silent board. *)
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

(* The same gate at two layers. It is not a wider shape for its own sake: at one layer the
   layer field of the ring address is empty and the per-layer ROM bases all read the first
   layer's, thus the [else] branch of [ring_row] and every address that carries a layer
   are dead in a one-layer simulation and live only on the board, which runs six. This
   gate elaborates them.

   The shape is the shape of [test/test_quantized_drift.ml], thus the frame gate of the
   circuit and the drift floors of the scheme stand on one model shape. The weights come
   from another seed than the gate above, thus the two gates do not share a model. The
   walk runs past the ring — sixteen slots — thus the second layer reads its own rows back
   after the wrap and not only inside the lead-in.

   The first drawn step reads the same frame as the gate above, and the two models are not
   the same model: the walk seed is the same, thus the uniforms are the same, and weights
   of scale 0.02 put the classes so near each other that a pick is almost the quantile of
   its uniform alone. The steps after it part. Each gate compares a circuit against its
   own reference in any case. *)
let%expect_test "the source agrees with the reference at two layers" =
  let config =
    { Transformer.Config.d = 16; layers = 2; heads = 4; context = 16; slope_span = 4 }
  in
  ring_geometry config;
  frames_agree ~model:(Quantized.Model.For_test.init config ~seed:23) ~seed:42 ~steps:24;
  [%expect
    {|
    layers 2: 512 ring rows, ring_bits 9 = layer_bits 1 + slot_bits 4 + dbits 4
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

(* The cycle bench: the circuit's measured cost against [Op.cycles], step by step. A step
   costs its forward pass, the chain behind it once the lead-in is past, and two cycles of
   command — the cycle that takes [step] and runs the first op's entry, and the cycle the
   bench spends dropping the strobe. The ring's fill [n] is the step index, capped at the
   slot count. *)
let bench ~steps () =
  let config = Quantized.Model.For_test.config in
  let model = Quantized.Model.For_test.init config ~seed:11 in
  let prog = schedule model in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 42)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let slots = config.context in
  let sum_ops ops ~n =
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
    let n = Int.min (index + 1) slots in
    let draws = index + 1 >= Jsb.bar_steps in
    let modeled =
      (* the chain is one seat and the machine runs it once for each of them *)
      2
      + sum_ops prog.forward ~n
      + if draws then Frame.voices * sum_ops prog.chain ~n else 0
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
    step  0: silent, measured 16754, model 16754, delta 0
    step 15: draws,  measured 31732, model 31732, delta 0
    step 16: draws,  measured 31732, model 31732, delta 0
    step 17: draws,  measured 31732, model 31732, delta 0
    18 steps, 0 disagree, total 354696
    |}]
;;
