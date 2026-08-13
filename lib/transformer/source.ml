(* The transformer note source — see source.mli for the contract. This file holds the
   design: why the machine has the shape it has.

   The mathematics is [Quantized], integer for integer. The machine is five layers:

   - L0, the primitives: [Divider], [Isqrt] and [Exp2] here, each in a file of its own
     with its contract in an interface file; [Sounding_state.Rtl] and [Prng.Rtl] in the
     core, each beside the software it must agree with. Real hardware with tiny
     interfaces.
   - L1, the datapath: the RAMs, the KV rings, the banked weight ROM, the one 25x18
     multiplier, and [Mac] (mac.ml) — the walk behind the multiplier as a unit: the issue
     counters, the tag line that follows the data through the pipe, and the accumulator,
     at one term a cycle. Built once; the ops mux its ports. No op owns a resource. The
     per-op facts stay in the ops: the address formulas, the operand sources, the landing
     writes.
   - L2, the schedule: the walk as a value — [schedule] gives the two programs, the
     sampler and the forward pass, as [Op.t] lists. One op holds the facts of one step:
     the tensor base and exponent, the address order, the loop bounds, the landing. The
     program is config-shaped: the layer count is a loop bound, and each op knows its
     layer at elaboration. Therefore no register carries a sub-step, a layer index or a
     return address, and every per-layer mux folds to a constant.
   - L3, the compiler: [chain] folds a program into cases of a program counter over the
     datapath, and the ops dispatch as one [switch] on the pc. The why: [Always] compiles
     a statement list into a linear chain of muxes, one level for each statement that
     writes a target, and that chain over every op was the thinnest timing path measured
     in this era; a switch with constant cases compiles into one parallel case. An op's
     finish runs the next op's entry actions in the same cycle — the one convention that
     replaces a hand-kept register reset for each op.
   - L4, the outer FSM, hand-written and small: Idle, Run, Decide, Emit, ForwardDone. The
     token walk, the seats and the socket handshake stay control; the mathematics is a
     program.

   The gates:

   - The stream test at the bottom: the events must equal [Quantized.Engine], event for
     event, on drawn weights. [Quantized] is the reference, and this is the gate that
     holds the circuit to it.
   - The block tests of the units, each in the unit's own file: [Mac], [Divider], [Isqrt]
     and [Exp2] against exact oracles, and [Sounding_state.Rtl] against
     [Sounding_state.legal_mask] over drawn walks.
   - The schedule prints: the state table is data, and an expect test pins it.
   - [Op.cycles], the cost model beside the schedule, pinned against the measured cycle
     bench.

   Cycle counts are not preserved, and need not be: the socket is latency-insensitive and
   the PRNG steps only on command, thus a draw cannot move.

   The timing design, decided against the measured paths (2026-08-13):

   - Every memory the walk reads stands two registers from the multiplier: the read
     register, then an output register that packs into the block RAM (DO_REG; the
     clock-to-out falls 2.46 -> 0.89 ns at speed grade -1). The six-layer build failed on
     the route from a far ROM bank into the DSP at 94 percent occupancy; the travel stage
     pays for that route, and at one term a cycle it costs only fill latency. A ROM bank
     registers its own data before the select mux — a register after the mux stays in the
     fabric and removes nothing. The bespoke chains (Exp, Temper) read the small RAMs at
     the one-register tap; they touch neither the ROM nor the rings.
   - The DSP stays a two-register multiply — the operand registers and [preg] — rated 257
     MHz at -1 against the 100 MHz clock. The accumulator is a fabric adder behind it,
     loaded on a row's first term by the tag. The full in-DSP accumulate was declined: the
     M register cannot drive the fabric, thus every reader of the product moves one
     register later, for headroom this clock does not spend. The bespoke readers of [preg]
     anchor it as the DSP's final register, thus the tools cannot fold the accumulator in.
   - The dormant debt, recorded: the product latency — two cycles from the operands to
     [preg] — is hand-encoded as tick numbers in the Exp, Temper and Threshold chains. If
     the pipe ever deepens, replace the tick counts with a wait on a valid bit; do not
     renumber.

   Design choices, decided here:

   - A repeated op is an inlined program step. [Rms_norm] appears twice per layer and once
     in the sampler; control is cheap, and the units it drives exist once. No return
     register.
   - [Attend] stays one op with an internal stage register, and its stages are walks of
     the engine: the scores; then the exp2 weight of every age — each weight lands over
     its score's vram row, and the total accumulates; then the weighted sum lane-major,
     one dot product over the ages for each lane, from the weight row and the value ring
     into the divider. Each lane's sum sees the ages in the same order as the reference,
     thus the lane-major order is exact. The age-major order — one exp2, then one MAC over
     the lanes, age by age — was declined: it needs a register file of lane sums, and this
     one needs none.
   - The op vocabulary is closed and concrete. The rule: when a field's meaning would
     depend on another field, stop extending and write a new op. *)

open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

let vocab = Token.vocab

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
  (* one weight tensor as the circuit sees it: where it starts, and its exponent *)
  type tensor =
    { base : int
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
    | To_logits (* one shift by [e]; the legal peak tracked for the temper *)
    | Add_to_h (* the residual join: the whole sum, rescaled onto the stream *)
  [@@deriving sexp_of]

  type matvec =
    { src : [ `Y | `Hidden ] (* the normed vector, or the FFN hidden *)
    ; w : tensor
    ; outer_major : bool (* the weight address order; true only for the tied head *)
    ; inner : int
    ; outer : int
    ; landing : landing
    }
  [@@deriving sexp_of]

  (* One step of the walk: the facts that one case of the pc switch needs. The bespoke ops
     close over the model at elaboration and carry no fields here. *)
  type t =
    | Embed of
        { token : tensor
        ; phase : tensor
        ; progress : tensor
        }
    | Rms_norm
    | Matvec of matvec
    | Attend of { layer : int }
    | Temper
    | Draw
    | Threshold
    | Pick
  [@@deriving sexp_of]

  (* The analytic cost of one op, in cycles from its go to its finish, both the cycle a
     predecessor's finish runs. [n] is the filled slot count of the ring at this token.
     The constants restate the builders: a walk retires its last term [Mac.read_latency]
     + 2 cycles after its last issue; the residual join lands one cycle later; the divider
       costs 42 from its start cycle to its wait's release, and the isqrt 22; a
       weighted-sum lane adds a 41-cycle hold for its pending divide. The cycle bench pins
       this model against the measured circuit. *)
  let cycles (config : Transformer.Config.t) ~n (op : t) =
    let { Transformer.Config.d; heads; _ } = config in
    let head_d = d / heads in
    let vocab = Token.vocab in
    match op with
    | Embed _ -> (3 * d) + 4
    | Rms_norm -> (44 * d) + 26
    | Matvec { inner; outer; landing; _ } ->
      (inner * outer)
      + 4
      +
        (match landing with
        | Add_to_h -> 1
        | To_q | To_ring _ | To_hidden | To_logits -> 0)
    | Attend _ -> heads * ((2 * n * head_d) + (7 * n) + (41 * head_d) + 8)
    | Temper -> 7 * vocab
    | Draw -> 4
    | Threshold -> 5
    | Pick -> 2 * vocab
  ;;
end

(* the two programs of the source: a draw, and the forward pass of the drawn token *)
type program =
  { sample : Op.t list
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
  let bases = Quantized.Model.rom_bases model in
  let t (q : Quantized.Model.quantized) base = { Op.base; e = q.e } in
  let layer l =
    let w = model.params.layers.(l) in
    let b = bases.Transformer.Params_data.layers.(l) in
    [ Op.Rms_norm
    ; Matvec
        { src = `Y
        ; w = t w.wq b.wq
        ; outer_major = false
        ; inner = d
        ; outer = d
        ; landing = To_q
        }
    ; Matvec
        { src = `Y
        ; w = t w.wk b.wk
        ; outer_major = false
        ; inner = d
        ; outer = d
        ; landing = To_ring { k = true; layer = l }
        }
    ; Matvec
        { src = `Y
        ; w = t w.wv b.wv
        ; outer_major = false
        ; inner = d
        ; outer = d
        ; landing = To_ring { k = false; layer = l }
        }
    ; Attend { layer = l }
    ; Matvec
        { src = `Y
        ; w = t w.wo b.wo
        ; outer_major = false
        ; inner = d
        ; outer = d
        ; landing = Add_to_h
        }
    ; Rms_norm
    ; Matvec
        { src = `Y
        ; w = t w.w1 b.w1
        ; outer_major = false
        ; inner = d
        ; outer = dff
        ; landing = To_hidden
        }
    ; Matvec
        { src = `Hidden
        ; w = t w.w2 b.w2
        ; outer_major = false
        ; inner = dff
        ; outer = d
        ; landing = Add_to_h
        }
    ]
  in
  { sample =
      [ Rms_norm
      ; Matvec
          { src = `Y
          ; w = t model.params.embed bases.embed
          ; outer_major = true (* the tied head reads the token table backward *)
          ; inner = d
          ; outer = vocab
          ; landing = To_logits
          }
      ; Temper
      ; Draw
      ; Threshold
      ; Pick
      ]
  ; forward =
      Embed
        { token = t model.params.embed bases.embed
        ; phase = t model.params.phase bases.phase
        ; progress = t model.params.progress bases.progress
        }
      :: List.concat (List.init layers ~f:layer)
  }
;;

(* ==================================================================== *)
(* L4 — the outer FSM; L1 and L3 live inside [create] *)
(* ==================================================================== *)

(* the token walk: the programs run under [Run], the grammar and the socket around it *)
module State = struct
  type t =
    | Idle
    | Run
    | Decide
    | Emit
    | ForwardDone
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(model : Quantized.Model.t) ~seed (i : _ I.t) : _ O.t =
  let { Quantized.Model.config; params; temper_q14; min_weight } = model in
  let { Transformer.Config.d; heads; context = slots; slope_span = span; layers } =
    config
  in
  let head_d = d / heads in
  let dff = 4 * d in
  (* the shift rules of the reference; the packing below derives every width *)
  assert (Int.is_pow2 d);
  assert (Int.is_pow2 slots);
  assert (Int.floor_log2 head_d % 2 = 0);
  assert (layers = Array.length params.layers);
  let dbits = Int.floor_log2 d in
  let lane_bits = Int.floor_log2 head_d in
  let head_bits = Int.floor_log2 heads in
  let slot_bits = Int.floor_log2 slots in
  let layer_bits = Int.max 1 (Int.ceil_log2 layers) in
  let ring_bits = layer_bits + slot_bits + dbits in
  (* vram serves the scores, the FFN hidden, the logits and the sampler weights *)
  let vram_size = Int.max vocab (Int.max dff slots) in
  let vbits = address_bits_for vram_size in
  let score_shift =
    (2 * Quantized.Constants.kv_q) - Quantized.Constants.y_q + (Int.floor_log2 head_d / 2)
  in
  let prog = schedule model in
  let sample_length = List.length prog.sample in
  let pc_bits = address_bits_for (sample_length + List.length prog.forward) in
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
  let tick = Variable.reg spec ~width:3 in
  let stage = Variable.reg spec ~width:2 in
  let ii = Variable.reg spec ~width:9 in
  let oo = Variable.reg spec ~width:9 in
  let hd = Variable.reg spec ~width:head_bits in
  let after_forward = Variable.reg spec ~width:1 in
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
  let chosen = Variable.reg spec ~width:8 in
  let found = Variable.reg spec ~width:1 in
  let pos = Variable.reg spec ~width:1 in
  (* the token walk *)
  let out_code = Variable.reg spec ~width:8 in
  let out_seat = Variable.reg spec ~width:2 in
  let s = Variable.reg spec ~width:16 in
  let cur = Variable.reg spec ~width:slot_bits in
  let filled = Variable.reg spec ~width:1 in
  let seat_pitch =
    Array.init Token.seats ~f:(fun (_ : int) -> Variable.reg spec ~width:7)
  in
  let seat_full =
    Array.init Token.seats ~f:(fun (_ : int) -> Variable.reg spec ~width:1)
  in
  let _ = sm.current -- "state" in
  let _ = pc.value -- "pc" in
  let _ = out_code.value -- "out_code" in
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
  let boot = Variable.wire ~default:gnd () in
  let land_ = Variable.wire ~default:gnd () in
  let legal_query = Variable.wire ~default:(zero 8) () in
  (* the walk engine's commands, and the freeze that keeps its tags with its data *)
  let mac_go = Variable.wire ~default:gnd () in
  let mac_inner = Variable.wire ~default:(zero 9) () in
  let mac_outer = Variable.wire ~default:(zero 9) () in
  let hold = Variable.wire ~default:gnd () in
  let nohold = ~:(hold.value) in
  let { Sounding_state.Rtl.O.legal } =
    Sounding_state.Rtl.create
      { Sounding_state.Rtl.I.clock = i.clock
      ; clear = i.clear
      ; boot = boot.value
      ; land_ = land_.value
      ; code = out_code.value
      ; query = legal_query.value
      }
  in
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
  (* Synthesis implements a ROM over the full power of two of its address space, thus the
     pad that [rom_bits] does not carry would return at the tools. The image splits at
     power-of-two boundaries into banks — a deeper power of two halves — and a bank holds
     at most 2^15 rows: one RAMB36 is 32K deep at one bit wide, thus no bank ever needs
     the block RAM cascade, whose wiring fails the tools' own check REQP-1962 at 2^16 and
     above. A bank is not a plain rom: the tools demoted deep write-portless arrays to
     slice logic under two different select shapes — the six-layer image in slice logic is
     69 percent of the device — so each bank is an initialized memory with one gated-off
     write port, the class the tools infer soundly at every depth here, and RAM_STYLE pins
     it to block RAM. The reads sit behind a nest of muxes on registered top address bits:
     two cycles from address to data, as one ROM — each bank registers its own data a
     second time before the mux, and that register packs into the block RAM's output
     register (the travel stage of the timing design). A read past the image selects a
     bank at a dead offset; no op makes one. *)
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
      reg spec ~enable:nohold (reg spec ~enable:nohold data))
    else (
      let split = if Int.is_pow2 n then n / 2 else 1 lsl Int.floor_log2 n in
      let low = rom_banked (Array.subo bits ~len:split) (lsbs addr) in
      let high =
        rom_banked
          (Array.subo bits ~pos:split)
          (sel_bottom (lsbs addr) ~width:(address_bits_for (n - split)))
      in
      mux2 (reg spec ~enable:nohold (reg spec ~enable:nohold (msb addr))) high low)
  in
  let romd = rom_banked rom_bits rom_addr.value in
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
  (* The rings store the top byte of a Q12 row — [Quantized.coarse_to_ring] — and the read
     restores the eight zero low bits, thus the format stays Q12 at a granularity of 2^-4
     and the rings cost half the block RAM. Every memory the walk reads stands two
     registers deep (the timing design); [nohold] freezes each stage with the walk's tags.
     The small RAMs keep the one-register tap for the bespoke chains, and a second
     register makes the walk's tap. *)
  let kcd =
    reg
      spec
      ~enable:nohold
      (reg
         spec
         ~enable:nohold
         (ram
            ~size:(layers * slots * d)
            ~waddr:ring_waddr.value
            ~wen:kc_wen.value
            ~wdata:ring_wdata.value
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
            ~waddr:ring_waddr.value
            ~wen:vc_wen.value
            ~wdata:ring_wdata.value
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
  let phase = sel_bottom s.value ~width:4 in
  let bucket = select s.value ~high:7 ~low:4 in
  let n =
    mux2
      filled.value
      (of_unsigned_int ~width:(slot_bits + 1) slots)
      (uresize cur.value ~width:(slot_bits + 1) +:. 1)
  in
  let n9 = uresize n ~width:9 in
  let alibi =
    (* the slope of head k is 2^-(span (k+1) / heads): a shift of the age, in Q12; the age
       is the retired row of the score walk *)
    mux
      hd.value
      (List.init heads ~f:(fun k ->
         let exponent = span * (k + 1) / heads in
         sll (uresize mac.row ~width:32) ~by:(Quantized.Constants.y_q - exponent)))
  in
  (* the seat of the drawn event: an On takes the highest free seat, an Off names the seat
     that holds its pitch *)
  let free_seat =
    mux2
      ~:(seat_full.(3).value)
      (of_unsigned_int ~width:2 3)
      (mux2
         ~:(seat_full.(2).value)
         (of_unsigned_int ~width:2 2)
         (mux2 ~:(seat_full.(1).value) (of_unsigned_int ~width:2 1) (zero 2)))
  in
  let match_seat p =
    let hit k = seat_full.(k).value &: (seat_pitch.(k).value ==: p) in
    mux2
      (hit 3)
      (of_unsigned_int ~width:2 3)
      (mux2
         (hit 2)
         (of_unsigned_int ~width:2 2)
         (mux2 (hit 1) (of_unsigned_int ~width:2 1) (zero 2)))
  in
  (* ================================================================== *)
  (* L3 — the compiler: one builder per op kind, then the chain *)
  (* ================================================================== *)
  (* [build op ~finish] gives the entry actions and the body of one program step. [finish]
     runs in the op's last cycle: the next op's entry, and the pc move — an op initializes
     its own counters, and its predecessor runs that entry. *)
  let build (op : Op.t) ~(finish : Always.t list) =
    match op with
    | Op.Embed { token; phase = ph; progress } ->
      (* h[row] = the three table rows summed on the walk: a term is a table, a row is an
         element *)
      let entry = [ mac_go <-- vdd ] in
      let body =
        [ mac_inner <--. 3
        ; mac_outer <--. d
        ; rom_addr
          <-- mux
                (sel_bottom mac.ii ~width:2)
                [ rom_const token.base
                  +: uresize (out_code.value @: mac_oo_d) ~width:rom_addr_bits
                ; rom_const ph.base +: uresize (phase @: mac_oo_d) ~width:rom_addr_bits
                ; rom_const progress.base
                  +: uresize (bucket @: mac_oo_d) ~width:rom_addr_bits
                ]
        ; mul_a <-- sresize romd ~width:25
        ; mul_b <-- of_signed_int ~width:18 1
        ; when_
            mac.row_done
            [ hram_wen <-- vdd
            ; hram_waddr <-- mac_row_d
            ; hram_wdata
              <-- sel_bottom
                    (rescale ~from:token.e ~target:Quantized.Constants.h_q mac.sum)
                    ~width:32
            ]
        ; when_ mac.done_ finish
        ]
      in
      entry, body
    | Rms_norm ->
      (* stage 0 sums the squares of the Q12 copy on the walk, one row of [d] terms; stage
         1 waits on the isqrt; stage 2 divides each element — y = (h << 8) / g, toward
         zero *)
      let entry = [ stage <--. 0; mac_go <-- vdd ] in
      let body =
        [ if_
            (stage.value ==:. 0)
            [ mac_inner <--. d
            ; mac_outer <--. 1
            ; hram_raddr <-- mac_ii_d
            ; mul_a <-- sel_bottom (sra hramd2 ~by:4) ~width:25
            ; mul_b <-- sel_bottom (sra hramd2 ~by:4) ~width:18
            ; when_
                mac.done_
                [ sq_start <-- vdd
                ; sq_value
                  <-- sel_bottom (srl mac.sum ~by:(Int.floor_log2 d) +: eps48) ~width:42
                ; ii <--. 0
                ; tick <--. 0
                ; stage <--. 1
                ]
            ]
            [ if_
                (stage.value ==:. 1)
                [ when_ ~:sq_busy [ stage <--. 2; ii <--. 0; tick <--. 0 ] ]
                [ hram_raddr <-- ii_d
                ; if_
                    (tick.value ==:. 0)
                    [ tick <--. 1 ]
                    [ if_
                        (tick.value ==:. 1)
                        [ div_start <-- vdd
                        ; div_num <-- sll (sresize hramd ~width:40) ~by:8
                        ; div_den <-- uresize sq_root ~width:24
                        ; tick <--. 2
                        ]
                        [ when_
                            ~:div_busy
                            [ yram_wen <-- vdd
                            ; yram_waddr <-- ii_d
                            ; yram_wdata <-- clamp16 div_quotient
                            ; tick <--. 0
                            ; if_ (ii.value ==:. d - 1) finish [ ii <-- ii.value +:. 1 ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
      in
      entry, body
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
          ; rom_addr <-- rom_const w.base +: uresize addr ~width:rom_addr_bits
          ; mul_a <-- sresize srcd ~width:25
          ; mul_b <-- sresize romd ~width:18
          ]
      in
      (match landing with
       | To_q | To_ring _ | To_hidden | To_logits ->
         let write v =
           match landing with
           | To_q ->
             [ qram_wen <-- vdd
             ; qram_waddr <-- mac_row_d
             ; qram_wdata
               <-- clamp16
                     (rescale
                        ~from:(Quantized.Constants.y_q + w.e)
                        ~target:Quantized.Constants.kv_q
                        v)
             ]
           | To_ring { k; layer } ->
             [ (if k then kc_wen else vc_wen) <-- vdd
             ; ring_waddr
               <-- of_unsigned_int ~width:layer_bits layer @: cur.value @: mac_row_d
             ; ring_wdata
               <-- sel_top
                     ~width:8
                     (clamp16
                        (rescale
                           ~from:(Quantized.Constants.y_q + w.e)
                           ~target:Quantized.Constants.kv_q
                           v))
             ]
           | To_hidden ->
             let shifted =
               rescale
                 ~from:(Quantized.Constants.y_q + w.e)
                 ~target:Quantized.Constants.hid_q
                 v
             in
             let relu = mux2 (shifted <+ zero 48) (zero 48) shifted in
             [ vram_wen <-- vdd
             ; vram_waddr <-- uresize row_o ~width:vbits
             ; vram_wdata <-- sresize (clamp16 relu) ~width:32
             ]
           | To_logits ->
             let logit = sel_bottom (sra v ~by:w.e) ~width:32 in
             [ vram_wen <-- vdd
             ; vram_waddr <-- uresize row_o ~width:vbits
             ; vram_wdata <-- logit
             ; legal_query <-- uresize row_o ~width:8
             ; when_ (legal &: (logit >+ peak.value)) [ peak <-- logit ]
             ]
           | Add_to_h -> assert false
         in
         let entry =
           match landing with
           | To_logits -> [ peak <-- min32; mac_go <-- vdd ]
           | _ -> [ mac_go <-- vdd ]
         in
         let body =
           common @ [ when_ mac.row_done (write mac.sum); when_ mac.done_ finish ]
         in
         entry, body
       | Add_to_h ->
         (* the residual join: the sum lands in a read-modify-write that overlaps the next
            row's terms — the walk reads y or the hidden, never h *)
         let from_q =
           match src with
           | `Y -> Quantized.Constants.kv_q
           | `Hidden -> Quantized.Constants.hid_q
         in
         let entry = [ rmw <-- gnd; done_p <-- gnd; mac_go <-- vdd ] in
         let body =
           common
           @ [ when_
                 mac.row_done
                 [ rmw <-- vdd
                 ; rmw_row <-- mac_row_d
                 ; rmw_sum <-- mac.sum
                 ; hram_raddr <-- mac_row_d
                 ]
             ; when_ mac.done_ [ done_p <-- vdd ]
             ; when_
                 rmw.value
                 [ hram_wen <-- vdd
                 ; hram_waddr <-- rmw_row.value
                 ; hram_wdata
                   <-- sel_bottom
                         (sresize hramd ~width:48
                          +: rescale
                               ~from:(from_q + w.e)
                               ~target:Quantized.Constants.h_q
                               rmw_sum.value)
                         ~width:32
                 ; rmw <-- gnd
                 ; when_ done_p.value ([ done_p <-- gnd ] @ finish)
                 ]
             ]
         in
         entry, body)
    | Attend { layer } ->
      (* the attention of one layer, head by head, in three walks: stage 0 scores the
         ages; stage 1 turns each score into its exp2 weight, over the score's own vram
         row, and accumulates the total; stage 2 sums the values lane-major — one dot
         product over the ages for each lane, weight row against value ring — and each
         lane's sum divides by the total as it lands. Age [a] reads slot
         [(cur - a) & (slots - 1)], thus the ALiBi distance is the age and the causal wall
         is the walk. *)
      let lconst = of_unsigned_int ~width:layer_bits layer in
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
      let body =
        [ if_
            (stage.value ==:. 0)
            (* Score: one row of [head_d] terms per age *)
            [ mac_inner <--. head_d
            ; mac_outer <-- n9
            ; qram_raddr <-- hd.value @: mac_ii_lane
            ; kc_raddr <-- lconst @: (cur.value -: mac_oo_slot) @: hd.value @: mac_ii_lane
            ; mul_a <-- sresize qd2 ~width:25
            ; mul_b <-- sresize kcd ~width:18
            ; when_
                mac.row_done
                [ (let score =
                     sel_bottom (sra mac.sum ~by:score_shift) ~width:32 -: alibi
                   in
                   proc
                     [ vram_wen <-- vdd
                     ; vram_waddr <-- uresize mac_row_slot ~width:vbits
                     ; vram_wdata <-- score
                     ; when_ (score >+ peak.value) [ peak <-- score ]
                     ])
                ]
            ; when_ mac.done_ [ ii <--. 0; tick <--. 0; stage <--. 1 ]
            ]
            [ if_
                (stage.value ==:. 1)
                (* ExpAll: the weight of age [ii] lands over its score; den accumulates *)
                [ vram_raddr <-- uresize ii_slot ~width:vbits
                ; mul_a <-- sel_bottom diff.value ~width:25
                ; mul_b <-- of_signed_int ~width:18 Quantized.Constants.log2e_q15
                ; if_
                    (tick.value ==:. 0)
                    [ tick <--. 1 ]
                    [ if_
                        (tick.value ==:. 1)
                        [ diff <-- vramd -: peak.value; tick <--. 2 ]
                        [ if_
                            (tick.value ==:. 2)
                            [ tick <--. 3 ]
                            [ if_
                                (tick.value ==:. 3)
                                [ tick <--. 4 ]
                                [ if_
                                    (tick.value ==:. 4)
                                    [ nn
                                      <-- sel_bottom
                                            (negate (sra mac.product ~by:15))
                                            ~width:22
                                    ; tick <--. 5
                                    ]
                                    [ if_
                                        (tick.value ==:. 5)
                                        [ tick <--. 6 ]
                                        [ vram_wen <-- vdd
                                        ; vram_waddr <-- uresize ii_slot ~width:vbits
                                        ; vram_wdata <-- uresize exp2_e ~width:32
                                        ; den <-- den.value +: uresize exp2_e ~width:24
                                        ; tick <--. 0
                                        ; if_
                                            (ii.value ==: n9 -:. 1)
                                            [ ii <--. 0; stage <--. 2; mac_go <-- vdd ]
                                            [ ii <-- ii.value +:. 1 ]
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
                (* WeightedSum: lane [row] = (sum over the ages of weight * value) / den;
                   the one pending divide holds the walk until its lane lands *)
                [ mac_inner <-- n9
                ; mac_outer <--. head_d
                ; hold <-- pending.value
                ; vram_raddr <-- uresize mac_ii_slot ~width:vbits
                ; vc_raddr
                  <-- lconst @: (cur.value -: mac_ii_slot) @: hd.value @: mac_oo_lane
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
                    ; yram_wdata <-- clamp16 div_quotient
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
            ]
        ]
      in
      entry, body
    | Temper ->
      (* the tempered weight of each code: masked, exp2, refused under min-p *)
      let entry = [ oo <--. 0; tick <--. 0; total <--. 0 ] in
      let body =
        [ vram_raddr <-- uresize oo8 ~width:vbits
        ; mul_a <-- sel_bottom diff.value ~width:25
        ; mul_b <-- of_signed_int ~width:18 temper_q14
        ; legal_query <-- oo8
        ; if_
            (tick.value ==:. 0)
            [ tick <--. 1 ]
            [ if_
                (tick.value ==:. 1)
                [ diff <-- vramd -: peak.value; tick <--. 2 ]
                [ if_
                    (tick.value ==:. 2)
                    [ tick <--. 3 ]
                    [ if_
                        (tick.value ==:. 3)
                        [ tick <--. 4 ]
                        [ if_
                            (tick.value ==:. 4)
                            [ nn
                              <-- sel_bottom (negate (sra mac.product ~by:14)) ~width:22
                            ; tick <--. 5
                            ]
                            [ if_
                                (tick.value ==:. 5)
                                [ tick <--. 6 ]
                                [ (let keep =
                                     legal
                                     &: (exp2_e >=: of_unsigned_int ~width:16 min_weight)
                                   in
                                   let w = mux2 keep exp2_e (zero 16) in
                                   proc
                                     [ vram_wen <-- vdd
                                     ; vram_waddr <-- uresize oo8 ~width:vbits
                                     ; vram_wdata <-- uresize w ~width:32
                                     ; total <-- total.value +: uresize w ~width:24
                                     ])
                                ; tick <--. 0
                                ; if_
                                    (oo.value ==:. vocab - 1)
                                    finish
                                    [ oo <-- oo.value +:. 1 ]
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
      in
      entry, body
    | Draw ->
      (* three PRNG bytes, high first: the walk of [Prng.uniform] *)
      let entry = [ tick <--. 0 ] in
      let body =
        [ if_
            (tick.value ==:. 0)
            [ prng_step <-- vdd; tick <--. 1 ]
            [ if_
                (tick.value ==:. 1)
                [ prng_step <-- vdd
                ; u24 <-- sel_bottom u24.value ~width:16 @: prng_byte
                ; tick <--. 2
                ]
                [ if_
                    (tick.value ==:. 2)
                    [ prng_step <-- vdd
                    ; u24 <-- sel_bottom u24.value ~width:16 @: prng_byte
                    ; tick <--. 3
                    ]
                    ([ u24 <-- sel_bottom u24.value ~width:16 @: prng_byte ] @ finish)
                ]
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
        ; if_
            (tick.value ==:. 0)
            [ tick <--. 1 ]
            [ if_
                (tick.value ==:. 1)
                [ tick <--. 2 ]
                [ if_
                    (tick.value ==:. 2)
                    [ thi <-- mac.product; tick <--. 3 ]
                    [ if_
                        (tick.value ==:. 3)
                        [ tick <--. 4 ]
                        ([ thr
                           <-- sel_bottom
                                 (srl
                                    (sll (uresize thi.value ~width:56) ~by:12
                                     +: uresize mac.product ~width:56)
                                    ~by:24)
                                 ~width:24
                         ]
                         @ finish)
                    ]
                ]
            ]
        ]
      in
      entry, body
    | Pick ->
      (* the first code whose running total passes the threshold; the last code catches a
         walk no weight stopped — the fallback decides at [Decide] *)
      let entry = [ oo <--. 0; tick <--. 0; cum <--. 0; found <--. 0 ] in
      let body =
        [ vram_raddr <-- uresize oo8 ~width:vbits
        ; if_
            (tick.value ==:. 0)
            [ tick <--. 1 ]
            [ (let w = sel_bottom vramd ~width:24 in
               let cum_next = cum.value +: uresize w ~width:25 in
               let passes = cum_next >: uresize thr.value ~width:25 in
               proc
                 [ cum <-- cum_next
                 ; when_
                     ~:(found.value)
                     [ when_ passes [ found <-- vdd; chosen <-- oo8; pos <-- (w <>:. 0) ]
                     ; when_
                         (oo.value ==:. vocab - 1)
                         [ chosen <--. 255; pos <-- (w <>:. 0) ]
                     ]
                 ])
            ; tick <--. 0
            ; if_ (oo.value ==:. vocab - 1) finish [ oo <-- oo.value +:. 1 ]
            ]
        ]
      in
      entry, body
  in
  (* the chain: op [k]'s finish is op [k+1]'s entry and the pc move; the last op of a
     segment leaves [Run] instead *)
  let rec chain index final = function
    | [] -> final, []
    | op :: rest ->
      let next_entry, tail = chain (index + 1) final rest in
      let entry, body = build op ~finish:next_entry in
      entry @ [ pc <--. index ], (index, body) :: tail
  in
  let sample_entry, sample_bodies = chain 0 [ sm.set_next Decide ] prog.sample in
  let forward_entry, forward_bodies =
    chain sample_length [ sm.set_next ForwardDone ] prog.forward
  in
  let enter_sample = sample_entry @ [ sm.set_next Run ] in
  let enter_forward = forward_entry @ [ sm.set_next Run ] in
  (* one parallel case, not a chain of guards: see the L3 note of the module comment *)
  let run_body =
    [ switch
        pc.value
        (List.map (sample_bodies @ forward_bodies) ~f:(fun (index, body) ->
           of_unsigned_int ~width:pc_bits index, body))
    ]
  in
  compile
    [ sm.switch
        [ ( Idle
          , [ when_
                i.rewind
                ([ cur <--. 0
                 ; filled <--. 0
                 ; s <--. 0
                 ; boot <-- vdd
                 ; out_code <--. 255
                 ; after_forward <--. 0
                 ]
                 @ List.concat
                     (List.init Token.seats ~f:(fun k ->
                        [ seat_full.(k) <--. 0; seat_pitch.(k) <--. 0 ]))
                 @ enter_forward)
            ; when_ (i.step &: ~:(i.rewind)) enter_sample
            ] )
        ; Run, run_body
        ; ( Decide
          , [ (let final = mux2 pos.value chosen.value (zero 8) in
               let p = sel_bottom final ~width:7 in
               proc
                 [ out_code <-- final
                 ; if_
                     (final ==:. 0)
                     ([ after_forward <--. 0 ] @ enter_forward)
                     [ out_seat <-- mux2 (msb final) free_seat (match_seat p)
                     ; sm.set_next Emit
                     ]
                 ])
            ] )
        ; ( Emit
          , [ when_
                i.ready
                ([ (let p = sel_bottom out_code.value ~width:7 in
                    proc
                      (List.init Token.seats ~f:(fun k ->
                         when_
                           (out_seat.value ==:. k)
                           [ seat_full.(k) <-- msb out_code.value
                           ; when_ (msb out_code.value) [ seat_pitch.(k) <-- p ]
                           ])))
                 ; after_forward <--. 1
                 ]
                 @ enter_forward)
            ] )
        ; ( ForwardDone
          , [ land_ <-- vdd
            ; when_ (out_code.value ==:. 0) [ s <-- s.value +:. 1 ]
            ; cur <-- cur.value +:. 1
            ; filled <-- (filled.value |: (cur.value ==:. slots - 1))
            ; if_ after_forward.value enter_sample [ sm.set_next Idle ]
            ] )
        ]
    ];
  { O.note =
      { Source_intf.Note.voice = out_seat.value
      ; pitch = uresize (sel_bottom out_code.value ~width:7) ~width:8
      ; on = msb out_code.value
      }
  ; valid = sm.is Emit
  ; idle = sm.is Idle
  }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the program is data: the state table prints" =
  let config =
    { Transformer.Config.d = 16; layers = 1; heads = 4; context = 16; slope_span = 8 }
  in
  let model = Quantized.Model.For_test.init config ~seed:11 in
  let { sample; forward } = schedule model in
  let show tag ops =
    List.iteri ops ~f:(fun index op ->
      Stdio.printf "%s%-2d %s\n" tag index (Sexp.to_string (Op.sexp_of_t op)))
  in
  show "s" sample;
  show "f" forward;
  let baseline =
    schedule (Quantized.Model.For_test.init Transformer.Config.baseline ~seed:11)
  in
  Stdio.printf
    "baseline: %d sample ops, %d forward ops\n"
    (List.length baseline.sample)
    (List.length baseline.forward);
  [%expect
    {|
    s0  Rms_norm
    s1  (Matvec((src Y)(w((base 0)(e 10)))(outer_major true)(inner 16)(outer 256)(landing To_logits)))
    s2  Temper
    s3  Draw
    s4  Threshold
    s5  Pick
    f0  (Embed(token((base 0)(e 10)))(phase((base 4096)(e 10)))(progress((base 4352)(e 10))))
    f1  Rms_norm
    f2  (Matvec((src Y)(w((base 4608)(e 11)))(outer_major false)(inner 16)(outer 16)(landing To_q)))
    f3  (Matvec((src Y)(w((base 4864)(e 11)))(outer_major false)(inner 16)(outer 16)(landing(To_ring(k true)(layer 0)))))
    f4  (Matvec((src Y)(w((base 5120)(e 11)))(outer_major false)(inner 16)(outer 16)(landing(To_ring(k false)(layer 0)))))
    f5  (Attend(layer 0))
    f6  (Matvec((src Y)(w((base 5376)(e 11)))(outer_major false)(inner 16)(outer 16)(landing Add_to_h)))
    f7  Rms_norm
    f8  (Matvec((src Y)(w((base 5632)(e 10)))(outer_major false)(inner 16)(outer 64)(landing To_hidden)))
    f9  (Matvec((src Hidden)(w((base 6656)(e 11)))(outer_major false)(inner 64)(outer 16)(landing Add_to_h)))
    baseline: 6 sample ops, 19 forward ops
    |}]
;;

(* The stream comparison: the circuit against the reference, event for event, on drawn
   weights. This is the gate that holds the circuit to [Quantized]. *)
let%expect_test "the source agrees with the reference, event for event" =
  let model = Quantized.Model.For_test.init Transformer.Config.baseline ~seed:11 in
  let seed = 42 in
  let steps = 3 in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 seed)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.ready := Bits.vdd;
  let budget = ref 30_000_000 in
  let cycle () =
    Cyclesim.cycle sim;
    Int.decr budget;
    assert (!budget > 0)
  in
  inp.rewind := Bits.vdd;
  cycle ();
  inp.rewind := Bits.gnd;
  cycle ();
  while not (Bits.to_bool !(out.idle)) do
    cycle ()
  done;
  let step () =
    inp.step := Bits.vdd;
    cycle ();
    inp.step := Bits.gnd;
    cycle ();
    let events = ref [] in
    while not (Bits.to_bool !(out.idle)) do
      if Bits.to_bool !(out.valid)
      then
        events
        := ( Bits.to_int_trunc !(out.note.voice)
           , Bits.to_int_trunc !(out.note.pitch)
           , Bits.to_bool !(out.note.on) )
           :: !events;
      cycle ()
    done;
    List.rev !events
  in
  let circuit =
    List.rev
      (List.fold (List.range 0 steps) ~init:[] ~f:(fun acc (_ : int) -> step () :: acc))
  in
  let engine = Quantized.Engine.init model ~seed in
  let (_ : Quantized.Engine.t), reference_reversed =
    List.fold (List.range 0 steps) ~init:(engine, []) ~f:(fun (engine, acc) (_ : int) ->
      let engine, events = Quantized.Engine.next_step engine in
      ( engine
      , List.map events ~f:(fun { Quantized.Engine.voice; pitch; on } -> voice, pitch, on)
        :: acc ))
  in
  let reference = List.rev reference_reversed in
  Stdio.printf
    "%d steps, %d events, the streams agree: %b\n"
    steps
    (List.length (List.concat circuit))
    ([%compare.equal: (int * int * bool) list list] circuit reference);
  if not ([%compare.equal: (int * int * bool) list list] circuit reference)
  then (
    Stdio.print_s ([%sexp_of: (int * int * bool) list list] circuit);
    Stdio.print_s ([%sexp_of: (int * int * bool) list list] reference));
  [%expect {| 3 steps, 8 events, the streams agree: true |}]
;;

(* The cycle bench: the circuit's measured cost against [Op.cycles], phase by phase. A
   phase counts from its command cycle to the cycle [idle] reads true. A round of the walk
   is one draw and its forward: a note round costs the two programs plus three control
   cycles (Decide, Emit, ForwardDone); the closing round (the drawn 0) has no Emit but
   ends over ForwardDone and Idle, thus three as well; the observation lands one cycle
   after Idle sets. The ring's fill [n] is the forward count so far, capped at the slot
   count. *)
let%expect_test "the cycle bench: the measured walk against the cost model" =
  let config = Transformer.Config.baseline in
  let model = Quantized.Model.For_test.init config ~seed:11 in
  let prog = schedule model in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 42)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.ready := Bits.vdd;
  let slots = config.context in
  let forwards = ref 0 in
  let sum_ops ops ~n =
    List.fold ops ~init:0 ~f:(fun total op -> total + Op.cycles config ~n op)
  in
  let model_round ~closing =
    let n = Int.min (!forwards + 1) slots in
    Int.incr forwards;
    sum_ops prog.sample ~n + sum_ops prog.forward ~n + if closing then 4 else 3
  in
  let events = ref 0 in
  let count_until_idle () =
    let cycles = ref 0 in
    while not (Bits.to_bool !(out.idle)) do
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.valid) then Int.incr events;
      Int.incr cycles;
      assert (!cycles < 20_000_000)
    done;
    !cycles
  in
  inp.rewind := Bits.vdd;
  Cyclesim.cycle sim;
  inp.rewind := Bits.gnd;
  Cyclesim.cycle sim;
  let boot_measured = 2 + count_until_idle () in
  let n_boot = Int.min (!forwards + 1) slots in
  Int.incr forwards;
  let boot_model = sum_ops prog.forward ~n:n_boot + 3 in
  Stdio.printf
    "boot: measured %d, model %d, delta %d\n"
    boot_measured
    boot_model
    (boot_measured - boot_model);
  let step k =
    events := 0;
    inp.step := Bits.vdd;
    Cyclesim.cycle sim;
    inp.step := Bits.gnd;
    Cyclesim.cycle sim;
    let measured = 2 + count_until_idle () in
    let rounds = !events + 1 in
    let modeled =
      List.init rounds ~f:(fun r -> model_round ~closing:(r = rounds - 1))
      |> List.fold ~init:0 ~f:( + )
    in
    Stdio.printf
      "step %d: %d events, measured %d, model %d, delta %d\n"
      k
      !events
      measured
      modeled
      (measured - modeled);
    measured
  in
  let total = List.fold (List.range 1 4) ~init:boot_measured ~f:(fun t k -> t + step k) in
  Stdio.printf "total %d\n" total;
  [%expect
    {|
    boot: measured 115547, model 115547, delta 0
    step 1: 4 events, measured 690131, model 690131, delta 0
    step 2: 2 events, measured 417823, model 417823, delta 0
    step 3: 2 events, measured 420631, model 420631, delta 0
    total 1644132
    |}]
;;
