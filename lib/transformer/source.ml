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
     appears twice per layer and once in the sampler; control is cheap, and the units it
     drives exist once.
   - [Attend] walks lane-major. The age-major order — one exp2, then one MAC over the
     lanes, age by age — was declined: it needs a register file of lane sums, and
     lane-major needs none.

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
     [preg] — is hand-encoded as tick positions, in [exp_weight_chain] for the Exp and
     Temper chains and in the Threshold chain. If the pipe ever deepens, replace the tick
     counts with a wait on a valid bit; do not renumber. *)

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
    (* the three tables add row for row, thus one exponent covers all of them; the walk
       reads them as one tensor of three bases. [Quantized.Model.check_shape] holds the
       rule, and [create] calls it before it reads a base. *)
    | Embed of
        { token : int
        ; phase : int
        ; progress : int
        ; e : int
        }
    | Rms_norm
    | Matvec of matvec
    | Attend of { layer : int }
    | Temper
    | Draw
    | Threshold
    | Pick
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
     predecessor's finish runs. [n] is the filled slot count of the ring at this token. *)
  let cycles (config : Transformer.Config.t) ~n (op : t) =
    let { Transformer.Config.d; heads; _ } = config in
    let head_d = d / heads in
    let vocab = Token.vocab in
    match op with
    | Embed _ -> (3 * d) + Cost.drain
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
    | Temper -> Cost.exp_weight * vocab
    | Draw -> Cost.draw
    | Threshold -> Cost.threshold
    | Pick -> Cost.pick * vocab
  ;;

  (* The cycles a piece boundary costs, before the step it opens draws anything: the
     release states one OFF for each sounding pitch — a command cycle and a [ready] cycle
     for each — then one cycle finds none sounding and commands the forward of START,
     which walks a ring of one, and one cycle lands it. [forward] is the forward program.
     The cycle that leaves [Idle] is the one a step without a boundary spends entering its
     draw, thus it is not here. *)
  let boundary_cycles config forward ~released =
    (2 * released)
    + 1
    + List.fold forward ~init:0 ~f:(fun total op -> total + cycles config ~n:1 op)
    + 1
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
  let tensor_at (q : Quantized.Model.quantized) base = { Op.base; e = q.e } in
  (* every matvec reads [`Y] into [d] rows of [d] terms, inner-major; the defaults hold
     for all but the two FFN weights and the tied head *)
  let matvec ?(src = `Y) ?(outer_major = false) ?(inner = d) ?(outer = d) w base landing =
    Op.Matvec { src; w = tensor_at w base; outer_major; inner; outer; landing }
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
  { sample =
      [ Rms_norm
      ; (* the tied head reads the token table backward *)
        matvec ~outer_major:true ~outer:vocab model.params.embed bases.embed To_logits
      ; Temper
      ; Draw
      ; Threshold
      ; Pick
      ]
  ; forward =
      Embed
        { token = bases.embed
        ; phase = bases.phase
        ; progress = bases.progress
        ; e = model.params.embed.e
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
    | Release
    | ForwardDone
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(model : Quantized.Model.t) ~seed (i : _ I.t) : _ O.t =
  let { Quantized.Model.config; params; temper; min_weight; piece_steps } = model in
  let { Transformer.Config.d; heads; context = slots; slope_span = span; layers } =
    config
  in
  let head_d = d / heads in
  let dff = 4 * d in
  (* the shift rules of the reference; the packing below derives every width *)
  Quantized.Model.check_shape model;
  let dbits = Int.floor_log2 d in
  let lane_bits = Int.floor_log2 head_d in
  let head_bits = Int.floor_log2 heads in
  let slot_bits = Int.floor_log2 slots in
  let layer_bits = Int.max 1 (Int.ceil_log2 layers) in
  let ring_bits = layer_bits + slot_bits + dbits in
  (* vram serves the scores, the FFN hidden, the logits and the sampler weights *)
  let vram_size = Int.max vocab (Int.max dff slots) in
  let vbits = address_bits_for vram_size in
  let score_shift = Quantized.Constants.score_shift ~head_d in
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
  (* 32 bits: the boundary test below reads [s <> 0], thus a wrap would skip one boundary
     and part the circuit from the reference for good. At 200 ms a step, 16 bits wrap in
     under four hours — an overnight walk — and 32 bits in 27 years. *)
  let s = Variable.reg spec ~width:32 in
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
  (* The banking rules are in [docs/transformer_rtl.md]; the measurements behind them are
     here. The tools demoted deep write-portless arrays to slice logic under two different
     select shapes — the six-layer image in slice logic is 69 percent of the device — thus
     a bank is an initialized memory with a gated-off write port, and RAM_STYLE pins it.
     The reads sit behind a nest of muxes on registered top address bits: two cycles from
     address to data, as one ROM, because each bank registers its own data a second time
     before the mux and that register packs into the block RAM's output register. A read
     past the image selects a bank at a dead offset; no op makes one. *)
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
  (* The read of a ring restores the eight zero low bits that [Quantized.coarse_to_ring]
     dropped at the write. Every memory the walk reads stands two registers deep, and
     [nohold] freezes each stage with the walk's tags; the small RAMs keep the
     one-register tap for the bespoke chains. *)
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
  (* The release of a piece boundary: the lowest sounding pitch and the seat that holds
     it, thus the OFFs climb as the grammar states them. The seats hold exactly the
     sounding pitches, thus these four registers answer it and the sounding mask needs no
     port of its own. The sentinel is 127, above every pitch an On may take. *)
  let release_any, release_pitch, release_seat =
    let lower (any, pitch, seat) k =
      let full = seat_full.(k).value in
      let held = seat_pitch.(k).value in
      let takes = full &: (~:any |: (held <: pitch)) in
      any |: full, mux2 takes held pitch, mux2 takes (of_unsigned_int ~width:2 k) seat
    in
    List.fold (List.range 0 Token.seats) ~init:(gnd, ones 7, zero 2) ~f:lower
  in
  (* The piece boundary: a positive multiple of the policy. The policy is a power of two,
     thus the test is a bit-slice of the step counter — the shift rule the width, the
     context and the two table periods all obey. *)
  let at_boundary =
    match piece_steps with
    | None -> gnd
    | Some steps ->
      (* the slice must sit strictly inside the counter, or the guard on zero would
         swallow every boundary *)
      assert (Int.floor_log2 steps < width s.value);
      sel_bottom s.value ~width:(Int.floor_log2 steps) ==:. 0 &: (s.value <>:. 0)
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
  (* [build op ~finish] gives the entry actions and the body of one program step. [finish]
     runs in the op's last cycle: the next op's entry, and the pc move — an op initializes
     its own counters, and its predecessor runs that entry. *)
  let build (op : Op.t) ~(finish : Always.t list) =
    match op with
    | Op.Embed { token; phase = ph; progress; e } ->
      (* h[row] = the three table rows summed on the walk: a term is a table, a row is an
         element *)
      let entry = [ mac_go <-- vdd ] in
      let body =
        [ mac_inner <--. 3
        ; mac_outer <--. d
        ; rom_addr
          <-- mux
                (sel_bottom mac.ii ~width:2)
                [ rom_const token
                  +: uresize (out_code.value @: mac_oo_d) ~width:rom_addr_bits
                ; rom_const ph +: uresize (phase @: mac_oo_d) ~width:rom_addr_bits
                ; rom_const progress +: uresize (bucket @: mac_oo_d) ~width:rom_addr_bits
                ]
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
          ; rom_addr <-- rom_const w.base +: uresize addr ~width:rom_addr_bits
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
           ; ring_waddr
             <-- of_unsigned_int ~width:layer_bits layer @: cur.value @: mac_row_d
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
         let logit = sel_bottom (sra mac.sum ~by:w.e) ~width:32 in
         simple
           ~entry_extra:[ peak <-- min32 ]
           [ vram_wen <-- vdd
           ; vram_waddr <-- uresize row_o ~width:vbits
           ; vram_wdata <-- logit
           ; legal_query <-- uresize row_o ~width:8
           ; when_ (legal &: (logit >+ peak.value)) [ peak <-- logit ]
           ]
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
      (* one row of [head_d] terms per age; the retired row is the age's score *)
      let score_ages =
        [ mac_inner <--. head_d
        ; mac_outer <-- n9
        ; qram_raddr <-- hd.value @: mac_ii_lane
        ; kc_raddr <-- lconst @: slot_of_oo @: hd.value @: mac_ii_lane
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
        ; vc_raddr <-- lconst @: slot_of_ii @: hd.value @: mac_oo_lane
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
      (* the tempered weight of each code: masked, exp2, refused under min-p *)
      let entry = [ oo <--. 0; tick <--. 0; total <--. 0 ] in
      (* the min-p refusal: a weight under the share of the peak weighs nothing *)
      let keep = legal &: (exp2_e >=: of_unsigned_int ~width:16 min_weight) in
      let w = mux2 keep exp2_e (zero 16) in
      let body =
        (legal_query <-- oo8)
        :: exp_weight_chain
             ~addr:oo8
             ~scale:temper
             ~land_:
               [ vram_wdata <-- uresize w ~width:32
               ; total <-- total.value +: uresize w ~width:24
               ]
             ~advance:[ if_ (oo.value ==:. vocab - 1) finish [ oo <-- oo.value +:. 1 ] ]
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
      (* the first code whose running total passes the threshold; the last code catches a
         walk no weight stopped — the fallback decides at [Decide] *)
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
                       [ when_
                           passes
                           [ found <-- vdd; chosen <-- oo8; pos <-- (w <>:. 0) ]
                       ; when_
                           (oo.value ==:. vocab - 1)
                           [ chosen <--. vocab - 1; pos <-- (w <>:. 0) ]
                       ]
                   ])
              ; tick <--. 0
              ; if_ (oo.value ==:. vocab - 1) finish [ oo <-- oo.value +:. 1 ]
              ]
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
                 ; out_code <--. Token.to_code Token.Start
                 ; after_forward <--. 0
                 ]
                 @ List.concat
                     (List.init Token.seats ~f:(fun k ->
                        [ seat_full.(k) <--. 0; seat_pitch.(k) <--. 0 ]))
                 @ enter_forward)
            ; when_
                (i.step &: ~:(i.rewind))
                [ if_ at_boundary [ tick <--. 0; sm.set_next Release ] enter_sample ]
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
        ; ( Release
          , (* The piece boundary. The release states its OFFs climbing, one to a [ready],
               and those events are not drawn tokens: the context they would enter is
               cleared below. Then the walk returns to the boot state and START feeds the
               engine, as it does at a rewind — but the step counter and the PRNG stand,
               thus the piece is new and the whole walk is still one function of the seed. *)
            [ if_
                release_any
                [ if_
                    (tick.value ==:. 0)
                    [ (* the code of an Off is its pitch *)
                      out_code <-- uresize release_pitch ~width:8
                    ; out_seat <-- release_seat
                    ; tick <--. 1
                    ]
                    [ when_
                        i.ready
                        [ proc
                            (List.init Token.seats ~f:(fun k ->
                               when_ (out_seat.value ==:. k) [ seat_full.(k) <--. 0 ]))
                        ; tick <--. 0
                        ]
                    ]
                ]
                ([ cur <--. 0
                 ; filled <--. 0
                 ; boot <-- vdd
                 ; out_code <--. Token.to_code Token.Start
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
  ; valid = sm.is Emit |: (sm.is Release &: release_any &: (tick.value ==:. 1))
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
    f0  (Embed(token 0)(phase 4096)(progress 4352)(e 10))
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
let stream_agrees ~model ~seed ~steps =
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
    Stdio.print_s ([%sexp_of: (int * int * bool) list list] reference))
;;

let%expect_test "the source agrees with the reference, event for event" =
  stream_agrees
    ~model:(Quantized.Model.For_test.init Transformer.Config.baseline ~seed:11)
    ~seed:42
    ~steps:3;
  [%expect {| 3 steps, 8 events, the streams agree: true |}]
;;

(* The same gate over a piece boundary. The arc is short here and 256 on the board: a
   boundary costs a whole forward pass in simulation, and the rule under test is the
   boundary and not its period. Twenty steps cross two of them. *)
let%expect_test "the source agrees with the reference over a piece boundary" =
  stream_agrees
    ~model:
      (Quantized.Model.For_test.init
         ~piece_steps:(Some 8)
         Transformer.Config.baseline
         ~seed:11)
    ~seed:42
    ~steps:20;
  [%expect {| 20 steps, 44 events, the streams agree: true |}]
;;

(* The cycle bench: the circuit's measured cost against [Op.cycles], phase by phase. A
   phase counts from its command cycle to the cycle [idle] reads true. A round of the walk
   is one draw and its forward: a note round costs the two programs plus three control
   cycles (Decide, Emit, ForwardDone); the closing round (the drawn 0) has no Emit but
   ends over ForwardDone and Idle, thus three as well; the observation lands one cycle
   after Idle sets. The ring's fill [n] is the forward count so far, capped at the slot
   count. *)
let bench ~piece_steps ~steps () =
  let config = Transformer.Config.baseline in
  let model = Quantized.Model.For_test.init ~piece_steps config ~seed:11 in
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
  (* the sounding set follows the socket, thus a boundary step knows what it will release
     before it runs *)
  let sounding = ref (Set.empty (module Int)) in
  let count_until_idle () =
    let cycles = ref 0 in
    while not (Bits.to_bool !(out.idle)) do
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.valid)
      then (
        Int.incr events;
        let pitch = Bits.to_int_trunc !(out.note.pitch) in
        sounding
        := (if Bits.to_bool !(out.note.on) then Set.add else Set.remove) !sounding pitch);
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
  (* [k] counts the steps the bench has commanded, thus the step counter of the source is
     [k - 1] when this one draws *)
  let step k =
    let boundary =
      match piece_steps with
      | Some period when k - 1 > 0 && (k - 1) % period = 0 -> Some (Set.length !sounding)
      | Some (_ : int) | None -> None
    in
    events := 0;
    inp.step := Bits.vdd;
    Cyclesim.cycle sim;
    inp.step := Bits.gnd;
    Cyclesim.cycle sim;
    let measured = 2 + count_until_idle () in
    (* the release is a socket event and not a draw, thus the rounds count the rest; the
       ring is empty behind the boundary, and its START forward fills the one slot *)
    let released = Option.value boundary ~default:0 in
    let opening =
      match boundary with
      | None -> 0
      | Some released ->
        forwards := 1;
        Op.boundary_cycles config prog.forward ~released
    in
    let rounds = !events - released + 1 in
    let modeled =
      opening
      + (List.init rounds ~f:(fun r -> model_round ~closing:(r = rounds - 1))
         |> List.fold ~init:0 ~f:( + ))
    in
    Stdio.printf
      "step %d:%s %d events, measured %d, model %d, delta %d\n"
      k
      (match boundary with
       | None -> ""
       | Some released -> Printf.sprintf " boundary releases %d," released)
      !events
      measured
      modeled
      (measured - modeled);
    measured
  in
  let total =
    List.fold (List.range 1 (steps + 1)) ~init:boot_measured ~f:(fun t k -> t + step k)
  in
  Stdio.printf "total %d\n" total
;;

let%expect_test "the cycle bench: the measured walk against the cost model" =
  bench ~piece_steps:None ~steps:3 ();
  [%expect
    {|
    boot: measured 115547, model 115547, delta 0
    step 1: 4 events, measured 690131, model 690131, delta 0
    step 2: 2 events, measured 417823, model 417823, delta 0
    step 3: 2 events, measured 420631, model 420631, delta 0
    total 1644132
    |}]
;;

(* The same bench over a piece boundary. A boundary costs the release, the forward of
   START over an empty ring, and the landing — the cycles a step of the board pays four
   times an hour. The arc is short here for the same reason the stream gate's is. *)
let%expect_test "the cycle bench over a piece boundary" =
  bench ~piece_steps:(Some 2) ~steps:4 ();
  [%expect
    {|
    boot: measured 115547, model 115547, delta 0
    step 1: 4 events, measured 690131, model 690131, delta 0
    step 2: 2 events, measured 417823, model 417823, delta 0
    step 3: boundary releases 4, 8 events, measured 805685, model 805685, delta 0
    step 4: 2 events, measured 417823, model 417823, delta 0
    total 2447009
    |}]
;;
