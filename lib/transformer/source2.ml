(* Source2: the transformer note source, in the production organization. The same socket,
   the same integers as [Quantized]; only the shape of the code changes.

   The prototype [Source] is one hand-written FSM of 23 states. This file splits the same
   machine into five layers:

   - L0, the primitives: the divider, the isqrt, the exp2 lookup, the sounding mask, and
     [Prng.Rtl]. Real hardware with tiny interfaces, each block-tested against an exact
     oracle. They stay in this file for now and can move to files of their own.
   - L1, the datapath: the one 25x18 MAC behind operand registers, the RAMs, the KV rings,
     the weight ROM and the shared walk registers. Built once; the ops mux its ports. No
     op owns a resource.
   - L2, the schedule: the walk as a value — [schedule] gives the two programs, the
     sampler and the forward pass, as [Op.t] lists. One op holds the facts one switch arm
     of the prototype hand-encodes: the tensor base and exponent, the address order, the
     loop bounds, the landing. The program is config-shaped: the layer count is a loop
     bound, and each op knows its layer at elaboration. Therefore the [sub], [lyr] and
     [rms_ret] registers of the prototype do not exist here, and every per-layer mux folds
     to a constant.
   - L3, the compiler: [chain] folds a program into states of a program counter over the
     datapath. An op's finish runs the next op's entry actions in the same cycle — the one
     convention that replaces the prototype's hand-kept register resets.
   - L4, the outer FSM, hand-written and small: Idle, Run, Decide, Emit, ForwardDone. The
     token walk, the seats and the socket handshake stay control; the mathematics is a
     program.

   The gates:

   - The stream test at the bottom: the events must equal [Quantized.Engine], event for
     event, on drawn weights — the same gate [Source] passes, thus the two circuits agree
     with each other through the one reference.
   - The block tests of the L0 units against exact oracles.
   - The schedule prints: the state table is data, and an expect test pins it.

   Cycle counts are not preserved, and need not be: the socket is latency-insensitive and
   the PRNG steps only on command, thus a draw cannot move.

   Design choices, decided here:

   - A repeated op is an inlined program step. [Rms_norm] appears twice per layer and once
     in the sampler; control is cheap, and the units it drives exist once. No return
     register.
   - [Attend] stays one bespoke op with an internal stage register: its interleave of
     score, exp and MAC is not a matvec, and forcing one shape onto it would put an
     interpreter in the generator.
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
(* L0 — the primitives *)
(* ==================================================================== *)

(* The restoring divider, toward zero: the magnitude walks bit by bit, the sign lands at
   the end — the one division rule of the circuit. [start] loads; [busy] falls when
   [quotient] holds, and the result stands until the next start. *)
module Divider = struct
  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      ; numerator : 'a [@bits 40]
      ; denominator : 'a [@bits 24]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { quotient : 'a [@bits 40]
      ; busy : 'a
      }
    [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let open Always in
    let m = Variable.reg spec ~width:40 in
    let d = Variable.reg spec ~width:24 in
    let q = Variable.reg spec ~width:40 in
    let r = Variable.reg spec ~width:25 in
    let n = Variable.reg spec ~width:6 in
    let sign = Variable.reg spec ~width:1 in
    let busy = Variable.reg spec ~width:1 in
    compile
      [ if_
          i.start
          [ m <-- mux2 (i.numerator <+ zero 40) (negate i.numerator) i.numerator
          ; sign <-- (i.numerator <+ zero 40)
          ; d <-- i.denominator
          ; q <--. 0
          ; r <--. 0
          ; n <--. 40
          ; busy <-- vdd
          ]
          [ when_
              busy.value
              [ (let r' = sel_bottom (r.value @: msb m.value) ~width:25 in
                 let fits = r' >=: uresize d.value ~width:25 in
                 proc
                   [ m <-- sll m.value ~by:1
                   ; n <-- n.value -:. 1
                   ; r <-- mux2 fits (r' -: uresize d.value ~width:25) r'
                   ; q <-- sel_bottom (q.value @: fits) ~width:40
                   ; when_ (n.value ==:. 1) [ busy <-- gnd ]
                   ])
              ]
          ]
      ];
    { O.quotient = mux2 sign.value (negate q.value) q.value; busy = busy.value }
  ;;
end

(* The restoring square root: one radicand bit pair a cycle. [busy] falls when [root]
   holds, and the result stands until the next start. *)
module Isqrt = struct
  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      ; value : 'a [@bits 42]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { root : 'a [@bits 21]
      ; busy : 'a
      }
    [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let open Always in
    let m = Variable.reg spec ~width:42 in
    let root = Variable.reg spec ~width:21 in
    let r = Variable.reg spec ~width:25 in
    let n = Variable.reg spec ~width:5 in
    let busy = Variable.reg spec ~width:1 in
    compile
      [ if_
          i.start
          [ m <-- i.value; root <--. 0; r <--. 0; n <--. 21; busy <-- vdd ]
          [ when_
              busy.value
              [ (let r' =
                   sel_bottom (r.value @: select m.value ~high:41 ~low:40) ~width:25
                 in
                 let trial =
                   uresize (root.value @: of_unsigned_int ~width:2 1) ~width:25
                 in
                 let fits = r' >=: trial in
                 proc
                   [ m <-- sll m.value ~by:2
                   ; n <-- n.value -:. 1
                   ; r <-- mux2 fits (r' -: trial) r'
                   ; root <-- sel_bottom (root.value @: fits) ~width:21
                   ; when_ (n.value ==:. 1) [ busy <-- gnd ]
                   ])
              ]
          ]
      ];
    { O.root = root.value; busy = busy.value }
  ;;
end

(* exp2 of a nonpositive Q12 value, as its magnitude [nn]: the top eight fraction bits
   read the table, the integer part shifts, 16 or more is 0. The table read registers,
   thus [nn] must stand for two cycles and [e] holds on the second. *)
module Exp2 = struct
  module I = struct
    type 'a t =
      { clock : 'a
      ; nn : 'a [@bits 22]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { e : 'a [@bits 16] } [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock () in
    let data =
      reg
        spec
        (rom
           ~read_addresses:[| select i.nn ~high:11 ~low:4 |]
           Quantized.Constants.exp2_bits).(0)
    in
    let big = select i.nn ~high:21 ~low:16 <>:. 0 in
    let shifted =
      mux (select i.nn ~high:15 ~low:12) (List.init 16 ~f:(fun k -> srl data ~by:k))
    in
    { O.e = mux2 big (zero 16) shifted }
  ;;
end

(* The sounding state as registers: the mask, the count, and the last-on/last-off pair —
   the rules of [Sounding_state]. [land_] applies one forwarded token (START changes
   nothing, END clears the valids); [query] asks the legality of one code,
   combinationally. *)
module Sounding = struct
  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; boot : 'a
      ; land_ : 'a
      ; code : 'a [@bits 8]
      ; query : 'a [@bits 8]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { legal : 'a } [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let open Always in
    let mask = Variable.reg spec ~width:128 in
    let count = Variable.reg spec ~width:3 in
    let last_on = Variable.reg spec ~width:7 in
    let lov = Variable.reg spec ~width:1 in
    let last_off = Variable.reg spec ~width:7 in
    let lofv = Variable.reg spec ~width:1 in
    let p = sel_bottom i.code ~width:7 in
    let hot = binary_to_onehot p in
    compile
      [ when_
          i.boot
          [ mask <-- zero 128
          ; count <--. 0
          ; lov <--. 0
          ; lofv <--. 0
          ; last_on <--. 0
          ; last_off <--. 0
          ]
      ; when_
          (i.land_ &: ~:(i.boot))
          [ if_
              (i.code ==:. 0)
              [ lov <--. 0; lofv <--. 0 ]
              [ when_
                  (i.code <>:. 255)
                  [ if_
                      (msb i.code)
                      [ mask <-- (mask.value |: hot)
                      ; count <-- count.value +:. 1
                      ; last_on <-- p
                      ; lov <--. 1
                      ]
                      [ mask <-- (mask.value &: ~:hot)
                      ; count <-- count.value -:. 1
                      ; last_off <-- p
                      ; lofv <--. 1
                      ]
                  ]
              ]
          ]
      ];
    let q = sel_bottom i.query ~width:7 in
    let bit = mux q (bits_lsb mask.value) in
    let off_ok = bit &: ~:(lov.value) &: (~:(lofv.value) |: (q >: last_off.value)) in
    let on_ok =
      ~:bit &: (count.value <:. Token.seats) &: (~:(lov.value) |: (q <: last_on.value))
    in
    { O.legal =
        mux2
          (i.query ==:. 0)
          vdd
          (mux2 (i.query ==:. 255) gnd (mux2 (msb i.query) on_ok off_ok))
    }
  ;;
end

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

  (* One step of the walk: the facts one switch arm of the prototype hand-encodes. The
     bespoke ops close over the model at elaboration and carry no fields here. *)
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
  let age = Variable.reg spec ~width:(slot_bits + 1) in
  let after_forward = Variable.reg spec ~width:1 in
  let acc = Variable.reg spec ~width:48 in
  let preg = Variable.reg spec ~width:43 in
  let thi = Variable.reg spec ~width:43 in
  let nums = Array.init head_d ~f:(fun (_ : int) -> Variable.reg spec ~width:40) in
  let den = Variable.reg spec ~width:24 in
  let peak = Variable.reg spec ~width:32 in
  let diff = Variable.reg spec ~width:32 in
  let nn = Variable.reg spec ~width:22 in
  let e_reg = Variable.reg spec ~width:16 in
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
  let { Sounding.O.legal } =
    Sounding.create
      { Sounding.I.clock = i.clock
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
     one cycle from address to data, as one ROM. A read past the image selects a bank at a
     dead offset; no op makes one. *)
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
      reg spec data)
    else (
      let split = if Int.is_pow2 n then n / 2 else 1 lsl Int.floor_log2 n in
      let low = rom_banked (Array.subo bits ~len:split) (lsbs addr) in
      let high =
        rom_banked
          (Array.subo bits ~pos:split)
          (sel_bottom (lsbs addr) ~width:(address_bits_for (n - split)))
      in
      mux2 (reg spec (msb addr)) high low)
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
     and the rings cost half the block RAM. *)
  let kcd =
    reg
      spec
      (ram
         ~size:(layers * slots * d)
         ~waddr:ring_waddr.value
         ~wen:kc_wen.value
         ~wdata:ring_wdata.value
         ~raddr:kc_raddr.value)
    @: zero 8
  in
  let vcd =
    reg
      spec
      (ram
         ~size:(layers * slots * d)
         ~waddr:ring_waddr.value
         ~wen:vc_wen.value
         ~wdata:ring_wdata.value
         ~raddr:vc_raddr.value)
    @: zero 8
  in
  let vramd =
    reg
      spec
      (ram
         ~size:vram_size
         ~waddr:vram_waddr.value
         ~wen:vram_wen.value
         ~wdata:vram_wdata.value
         ~raddr:vram_raddr.value)
  in
  let hramd =
    reg
      spec
      (ram
         ~size:d
         ~waddr:hram_waddr.value
         ~wen:hram_wen.value
         ~wdata:hram_wdata.value
         ~raddr:hram_raddr.value)
  in
  let yd =
    reg
      spec
      (ram
         ~size:d
         ~waddr:yram_waddr.value
         ~wen:yram_wen.value
         ~wdata:yram_wdata.value
         ~raddr:yram_raddr.value)
  in
  let qd =
    reg
      spec
      (ram
         ~size:d
         ~waddr:qram_waddr.value
         ~wen:qram_wen.value
         ~wdata:qram_wdata.value
         ~raddr:qram_raddr.value)
  in
  (* L1 — the one DSP-sized product, 25 by 18 signed, behind operand registers *)
  let mul_a = Variable.wire ~default:(zero 25) () in
  let mul_b = Variable.wire ~default:(zero 18) () in
  let opa = Variable.reg spec ~width:25 in
  let opb = Variable.reg spec ~width:18 in
  let mul_out = opa.value *+ opb.value in
  let preg48 = sresize preg.value ~width:48 in
  let acc_full = acc.value +: preg48 in
  (* the walk slices *)
  let ii_d = sel_bottom ii.value ~width:dbits in
  let ii_lane = sel_bottom ii.value ~width:lane_bits in
  let oo_d = sel_bottom oo.value ~width:dbits in
  let oo8 = sel_bottom oo.value ~width:8 in
  let phase = sel_bottom s.value ~width:4 in
  let bucket = select s.value ~high:7 ~low:4 in
  let age_s = sel_bottom age.value ~width:slot_bits in
  let slot = cur.value -: age_s in
  let n =
    mux2
      filled.value
      (of_unsigned_int ~width:(slot_bits + 1) slots)
      (uresize cur.value ~width:(slot_bits + 1) +:. 1)
  in
  let numv = mux ii_lane (Array.to_list (Array.map nums ~f:(fun v -> v.value))) in
  let alibi =
    (* the slope of head k is 2^-(span (k+1) / heads): a shift of the age, in Q12 *)
    mux
      hd.value
      (List.init heads ~f:(fun k ->
         let exponent = span * (k + 1) / heads in
         sll (uresize age.value ~width:32) ~by:(Quantized.Constants.y_q - exponent)))
  in
  (* One three-phase multiply-accumulate iteration: tick 0 presents the addresses, tick 1
     registers the operands, tick 2 the product, tick 3 accumulates. [keep] holds the
     whole sum at the last iteration instead of zeroing — the two-state residual join
     reads it. The one definition every matvec-family op walks. *)
  let mac_walk ?(keep = false) ~last ~at_last ~else_next () =
    [ if_
        (tick.value ==:. 0)
        [ tick <--. 1 ]
        [ if_
            (tick.value ==:. 1)
            [ opa <-- mul_a.value; opb <-- mul_b.value; tick <--. 2 ]
            [ if_
                (tick.value ==:. 2)
                [ preg <-- mul_out; tick <--. 3 ]
                [ tick <--. 0
                ; if_
                    last
                    (at_last @ [ (acc <-- if keep then acc_full else zero 48) ])
                    (else_next @ [ acc <-- acc_full ])
                ]
            ]
        ]
    ]
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
      (* h[ii] = the three table rows summed on the MAC; oo walks the tables *)
      let entry = [ ii <--. 0; oo <--. 0; tick <--. 0; acc <--. 0 ] in
      let body =
        [ rom_addr
          <-- mux
                (sel_bottom oo.value ~width:2)
                [ rom_const token.base
                  +: uresize (out_code.value @: ii_d) ~width:rom_addr_bits
                ; rom_const ph.base +: uresize (phase @: ii_d) ~width:rom_addr_bits
                ; rom_const progress.base +: uresize (bucket @: ii_d) ~width:rom_addr_bits
                ]
        ; mul_a <-- sresize romd ~width:25
        ; mul_b <-- of_signed_int ~width:18 1
        ]
        @ mac_walk
            ~last:(oo.value ==:. 2)
            ~at_last:
              [ hram_wen <-- vdd
              ; hram_waddr <-- ii_d
              ; hram_wdata
                <-- sel_bottom
                      (rescale ~from:token.e ~target:Quantized.Constants.h_q acc_full)
                      ~width:32
              ; oo <--. 0
              ; if_ (ii.value ==:. d - 1) finish [ ii <-- ii.value +:. 1 ]
              ]
            ~else_next:[ oo <-- oo.value +:. 1 ]
            ()
      in
      entry, body
    | Rms_norm ->
      (* stage 0 sums the squares of the Q12 copy; stage 1 waits on the isqrt; stage 2
         divides each element — y = (h << 8) / g, toward zero *)
      let entry = [ ii <--. 0; tick <--. 0; acc <--. 0; stage <--. 0 ] in
      let body =
        [ if_
            (stage.value ==:. 0)
            ([ hram_raddr <-- ii_d
             ; mul_a <-- sel_bottom (sra hramd ~by:4) ~width:25
             ; mul_b <-- sel_bottom (sra hramd ~by:4) ~width:18
             ]
             @ mac_walk
                 ~last:(ii.value ==:. d - 1)
                 ~at_last:
                   [ sq_start <-- vdd
                   ; sq_value
                     <-- sel_bottom
                           (srl acc_full ~by:(Int.floor_log2 d) +: eps48)
                           ~width:42
                   ; ii <--. 0
                   ; tick <--. 0
                   ; stage <--. 1
                   ]
                 ~else_next:[ ii <-- ii.value +:. 1 ]
                 ())
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
      let ii_i = sel_bottom ii.value ~width:ibits in
      let oo_o = sel_bottom oo.value ~width:obits in
      let addr = if outer_major then oo_o @: ii_i else ii_i @: oo_o in
      let read_src, srcd =
        match src with
        | `Y -> [ yram_raddr <-- ii_d ], yd
        | `Hidden ->
          [ vram_raddr <-- uresize ii_i ~width:vbits ], sel_bottom vramd ~width:16
      in
      let common =
        read_src
        @ [ rom_addr <-- rom_const w.base +: uresize addr ~width:rom_addr_bits
          ; mul_a <-- sresize srcd ~width:25
          ; mul_b <-- sresize romd ~width:18
          ]
      in
      let outer_last = oo.value ==:. outer - 1 in
      let entry = [ ii <--. 0; oo <--. 0; tick <--. 0; acc <--. 0; stage <--. 0 ] in
      (match landing with
       | To_q | To_ring _ | To_hidden | To_logits ->
         let write v =
           match landing with
           | To_q ->
             [ qram_wen <-- vdd
             ; qram_waddr <-- oo_d
             ; qram_wdata
               <-- clamp16
                     (rescale
                        ~from:(Quantized.Constants.y_q + w.e)
                        ~target:Quantized.Constants.kv_q
                        v)
             ]
           | To_ring { k; layer } ->
             [ (if k then kc_wen else vc_wen) <-- vdd
             ; ring_waddr <-- of_unsigned_int ~width:layer_bits layer @: cur.value @: oo_d
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
             ; vram_waddr <-- uresize oo_o ~width:vbits
             ; vram_wdata <-- sresize (clamp16 relu) ~width:32
             ]
           | To_logits ->
             let logit = sel_bottom (sra v ~by:w.e) ~width:32 in
             [ vram_wen <-- vdd
             ; vram_waddr <-- uresize oo_o ~width:vbits
             ; vram_wdata <-- logit
             ; legal_query <-- uresize oo_o ~width:8
             ; when_ (legal &: (logit >+ peak.value)) [ peak <-- logit ]
             ]
           | Add_to_h -> assert false
         in
         let entry =
           match landing with
           | To_logits -> entry @ [ peak <-- min32 ]
           | _ -> entry
         in
         let body =
           common
           @ mac_walk
               ~last:(ii.value ==:. inner - 1)
               ~at_last:
                 (write acc_full
                  @ [ ii <--. 0; if_ outer_last finish [ oo <-- oo.value +:. 1 ] ])
               ~else_next:[ ii <-- ii.value +:. 1 ]
               ()
         in
         entry, body
       | Add_to_h ->
         (* the residual join: the whole sum kept, then a two-tick read-modify-write *)
         let from_q =
           match src with
           | `Y -> Quantized.Constants.kv_q
           | `Hidden -> Quantized.Constants.hid_q
         in
         let body =
           [ if_
               (stage.value ==:. 0)
               (common
                @ mac_walk
                    ~keep:true
                    ~last:(ii.value ==:. inner - 1)
                    ~at_last:[ ii <--. 0; tick <--. 0; stage <--. 1 ]
                    ~else_next:[ ii <-- ii.value +:. 1 ]
                    ())
               [ hram_raddr <-- oo_d
               ; if_
                   (tick.value ==:. 0)
                   [ tick <--. 1 ]
                   [ hram_wen <-- vdd
                   ; hram_waddr <-- oo_d
                   ; hram_wdata
                     <-- sel_bottom
                           (sresize hramd ~width:48
                            +: rescale
                                 ~from:(from_q + w.e)
                                 ~target:Quantized.Constants.h_q
                                 acc.value)
                           ~width:32
                   ; acc <--. 0
                   ; tick <--. 0
                   ; if_ outer_last finish [ oo <-- oo.value +:. 1; stage <--. 0 ]
                   ]
               ]
           ]
         in
         entry, body)
    | Attend { layer } ->
      (* the attention of one layer, head by head: stage 0 scores the ages, stages 1 and 2
         interleave the exp2 weight and its MAC over the values, stage 3 divides the lanes
         by the total weight. Age [a] reads slot [(cur - a) & (slots - 1)], thus the ALiBi
         distance is the age and the causal wall is the walk. *)
      let lconst = of_unsigned_int ~width:layer_bits layer in
      let entry =
        [ hd <--. 0
        ; age <--. 0
        ; ii <--. 0
        ; tick <--. 0
        ; acc <--. 0
        ; den <--. 0
        ; peak <-- min32
        ; stage <--. 0
        ]
        @ Array.to_list (Array.map nums ~f:(fun v -> v <--. 0))
      in
      let body =
        [ if_
            (stage.value ==:. 0)
            (* Score *)
            ([ qram_raddr <-- hd.value @: ii_lane
             ; kc_raddr <-- lconst @: slot @: hd.value @: ii_lane
             ; mul_a <-- sresize qd ~width:25
             ; mul_b <-- sresize kcd ~width:18
             ]
             @ mac_walk
                 ~last:(ii.value ==:. head_d - 1)
                 ~at_last:
                   [ (let score =
                        sel_bottom (sra acc_full ~by:score_shift) ~width:32 -: alibi
                      in
                      proc
                        [ vram_wen <-- vdd
                        ; vram_waddr <-- uresize age_s ~width:vbits
                        ; vram_wdata <-- score
                        ; when_ (score >+ peak.value) [ peak <-- score ]
                        ])
                   ; ii <--. 0
                   ; if_
                       (age.value +:. 1 ==: n)
                       [ age <--. 0; tick <--. 0; stage <--. 1 ]
                       [ age <-- age.value +:. 1 ]
                   ]
                 ~else_next:[ ii <-- ii.value +:. 1 ]
                 ())
            [ if_
                (stage.value ==:. 1)
                (* Exp: the weight of one age *)
                [ vram_raddr <-- uresize age_s ~width:vbits
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
                            [ opa <-- mul_a.value; opb <-- mul_b.value; tick <--. 3 ]
                            [ if_
                                (tick.value ==:. 3)
                                [ preg <-- mul_out; tick <--. 4 ]
                                [ if_
                                    (tick.value ==:. 4)
                                    [ nn
                                      <-- sel_bottom
                                            (negate (sra preg.value ~by:15))
                                            ~width:22
                                    ; tick <--. 5
                                    ]
                                    [ if_
                                        (tick.value ==:. 5)
                                        [ tick <--. 6 ]
                                        [ e_reg <-- exp2_e
                                        ; den <-- den.value +: uresize exp2_e ~width:24
                                        ; ii <--. 0
                                        ; tick <--. 0
                                        ; stage <--. 2
                                        ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
                [ if_
                    (stage.value ==:. 2)
                    (* ExpMac: nums += e * v over the head's lanes *)
                    [ vc_raddr <-- lconst @: slot @: hd.value @: ii_lane
                    ; mul_a <-- uresize e_reg.value ~width:25
                    ; mul_b <-- sresize vcd ~width:18
                    ; if_
                        (tick.value ==:. 0)
                        [ tick <--. 1 ]
                        [ if_
                            (tick.value ==:. 1)
                            [ opa <-- mul_a.value; opb <-- mul_b.value; tick <--. 2 ]
                            [ if_
                                (tick.value ==:. 2)
                                [ preg <-- mul_out; tick <--. 3 ]
                                [ tick <--. 0
                                ; proc
                                    (List.init head_d ~f:(fun k ->
                                       when_
                                         (ii.value ==:. k)
                                         [ nums.(k)
                                           <-- nums.(k).value
                                               +: sel_bottom preg.value ~width:40
                                         ]))
                                ; if_
                                    (ii.value ==:. head_d - 1)
                                    [ ii <--. 0
                                    ; if_
                                        (age.value +:. 1 ==: n)
                                        [ age <--. 0; tick <--. 0; stage <--. 3 ]
                                        [ age <-- age.value +:. 1
                                        ; tick <--. 0
                                        ; stage <--. 1
                                        ]
                                    ]
                                    [ ii <-- ii.value +:. 1 ]
                                ]
                            ]
                        ]
                    ]
                    (* CtxDiv: ctx = nums / den, toward zero, into the y RAM *)
                    [ if_
                        (tick.value ==:. 0)
                        [ div_start <-- vdd
                        ; div_num <-- numv
                        ; div_den <-- den.value
                        ; tick <--. 1
                        ]
                        [ when_
                            ~:div_busy
                            [ yram_wen <-- vdd
                            ; yram_waddr <-- hd.value @: ii_lane
                            ; yram_wdata <-- clamp16 div_quotient
                            ; tick <--. 0
                            ; if_
                                (ii.value ==:. head_d - 1)
                                [ ii <--. 0
                                ; if_
                                    (hd.value ==:. heads - 1)
                                    finish
                                    ([ hd <-- hd.value +:. 1
                                     ; age <--. 0
                                     ; den <--. 0
                                     ; peak <-- min32
                                     ; stage <--. 0
                                     ]
                                     @ Array.to_list
                                         (Array.map nums ~f:(fun v -> v <--. 0)))
                                ]
                                [ ii <-- ii.value +:. 1 ]
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
                    [ opa <-- mul_a.value; opb <-- mul_b.value; tick <--. 3 ]
                    [ if_
                        (tick.value ==:. 3)
                        [ preg <-- mul_out; tick <--. 4 ]
                        [ if_
                            (tick.value ==:. 4)
                            [ nn <-- sel_bottom (negate (sra preg.value ~by:14)) ~width:22
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
            [ opa <-- mul_a.value; opb <-- mul_b.value; tick <--. 1 ]
            [ if_
                (tick.value ==:. 1)
                [ preg <-- mul_out; tick <--. 2 ]
                [ if_
                    (tick.value ==:. 2)
                    [ thi <-- preg.value; opb <-- mul_b.value; tick <--. 3 ]
                    [ if_
                        (tick.value ==:. 3)
                        [ preg <-- mul_out; tick <--. 4 ]
                        ([ thr
                           <-- sel_bottom
                                 (srl
                                    (sll (uresize thi.value ~width:56) ~by:12
                                     +: uresize preg.value ~width:56)
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
  let run_body =
    List.map (sample_bodies @ forward_bodies) ~f:(fun (index, body) ->
      when_ (pc.value ==:. index) body)
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

let%expect_test "the divider is the reference division, toward zero" =
  let module Sim = Cyclesim.With_interface (Divider.I) (Divider.O) in
  let sim = Sim.create Divider.create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let signed_of bits =
    let v = Bits.to_int_trunc bits in
    if v land (1 lsl 39) <> 0 then v - (1 lsl 40) else v
  in
  let divide n d =
    inp.numerator := Bits.of_signed_int ~width:40 n;
    inp.denominator := Bits.of_unsigned_int ~width:24 d;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim
    done;
    signed_of !(out.quotient)
  in
  List.iter
    [ 100, 7; -100, 7; 0, 5; 1234567, 89; -(1 lsl 38), 3 ]
    ~f:(fun (n, d) ->
      Stdio.printf "%d / %d = %d, the reference gives %d\n" n d (divide n d) (n / d));
  [%expect
    {|
    100 / 7 = 14, the reference gives 14
    -100 / 7 = -14, the reference gives -14
    0 / 5 = 0, the reference gives 0
    1234567 / 89 = 13871, the reference gives 13871
    -274877906944 / 3 = -91625968981, the reference gives -91625968981
    |}]
;;

let%expect_test "the isqrt floors, as the reference does" =
  let module Sim = Cyclesim.With_interface (Isqrt.I) (Isqrt.O) in
  let sim = Sim.create Isqrt.create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let isqrt v =
    inp.value := Bits.of_unsigned_int ~width:42 v;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim
    done;
    Bits.to_int_trunc !(out.root)
  in
  (* the oracle of the reference: floor of the square root *)
  let floor_sqrt v =
    let rec shrink g = if g * g > v then shrink (g - 1) else g in
    shrink (Float.to_int (Float.sqrt (Float.of_int v)) + 1)
  in
  List.iter
    [ 0; 15; 16; 4295; (1 lsl 41) + 12345 ]
    ~f:(fun v -> Stdio.printf "%d -> %d (oracle %d)\n" v (isqrt v) (floor_sqrt v));
  [%expect
    {|
    0 -> 0 (oracle 0)
    15 -> 3 (oracle 3)
    16 -> 4 (oracle 4)
    4295 -> 65 (oracle 65)
    2199023267897 -> 1482910 (oracle 1482910)
    |}]
;;

let%expect_test "the exp2 unit is the table and the shift" =
  let module Sim = Cyclesim.With_interface (Exp2.I) (Exp2.O) in
  let sim = Sim.create Exp2.create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let e nn =
    inp.nn := Bits.of_unsigned_int ~width:22 nn;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim;
    Bits.to_int_trunc !(out.e)
  in
  (* the oracle: exp2 of -nn/4096, in Q15 — [Quantized.Engine.exp2_q] *)
  let oracle nn =
    if nn lsr 16 <> 0
    then 0
    else (
      let entry =
        Float.iround_nearest_exn
          Float.(32768.0 * (2.0 ** (-of_int ((nn asr 4) land 255) / 256.0)))
      in
      entry asr ((nn asr 12) land 15))
  in
  List.iter [ 0; 2048; 4096; 8192; 70000 ] ~f:(fun nn ->
    Stdio.printf "%d -> %d (oracle %d)\n" nn (e nn) (oracle nn));
  [%expect
    {|
    0 -> 32768 (oracle 32768)
    2048 -> 23170 (oracle 23170)
    4096 -> 16384 (oracle 16384)
    8192 -> 8192 (oracle 8192)
    70000 -> 0 (oracle 0)
    |}]
;;

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
   weights — the same gate [Source] passes, thus the two circuits agree with each other
   through the one reference. *)
let%expect_test "source2 agrees with the reference, event for event" =
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
