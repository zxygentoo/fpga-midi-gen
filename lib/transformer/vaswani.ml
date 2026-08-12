open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

(* The walk of the engine, in the order of the twin. [Embed] sums the three table rows.
   The rms states serve three sites — the attention, the feed-forward and the head — and
   [rms_ret] names the state that follows. [Qkv] projects and fills the ring; [Score],
   [Exp], [ExpMac] and [CtxDiv] are the attention of one head at a time; the Wo and Ffn
   states join the residual stream. [Logits], [Weights], [Draw], [Thresh] and [Pick] are
   the sampler; [Decide] decodes the drawn code, [Emit] holds one event for the sequencer,
   and [ForwardDone] lands the model state of the forwarded token. *)
module State = struct
  type t =
    | Idle
    | Embed
    | RmsSum
    | RmsSqrt
    | RmsScale
    | Qkv
    | Score
    | Exp
    | ExpMac
    | CtxDiv
    | WoMac
    | WoAdd
    | Ffn1
    | Ffn2Mac
    | Ffn2Add
    | Logits
    | Weights
    | Draw
    | Thresh
    | Pick
    | Decide
    | Emit
    | ForwardDone
  [@@deriving compare ~localize, enumerate, sexp_of]
end

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

let create ~(model : Fixed.Model.t) ~seed (i : _ I.t) : _ O.t =
  let { Fixed.Model.config; params; temper_q14; min_weight } = model in
  (* The dimensions come from the configuration; the derived pair follows the network
     rules. The address packing of the prototype fixes the shape — one layer bit, an 8-bit
     slot, a 2-bit head, a 4-bit lane, a 6-bit dimension — and the checks state it loudly. *)
  let { Transformer.Config.d; heads; context = slots; slope_span = span; layers } =
    config
  in
  let head_d = d / heads in
  let dff = 4 * d in
  assert (layers = 2 && d = 64 && heads = 4 && slots = 256 && span = 8);
  assert (layers = Array.length params.layers);
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let e_tbl = params.embed.e in
  (* the ROM of the model, and the address width of its padded depth *)
  let rom_bits = Fixed.Model.rom_bits model in
  let rom_addr_bits = address_bits_for (Array.length rom_bits) in
  (* the address book: the base of each tensor, in the shape of the parameters *)
  let bases = Fixed.Model.rom_bases model in
  let layer_e f = [| f params.layers.(0); f params.layers.(1) |] in
  let eq = layer_e (fun (l : Fixed.Model.layer) -> l.wq.e) in
  let ek = layer_e (fun (l : Fixed.Model.layer) -> l.wk.e) in
  let ev = layer_e (fun (l : Fixed.Model.layer) -> l.wv.e) in
  let eo = layer_e (fun (l : Fixed.Model.layer) -> l.wo.e) in
  let e1 = layer_e (fun (l : Fixed.Model.layer) -> l.w1.e) in
  let e2 = layer_e (fun (l : Fixed.Model.layer) -> l.w2.e) in
  let min32 = of_signed_int ~width:32 (-(1 lsl 31)) in
  (* the counters and the walk registers *)
  let tick = Variable.reg spec ~width:3 in
  let ii = Variable.reg spec ~width:9 in
  let oo = Variable.reg spec ~width:9 in
  let sub = Variable.reg spec ~width:2 in
  let hd = Variable.reg spec ~width:2 in
  let age = Variable.reg spec ~width:9 in
  let lyr = Variable.reg spec ~width:1 in
  let rms_ret = Variable.reg spec ~width:2 in
  let after_forward = Variable.reg spec ~width:1 in
  let acc = Variable.reg spec ~width:48 in
  let preg = Variable.reg spec ~width:43 in
  let thi = Variable.reg spec ~width:43 in
  let nums = Array.init head_d ~f:(fun _ -> Variable.reg spec ~width:40) in
  let den = Variable.reg spec ~width:24 in
  let peak = Variable.reg spec ~width:32 in
  let diff = Variable.reg spec ~width:32 in
  let nn = Variable.reg spec ~width:22 in
  let e_reg = Variable.reg spec ~width:16 in
  (* the isqrt: the shifting radicand, the root and the remainder *)
  let m = Variable.reg spec ~width:42 in
  let sq_root = Variable.reg spec ~width:21 in
  let sq_rem = Variable.reg spec ~width:25 in
  let sq_i = Variable.reg spec ~width:5 in
  (* the divider: restoring on the magnitude with a sign — toward zero, as the twin *)
  let div_m = Variable.reg spec ~width:40 in
  let div_d = Variable.reg spec ~width:24 in
  let div_q = Variable.reg spec ~width:40 in
  let div_rem = Variable.reg spec ~width:25 in
  let div_i = Variable.reg spec ~width:6 in
  let div_sign = Variable.reg spec ~width:1 in
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
  let cur = Variable.reg spec ~width:8 in
  let filled = Variable.reg spec ~width:1 in
  (* the sounding state: the registers of the legality mask *)
  let sounding = Variable.reg spec ~width:128 in
  let scount = Variable.reg spec ~width:3 in
  let last_on = Variable.reg spec ~width:7 in
  let lov = Variable.reg spec ~width:1 in
  let last_off = Variable.reg spec ~width:7 in
  let lofv = Variable.reg spec ~width:1 in
  let seat_pitch = Array.init Token.seats ~f:(fun _ -> Variable.reg spec ~width:7) in
  let seat_full = Array.init Token.seats ~f:(fun _ -> Variable.reg spec ~width:1) in
  let _ = sm.current -- "state" in
  let _ = out_code.value -- "out_code" in
  let _ = s.value -- "step" in
  (* the draw *)
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
  (* the memory address and write wires *)
  let rom_addr = Variable.wire ~default:(zero rom_addr_bits) () in
  let exp2_addr = Variable.wire ~default:(zero 8) () in
  let kc_raddr = Variable.wire ~default:(zero 15) () in
  let vc_raddr = Variable.wire ~default:(zero 15) () in
  let kc_wen = Variable.wire ~default:gnd () in
  let vc_wen = Variable.wire ~default:gnd () in
  let ring_waddr = Variable.wire ~default:(zero 15) () in
  let ring_wdata = Variable.wire ~default:(zero 16) () in
  let vram_raddr = Variable.wire ~default:(zero 8) () in
  let vram_wen = Variable.wire ~default:gnd () in
  let vram_waddr = Variable.wire ~default:(zero 8) () in
  let vram_wdata = Variable.wire ~default:(zero 32) () in
  let hram_raddr = Variable.wire ~default:(zero 6) () in
  let hram_wen = Variable.wire ~default:gnd () in
  let hram_waddr = Variable.wire ~default:(zero 6) () in
  let hram_wdata = Variable.wire ~default:(zero 32) () in
  let yram_raddr = Variable.wire ~default:(zero 6) () in
  let yram_wen = Variable.wire ~default:gnd () in
  let yram_waddr = Variable.wire ~default:(zero 6) () in
  let yram_wdata = Variable.wire ~default:(zero 16) () in
  let qram_raddr = Variable.wire ~default:(zero 6) () in
  let qram_wen = Variable.wire ~default:gnd () in
  let qram_waddr = Variable.wire ~default:(zero 6) () in
  let qram_wdata = Variable.wire ~default:(zero 16) () in
  (* the memories; every read lands in a register, thus block RAM is inferred *)
  let romd = reg spec (rom ~read_addresses:[| rom_addr.value |] rom_bits).(0) in
  let exp2d =
    reg spec (rom ~read_addresses:[| exp2_addr.value |] Fixed.Constants.exp2_bits).(0)
  in
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
  let kcd =
    reg
      spec
      (ram
         ~size:(2 * slots * d)
         ~waddr:ring_waddr.value
         ~wen:kc_wen.value
         ~wdata:ring_wdata.value
         ~raddr:kc_raddr.value)
  in
  let vcd =
    reg
      spec
      (ram
         ~size:(2 * slots * d)
         ~waddr:ring_waddr.value
         ~wen:vc_wen.value
         ~wdata:ring_wdata.value
         ~raddr:vc_raddr.value)
  in
  let vramd =
    reg
      spec
      (ram
         ~size:vocab
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
  (* the shared multiplier: the operands are muxed by state, the product registers *)
  (* one DSP-sized product: 25 by 18, signed — the timing of the whole engine rests on
     every operand fitting these two widths *)
  let mul_a = Variable.wire ~default:(zero 25) () in
  let mul_b = Variable.wire ~default:(zero 18) () in
  (* the operand registers land the state mux chains before the DSP, thus no path runs
     memory to mux to multiplier in one cycle *)
  let opa = Variable.reg spec ~width:25 in
  let opb = Variable.reg spec ~width:18 in
  let mul_out = opa.value *+ opb.value in
  let preg48 = sresize preg.value ~width:48 in
  let acc_full = acc.value +: preg48 in
  (* the legality of one code, from the mask registers — the rules of [Sounding_state] *)
  let legal_of code =
    let p = sel_bottom code ~width:7 in
    let bit = mux p (bits_lsb sounding.value) in
    let off_ok = bit &: ~:(lov.value) &: (~:(lofv.value) |: (p >: last_off.value)) in
    let on_ok =
      ~:bit &: (scount.value <:. Token.seats) &: (~:(lov.value) |: (p <: last_on.value))
    in
    mux2 (code ==:. 0) vdd (mux2 (code ==:. 255) gnd (mux2 (msb code) on_ok off_ok))
  in
  (* the window: the count of the valid ring entries, and the slot of an age *)
  let n =
    mux2 filled.value (of_unsigned_int ~width:9 slots) (uresize cur.value ~width:9 +:. 1)
  in
  let age8 = sel_bottom age.value ~width:8 in
  let slot = cur.value -: age8 in
  let ii4 = sel_bottom ii.value ~width:4 in
  let ii6 = sel_bottom ii.value ~width:6 in
  let ii8 = sel_bottom ii.value ~width:8 in
  let oo6 = sel_bottom oo.value ~width:6 in
  let oo8 = sel_bottom oo.value ~width:8 in
  let phase = sel_bottom s.value ~width:4 in
  let bucket = select s.value ~high:7 ~low:4 in
  let rom_const at = of_unsigned_int ~width:rom_addr_bits at in
  let by_layer field =
    mux
      lyr.value
      (Array.to_list (Array.map bases.layers ~f:(fun l -> rom_const (field l))))
  in
  (* One three-phase multiply-accumulate iteration: tick 0 presents the addresses, tick 1
     registers the product, tick 2 accumulates; [last] closes the loop. [at_last] owns the
     accumulator: a writeback state reads [acc_full] and zeroes it, a two-state join keeps
     the whole sum with [acc <-- acc_full]. *)
  let mac3 ~last ~at_last ~else_next =
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
                ; if_ last (at_last @ [ acc <--. 0 ]) (else_next @ [ acc <-- acc_full ])
                ]
            ]
        ]
    ]
  in
  (* the last iteration of a two-state join keeps the whole sum instead *)
  let mac3_keep ~last ~at_last ~else_next =
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
                    (at_last @ [ acc <-- acc_full ])
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
  let enter_sample =
    [ rms_ret <--. 2; ii <--. 0; acc <--. 0; tick <--. 0; sm.set_next RmsSum ]
  in
  let enter_forward =
    [ lyr <--. 0; ii <--. 0; sub <--. 0; acc <--. 0; tick <--. 0; sm.set_next Embed ]
  in
  let qkv_shift value =
    (* from Q(12 + e) to Q12: one arithmetic shift for each tensor of each layer *)
    let variant table =
      mux lyr.value (Array.to_list (Array.map table ~f:(fun e -> sra value ~by:e)))
    in
    mux sub.value [ variant eq; variant ek; variant ev ]
  in
  let ffn1_hidden value =
    (* from Q(12 + e1) to the hidden Q, then the ReLU and the clamp of the twin *)
    let shifted =
      mux
        lyr.value
        (Array.to_list
           (Array.map e1 ~f:(fun e ->
              rescale ~from:(Fixed.Constants.y_q + e) ~target:Fixed.Constants.hid_q value)))
    in
    let relu = mux2 (shifted <+ zero 48) (zero 48) shifted in
    sresize (clamp16 relu) ~width:32
  in
  let resid_shift ~e_layer ~from value =
    let variant =
      Array.map e_layer ~f:(fun e ->
        rescale ~from:(from + e) ~target:Fixed.Constants.h_q value)
    in
    mux lyr.value (Array.to_list variant)
  in
  let numv = mux ii4 (Array.to_list (Array.map nums ~f:(fun v -> v.value))) in
  let alibi =
    (* the slope of head k is 2^-(span (k+1) / heads): a shift of the age, in Q12 *)
    mux
      hd.value
      (List.init heads ~f:(fun k ->
         let exponent = span * (k + 1) / heads in
         sll (uresize age.value ~width:32) ~by:(Fixed.Constants.y_q - exponent)))
  in
  let exp_of_nn =
    (* exp2 through the table: the integer part shifts the table entry; 16 or more is 0 *)
    let big = select nn.value ~high:21 ~low:16 <>:. 0 in
    let shift_i = select nn.value ~high:15 ~low:12 in
    let shifted = mux shift_i (List.init 16 ~f:(fun k -> srl exp2d ~by:k)) in
    mux2 big (zero 16) shifted
  in
  compile
    [ sm.switch
        [ ( Idle
          , [ when_
                i.rewind
                ([ cur <--. 0
                 ; filled <--. 0
                 ; s <--. 0
                 ; sounding <-- zero 128
                 ; scount <--. 0
                 ; lov <--. 0
                 ; lofv <--. 0
                 ; out_code <--. 255
                 ; after_forward <--. 0
                 ]
                 @ List.concat
                     (List.init Token.seats ~f:(fun k ->
                        [ seat_full.(k) <--. 0; seat_pitch.(k) <--. 0 ]))
                 @ enter_forward)
            ; when_ (i.step &: ~:(i.rewind)) enter_sample
            ] )
        ; ( Embed
          , [ rom_addr
              <-- mux
                    sub.value
                    [ rom_const bases.embed
                      +: uresize (out_code.value @: ii6) ~width:rom_addr_bits
                    ; rom_const bases.phase +: uresize (phase @: ii6) ~width:rom_addr_bits
                    ; rom_const bases.progress
                      +: uresize (bucket @: ii6) ~width:rom_addr_bits
                    ]
            ; mul_a <-- sresize romd ~width:25
            ; mul_b <-- of_signed_int ~width:18 1
            ]
            @ mac3
                ~last:(sub.value ==:. 2)
                ~at_last:
                  [ hram_wen <-- vdd
                  ; hram_waddr <-- ii6
                  ; hram_wdata
                    <-- sel_bottom
                          (rescale ~from:e_tbl ~target:Fixed.Constants.h_q acc_full)
                          ~width:32
                  ; sub <--. 0
                  ; if_
                      (ii.value ==:. d - 1)
                      [ ii <--. 0; rms_ret <--. 0; sm.set_next RmsSum ]
                      [ ii <-- ii.value +:. 1 ]
                  ]
                ~else_next:[ sub <-- sub.value +:. 1 ] )
        ; ( RmsSum
          , [ hram_raddr <-- ii6
            ; mul_a <-- sel_bottom (sra hramd ~by:4) ~width:25
            ; mul_b <-- sel_bottom (sra hramd ~by:4) ~width:18
            ]
            @ mac3
                ~last:(ii.value ==:. d - 1)
                ~at_last:
                  [ m
                    <-- sel_bottom
                          (srl acc_full ~by:6
                           +: of_unsigned_int ~width:48 Fixed.Constants.eps_q)
                          ~width:42
                  ; sq_root <--. 0
                  ; sq_rem <--. 0
                  ; sq_i <--. 21
                  ; ii <--. 0
                  ; sm.set_next RmsSqrt
                  ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( RmsSqrt
          , [ (* one bit pair of the radicand for each cycle: the restoring square root *)
              (let rem' =
                 sel_bottom (sq_rem.value @: select m.value ~high:41 ~low:40) ~width:25
               in
               let trial =
                 uresize (sq_root.value @: of_unsigned_int ~width:2 1) ~width:25
               in
               let fits = rem' >=: trial in
               let root' = sel_bottom (sq_root.value @: fits) ~width:21 in
               proc
                 [ m <-- sll m.value ~by:2
                 ; sq_i <-- sq_i.value -:. 1
                 ; sq_rem <-- mux2 fits (rem' -: trial) rem'
                 ; sq_root <-- root'
                 ; when_
                     (sq_i.value ==:. 1)
                     [ (* the completed root is the divisor of every element *)
                       div_d <-- uresize root' ~width:24
                     ; ii <--. 0
                     ; tick <--. 0
                     ; sm.set_next RmsScale
                     ]
                 ])
            ] )
        ; ( RmsScale
          , [ (* y = (x << 8) / g for each element, toward zero: the twin's division *)
              hram_raddr <-- ii6
            ; if_
                (tick.value ==:. 0)
                [ tick <--. 1 ]
                [ if_
                    (tick.value ==:. 1)
                    [ div_m
                      <-- sll
                            (uresize
                               (mux2 (hramd <+ zero 32) (negate hramd) hramd)
                               ~width:40)
                            ~by:8
                    ; div_sign <-- (hramd <+ zero 32)
                    ; div_q <--. 0
                    ; div_rem <--. 0
                    ; div_i <--. 40
                    ; tick <--. 2
                    ]
                    [ (let rem' =
                         sel_bottom (div_rem.value @: msb div_m.value) ~width:25
                       in
                       let fits = rem' >=: uresize div_d.value ~width:25 in
                       proc
                         [ div_m <-- sll div_m.value ~by:1
                         ; div_i <-- div_i.value -:. 1
                         ; div_rem
                           <-- mux2 fits (rem' -: uresize div_d.value ~width:25) rem'
                         ; div_q <-- sel_bottom (div_q.value @: fits) ~width:40
                         ])
                    ; when_
                        (div_i.value ==:. 1)
                        [ (let quotient =
                             sel_bottom
                               (div_q.value
                                @: (sel_bottom
                                      (div_rem.value @: msb div_m.value)
                                      ~width:25
                                    >=: uresize div_d.value ~width:25))
                               ~width:40
                           in
                           let signed = mux2 div_sign.value (negate quotient) quotient in
                           proc
                             [ yram_wen <-- vdd
                             ; yram_waddr <-- ii6
                             ; yram_wdata <-- clamp16 signed
                             ])
                        ; tick <--. 0
                        ; if_
                            (ii.value ==:. d - 1)
                            [ ii <--. 0
                            ; oo <--. 0
                            ; sub <--. 0
                            ; acc <--. 0
                            ; if_
                                (rms_ret.value ==:. 0)
                                [ sm.set_next Qkv ]
                                [ if_
                                    (rms_ret.value ==:. 1)
                                    [ sm.set_next Ffn1 ]
                                    [ peak <-- min32; sm.set_next Logits ]
                                ]
                            ]
                            [ ii <-- ii.value +:. 1 ]
                        ]
                    ]
                ]
            ] )
        ; ( Qkv
          , [ yram_raddr <-- ii6
            ; rom_addr
              <-- mux
                    sub.value
                    [ by_layer (fun l -> l.wq)
                      +: uresize (ii6 @: oo6) ~width:rom_addr_bits
                    ; by_layer (fun l -> l.wk)
                      +: uresize (ii6 @: oo6) ~width:rom_addr_bits
                    ; by_layer (fun l -> l.wv)
                      +: uresize (ii6 @: oo6) ~width:rom_addr_bits
                    ]
            ; mul_a <-- sresize yd ~width:25
            ; mul_b <-- sresize romd ~width:18
            ]
            @ mac3
                ~last:(ii.value ==:. d - 1)
                ~at_last:
                  [ (let value = clamp16 (qkv_shift acc_full) in
                     proc
                       [ qram_wen <-- (sub.value ==:. 0)
                       ; qram_waddr <-- oo6
                       ; qram_wdata <-- value
                       ; kc_wen <-- (sub.value ==:. 1)
                       ; vc_wen <-- (sub.value ==:. 2)
                       ; ring_waddr <-- lyr.value @: cur.value @: oo6
                       ; ring_wdata <-- value
                       ])
                  ; ii <--. 0
                  ; if_
                      (oo.value ==:. d - 1)
                      [ oo <--. 0
                      ; if_
                          (sub.value ==:. 2)
                          [ sub <--. 0
                          ; hd <--. 0
                          ; age <--. 0
                          ; peak <-- min32
                          ; den <--. 0
                          ; proc (Array.to_list (Array.map nums ~f:(fun v -> v <--. 0)))
                          ; sm.set_next Score
                          ]
                          [ sub <-- sub.value +:. 1 ]
                      ]
                      [ oo <-- oo.value +:. 1 ]
                  ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( Score
          , [ qram_raddr <-- hd.value @: ii4
            ; kc_raddr <-- lyr.value @: slot @: hd.value @: ii4
            ; mul_a <-- sresize qd ~width:25
            ; mul_b <-- sresize kcd ~width:18
            ]
            @ mac3
                ~last:(ii.value ==:. head_d - 1)
                ~at_last:
                  [ (let score = sel_bottom (sra acc_full ~by:14) ~width:32 -: alibi in
                     proc
                       [ vram_wen <-- vdd
                       ; vram_waddr <-- age8
                       ; vram_wdata <-- score
                       ; when_ (score >+ peak.value) [ peak <-- score ]
                       ])
                  ; ii <--. 0
                  ; if_
                      (age.value +:. 1 ==: n)
                      [ age <--. 0; tick <--. 0; sm.set_next Exp ]
                      [ age <-- age.value +:. 1 ]
                  ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( Exp
          , [ vram_raddr <-- age8
            ; mul_a <-- sel_bottom diff.value ~width:25
            ; mul_b <-- of_signed_int ~width:18 Fixed.Constants.log2e_q15
            ; exp2_addr <-- uresize (select nn.value ~high:11 ~low:4) ~width:8
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
                                    [ e_reg <-- exp_of_nn
                                    ; den <-- den.value +: uresize exp_of_nn ~width:24
                                    ; ii <--. 0
                                    ; tick <--. 0
                                    ; sm.set_next ExpMac
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ] )
        ; ( ExpMac
          , [ vc_raddr <-- lyr.value @: slot @: hd.value @: ii4
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
                                   <-- nums.(k).value +: sel_bottom preg.value ~width:40
                                 ]))
                        ; if_
                            (ii.value ==:. head_d - 1)
                            [ ii <--. 0
                            ; if_
                                (age.value +:. 1 ==: n)
                                [ age <--. 0; tick <--. 0; sm.set_next CtxDiv ]
                                [ age <-- age.value +:. 1; tick <--. 0; sm.set_next Exp ]
                            ]
                            [ ii <-- ii.value +:. 1 ]
                        ]
                    ]
                ]
            ] )
        ; ( CtxDiv
          , [ if_
                (tick.value ==:. 0)
                [ div_m
                  <-- sel_bottom (mux2 (numv <+ zero 40) (negate numv) numv) ~width:40
                ; div_sign <-- (numv <+ zero 40)
                ; div_d <-- den.value
                ; div_q <--. 0
                ; div_rem <--. 0
                ; div_i <--. 40
                ; tick <--. 1
                ]
                [ (let rem' = sel_bottom (div_rem.value @: msb div_m.value) ~width:25 in
                   let fits = rem' >=: uresize div_d.value ~width:25 in
                   proc
                     [ div_m <-- sll div_m.value ~by:1
                     ; div_i <-- div_i.value -:. 1
                     ; div_rem <-- mux2 fits (rem' -: uresize div_d.value ~width:25) rem'
                     ; div_q <-- sel_bottom (div_q.value @: fits) ~width:40
                     ; when_
                         (div_i.value ==:. 1)
                         [ (let quotient = sel_bottom (div_q.value @: fits) ~width:40 in
                            let signed = mux2 div_sign.value (negate quotient) quotient in
                            proc
                              [ yram_wen <-- vdd
                              ; yram_waddr <-- hd.value @: ii4
                              ; yram_wdata <-- clamp16 signed
                              ])
                         ; tick <--. 0
                         ; if_
                             (ii.value ==:. head_d - 1)
                             [ ii <--. 0
                             ; if_
                                 (hd.value ==:. heads - 1)
                                 [ hd <--. 0; oo <--. 0; acc <--. 0; sm.set_next WoMac ]
                                 [ hd <-- hd.value +:. 1
                                 ; age <--. 0
                                 ; peak <-- min32
                                 ; den <--. 0
                                 ; proc
                                     (Array.to_list
                                        (Array.map nums ~f:(fun v -> v <--. 0)))
                                 ; acc <--. 0
                                 ; sm.set_next Score
                                 ]
                             ]
                             [ ii <-- ii.value +:. 1 ]
                         ]
                     ])
                ]
            ] )
        ; ( WoMac
          , [ yram_raddr <-- ii6
            ; rom_addr
              <-- by_layer (fun l -> l.wo) +: uresize (ii6 @: oo6) ~width:rom_addr_bits
            ; mul_a <-- sresize yd ~width:25
            ; mul_b <-- sresize romd ~width:18
            ]
            @ mac3_keep
                ~last:(ii.value ==:. d - 1)
                ~at_last:[ ii <--. 0; tick <--. 0; sm.set_next WoAdd ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( WoAdd
          , [ hram_raddr <-- oo6
            ; if_
                (tick.value ==:. 0)
                [ tick <--. 1 ]
                [ tick <--. 0
                ; hram_wen <-- vdd
                ; hram_waddr <-- oo6
                ; hram_wdata
                  <-- sel_bottom
                        (sresize hramd ~width:48
                         +: resid_shift ~e_layer:eo ~from:Fixed.Constants.kv_q acc.value)
                        ~width:32
                ; acc <--. 0
                ; if_
                    (oo.value ==:. d - 1)
                    [ oo <--. 0; ii <--. 0; rms_ret <--. 1; sm.set_next RmsSum ]
                    [ oo <-- oo.value +:. 1; sm.set_next WoMac ]
                ]
            ] )
        ; ( Ffn1
          , [ yram_raddr <-- ii6
            ; rom_addr
              <-- by_layer (fun l -> l.w1) +: uresize (ii6 @: oo8) ~width:rom_addr_bits
            ; mul_a <-- sresize yd ~width:25
            ; mul_b <-- sresize romd ~width:18
            ]
            @ mac3
                ~last:(ii.value ==:. d - 1)
                ~at_last:
                  [ (let hidden = ffn1_hidden acc_full in
                     proc [ vram_wen <-- vdd; vram_waddr <-- oo8; vram_wdata <-- hidden ])
                  ; ii <--. 0
                  ; if_
                      (oo.value ==:. dff - 1)
                      [ oo <--. 0; sm.set_next Ffn2Mac ]
                      [ oo <-- oo.value +:. 1 ]
                  ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( Ffn2Mac
          , [ vram_raddr <-- ii8
            ; rom_addr
              <-- by_layer (fun l -> l.w2) +: uresize (ii8 @: oo6) ~width:rom_addr_bits
            ; mul_a <-- sresize (sel_bottom vramd ~width:16) ~width:25
            ; mul_b <-- sresize romd ~width:18
            ]
            @ mac3_keep
                ~last:(ii.value ==:. dff - 1)
                ~at_last:[ ii <--. 0; tick <--. 0; sm.set_next Ffn2Add ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( Ffn2Add
          , [ hram_raddr <-- oo6
            ; if_
                (tick.value ==:. 0)
                [ tick <--. 1 ]
                [ tick <--. 0
                ; hram_wen <-- vdd
                ; hram_waddr <-- oo6
                ; hram_wdata
                  <-- sel_bottom
                        (sresize hramd ~width:48
                         +: resid_shift ~e_layer:e2 ~from:Fixed.Constants.hid_q acc.value
                        )
                        ~width:32
                ; acc <--. 0
                ; if_
                    (oo.value ==:. d - 1)
                    [ oo <--. 0
                    ; ii <--. 0
                    ; if_
                        (lyr.value ==:. 0)
                        [ lyr <--. 1; rms_ret <--. 0; sm.set_next RmsSum ]
                        [ lyr <--. 0; sm.set_next ForwardDone ]
                    ]
                    [ oo <-- oo.value +:. 1; sm.set_next Ffn2Mac ]
                ]
            ] )
        ; ( Logits
          , [ yram_raddr <-- ii6
            ; rom_addr
              <-- rom_const bases.embed +: uresize (oo8 @: ii6) ~width:rom_addr_bits
            ; mul_a <-- sresize yd ~width:25
            ; mul_b <-- sresize romd ~width:18
            ]
            @ mac3
                ~last:(ii.value ==:. d - 1)
                ~at_last:
                  [ (let logit = sel_bottom (sra acc_full ~by:e_tbl) ~width:32 in
                     proc
                       [ vram_wen <-- vdd
                       ; vram_waddr <-- oo8
                       ; vram_wdata <-- logit
                       ; when_ (legal_of oo8 &: (logit >+ peak.value)) [ peak <-- logit ]
                       ])
                  ; ii <--. 0
                  ; if_
                      (oo.value ==:. vocab - 1)
                      [ oo <--. 0; total <--. 0; tick <--. 0; sm.set_next Weights ]
                      [ oo <-- oo.value +:. 1 ]
                  ]
                ~else_next:[ ii <-- ii.value +:. 1 ] )
        ; ( Weights
          , [ vram_raddr <-- oo8
            ; mul_a <-- sel_bottom diff.value ~width:25
            ; mul_b <-- of_signed_int ~width:18 temper_q14
            ; exp2_addr <-- uresize (select nn.value ~high:11 ~low:4) ~width:8
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
                                        (negate (sra preg.value ~by:14))
                                        ~width:22
                                ; tick <--. 5
                                ]
                                [ if_
                                    (tick.value ==:. 5)
                                    [ tick <--. 6 ]
                                    [ (let e = exp_of_nn in
                                       let keep =
                                         legal_of oo8
                                         &: (e >=: of_unsigned_int ~width:16 min_weight)
                                       in
                                       let w = mux2 keep e (zero 16) in
                                       proc
                                         [ vram_wen <-- vdd
                                         ; vram_waddr <-- oo8
                                         ; vram_wdata <-- uresize w ~width:32
                                         ; total <-- total.value +: uresize w ~width:24
                                         ])
                                    ; tick <--. 0
                                    ; if_
                                        (oo.value ==:. vocab - 1)
                                        [ oo <--. 0; sm.set_next Draw ]
                                        [ oo <-- oo.value +:. 1 ]
                                    ]
                                ]
                            ]
                        ]
                    ]
                ]
            ] )
        ; ( Draw
          , [ if_
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
                        [ u24 <-- sel_bottom u24.value ~width:16 @: prng_byte
                        ; tick <--. 0
                        ; sm.set_next Thresh
                        ]
                    ]
                ]
            ] )
        ; ( Thresh
          , [ (* (u24 * total) >> 24 in two DSP passes: the high twelve bits of the total,
                 then the low twelve — the same integer as one wide multiply *)
              mul_a <-- uresize u24.value ~width:25
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
                            [ thr
                              <-- sel_bottom
                                    (srl
                                       (sll (uresize thi.value ~width:56) ~by:12
                                        +: uresize preg.value ~width:56)
                                       ~by:24)
                                    ~width:24
                            ; cum <--. 0
                            ; oo <--. 0
                            ; found <--. 0
                            ; tick <--. 0
                            ; sm.set_next Pick
                            ]
                        ]
                    ]
                ]
            ] )
        ; ( Pick
          , [ vram_raddr <-- oo8
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
                         [ when_
                             passes
                             [ found <-- vdd; chosen <-- oo8; pos <-- (w <>:. 0) ]
                         ; when_
                             (oo.value ==:. vocab - 1)
                             [ chosen <--. 255; pos <-- (w <>:. 0) ]
                         ]
                     ])
                ; tick <--. 0
                ; if_
                    (oo.value ==:. vocab - 1)
                    [ oo <--. 0; sm.set_next Decide ]
                    [ oo <-- oo.value +:. 1 ]
                ]
            ] )
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
          , [ (let p = sel_bottom out_code.value ~width:7 in
               let hot = binary_to_onehot p in
               proc
                 [ if_
                     (out_code.value ==:. 0)
                     [ lov <--. 0; lofv <--. 0; s <-- s.value +:. 1 ]
                     [ when_
                         (out_code.value <>:. 255)
                         [ if_
                             (msb out_code.value)
                             [ sounding <-- (sounding.value |: hot)
                             ; scount <-- scount.value +:. 1
                             ; last_on <-- p
                             ; lov <--. 1
                             ]
                             [ sounding <-- (sounding.value &: ~:hot)
                             ; scount <-- scount.value -:. 1
                             ; last_off <-- p
                             ; lofv <--. 1
                             ]
                         ]
                     ]
                 ])
            ; cur <-- cur.value +:. 1
            ; filled <-- (filled.value |: (cur.value ==:. 255))
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

(* The stream comparison: the circuit against the twin, event for event, on drawn weights.
   Every integer of the engine crosses this test — the embedding, the ring, the softmax,
   the sampler and the seats — because one wrong bit moves a draw and the streams part.
   The budget catches a stall. *)
let%expect_test "the source agrees with the twin, event for event" =
  let model = Fixed.Model.For_test.init Transformer.Config.baseline ~seed:11 in
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
  (* the source is still in Idle in the cycle that takes a strobe, thus each wait must
     cycle once before it reads [idle] — the rule of the Voss harness *)
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
  let twin = Fixed.Engine.init model ~seed in
  let (_ : Fixed.Engine.t), reference_reversed =
    List.fold (List.range 0 steps) ~init:(twin, []) ~f:(fun (twin, acc) (_ : int) ->
      let twin, events = Fixed.Engine.next_step twin in
      ( twin
      , List.map events ~f:(fun { Fixed.Engine.voice; pitch; on } -> voice, pitch, on)
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
