(* The draw of the chain — see sampler.mli for the contract. Every builder here is the
   text the two frozen sources carried under two module names, moved once and read twice.

   NOTHING HERE CREATES A REGISTER OR A WIRE. An era declares its own and passes them in,
   because Hardcaml names an unnamed signal by the order of its creation: a declaration
   moved into this module would create its signal at another point of the elaboration and
   rename every signal after it. The only signals these functions make are the constants
   and the combinational terms the era's own text already made, in the same order. *)
open Base
open Hardcaml
open Signal
open Always

type t =
  { classes : int
  ; temper : Quantized.Constants.scale
  ; min_weight : int
  ; tick : Variable.t
  ; oo : Variable.t
  ; u24 : Variable.t
  ; total : Variable.t
  ; thi : Variable.t
  ; thr : Variable.t
  ; cum : Variable.t
  ; found : Variable.t
  ; diff : Variable.t
  ; nn : Variable.t
  ; vram_raddr : Variable.t
  ; vram_wen : Variable.t
  ; vram_waddr : Variable.t
  ; vram_wdata : Variable.t
  ; vramd : Signal.t
  ; below_peak : Signal.t
  ; mul_a : Variable.t
  ; mul_b : Variable.t
  ; product : Signal.t
  ; prng_step : Variable.t
  ; prng_byte : Signal.t
  ; exp2_e : Signal.t
  ; weight_addr : Signal.t
  ; oo_class : Signal.t
  ; write_drawn : Signal.t -> Always.t
  }

let exp_weight_chain t ~addr ~(scale : Quantized.Constants.scale) ~land_ ~advance =
  let at_addr = uresize addr ~width:(width t.vram_raddr.value) in
  [ t.vram_raddr <-- at_addr
  ; t.mul_a <-- sel_bottom t.diff.value ~width:25
  ; t.mul_b <-- of_signed_int ~width:18 scale.q_value
  ; Program.chain_over
      t.tick
      [ []
      ; [ t.diff <-- t.below_peak ]
      ; []
      ; []
      ; [ t.nn <-- sel_bottom (negate (sra t.product ~by:scale.q)) ~width:22 ]
      ; []
      ; (* [Exp2.latency] cycles of it, and the magnitude stands from the tick after it is
           written: the weight is whole here and the landing below reads it *)
        []
      ; [ t.vram_wen <-- vdd; t.vram_waddr <-- at_addr ]
        @ land_
        @ [ t.tick <--. 0 ]
        @ advance
      ]
  ]
;;

let tempered_weights t ~finish =
  let entry = [ t.oo <--. 0; t.tick <--. 0; t.total <--. 0 ] in
  (* the min-p refusal: a weight under the share of the peak weighs nothing *)
  let keep = t.exp2_e >=: of_unsigned_int ~width:16 t.min_weight in
  let w = mux2 keep t.exp2_e (zero 16) in
  let body =
    exp_weight_chain
      t
      ~addr:t.weight_addr
      ~scale:t.temper
      ~land_:
        [ t.vram_wdata <-- uresize w ~width:32
        ; t.total <-- t.total.value +: uresize w ~width:24
        ]
      ~advance:
        [ if_ (t.oo.value ==:. t.classes - 1) finish [ t.oo <-- t.oo.value +:. 1 ] ]
  in
  entry, body
;;

let uniform_word t ~finish =
  let entry = [ t.tick <--. 0 ] in
  (* the shift is written out at each of the three ticks and not named once: a name would
     be ONE concatenation where the era's text made three, which is a netlist of its own *)
  let shifted_in () = sel_bottom t.u24.value ~width:16 @: t.prng_byte in
  let body =
    [ Program.chain_over
        t.tick
        [ [ t.prng_step <-- vdd ]
        ; [ t.prng_step <-- vdd; t.u24 <-- shifted_in () ]
        ; [ t.prng_step <-- vdd; t.u24 <-- shifted_in () ]
        ; [ t.u24 <-- shifted_in () ] @ finish
        ]
    ]
  in
  entry, body
;;

let threshold t ~finish =
  let entry = [ t.tick <--. 0 ] in
  let body =
    [ t.mul_a <-- uresize t.u24.value ~width:25
    ; t.mul_b
      <-- uresize
            (mux2
               (t.tick.value <:. 2)
               (select t.total.value ~high:23 ~low:12)
               (sel_bottom t.total.value ~width:12))
            ~width:18
    ; Program.chain_over
        t.tick
        [ []
        ; []
        ; [ t.thi <-- t.product ]
        ; []
        ; [ t.thr
            <-- sel_bottom
                  (srl
                     (sll (uresize t.thi.value ~width:56) ~by:12
                      +: uresize t.product ~width:56)
                     ~by:24)
                  ~width:24
          ]
          @ finish
        ]
    ]
  in
  entry, body
;;

let pick t ~finish =
  let entry = [ t.oo <--. 0; t.tick <--. 0; t.cum <--. 0; t.found <--. 0 ] in
  let last_class = of_unsigned_int ~width:(width t.oo_class) (t.classes - 1) in
  let body =
    [ t.vram_raddr <-- uresize t.weight_addr ~width:(width t.vram_raddr.value)
    ; Program.chain_over
        t.tick
        [ []
        ; [ (let w = sel_bottom t.vramd ~width:24 in
             let cum_next = t.cum.value +: uresize w ~width:25 in
             let passes = cum_next >: uresize t.thr.value ~width:25 in
             proc
               [ t.cum <-- cum_next
               ; when_
                   ~:(t.found.value)
                   [ when_ passes [ t.found <-- vdd; t.write_drawn t.oo_class ]
                   ; when_ (t.oo.value ==:. t.classes - 1) [ t.write_drawn last_class ]
                   ]
               ])
          ; t.tick <--. 0
          ; if_ (t.oo.value ==:. t.classes - 1) finish [ t.oo <-- t.oo.value +:. 1 ]
          ]
        ]
    ]
  in
  entry, body
;;

let drawn_at_seat ~(seat : Variable.t) ~drawn =
  mux seat.value (List.map (Array.to_list drawn) ~f:(fun c -> c.Variable.value))
;;

let write_drawn ~(seat : Variable.t) ~seats value =
  switch
    seat.value
    (List.init (Array.length seats) ~f:(fun s ->
       of_unsigned_int ~width:(width seat.value) s, [ seats.(s) <-- value ]))
;;

let frame_word ~code_of_class ~drawn =
  concat_msb
    (List.rev_map (Array.to_list drawn) ~f:(fun c -> code_of_class c.Variable.value))
;;
