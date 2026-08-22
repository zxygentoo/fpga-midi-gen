(* The restoring divider, with the magnitude inside the walk — see divider.mli for the
   contract and for why the magnitude moved there. The start cycle only latches the raw
   operands; the first busy cycle takes the magnitude reg-to-reg; the sign lands at the
   output, thus the quotient truncates toward zero at every sign — the one division rule
   of both circuits. *)

open Base
open Hardcaml
open Signal

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

(* one cycle for the magnitude, then one for each bit of the quotient *)
let busy_cycles = 41

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let m = Variable.reg spec ~width:40 in
  let d = Variable.reg spec ~width:24 in
  let q = Variable.reg spec ~width:40 in
  let r = Variable.reg spec ~width:25 in
  let n = Variable.reg spec ~width:(Int.ceil_log2 (busy_cycles + 1)) in
  let sign = Variable.reg spec ~width:1 in
  let busy = Variable.reg spec ~width:1 in
  compile
    [ if_
        i.start
        [ (* the raw numerator, sign and all: no carry chain stands between the caller's
             operand mux and this register *)
          m <-- i.numerator
        ; sign <-- msb i.numerator
        ; d <-- i.denominator
        ; q <--. 0
        ; r <--. 0
        ; n <--. busy_cycles
        ; busy <-- vdd
        ]
        [ when_
            busy.value
            [ if_
                (n.value ==:. busy_cycles)
                [ m <-- mux2 sign.value (negate m.value) m.value; n <-- n.value -:. 1 ]
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
        ]
    ];
  { O.quotient = mux2 sign.value (negate q.value) q.value; busy = busy.value }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

(* One simulator, and one walk on it: [start], then wait on [busy] and read the quotient
   in the cycle the wait releases — the contract's own reading order. Both gates below
   drive the unit through this. *)
let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  fun n d ->
    inp.numerator := Bits.of_signed_int ~width:40 n;
    inp.denominator := Bits.of_unsigned_int ~width:24 d;
    inp.start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      Cyclesim.cycle sim
    done;
    Bits.to_signed_int !(out.quotient)
;;

let%expect_test "the divider is the reference division, toward zero" =
  let divide = harness () in
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

(* the widths of the contract: the numerator is 40 bits signed and the denominator 24 bits
   unsigned, thus these are the corners a walk must survive *)
let numerator_low = -(1 lsl 39)
let numerator_high = (1 lsl 39) - 1
let denominator_high = (1 lsl 24) - 1

let%expect_test "the fuzz: every walk truncates toward zero, as the reference does" =
  let divide = harness () in
  (* The seed is fixed, thus the gate is the same gate on every machine. The report is a
     verdict and never the drawn values: a generator that changes under us moves the cases
     and not this output, and a disagreement prints itself. *)
  let state = Random.State.make [| 20260819 |] in
  let drawn (_ : int) =
    ( Random.State.int_incl state numerator_low numerator_high
    , Random.State.int_incl state 1 denominator_high )
  in
  (* the corners the contract names, and the two the sign rule turns on: a magnitude under
     the denominator gives zero, and the low numerator is its own negation in 40 bits *)
  let edges =
    [ 0, 1
    ; 0, denominator_high
    ; 1, 1
    ; -1, 1
    ; numerator_high, 1
    ; numerator_low, 1
    ; numerator_high, denominator_high
    ; numerator_low, denominator_high
    ; 5, 7
    ; -5, 7
    ; numerator_low, 2
    ]
  in
  let cases = edges @ List.map (List.range 0 1000) ~f:drawn in
  let disagreement (n, d) =
    let got = divide n d in
    if got = n / d then None else Some (n, d, got)
  in
  (match List.filter_map cases ~f:disagreement with
   | [] ->
     Stdio.printf
       "%d walks over the whole range: every quotient is the reference quotient\n"
       (List.length cases)
   | (n, d, got) :: (_ : (int * int * int) list) ->
     Stdio.printf "%d / %d gave %d, the reference gives %d\n" n d got (n / d));
  [%expect
    {| 1011 walks over the whole range: every quotient is the reference quotient |}]
;;

let%expect_test "the waveform of one walk: the strobe, the busy window, the landing" =
  (* The shape of the contract, 1000 / 7. [start] is one cycle; [busy] rises in the cycle
     after it and falls [busy_cycles] later — one cycle for the magnitude and forty for
     the bits. The quotient shifts a bit in on every cycle of the walk, thus it is
     meaningless while [busy] stands — the picture shows it flat at 0 while the high bits
     are still zero, then moving each cycle as the low bits land. It is whole in the cycle
     [busy] reads 0, and the line under the picture reads it there. Forty-one cycles leave
     one column each, thus the value is a shape here and a number below. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.numerator := Bits.of_signed_int ~width:40 1000;
  inp.denominator := Bits.of_unsigned_int ~width:24 7;
  inp.start := Bits.vdd;
  Cyclesim.cycle sim;
  inp.start := Bits.gnd;
  let cycles_waited = ref 0 in
  while Bits.to_bool !(out.busy) do
    Cyclesim.cycle sim;
    Int.incr cycles_waited
  done;
  let landed = Bits.to_signed_int !(out.quotient) in
  Cyclesim.cycle sim;
  Cyclesim.cycle sim;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Hex)
          [ "clock"; "start"; "busy"; "quotient" ]
      ]
    ~show_digest:false
    ~wave_width:(-1)
    waves;
  Stdio.printf
    "the wait released after %d cycles and the quotient read %d\n"
    !cycles_waited
    landed;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥│
    │               ││╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨│
    │start          ││─┐                                                 │
    │               ││ └──────────────────────────────────────────       │
    │busy           ││ ┌────────────────────────────────────────┐        │
    │               ││─┘                                        └─       │
    │               ││───────────────────────────────────┬┬┬┬┬┬┬┬─       │
    │quotient       ││ 0000000000                        ││││││││.       │
    │               ││───────────────────────────────────┴┴┴┴┴┴┴┴─       │
    └───────────────┘└───────────────────────────────────────────────────┘
    the wait released after 41 cycles and the quotient read 142
    |}]
;;
