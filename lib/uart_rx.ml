open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; serial : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { data : 'a [@bits 8]
    ; valid : 'a
    }
  [@@deriving hardcaml]
end

let create ~clocks_per_bit (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  (* the two-flop synchronizer; the shadow hides the raw pin from the code below *)
  let serial = reg spec (reg spec i.serial) in
  let timer_width = address_bits_for clocks_per_bit in
  let open Always in
  let busy = Variable.reg spec ~width:1 in
  let bit_timer = Variable.reg spec ~width:timer_width in
  (* 0: the start bit; 1 to 8: the data bits; 9: the stop bit *)
  let bit_index = Variable.reg spec ~width:4 in
  let byte = Variable.reg spec ~width:8 in
  let valid = Variable.wire ~default:gnd () in
  (* the name puts the frame envelope into the waveform test *)
  let _ = busy.value -- "busy" in
  (* the first sample point is the center of the start bit *)
  let sample_point =
    mux2
      (bit_index.value ==:. 0)
      (of_unsigned_int ~width:timer_width ((clocks_per_bit / 2) - 1))
      (of_unsigned_int ~width:timer_width (clocks_per_bit - 1))
  in
  compile
    [ if_
        busy.value
        [ if_
            (bit_timer.value ==: sample_point)
            [ bit_timer <--. 0
            ; if_
                (bit_index.value ==:. 0)
                [ if_ serial [ busy <-- gnd ] [ bit_index <--. 1 ] ]
                [ if_
                    (bit_index.value ==:. 9)
                    [ busy <-- gnd; when_ serial [ valid <-- vdd ] ]
                    [ byte <-- concat_msb [ serial; select byte.value ~high:7 ~low:1 ]
                    ; bit_index <-- bit_index.value +:. 1
                    ]
                ]
            ]
            [ bit_timer <-- bit_timer.value +:. 1 ]
        ]
        [ when_ ~:serial [ busy <-- vdd; bit_timer <--. 0; bit_index <--. 0 ] ]
    ];
  { O.data = byte.value; valid = valid.value }
;;

let%expect_test "frames at 4 clocks per bit" =
  let clocks_per_bit = 4 in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_bit) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.serial := Bits.vdd;
  Cyclesim.cycle ~n:8 sim;
  let send_frame ~stop byte =
    let level b = inp.serial := if b then Bits.vdd else Bits.gnd in
    let bit b =
      level b;
      for _ = 1 to clocks_per_bit do
        Cyclesim.cycle sim;
        if Bits.to_bool !(out.valid)
        then Printf.printf "byte %02x\n" (Bits.to_int_trunc !(out.data))
      done
    in
    bit false;
    for i = 0 to 7 do
      bit ((byte lsr i) land 1 = 1)
    done;
    bit stop;
    (* idle so the synchronizer and the state settle *)
    level true;
    for _ = 1 to 8 do
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.valid)
      then Printf.printf "byte %02x\n" (Bits.to_int_trunc !(out.data))
    done
  in
  send_frame ~stop:true 0xa3;
  send_frame ~stop:true 0x00;
  [%expect {|
    byte a3
    byte 00
    |}];
  (* a frame with a bad stop bit is silent *)
  send_frame ~stop:false 0x5a;
  [%expect {| |}]
;;

let%expect_test "the waveform of one frame" =
  (* the line carries a5 at 4 clocks per bit, the lsb first. [data] assembles from the msb
     side as each sampled bit shifts in, and [valid] strobes at the center of the stop
     bit. One character is one clock cycle. The [busy] blip at the left edge is the
     start-bit check at work: the synchronizer wakes at 0, the machine starts, samples the
     center, sees the idle line and aborts. *)
  let clocks_per_bit = 4 in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all (create ~clocks_per_bit) in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  let bit b =
    inp.serial := if b then Bits.vdd else Bits.gnd;
    Cyclesim.cycle ~n:clocks_per_bit sim
  in
  inp.serial := Bits.vdd;
  Cyclesim.cycle ~n:4 sim;
  bit false;
  List.iter bit (List.init 8 (fun k -> (0xa5 lsr k) land 1 = 1));
  bit true;
  Cyclesim.cycle ~n:4 sim;
  let rules =
    let signal name =
      Hardcaml_waveterm.Display_rule.port_name_is
        name
        ~wave_format:Wave_format.(Bit_or Hex)
    in
    [ signal "serial"; signal "busy"; signal "data"; signal "valid" ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:(-1)
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │serial         ││────┐   ┌───┐   ┌───┐       ┌───┐   ┌───────────   │
    │               ││    └───┘   └───┘   └───────┘   └───┘              │
    │busy           ││ ┌─┐   ┌─────────────────────────────────────┐     │
    │               ││─┘ └───┘                                     └──   │
    │               ││─────────────┬───┬───┬───┬───┬───┬───┬───┬──────   │
    │data           ││ 00          │80 │40 │A0 │50 │28 │94 │4A │A5       │
    │               ││─────────────┴───┴───┴───┴───┴───┴───┴───┴──────   │
    │valid          ││                                            ┌┐     │
    │               ││────────────────────────────────────────────┘└──   │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
