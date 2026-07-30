open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; data : 'a [@bits 8]
    ; valid : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { serial : 'a
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

let create ~clocks_per_bit (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  (* start bit, 8 data bits, stop bit; the lsb goes out first *)
  let frame = Variable.reg spec ~width:10 in
  let bits_left = Variable.reg spec ~width:4 in
  let bit_timer = Variable.reg spec ~width:(address_bits_for clocks_per_bit) in
  let busy = bits_left.value <>:. 0 in
  let bit_end = bit_timer.value ==:. clocks_per_bit - 1 in
  compile
    [ when_
        busy
        [ if_
            bit_end
            [ bit_timer <--. 0
            ; bits_left <-- bits_left.value -:. 1
            ; frame <-- concat_msb [ vdd; select frame.value ~high:9 ~low:1 ]
            ]
            [ bit_timer <-- bit_timer.value +:. 1 ]
        ]
    ; when_
        (i.valid &: ~:busy)
        [ frame <-- concat_msb [ vdd; i.data; gnd ]; bits_left <--. 10; bit_timer <--. 0 ]
    ];
  { O.serial = mux2 busy (lsb frame.value) vdd; busy }
;;

let%expect_test "one frame at 4 clocks per bit" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_bit:4) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.data := Bits.of_unsigned_int ~width:8 0x55;
  inp.valid := Bits.vdd;
  Cyclesim.cycle sim;
  inp.valid := Bits.gnd;
  (* 10 bits of 4 cycles each, then 4 cycles of idle *)
  let wave = Buffer.create 44 in
  for _ = 1 to 44 do
    Buffer.add_string wave (if Bits.to_bool !(out.serial) then "1" else "0");
    Cyclesim.cycle sim
  done;
  Printf.printf "serial  %s\n" (Buffer.contents wave);
  Printf.printf "busy %b\n" (Bits.to_bool !(out.busy));
  [%expect
    {|
    serial  00001111000011110000111100001111000011111111
    busy false
    |}]
;;

let%expect_test "the waveform of one frame" =
  (* the byte is a5. Between the start and the stop bit the line shows 1 0 1 0 0 1 0 1,
     the lsb first. One character is one clock cycle, thus a bit is 4 characters wide. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_bit:4) in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  inp.data := Bits.of_unsigned_int ~width:8 0xa5;
  inp.valid := Bits.vdd;
  Cyclesim.cycle sim;
  inp.valid := Bits.gnd;
  Cyclesim.cycle ~n:44 sim;
  let rules =
    let signal name =
      Hardcaml_waveterm.Display_rule.port_name_is
        name
        ~wave_format:Wave_format.(Bit_or Hex)
    in
    [ signal "data"; signal "valid"; signal "serial"; signal "busy" ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:(-1)
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │               ││─────────────────────────────────────────────      │
    │data           ││ A5                                                │
    │               ││─────────────────────────────────────────────      │
    │valid          ││─┐                                                 │
    │               ││ └───────────────────────────────────────────      │
    │serial         ││─┐   ┌───┐   ┌───┐       ┌───┐   ┌───────────      │
    │               ││ └───┘   └───┘   └───────┘   └───┘                 │
    │busy           ││ ┌───────────────────────────────────────┐         │
    │               ││─┘                                       └───      │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
