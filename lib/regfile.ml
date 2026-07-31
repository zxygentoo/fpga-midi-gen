open Base
open Hardcaml
open Signal

module Make (Config : sig
    val size : int
  end) =
struct
  let address_bits = address_bits_for Config.size

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; write_enable : 'a
      ; address : 'a [@bits address_bits]
      ; write_data : 'a [@bits 8]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t = { read_data : 'a [@bits 8] } [@@deriving hardcaml]
  end

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    let open Always in
    let cells = Array.init Config.size ~f:(fun _ -> Variable.reg spec ~width:8) in
    compile
      [ when_
          i.write_enable
          [ proc
              (List.init Config.size ~f:(fun k ->
                 when_ (i.address ==:. k) [ cells.(k) <-- i.write_data ]))
          ]
      ];
    { O.read_data = mux i.address (Array.to_list (Array.map cells ~f:Variable.value)) }
  ;;
end

let%expect_test "zeros, write and read" =
  let module R =
    Make (struct
      let size = 16
    end)
  in
  let module Sim = Cyclesim.With_interface (R.I) (R.O) in
  let sim = Sim.create R.create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inp.clear := Bits.gnd;
  let dump () =
    for k = 0 to 15 do
      inp.address := Bits.of_unsigned_int ~width:4 k;
      Cyclesim.cycle sim;
      Stdio.printf "%02x " (Bits.to_int_trunc !(out.read_data))
    done;
    Stdio.printf "\n"
  in
  (* all cells are 0 at power-on *)
  dump ();
  [%expect {| 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 |}];
  (* write one cell, the others hold *)
  inp.write_enable := Bits.vdd;
  inp.address := Bits.of_unsigned_int ~width:4 9;
  inp.write_data := Bits.of_unsigned_int ~width:8 0x30;
  Cyclesim.cycle sim;
  inp.write_enable := Bits.gnd;
  dump ();
  [%expect {| 00 00 00 00 00 00 00 00 00 30 00 00 00 00 00 00 |}];
  (* during the write cycle the read still shows the old value *)
  let out_before = Cyclesim.outputs ~clock_edge:Before sim in
  inp.write_enable := Bits.vdd;
  inp.address := Bits.of_unsigned_int ~width:4 9;
  inp.write_data := Bits.of_unsigned_int ~width:8 0x2a;
  Cyclesim.cycle sim;
  inp.write_enable := Bits.gnd;
  Stdio.printf
    "during %02x after %02x\n"
    (Bits.to_int_trunc !(out_before.read_data))
    (Bits.to_int_trunc !(out.read_data));
  [%expect {| during 30 after 2a |}]
;;

let%expect_test "the waveform of the port timing" =
  (* the write is registered: during the write cycle [read_data] shows the old value, and
     the new value appears in the next cycle. The read is combinational: [read_data] moves
     in the same cycle as [address]. *)
  let module R =
    Make (struct
      let size = 4
    end)
  in
  let module Sim = Cyclesim.With_interface (R.I) (R.O) in
  let sim = Sim.create R.create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  let address k = inp.address := Bits.of_unsigned_int ~width:2 k in
  address 2;
  Cyclesim.cycle sim;
  (* the write cycle *)
  inp.write_enable := Bits.vdd;
  inp.write_data := Bits.of_unsigned_int ~width:8 0xaa;
  Cyclesim.cycle sim;
  inp.write_enable := Bits.gnd;
  Cyclesim.cycle sim;
  (* the combinational read *)
  address 0;
  Cyclesim.cycle sim;
  address 2;
  Cyclesim.cycle sim;
  Cyclesim.cycle sim;
  let display_rules =
    List.map
      ~f:(fun name ->
        Hardcaml_waveterm.Display_rule.port_name_is
          name
          ~wave_format:Wave_format.(Bit_or Hex))
      [ "clock"; "write_enable"; "address"; "write_data"; "read_data" ]
  in
  Hardcaml_waveterm.Waveform.expect ~display_rules ~show_digest:false ~wave_width:2 waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──│
    │               ││   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  │
    │write_enable   ││      ┌─────┐                                      │
    │               ││──────┘     └───────────────────────               │
    │               ││──────────────────┬─────┬───────────               │
    │address        ││ 2                │0    │2                         │
    │               ││──────────────────┴─────┴───────────               │
    │               ││──────┬─────────────────────────────               │
    │write_data     ││ 00   │AA                                          │
    │               ││──────┴─────────────────────────────               │
    │               ││────────────┬─────┬─────┬───────────               │
    │read_data      ││ 00         │AA   │00   │AA                        │
    │               ││────────────┴─────┴─────┴───────────               │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the address width follows the size" =
  let module R =
    Make (struct
      let size = 4
    end)
  in
  let module Sim = Cyclesim.With_interface (R.I) (R.O) in
  let sim = Sim.create R.create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.write_enable := Bits.vdd;
  inp.address := Bits.of_unsigned_int ~width:2 2;
  inp.write_data := Bits.of_unsigned_int ~width:8 0x5a;
  Cyclesim.cycle sim;
  inp.write_enable := Bits.gnd;
  for k = 0 to 3 do
    inp.address := Bits.of_unsigned_int ~width:2 k;
    Cyclesim.cycle sim;
    Stdio.printf "%02x " (Bits.to_int_trunc !(out.read_data))
  done;
  Stdio.printf "\n";
  [%expect {| 00 00 5a 00 |}]
;;
