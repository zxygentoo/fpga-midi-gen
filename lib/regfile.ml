open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; write_enable : 'a
    ; address : 'a [@bits 4]
    ; write_data : 'a [@bits 8]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { read_data : 'a [@bits 8] } [@@deriving hardcaml]
end

(* The power-on values of the register file, from the ABI constants. *)
let defaults =
  let d = Array.make Abi.Reg.Ctl.size 0 in
  let set_byte addr value = d.(addr - Abi.Reg.Ctl.base) <- value land 0xff in
  let set_bytes addr n value =
    for i = 0 to n - 1 do
      set_byte (addr + i) (value lsr (8 * i))
    done
  in
  set_byte Abi.Reg.Ctl.channel Abi.Default.channel;
  set_bytes Abi.Reg.Ctl.step_ms 2 Abi.Default.step_ms;
  set_bytes Abi.Reg.Ctl.gate_ms 2 Abi.Default.gate_ms;
  set_byte Abi.Reg.Ctl.velocity Abi.Default.velocity;
  set_bytes Abi.Reg.Ctl.seed 4 Abi.Default.seed;
  d
;;

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let cells =
    Array.init Abi.Reg.Ctl.size (fun k ->
      (* the default holds at power-on and at clear *)
      Variable.reg
        ~initialize_to:(Bits.of_unsigned_int ~width:8 defaults.(k))
        ~clear_to:(of_unsigned_int ~width:8 defaults.(k))
        spec
        ~width:8)
  in
  compile
    [ when_
        i.write_enable
        [ proc
            (List.init Abi.Reg.Ctl.size (fun k ->
               when_ (i.address ==:. k) [ cells.(k) <-- i.write_data ]))
        ]
    ];
  { O.read_data = mux i.address (Array.to_list (Array.map Variable.value cells)) }
;;

let%expect_test "defaults, write and read" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inp.clear := Bits.gnd;
  let dump () =
    for k = 0 to Abi.Reg.Ctl.size - 1 do
      inp.address := Bits.of_unsigned_int ~width:4 k;
      Cyclesim.cycle sim;
      Printf.printf "%02x " (Bits.to_int_trunc !(out.read_data))
    done;
    print_newline ()
  in
  (* all cells at power-on *)
  dump ();
  [%expect {| 00 00 00 00 00 2a 00 00 00 64 7d 00 fa 00 02 00 |}];
  (* write one cell, the others hold *)
  inp.write_enable := Bits.vdd;
  inp.address := Bits.of_unsigned_int ~width:4 (Abi.Reg.Ctl.velocity - Abi.Reg.Ctl.base);
  inp.write_data := Bits.of_unsigned_int ~width:8 0x30;
  Cyclesim.cycle sim;
  inp.write_enable := Bits.gnd;
  dump ();
  [%expect {| 00 00 00 00 00 2a 00 00 00 30 7d 00 fa 00 02 00 |}];
  (* during the write cycle the read still shows the old value *)
  let out_before = Cyclesim.outputs ~clock_edge:Before sim in
  inp.write_enable := Bits.vdd;
  inp.address := Bits.of_unsigned_int ~width:4 (Abi.Reg.Ctl.velocity - Abi.Reg.Ctl.base);
  inp.write_data := Bits.of_unsigned_int ~width:8 0x2a;
  Cyclesim.cycle sim;
  inp.write_enable := Bits.gnd;
  Printf.printf
    "during %02x after %02x\n"
    (Bits.to_int_trunc !(out_before.read_data))
    (Bits.to_int_trunc !(out.read_data));
  [%expect {| during 30 after 2a |}]
;;
