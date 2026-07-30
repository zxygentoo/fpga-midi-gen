open Hardcaml
open Signal

type t =
  { rd_data : Signal.t
  ; cells : Signal.t
  }

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

let create ~clock ~clear ~wr_en ~wr_idx ~wr_data ~rd_idx =
  let spec = Reg_spec.create ~clock ~clear () in
  let open Always in
  let cells =
    Array.init Abi.Reg.Ctl.size (fun i ->
      Variable.reg ~clear_to:(of_unsigned_int ~width:8 defaults.(i)) spec ~width:8)
  in
  compile
    [ when_
        wr_en
        [ proc
            (List.init Abi.Reg.Ctl.size (fun k ->
               when_ (wr_idx ==:. k) [ cells.(k) <-- wr_data ]))
        ]
    ];
  { rd_data = mux rd_idx (Array.to_list (Array.map Variable.value cells))
  ; cells = concat_lsb (Array.to_list (Array.map Variable.value cells))
  }
;;

let%expect_test "defaults, write and read" =
  let circuit =
    let t =
      create
        ~clock:(input "clock" 1)
        ~clear:(input "clear" 1)
        ~wr_en:(input "wr_en" 1)
        ~wr_idx:(input "wr_idx" 4)
        ~wr_data:(input "wr_data" 8)
        ~rd_idx:(input "rd_idx" 4)
    in
    Circuit.create_exn ~name:"regfile" [ output "rd_data" t.rd_data ]
  in
  let sim = Cyclesim.create circuit in
  let clear = Cyclesim.in_port sim "clear" in
  let wr_en = Cyclesim.in_port sim "wr_en" in
  let wr_idx = Cyclesim.in_port sim "wr_idx" in
  let wr_data = Cyclesim.in_port sim "wr_data" in
  let rd_idx = Cyclesim.in_port sim "rd_idx" in
  let rd_data = Cyclesim.out_port sim "rd_data" in
  clear := Bits.vdd;
  Cyclesim.cycle sim;
  clear := Bits.gnd;
  (* all cells at power-on *)
  let dump () =
    for i = 0 to Abi.Reg.Ctl.size - 1 do
      rd_idx := Bits.of_unsigned_int ~width:4 i;
      Cyclesim.cycle sim;
      Printf.printf "%02x " (Bits.to_int_trunc !rd_data)
    done;
    print_newline ()
  in
  dump ();
  [%expect {| 00 00 00 00 00 ee ff c0 00 64 7d 00 fa 00 02 00 |}];
  (* write one cell, the others hold *)
  wr_en := Bits.vdd;
  wr_idx := Bits.of_unsigned_int ~width:4 (Abi.Reg.Ctl.velocity - Abi.Reg.Ctl.base);
  wr_data := Bits.of_unsigned_int ~width:8 0x30;
  Cyclesim.cycle sim;
  wr_en := Bits.gnd;
  dump ();
  [%expect {| 00 00 00 00 00 ee ff c0 00 30 7d 00 fa 00 02 00 |}]
;;
