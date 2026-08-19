open Base
open Hardcaml
open Signal

module Params = struct
  type 'a t =
    { run : 'a [@bits 1]
    ; channel : 'a [@bits 4]
    ; step_ms : 'a [@bits 16]
    ; velocity : 'a [@bits 8]
    ; seed : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let size = Control_intf.Reg.size
let cell_bits = address_bits_for size

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; write_enable : 'a
    ; write_address : 'a [@bits cell_bits]
    ; write_data : 'a [@bits 8]
    ; commit : 'a
    ; read_address : 'a [@bits cell_bits]
    ; run_toggle : 'a
    ; seed_write : 'a
    ; seed_value : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { params : 'a Params.t
    ; read_data : 'a [@bits 8]
    }
  [@@deriving hardcaml]
end

(* the cell index of an address in the control section *)
let cell address = address - Control_intf.Reg.base

(* the power-on value of each cell, in the little-endian order of the host control.
   [Control_intf.Reg.fields] is the one table of the widths and the values, and its
   coverage assert proves that each cell is here exactly one time. *)
let defaults =
  let bytes =
    List.concat_map Control_intf.Reg.fields ~f:(fun (f : Control_intf.Reg.field) ->
      List.init f.width ~f:(fun k ->
        cell f.address + k, (f.default lsr (8 * k)) land 0xff))
  in
  List.init size ~f:(fun k -> List.Assoc.find_exn bytes k ~equal:Int.equal)
;;

let run_cell = cell Control_intf.Reg.run
let seed_cell = cell Control_intf.Reg.seed

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  (* each cell carries its power-on value in the bitstream, and the clear gives the same
     value; thus the section needs no init walk *)
  let cells () =
    Array.of_list
      (List.map defaults ~f:(fun d ->
         Variable.reg
           spec
           ~initialize_to:(Bits.of_unsigned_int ~width:8 d)
           ~clear_to:(of_unsigned_int ~width:8 d)
           ~width:8))
  in
  let live = cells () in
  let shadow = cells () in
  (* The shadow follows the live cells, except while a burst fills it and in the commit
     cycle. A rule that looks only at [write_enable] leaves the shadow stale for one cycle
     after the commit. *)
  let follow = ~:(i.write_enable) &: ~:(i.commit) in
  (* RUN has two writers: the host burst and the board button. A commit and a push in the
     same cycle both apply. *)
  let run_next = mux2 i.commit shadow.(run_cell).value live.(run_cell).value in
  (* the bytes of the panel, in the little-endian order of the cells *)
  let seed_bytes = split_lsb ~part_width:8 i.seed_value in
  compile
    [ proc
        (List.init size ~f:(fun k ->
           proc
             [ when_ follow [ shadow.(k) <-- live.(k).value ]
             ; when_
                 (i.write_enable &: (i.write_address ==:. k))
                 [ shadow.(k) <-- i.write_data ]
             ; when_ i.commit [ live.(k) <-- shadow.(k).value ]
             ]))
    ; when_ i.run_toggle [ live.(run_cell) <-- run_next ^: of_unsigned_int ~width:8 1 ]
      (* SEED has two writers as RUN has: the host burst and the slide switches. This rule
         stands after the commit, thus the panel wins a cycle that carries both — the last
         writer wins, and the panel is the writer a person can see. The shadow follows the
         live cells in the cycle after, thus a later commit cannot bring the host value
         back. *)
    ; when_
        i.seed_write
        (List.mapi seed_bytes ~f:(fun k byte -> live.(seed_cell + k) <-- byte))
    ];
  let view address =
    concat_lsb
      (List.init (Control_intf.Reg.width_of address) ~f:(fun k ->
         live.(cell address + k).value))
  in
  (* each view has the natural width of its value, thus no consumer knows where the value
     sits in the cell byte *)
  { O.params =
      { Params.run = lsb (view Control_intf.Reg.run)
      ; channel = sel_bottom (view Control_intf.Reg.channel) ~width:4
      ; step_ms = view Control_intf.Reg.step_ms
      ; velocity = view Control_intf.Reg.velocity
      ; seed = view Control_intf.Reg.seed
      }
  ; read_data = mux i.read_address (List.init size ~f:(fun k -> live.(k).value))
  }
;;

(* The test harness: [dump] reads every cell, and [burst] fills the shadow one byte in
   each cycle and then commits, as [Control_port] does. *)

let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let dump () =
    Bytes.of_char_list
      (List.init size ~f:(fun k ->
         inp.read_address := Bits.of_unsigned_int ~width:cell_bits k;
         Cyclesim.cycle sim;
         Char.of_int_exn (Bits.to_int_trunc !(out.read_data))))
  in
  let burst address bytes =
    List.iteri bytes ~f:(fun k b ->
      inp.write_enable := Bits.vdd;
      inp.write_address := Bits.of_unsigned_int ~width:cell_bits (cell address + k);
      inp.write_data := Bits.of_unsigned_int ~width:8 b;
      Cyclesim.cycle sim);
    inp.write_enable := Bits.gnd;
    inp.commit := Bits.vdd;
    Cyclesim.cycle sim;
    inp.commit := Bits.gnd;
    Cyclesim.cycle sim
  in
  sim, inp, out, dump, burst
;;

let%expect_test "the defaults need no init walk" =
  let sim, inp, _out, dump, burst = harness () in
  (* the first cycle: the cells are already correct, with no walk and no clear *)
  Stdio.printf "power-on   %s\n" (Bytes_util.hex (dump ()));
  burst Control_intf.Reg.velocity [ 0x30 ];
  Stdio.printf "after write %s\n" (Bytes_util.hex (dump ()));
  inp.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inp.clear := Bits.gnd;
  Stdio.printf "after clear %s\n" (Bytes_util.hex (dump ()));
  [%expect
    {|
    power-on   00 00 00 00 64 c8 00 02 00
    after write 00 00 00 00 30 c8 00 02 00
    after clear 00 00 00 00 64 c8 00 02 00
    |}]
;;

let%expect_test "the write is atomic" =
  (* a two-byte write to STEP_MS. The view holds the old value for the whole burst, and it
     takes the new value at the commit; thus it never shows one new byte and one old byte. *)
  let sim, inp, out, _dump, _burst = harness () in
  let show tag =
    Stdio.printf "%-14s step_ms %d\n" tag (Bits.to_int_trunc !(out.params.step_ms))
  in
  (* the simulator computes the outputs at the first cycle; the cells already hold their
     power-on value, because it comes from the bitstream *)
  Cyclesim.cycle sim;
  show "power-on";
  let write k b =
    inp.write_enable := Bits.vdd;
    inp.write_address := Bits.of_unsigned_int ~width:cell_bits k;
    inp.write_data := Bits.of_unsigned_int ~width:8 b;
    Cyclesim.cycle sim
  in
  write (cell Control_intf.Reg.step_ms) 0x11;
  show "byte 0 written";
  write (cell Control_intf.Reg.step_ms + 1) 0x22;
  show "byte 1 written";
  inp.write_enable := Bits.gnd;
  inp.commit := Bits.vdd;
  Cyclesim.cycle sim;
  inp.commit := Bits.gnd;
  Cyclesim.cycle sim;
  show "committed";
  [%expect
    {|
    power-on       step_ms 200
    byte 0 written step_ms 200
    byte 1 written step_ms 200
    committed      step_ms 8721
    |}]
;;

let%expect_test "the RUN toggle" =
  let sim, inp, out, _dump, burst = harness () in
  let show tag =
    Stdio.printf "%-18s run %d\n" tag (Bits.to_int_trunc !(out.params.run))
  in
  let toggle () =
    inp.run_toggle := Bits.vdd;
    Cyclesim.cycle sim;
    inp.run_toggle := Bits.gnd;
    Cyclesim.cycle sim
  in
  Cyclesim.cycle sim;
  show "power-on";
  toggle ();
  show "one push";
  toggle ();
  show "two pushes";
  burst Control_intf.Reg.run [ 0x01 ];
  show "the host writes 1";
  toggle ();
  show "and one push";
  [%expect
    {|
    power-on           run 0
    one push           run 1
    two pushes         run 0
    the host writes 1  run 1
    and one push       run 0
    |}]
;;

let%expect_test "the two writers of SEED" =
  let sim, inp, out, _dump, burst = harness () in
  let show tag =
    Stdio.printf "%-20s seed %08x\n" tag (Bits.to_int_trunc !(out.params.seed))
  in
  (* the panel strobes one cycle, as [Seed_switches] gives it *)
  let panel value =
    inp.seed_write := Bits.vdd;
    inp.seed_value := Bits.of_unsigned_int ~width:32 value;
    Cyclesim.cycle sim;
    inp.seed_write := Bits.gnd;
    Cyclesim.cycle sim
  in
  Cyclesim.cycle sim;
  show "power-on";
  panel 0x1234;
  show "the switches";
  burst Control_intf.Reg.seed [ 0x78; 0x56; 0x34; 0x12 ];
  show "the host writes";
  panel 0x0001;
  show "and a switch moves";
  [%expect
    {|
    power-on             seed 00000000
    the switches         seed 00001234
    the host writes      seed 12345678
    and a switch moves   seed 00000001
    |}]
;;

let%expect_test "the panel wins the cycle of a commit" =
  (* A host commit and a switch move in one cycle both apply, as the RUN test states it.
     SEED has no toggle, thus both applying means the panel writes after the commit and
     the host value never stands. The burst that comes after proves the shadow followed: a
     stale shadow would bring the host value back at the next commit. *)
  let sim, inp, out, _dump, burst = harness () in
  let show tag =
    Stdio.printf "%-20s seed %08x\n" tag (Bits.to_int_trunc !(out.params.seed))
  in
  List.iteri [ 0x78; 0x56; 0x34; 0x12 ] ~f:(fun k byte ->
    inp.write_enable := Bits.vdd;
    inp.write_address := Bits.of_unsigned_int ~width:cell_bits (seed_cell + k);
    inp.write_data := Bits.of_unsigned_int ~width:8 byte;
    Cyclesim.cycle sim);
  inp.write_enable := Bits.gnd;
  inp.commit := Bits.vdd;
  inp.seed_write := Bits.vdd;
  inp.seed_value := Bits.of_unsigned_int ~width:32 0x1234;
  Cyclesim.cycle sim;
  inp.commit := Bits.gnd;
  inp.seed_write := Bits.gnd;
  Cyclesim.cycle sim;
  show "commit and switch";
  burst Control_intf.Reg.velocity [ 0x30 ];
  show "another commit";
  [%expect
    {|
    commit and switch    seed 00001234
    another commit       seed 00001234
    |}]
;;

let%expect_test "the waveform of the atomic commit" =
  (* a two-byte write to STEP_MS. The port fills the shadow one byte in each cycle, and
     [commit] moves both bytes at one time: [params$step_ms] holds 250 for the whole burst
     and then takes 0x2211. A consumer never sees one new byte and one old byte. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  Cyclesim.cycle sim;
  let write k b =
    inp.write_enable := Bits.vdd;
    inp.write_address := Bits.of_unsigned_int ~width:cell_bits k;
    inp.write_data := Bits.of_unsigned_int ~width:8 b;
    Cyclesim.cycle sim
  in
  write (cell Control_intf.Reg.step_ms) 0x11;
  write (cell Control_intf.Reg.step_ms + 1) 0x22;
  inp.write_enable := Bits.gnd;
  inp.commit := Bits.vdd;
  Cyclesim.cycle sim;
  inp.commit := Bits.gnd;
  Cyclesim.cycle ~n:3 sim;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Hex)
          [ "write_enable"; "write_address"; "write_data"; "commit"; "params$step_ms" ]
      ]
    ~show_digest:false
    ~wave_width:2
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │write_enable   ││      ┌───────────┐                                │
    │               ││──────┘           └───────────────────────         │
    │               ││──────┬─────┬─────────────────────────────         │
    │write_address  ││ 0    │5    │6                                     │
    │               ││──────┴─────┴─────────────────────────────         │
    │               ││──────┬─────┬─────────────────────────────         │
    │write_data     ││ 00   │11   │22                                    │
    │               ││──────┴─────┴─────────────────────────────         │
    │commit         ││                  ┌─────┐                          │
    │               ││──────────────────┘     └─────────────────         │
    │               ││────────────────────────┬─────────────────         │
    │params$step_ms ││ 00C8                   │2211                      │
    │               ││────────────────────────┴─────────────────         │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
