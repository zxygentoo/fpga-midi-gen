open Base
open Hardcaml
open Signal

module Params = struct
  type 'a t =
    { run : 'a [@bits 1]
    ; channel : 'a [@bits 4]
    ; step_ms : 'a [@bits 16]
    ; gate_ms : 'a [@bits 16]
    ; velocity : 'a [@bits 8]
    ; seed : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

let size = Control.Reg.size
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
    ; doorbell_ready : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { params : 'a Params.t
    ; read_data : 'a [@bits 8]
    ; doorbell : 'a Midi.Rtl.Message.t
    }
  [@@deriving hardcaml]
end

(* the cell index of an address in the control section *)
let cell address = address - Control.Reg.base

(* Each field of the control section: the cell of its first byte, the width in bytes, and
   the power-on value. The views and the defaults both come from this one table. *)
let fields =
  [ cell Control.Reg.run, 1, 0
  ; cell Control.Reg.channel, 1, Control.Default.channel
  ; cell Control.Reg.step_ms, 2, Control.Default.step_ms
  ; cell Control.Reg.gate_ms, 2, Control.Default.gate_ms
  ; cell Control.Reg.velocity, 1, Control.Default.velocity
  ; cell Control.Reg.seed, 4, Control.Default.seed
  ; cell Control.Reg.midi_msg, Midi.max_message_bytes, 0
  ; cell Control.Reg.midi_len, 1, 0
  ; cell Control.Reg.midi_go, 1, 0
  ]
;;

(* the table must cover each cell of the section one time: this catches a width that does
   not agree with the addresses *)
let () =
  let covered =
    List.concat_map fields ~f:(fun (first, width, _) ->
      List.init width ~f:(fun k -> first + k))
  in
  assert (List.length covered = size);
  assert (List.length (List.dedup_and_sort covered ~compare:Int.compare) = size)
;;

let width_of address =
  List.find_map_exn fields ~f:(fun (first, width, _) ->
    if first = cell address then Some width else None)
;;

(* the power-on value of each cell, in the little-endian order of the host control *)
let defaults =
  let bytes =
    List.concat_map fields ~f:(fun (first, width, value) ->
      List.init width ~f:(fun k -> first + k, (value lsr (8 * k)) land 0xff))
  in
  (* the coverage assert above proves that each cell is in [bytes] exactly one time *)
  List.init size ~f:(fun k -> List.Assoc.find_exn bytes k ~equal:Int.equal)
;;

let msg_cell = cell Control.Reg.midi_msg
let len_cell = cell Control.Reg.midi_len
let go_cell = cell Control.Reg.midi_go
let run_cell = cell Control.Reg.run

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
  let doorbell_data = Variable.reg spec ~width:(Midi.max_message_bytes * 8) in
  let doorbell_len = Variable.reg spec ~width:8 in
  let pending = Variable.reg spec ~width:1 in
  (* the names put the doorbell state into the waveform tests *)
  let _ = pending.value -- "pending" in
  (* The shadow follows the live cells, except while a burst fills it and in the commit
     cycle. A rule that looks only at [write_enable] leaves the shadow stale for one cycle
     after the commit. *)
  let follow = ~:(i.write_enable) &: ~:(i.commit) in
  (* RUN has two writers: the host burst and the board button. A commit and a push in the
     same cycle both apply. *)
  let run_next = mux2 i.commit shadow.(run_cell).value live.(run_cell).value in
  (* The ring: a commit that covers MIDI_GO with bit 0 set, with MIDI_LEN in range, and
     with no message pending. The whole burst commits at one time, thus MIDI_MSG and
     MIDI_LEN of the same burst are already in the shadow. *)
  let shadow_len = shadow.(len_cell).value in
  let ring =
    i.commit
    &: lsb shadow.(go_cell).value
    &: (shadow_len >=:. 1)
    &: (shadow_len <=:. Midi.max_message_bytes)
    &: ~:(pending.value)
  in
  let taken = pending.value &: i.doorbell_ready in
  compile
    [ proc
        (List.init size ~f:(fun k ->
           proc
             [ when_ follow [ shadow.(k) <-- live.(k).value ]
             ; when_
                 (i.write_enable &: (i.write_address ==:. k))
                 [ shadow.(k) <-- i.write_data ]
             ; when_
                 i.commit
                 (if k = go_cell
                  then
                    (* MIDI_GO holds no value: it is a write strobe and a read status,
                       thus each copy goes back to 0 at the commit *)
                    [ live.(k) <-- zero 8; shadow.(k) <-- zero 8 ]
                  else [ live.(k) <-- shadow.(k).value ])
             ]))
    ; when_ i.run_toggle [ live.(run_cell) <-- run_next ^: of_unsigned_int ~width:8 1 ]
    ; when_
        ring
        [ doorbell_data
          <-- concat_lsb
                (List.init Midi.max_message_bytes ~f:(fun k ->
                   shadow.(msg_cell + k).value))
        ; doorbell_len <-- shadow_len
        ; pending <-- vdd
        ]
    ; when_ taken [ pending <-- gnd ]
    ];
  let view address =
    concat_lsb (List.init (width_of address) ~f:(fun k -> live.(cell address + k).value))
  in
  (* each view has the natural width of its value, thus no consumer knows where the value
     sits in the cell byte *)
  { O.params =
      { Params.run = lsb (view Control.Reg.run)
      ; channel = sel_bottom (view Control.Reg.channel) ~width:4
      ; step_ms = view Control.Reg.step_ms
      ; gate_ms = view Control.Reg.gate_ms
      ; velocity = view Control.Reg.velocity
      ; seed = view Control.Reg.seed
      }
      (* MIDI_GO gives the pending flag; each other cell gives its stored byte *)
  ; read_data =
      mux
        i.read_address
        (List.init size ~f:(fun k ->
           if k = go_cell then uresize pending.value ~width:8 else live.(k).value))
  ; doorbell =
      { Midi.Rtl.Message.data = doorbell_data.value
      ; len = doorbell_len.value
      ; valid = pending.value
      }
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
  burst Control.Reg.velocity [ 0x30 ];
  Stdio.printf "after write %s\n" (Bytes_util.hex (dump ()));
  inp.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inp.clear := Bits.gnd;
  Stdio.printf "after clear %s\n" (Bytes_util.hex (dump ()));
  [%expect
    {|
    power-on   00 00 00 00 00 2a 00 00 00 64 7d 00 fa 00 02 00
    after write 00 00 00 00 00 2a 00 00 00 30 7d 00 fa 00 02 00
    after clear 00 00 00 00 00 2a 00 00 00 64 7d 00 fa 00 02 00
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
  write (cell Control.Reg.step_ms) 0x11;
  show "byte 0 written";
  write (cell Control.Reg.step_ms + 1) 0x22;
  show "byte 1 written";
  inp.write_enable := Bits.gnd;
  inp.commit := Bits.vdd;
  Cyclesim.cycle sim;
  inp.commit := Bits.gnd;
  Cyclesim.cycle sim;
  show "committed";
  [%expect
    {|
    power-on       step_ms 250
    byte 0 written step_ms 250
    byte 1 written step_ms 250
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
  burst Control.Reg.run [ 0x01 ];
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

let%expect_test "the doorbell" =
  let sim, inp, out, _dump, burst = harness () in
  let show tag =
    inp.read_address := Bits.of_unsigned_int ~width:cell_bits (cell Control.Reg.midi_go);
    Cyclesim.cycle sim;
    Stdio.printf
      "%-22s MIDI_GO %d | valid %b data %06x len %d\n"
      tag
      (Bits.to_int_trunc !(out.read_data))
      (Bits.to_bool !(out.doorbell.valid))
      (Bits.to_int_trunc !(out.doorbell.data))
      (Bits.to_int_trunc !(out.doorbell.len))
  in
  let take () =
    inp.doorbell_ready := Bits.vdd;
    Cyclesim.cycle sim;
    inp.doorbell_ready := Bits.gnd;
    Cyclesim.cycle sim
  in
  (* one ascending burst does the whole operation: MIDI_MSG, MIDI_LEN, MIDI_GO *)
  burst Control.Reg.midi_msg [ 0x92; 0x3C; 0x64; 0x03; 0x01 ];
  show "after the ring";
  take ();
  show "after the transfer";
  (* the send bit is bit 0 alone *)
  burst Control.Reg.midi_go [ 0x00 ];
  show "go byte 00";
  burst Control.Reg.midi_go [ 0x02 ];
  show "go byte 02";
  (* MIDI_LEN outside 1 to 3 *)
  burst Control.Reg.midi_len [ 0x00; 0x01 ];
  show "len 0";
  burst Control.Reg.midi_len [ 0x04; 0x01 ];
  show "len 4";
  (* a ring while a message waits is ignored, and the first message holds *)
  burst Control.Reg.midi_msg [ 0xB2; 0x4A; 0x00; 0x02; 0x01 ];
  show "a good ring";
  burst Control.Reg.midi_msg [ 0xF8; 0x00; 0x00; 0x01; 0x01 ];
  show "a ring while pending";
  take ();
  show "after the transfer";
  [%expect
    {|
    after the ring         MIDI_GO 1 | valid true data 643c92 len 3
    after the transfer     MIDI_GO 0 | valid false data 643c92 len 3
    go byte 00             MIDI_GO 0 | valid false data 643c92 len 3
    go byte 02             MIDI_GO 0 | valid false data 643c92 len 3
    len 0                  MIDI_GO 0 | valid false data 643c92 len 3
    len 4                  MIDI_GO 0 | valid false data 643c92 len 3
    a good ring            MIDI_GO 1 | valid true data 004ab2 len 2
    a ring while pending   MIDI_GO 1 | valid true data 004ab2 len 2
    after the transfer     MIDI_GO 0 | valid false data 004ab2 len 2
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
  write (cell Control.Reg.step_ms) 0x11;
  write (cell Control.Reg.step_ms + 1) 0x22;
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
    │write_address  ││ 0    │C    │D                                     │
    │               ││──────┴─────┴─────────────────────────────         │
    │               ││──────┬─────┬─────────────────────────────         │
    │write_data     ││ 00   │11   │22                                    │
    │               ││──────┴─────┴─────────────────────────────         │
    │commit         ││                  ┌─────┐                          │
    │               ││──────────────────┘     └─────────────────         │
    │               ││────────────────────────┬─────────────────         │
    │params$step_ms ││ 00FA                   │2211                      │
    │               ││────────────────────────┴─────────────────         │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
