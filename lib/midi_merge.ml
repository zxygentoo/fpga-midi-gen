open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { doorbell : 'a Midi.Rtl.Message.t
    ; model : 'a Midi.Rtl.Message.t
    ; out_ready : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { out : 'a Midi.Rtl.Message.t
    ; doorbell_ready : 'a
    ; model_ready : 'a
    }
  [@@deriving hardcaml]
end

let create (i : _ I.t) : _ O.t =
  let granted_doorbell = i.doorbell.valid in
  let granted_model = i.model.valid &: ~:(i.doorbell.valid) in
  { O.out =
      { Midi.Rtl.Message.data = mux2 granted_doorbell i.doorbell.data i.model.data
      ; len = mux2 granted_doorbell i.doorbell.len i.model.len
      ; valid = i.doorbell.valid |: i.model.valid
      }
      (* the [ready] of the sink goes to the source that has the grant, thus a transfer
         happens for one source at a time *)
  ; doorbell_ready = i.out_ready &: granted_doorbell
  ; model_ready = i.out_ready &: granted_model
  }
;;

let%expect_test "the priority and the ready of each source" =
  (* every combination of the three control bits. [out.data] and [out.len] have no meaning
     when [out.valid] is 0. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let show b = if b then "1" else "0" in
  let bit b = if b then Bits.vdd else Bits.gnd in
  let width = Midi.max_message_bytes * 8 in
  Stdio.print_endline "db mdl rdy | out.data len valid | db_rdy mdl_rdy";
  let step db mdl rdy =
    inp.doorbell.data := Bits.of_unsigned_int ~width 0xAAAAAA;
    inp.doorbell.len := Bits.of_unsigned_int ~width:8 3;
    inp.doorbell.valid := bit db;
    inp.model.data := Bits.of_unsigned_int ~width 0x555555;
    inp.model.len := Bits.of_unsigned_int ~width:8 2;
    inp.model.valid := bit mdl;
    inp.out_ready := bit rdy;
    Cyclesim.cycle sim;
    Stdio.printf
      " %s   %s   %s  |   %06x   %d    %s   |   %s      %s\n"
      (show db)
      (show mdl)
      (show rdy)
      (Bits.to_int_trunc !(out.out.data))
      (Bits.to_int_trunc !(out.out.len))
      (show (Bits.to_bool !(out.out.valid)))
      (show (Bits.to_bool !(out.doorbell_ready)))
      (show (Bits.to_bool !(out.model_ready)))
  in
  List.iter [ false; true ] ~f:(fun db ->
    List.iter [ false; true ] ~f:(fun mdl ->
      List.iter [ false; true ] ~f:(fun rdy -> step db mdl rdy)));
  [%expect
    {|
    db mdl rdy | out.data len valid | db_rdy mdl_rdy
     0   0   0  |   555555   2    0   |   0      0
     0   0   1  |   555555   2    0   |   0      0
     0   1   0  |   555555   2    1   |   0      0
     0   1   1  |   555555   2    1   |   0      1
     1   0   0  |   aaaaaa   3    1   |   0      0
     1   0   1  |   aaaaaa   3    1   |   1      0
     1   1   0  |   aaaaaa   3    1   |   0      0
     1   1   1  |   aaaaaa   3    1   |   1      0
    |}]
;;
