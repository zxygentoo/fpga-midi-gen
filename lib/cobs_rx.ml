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
    { data : 'a [@bits 8] [@rtlname "out_data"]
    ; valid : 'a [@rtlname "out_valid"]
    ; frame_end : 'a
    ; abort : 'a
    }
  [@@deriving hardcaml]
end

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let remaining = Variable.reg spec ~width:8 in
  let insert_zero = Variable.reg spec ~width:1 in
  let out_data = Variable.wire ~default:(zero 8) () in
  let out_valid = Variable.wire ~default:gnd () in
  let frame_end = Variable.wire ~default:gnd () in
  let abort = Variable.wire ~default:gnd () in
  compile
    [ when_
        i.valid
        [ if_
            (i.data ==:. 0)
            [ (* the delimiter; a pending implicit zero is discarded *)
              if_ (remaining.value ==:. 0) [ frame_end <-- vdd ] [ abort <-- vdd ]
            ; remaining <--. 0
            ; insert_zero <-- gnd
            ]
            [ if_
                (remaining.value ==:. 0)
                [ (* a group code; a pending zero goes out first *)
                  when_ insert_zero.value [ out_valid <-- vdd ]
                ; remaining <-- i.data -:. 1
                ; insert_zero <-- (i.data <>:. 0xff)
                ]
                [ out_data <-- i.data
                ; out_valid <-- vdd
                ; remaining <-- remaining.value -:. 1
                ]
            ]
        ]
    ];
  { O.data = out_data.value
  ; valid = out_valid.value
  ; frame_end = frame_end.value
  ; abort = abort.value
  }
;;

let%expect_test "the deframer agrees with Cobs.decode" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let run frame =
    let bytes = Buffer.create 8 in
    let events = Buffer.create 8 in
    String.iter
      (fun c ->
        inp.data := Bits.of_unsigned_int ~width:8 (Char.code c);
        inp.valid := Bits.vdd;
        Cyclesim.cycle sim;
        if Bits.to_bool !(out.valid)
        then
          Buffer.add_string bytes (Printf.sprintf "%02x " (Bits.to_int_trunc !(out.data)));
        if Bits.to_bool !(out.frame_end) then Buffer.add_string events "end ";
        if Bits.to_bool !(out.abort) then Buffer.add_string events "abort ")
      frame;
    inp.valid := Bits.gnd;
    Cyclesim.cycle sim;
    let sw =
      match Cobs.decode (Bytes.of_string frame) with
      | Ok b ->
        b
        |> Bytes.to_seq
        |> Seq.map (fun c -> Printf.sprintf "%02x " (Char.code c))
        |> List.of_seq
        |> String.concat ""
      | Error e -> "error: " ^ e
    in
    Printf.printf "hw %s| %ssw %s\n" (Buffer.contents bytes) (Buffer.contents events) sw
  in
  run "\x01\x00";
  [%expect {| hw | end sw |}];
  run "\x03\x11\x22\x02\x33\x00";
  [%expect {| hw 11 22 00 33 | end sw 11 22 00 33 |}];
  (* the phantom-zero form: the trailing empty group emits the zero *)
  run "\x02\x61\x01\x00";
  [%expect {| hw 61 00 | end sw 61 00 |}];
  (* a frame that ends inside a group *)
  run "\x05\x11\x00";
  [%expect {| hw 11 | abort sw error: the frame is too short |}];
  (* the deframer recovers after an abort *)
  run "\x02\x41\x00";
  [%expect {| hw 41 | end sw 41 |}]
;;
