open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a [@bits 8]
    ; in_valid : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { out_data : 'a [@bits 8]
    ; out_valid : 'a
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
  (* the names put the two state registers into the waveform tests *)
  let _ = remaining.value -- "remaining" in
  let _ = insert_zero.value -- "insert_zero" in
  let at_boundary = remaining.value ==:. 0 in
  let out_data = Variable.wire ~default:(zero 8) () in
  let out_valid = Variable.wire ~default:gnd () in
  let frame_end = Variable.wire ~default:gnd () in
  let abort = Variable.wire ~default:gnd () in
  compile
    [ when_
        i.in_valid
        [ if_
            (i.in_data ==:. 0)
            [ (* the delimiter; a pending implicit zero is discarded *)
              if_ at_boundary [ frame_end <-- vdd ] [ abort <-- vdd ]
            ; remaining <--. 0
            ; insert_zero <-- gnd
            ]
            [ if_
                at_boundary
                [ (* a group code; a pending zero goes out first *)
                  when_ insert_zero.value [ out_valid <-- vdd ]
                ; remaining <-- i.in_data -:. 1
                ; insert_zero <-- (i.in_data <>:. 0xff)
                ]
                [ out_data <-- i.in_data
                ; out_valid <-- vdd
                ; remaining <-- remaining.value -:. 1
                ]
            ]
        ]
    ];
  { O.out_data = out_data.value
  ; out_valid = out_valid.value
  ; frame_end = frame_end.value
  ; abort = abort.value
  }
;;

let%expect_test "the decoder agrees with Cobs.decode" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let run frame =
    let bytes = Buffer.create 8 in
    let events = Buffer.create 8 in
    String.iter
      (fun c ->
        inp.in_data := Bits.of_unsigned_int ~width:8 (Char.code c);
        inp.in_valid := Bits.vdd;
        Cyclesim.cycle sim;
        if Bits.to_bool !(out.out_valid)
        then
          Buffer.add_string
            bytes
            (Printf.sprintf "%02x " (Bits.to_int_trunc !(out.out_data)));
        if Bits.to_bool !(out.frame_end) then Buffer.add_string events "end ";
        if Bits.to_bool !(out.abort) then Buffer.add_string events "abort ")
      frame;
    inp.in_valid := Bits.gnd;
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
  (* the decoder recovers after an abort *)
  run "\x02\x41\x00";
  [%expect {| hw 41 | end sw 41 |}]
;;

(* The two waveform tests below are visual documentation. A fresh simulator for each test
   keeps cycle 0 at the first frame byte. *)

let waveform_rules =
  let signal name =
    Hardcaml_waveterm.Display_rule.port_name_is name ~wave_format:Wave_format.(Bit_or Hex)
  in
  List.map
    signal
    [ "clock"
    ; "in_data"
    ; "in_valid"
    ; "out_data"
    ; "out_valid"
    ; "frame_end"
    ; "abort"
    ; "remaining"
    ; "insert_zero"
    ]
;;

let waveform_sim () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  let feed byte =
    inp.in_data := Bits.of_unsigned_int ~width:8 byte;
    inp.in_valid := Bits.vdd;
    Cyclesim.cycle sim
  in
  let idle () =
    inp.in_valid := Bits.gnd;
    Cyclesim.cycle sim
  in
  waves, feed, idle
;;

let%expect_test "the waveform of a frame with a zero in the body" =
  (* the frame encodes 11 22 00 33. [remaining] counts down inside each group. The code
     byte 02 emits the pending zero of the group before it: [out_valid] with [out_data] 0.
     At the delimiter the pending zero of the last group is discarded, and [frame_end]
     strobes. *)
  let waves, feed, idle = waveform_sim () in
  List.iter feed [ 0x03; 0x11; 0x22; 0x02; 0x33; 0x00 ];
  idle ();
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:waveform_rules
    ~show_digest:false
    ~wave_width:2
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──│
    │               ││   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  │
    │               ││──────┬─────┬─────┬─────┬─────┬───────────         │
    │in_data        ││ 03   │11   │22   │02   │33   │00                  │
    │               ││──────┴─────┴─────┴─────┴─────┴───────────         │
    │in_valid       ││────────────────────────────────────┐              │
    │               ││                                    └─────         │
    │               ││──────┬─────┬─────┬─────┬─────┬───────────         │
    │out_data       ││ 00   │11   │22   │00   │33   │00                  │
    │               ││──────┴─────┴─────┴─────┴─────┴───────────         │
    │out_valid      ││      ┌───────────────────────┐                    │
    │               ││──────┘                       └───────────         │
    │frame_end      ││                              ┌─────┐              │
    │               ││──────────────────────────────┘     └─────         │
    │abort          ││                                                   │
    │               ││──────────────────────────────────────────         │
    │               ││──────┬─────┬─────┬─────┬─────┬───────────         │
    │remaining      ││ 00   │02   │01   │00   │01   │00                  │
    │               ││──────┴─────┴─────┴─────┴─────┴───────────         │
    │insert_zero    ││      ┌─────────────────────────────┐              │
    │               ││──────┘                             └─────         │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the waveform of the frame boundary" =
  (* the two byte streams decode to the same bytes, 41 42: first as one frame, then as two
     frames of one byte each. [out_data] and [out_valid] show no difference between the
     two cases; [frame_end] is the only signal that carries the boundary. The idle cycle
     in the first frame shows the converse: [out_valid] falls with no boundary, as it does
     after every byte at the real UART rate. *)
  let waves, feed, idle = waveform_sim () in
  List.iter feed [ 0x03; 0x41 ];
  idle ();
  List.iter feed [ 0x42; 0x00 ];
  List.iter feed [ 0x02; 0x41; 0x00; 0x02; 0x42; 0x00 ];
  idle ();
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:waveform_rules
    ~show_digest:false
    ~wave_width:1
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐│
    │               ││  └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └│
    │               ││────┬───────┬───┬───┬───┬───┬───┬───┬───┬───────   │
    │in_data        ││ 03 │41     │42 │00 │02 │41 │00 │02 │42 │00        │
    │               ││────┴───────┴───┴───┴───┴───┴───┴───┴───┴───────   │
    │in_valid       ││────────┐   ┌───────────────────────────────┐      │
    │               ││        └───┘                               └───   │
    │               ││────┬───┬───┬───┬───────┬───┬───────┬───┬───────   │
    │out_data       ││ 00 │41 │00 │42 │00     │41 │00     │42 │00        │
    │               ││────┴───┴───┴───┴───────┴───┴───────┴───┴───────   │
    │out_valid      ││    ┌───┐   ┌───┐       ┌───┐       ┌───┐          │
    │               ││────┘   └───┘   └───────┘   └───────┘   └───────   │
    │frame_end      ││                ┌───┐       ┌───┐       ┌───┐      │
    │               ││────────────────┘   └───────┘   └───────┘   └───   │
    │abort          ││                                                   │
    │               ││────────────────────────────────────────────────   │
    │               ││────┬───┬───────┬───────┬───┬───────┬───┬───────   │
    │remaining      ││ 00 │02 │01     │00     │01 │00     │01 │00        │
    │               ││────┴───┴───────┴───────┴───┴───────┴───┴───────   │
    │insert_zero    ││    ┌───────────────┐   ┌───────┐   ┌───────┐      │
    │               ││────┘               └───┘       └───┘       └───   │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the waveform of an abort and the recovery" =
  (* the code 05 promises four data bytes, and the delimiter comes after one: [abort]
     strobes in place of [frame_end]. The delimiter also clears the state, thus the good
     frame after it decodes with no extra reset. *)
  let waves, feed, idle = waveform_sim () in
  List.iter feed [ 0x05; 0x11; 0x00 ];
  List.iter feed [ 0x02; 0x41; 0x00 ];
  idle ();
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:waveform_rules
    ~show_digest:false
    ~wave_width:2
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──│
    │               ││   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  │
    │               ││──────┬─────┬─────┬─────┬─────┬───────────         │
    │in_data        ││ 05   │11   │00   │02   │41   │00                  │
    │               ││──────┴─────┴─────┴─────┴─────┴───────────         │
    │in_valid       ││────────────────────────────────────┐              │
    │               ││                                    └─────         │
    │               ││──────┬─────┬───────────┬─────┬───────────         │
    │out_data       ││ 00   │11   │00         │41   │00                  │
    │               ││──────┴─────┴───────────┴─────┴───────────         │
    │out_valid      ││      ┌─────┐           ┌─────┐                    │
    │               ││──────┘     └───────────┘     └───────────         │
    │frame_end      ││                              ┌─────┐              │
    │               ││──────────────────────────────┘     └─────         │
    │abort          ││            ┌─────┐                                │
    │               ││────────────┘     └───────────────────────         │
    │               ││──────┬─────┬─────┬─────┬─────┬───────────         │
    │remaining      ││ 00   │04   │03   │00   │01   │00                  │
    │               ││──────┴─────┴─────┴─────┴─────┴───────────         │
    │insert_zero    ││      ┌───────────┐     ┌───────────┐              │
    │               ││──────┘           └─────┘           └─────         │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
