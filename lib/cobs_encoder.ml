open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; frame_start : 'a
    ; payload_length : 'a [@bits 7]
    ; read_data : 'a [@bits 8]
    ; hold : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { address : 'a [@bits 7]
    ; data : 'a [@bits 8]
    ; valid : 'a
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

(* the two passes over each group: the Scan states find the group end, then Code and the
   Data states emit it *)
module State = struct
  type t =
    | Idle
    (** waits for [frame_start]. The first constructor encodes as 0, the value of the
        state register at power-on and at clear *)
    | Scan_req
    (** drives [scan_index] on [address]; the memory answers in the next cycle *)
    | Scan_eval (** examines [read_data]: a zero or the payload end closes the group *)
    | Code (** sends the group code: the group size plus one *)
    | Data_req (** drives [send_index] on [address] *)
    | Data_latch (** captures [read_data] into [byte_to_send] *)
    | Data_send (** sends the byte and advances [send_index] *)
    | Delim (** sends the zero delimiter; the frame is complete *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let payload_length = Variable.reg spec ~width:7 in
  let send_index = Variable.reg spec ~width:7 in
  let scan_index = Variable.reg spec ~width:7 in
  let group_end = Variable.reg spec ~width:7 in
  let ended_at_zero = Variable.reg spec ~width:1 in
  let byte_to_send = Variable.reg spec ~width:8 in
  let data = Variable.wire ~default:(zero 8) () in
  let valid = Variable.wire ~default:gnd () in
  (* the names put the machine state and the two cursors into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = scan_index.value -- "scan_index" in
  let _ = send_index.value -- "send_index" in
  let send byte next =
    proc [ data <-- byte; when_ ~:(i.hold) [ valid <-- vdd; proc next ] ]
  in
  let after_group =
    if_
      ended_at_zero.value
      [ send_index <-- group_end.value +:. 1
      ; scan_index <-- group_end.value +:. 1
      ; sm.set_next Scan_req
      ]
      [ sm.set_next Delim ]
  in
  compile
    [ sm.switch
        [ ( Idle
          , [ when_
                i.frame_start
                [ payload_length <-- i.payload_length
                ; send_index <--. 0
                ; scan_index <--. 0
                ; sm.set_next Scan_req
                ]
            ] )
        ; Scan_req, [ sm.set_next Scan_eval ]
        ; ( Scan_eval
          , [ if_
                (scan_index.value ==: payload_length.value |: (i.read_data ==:. 0))
                [ group_end <-- scan_index.value
                ; ended_at_zero <-- (scan_index.value <>: payload_length.value)
                ; sm.set_next Code
                ]
                [ scan_index <-- scan_index.value +:. 1; sm.set_next Scan_req ]
            ] )
        ; ( Code
          , [ send
                (uresize (group_end.value -: send_index.value +:. 1) ~width:8)
                [ if_
                    (send_index.value ==: group_end.value)
                    [ after_group ]
                    [ sm.set_next Data_req ]
                ]
            ] )
        ; Data_req, [ sm.set_next Data_latch ]
        ; Data_latch, [ byte_to_send <-- i.read_data; sm.set_next Data_send ]
        ; ( Data_send
          , [ send
                byte_to_send.value
                [ send_index <-- send_index.value +:. 1
                ; if_
                    (send_index.value +:. 1 ==: group_end.value)
                    [ after_group ]
                    [ sm.set_next Data_req ]
                ]
            ] )
        ; Delim, [ send (zero 8) [ sm.set_next Idle ] ]
        ]
    ];
  { O.address = mux2 (sm.is Scan_req) scan_index.value send_index.value
  ; data = data.value
  ; valid = valid.value
  ; busy = i.frame_start |: ~:(sm.is Idle)
  }
;;

(* The test harness: the payload lives in a small ROM with a registered read, per the
   contract of the block. *)

module Harness_i = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; frame_start : 'a
    ; payload_length : 'a [@bits 7]
    ; hold : 'a
    }
  [@@deriving hardcaml]
end

module Harness_o = struct
  type 'a t =
    { address : 'a [@bits 7]
    ; data : 'a [@bits 8]
    ; valid : 'a
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

let harness payload (h : _ Harness_i.t) : _ Harness_o.t =
  let bytes = List.map Char.code (List.of_seq (String.to_seq payload)) in
  let padded = bytes @ List.init (8 - List.length bytes) (fun _ -> 0) in
  let spec = Reg_spec.create ~clock:h.clock ~clear:h.clear () in
  let address = wire 7 in
  let read_data =
    reg
      spec
      (mux (select address ~high:2 ~low:0) (List.map (of_unsigned_int ~width:8) padded))
    -- "read_data"
  in
  let t =
    create
      { I.clock = h.clock
      ; clear = h.clear
      ; frame_start = h.frame_start
      ; payload_length = h.payload_length
      ; read_data
      ; hold = h.hold
      }
  in
  assign address t.address;
  { Harness_o.address = t.address; data = t.data; valid = t.valid; busy = t.busy }
;;

let%expect_test "the encoder agrees with Cobs.encode" =
  let run payload =
    let module Sim = Cyclesim.With_interface (Harness_i) (Harness_o) in
    let sim = Sim.create (harness payload) in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs ~clock_edge:Before sim in
    inp.payload_length := Bits.of_unsigned_int ~width:7 (String.length payload);
    inp.frame_start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.frame_start := Bits.gnd;
    let hw = Buffer.create 16 in
    let stop = ref 200 in
    while !stop > 0 do
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.valid)
      then (
        let b = Bits.to_int_trunc !(out.data) in
        Buffer.add_string hw (Printf.sprintf "%02x " b);
        if b = 0 then stop := 0);
      stop := max 0 (!stop - 1)
    done;
    let sw =
      Cobs.encode (Bytes.of_string payload)
      |> Bytes.to_seq
      |> Seq.map (fun c -> Printf.sprintf "%02x " (Char.code c))
      |> List.of_seq
      |> String.concat ""
    in
    Printf.printf "hw %s| sw %s\n" (Buffer.contents hw) sw
  in
  run "\x82\x00";
  [%expect {| hw 02 82 01 00 | sw 02 82 01 00 |}];
  run "\x81\x00\x64";
  [%expect {| hw 02 81 02 64 00 | sw 02 81 02 64 00 |}];
  run "\x11\x22\x33";
  [%expect {| hw 04 11 22 33 00 | sw 04 11 22 33 00 |}];
  run "\x00\x00";
  [%expect {| hw 01 01 01 00 | sw 01 01 01 00 |}];
  run "\x41\x00";
  [%expect {| hw 02 41 01 00 | sw 02 41 01 00 |}]
;;

(* The three waveform tests below are visual documentation. The state row shows the
   [State.t] constructors as three-letter tags, in declaration order: Idl ScR ScE Cod DaR
   DaL DaS Del. *)

let waveform_rules =
  let signal name =
    Hardcaml_waveterm.Display_rule.port_name_is name ~wave_format:Wave_format.(Bit_or Hex)
  in
  [ signal "clock"
  ; signal "frame_start"
  ; Hardcaml_waveterm.Display_rule.port_name_is
      "state"
      ~wave_format:
        (Wave_format.Index [ "Idl"; "ScR"; "ScE"; "Cod"; "DaR"; "DaL"; "DaS"; "Del" ])
  ; signal "address"
  ; signal "read_data"
  ; signal "scan_index"
  ; signal "send_index"
  ; signal "data"
  ; signal "valid"
  ; signal "hold"
  ; signal "busy"
  ]
;;

let waveform_sim payload =
  let module Sim = Cyclesim.With_interface (Harness_i) (Harness_o) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all (harness payload) in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  inp.payload_length := Bits.of_unsigned_int ~width:7 (String.length payload);
  let start () =
    inp.frame_start := Bits.vdd;
    Cyclesim.cycle sim;
    inp.frame_start := Bits.gnd
  in
  let cycles n =
    for _ = 1 to n do
      Cyclesim.cycle sim
    done
  in
  waves, inp, start, cycles
;;

let%expect_test "the waveform of the two passes" =
  (* the payload is 41 00 and the frame is 02 41 01 00. The scan pass reads ahead on
     [address] with [scan_index], and [valid] stays low; the emit pass re-reads with
     [send_index] and sends. After the first group both cursors jump over the zero at
     index 1, and the trailing empty group makes the code 01. *)
  let waves, _inp, start, cycles = waveform_sim "\x41\x00" in
  start ();
  cycles 12;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:waveform_rules
    ~show_digest:false
    ~wave_width:1
    ~display_width:76
    waves;
  [%expect
    {|
    ┌Signals──────────┐┌Waves──────────────────────────────────────────────────┐
    │clock            ││┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐│
    │                 ││  └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └│
    │frame_start      ││────┐                                                  │
    │                 ││    └───────────────────────────────────────────────   │
    │                 ││────┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───   │
    │state            ││ Idl│ScR│ScE│ScR│ScE│Cod│DaR│DaL│DaS│ScR│ScE│Cod│Del   │
    │                 ││────┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───   │
    │                 ││────────────┬───┬───────────────────┬───────────────   │
    │address          ││ 00         │01 │00                 │02                │
    │                 ││────────────┴───┴───────────────────┴───────────────   │
    │                 ││────┬───────────┬───┬───────────────────┬───────────   │
    │read_data        ││ 00 │41         │00 │41                 │00            │
    │                 ││────┴───────────┴───┴───────────────────┴───────────   │
    │                 ││────────────┬───────────────────────┬───────────────   │
    │scan_index       ││ 00         │01                     │02                │
    │                 ││────────────┴───────────────────────┴───────────────   │
    │                 ││────────────────────────────────────┬───────────────   │
    │send_index       ││ 00                                 │02                │
    │                 ││────────────────────────────────────┴───────────────   │
    │                 ││────────────────────┬───┬───────┬───┬───────┬───┬───   │
    │data             ││ 00                 │02 │00     │41 │00     │01 │00    │
    │                 ││────────────────────┴───┴───────┴───┴───────┴───┴───   │
    │valid            ││                    ┌───┐       ┌───┐       ┌───────   │
    │                 ││────────────────────┘   └───────┘   └───────┘          │
    │hold             ││                                                       │
    │                 ││────────────────────────────────────────────────────   │
    │busy             ││────────────────────────────────────────────────────   │
    │                 ││                                                       │
    └─────────────────┘└───────────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the waveform of the hold stall" =
  (* [hold] rises while the block is in Data_send: the state freezes, [data] keeps the
     byte, and [valid] strobes only in the first cycle with [hold] low again. *)
  let waves, inp, start, cycles = waveform_sim "\x41" in
  start ();
  cycles 7;
  inp.hold := Bits.vdd;
  cycles 2;
  inp.hold := Bits.gnd;
  cycles 3;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:waveform_rules
    ~show_digest:false
    ~wave_width:1
    ~display_width:76
    waves;
  [%expect
    {|
    ┌Signals──────────┐┌Waves──────────────────────────────────────────────────┐
    │clock            ││┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐│
    │                 ││  └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └│
    │frame_start      ││────┐                                                  │
    │                 ││    └───────────────────────────────────────────────   │
    │                 ││────┬───┬───┬───┬───┬───┬───┬───┬───────────┬───┬───   │
    │state            ││ Idl│ScR│ScE│ScR│ScE│Cod│DaR│DaL│DaS        │Del│Idl   │
    │                 ││────┴───┴───┴───┴───┴───┴───┴───┴───────────┴───┴───   │
    │                 ││────────────┬───┬───────────────────────────┬───────   │
    │address          ││ 00         │01 │00                         │01        │
    │                 ││────────────┴───┴───────────────────────────┴───────   │
    │                 ││────┬───────────┬───┬───────────────────────────┬───   │
    │read_data        ││ 00 │41         │00 │41                         │00    │
    │                 ││────┴───────────┴───┴───────────────────────────┴───   │
    │                 ││────────────┬───────────────────────────────────────   │
    │scan_index       ││ 00         │01                                        │
    │                 ││────────────┴───────────────────────────────────────   │
    │                 ││────────────────────────────────────────────┬───────   │
    │send_index       ││ 00                                         │01        │
    │                 ││────────────────────────────────────────────┴───────   │
    │                 ││────────────────────┬───┬───────┬───────────┬───────   │
    │data             ││ 00                 │02 │00     │41         │00        │
    │                 ││────────────────────┴───┴───────┴───────────┴───────   │
    │valid            ││                    ┌───┐               ┌───────┐      │
    │                 ││────────────────────┘   └───────────────┘       └───   │
    │hold             ││                                ┌───────┐              │
    │                 ││────────────────────────────────┘       └───────────   │
    │busy             ││────────────────────────────────────────────────┐      │
    │                 ││                                                └───   │
    └─────────────────┘└───────────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the waveform of the empty payload" =
  (* [payload_length] is 0 and the frame is 01 00: the empty group and the delimiter.
     [busy] is the envelope of the work: it rises with [frame_start] and falls after the
     delimiter. *)
  let waves, _inp, start, cycles = waveform_sim "" in
  start ();
  cycles 5;
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
    │frame_start    ││──────┐                                            │
    │               ││      └─────────────────────────────               │
    │               ││──────┬─────┬─────┬─────┬─────┬─────               │
    │state          ││ Idl  │ScR  │ScE  │Cod  │Del  │Idl                 │
    │               ││──────┴─────┴─────┴─────┴─────┴─────               │
    │               ││────────────────────────────────────               │
    │address        ││ 00                                                │
    │               ││────────────────────────────────────               │
    │               ││────────────────────────────────────               │
    │read_data      ││ 00                                                │
    │               ││────────────────────────────────────               │
    │               ││────────────────────────────────────               │
    │scan_index     ││ 00                                                │
    │               ││────────────────────────────────────               │
    │               ││────────────────────────────────────               │
    │send_index     ││ 00                                                │
    │               ││────────────────────────────────────               │
    │               ││──────────────────┬─────┬───────────               │
    │data           ││ 00               │01   │00                        │
    │               ││──────────────────┴─────┴───────────               │
    │valid          ││                  ┌───────────┐                    │
    │               ││──────────────────┘           └─────               │
    │hold           ││                                                   │
    │               ││────────────────────────────────────               │
    │busy           ││──────────────────────────────┐                    │
    │               ││                              └─────               │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
