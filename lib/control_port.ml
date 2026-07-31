open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a [@bits 8]
    ; in_valid : 'a
    ; hold : 'a
    ; midi_hold : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { out_data : 'a [@bits 8]
    ; out_valid : 'a
    ; midi_data : 'a [@bits 8]
    ; midi_valid : 'a
    ; state : 'a [@bits 2]
    }
  [@@deriving hardcaml]
end

module State = struct
  type t =
    | Init
    | Ready
    | Busy

  let to_code = function
    | Init -> 0
    | Ready -> 1
    | Busy -> 2
  ;;
end

(* The power-on values of the control cells, from the ABI constants. The port writes them
   into the register file in the init state. A field is (address, width, value);
   multi-byte values take the little-endian order of the ABI. A cell of no field is 0. *)
let defaults =
  let fields =
    [ Abi.Reg.Ctl.channel, 1, Abi.Default.channel
    ; Abi.Reg.Ctl.step_ms, 2, Abi.Default.step_ms
    ; Abi.Reg.Ctl.gate_ms, 2, Abi.Default.gate_ms
    ; Abi.Reg.Ctl.velocity, 1, Abi.Default.velocity
    ; Abi.Reg.Ctl.seed, 4, Abi.Default.seed
    ]
  in
  let bytes =
    List.concat_map
      ~f:(fun (address, width, value) ->
        List.init width ~f:(fun k -> address + k, (value lsr (8 * k)) land 0xff))
      fields
  in
  List.init Abi.Reg.Ctl.size ~f:(fun k ->
    Option.value
      ~default:0
      (List.Assoc.find bytes (Abi.Reg.Ctl.base + k) ~equal:Int.equal))
;;

(* payload capacity: the header plus the largest write burst; the buffer size and its
   address width follow *)
let max_payload = 4 + Abi.Limits.max_data_len
let buffer_address_bits = address_bits_for (max_payload + 1)
let buffer_size = 1 lsl buffer_address_bits

(* the low bits of an ABI address select the cell: the base must be aligned *)
let () = assert (Abi.Reg.Ctl.base % Abi.Reg.Ctl.size = 0)
let cell_bits = address_bits_for Abi.Reg.Ctl.size

(* the doorbell cells, as indices into the control section. The read override and the
   write decode build in this layout: MSG at the bottom, then MSG_LEN, then MSG_GO. *)
let msg_cell = Abi.Reg.Ctl.msg - Abi.Reg.Ctl.base
let len_cell = Abi.Reg.Ctl.msg_len - Abi.Reg.Ctl.base
let go_cell = Abi.Reg.Ctl.msg_go - Abi.Reg.Ctl.base

let () =
  assert (msg_cell = 0 && len_cell = Abi.Limits.max_msg_len && go_cell = len_cell + 1)
;;

module Ctl_regfile = Regfile.Make (struct
    let size = Abi.Reg.Ctl.size
  end)

(* one transaction at a time: Receive buffers a frame, Parse judges it, Apply writes the
   cells, Respond and Sending run the encoder *)
module Fsm = struct
  type t =
    | Init
    (** writes the control defaults. The first constructor encodes as 0, the value of the
        state register at power-on and at clear *)
    | Receive (** buffers decoded bytes until [frame_end] *)
    | Parse (** judges the header and chooses the response *)
    | Apply (** writes one cell each cycle *)
    | Respond (** strobes [frame_start] to the encoder *)
    | Sending (** the encoder sends; back to [Receive] when it is done *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

(* the sender: one state for each byte of the test message, [Idle] when no message waits.
   The port registers of the doorbell cells are the one storage of the message: each
   [Byte_k] state offers the live MSG cell [k] on the message stream, and its exit
   compares the live MSG_LEN. The trigger is the [Idle] arm alone, thus a ring while a
   message waits has no arm to fire in — the ignore rule of the ABI is the shape of the
   machine — and [Byte_2] always ends, thus the send is bounded for every cell content. *)
module Send_fsm = struct
  type t =
    | Idle (** no message waits; the [Idle] arm holds the trigger *)
    | Byte_0
    | Byte_1
    | Byte_2
  [@@deriving compare ~localize, enumerate, sexp_of]
end

(* one [Byte_] state for each MSG cell *)
let () = assert (Abi.Limits.max_msg_len = 3)

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module Fsm) spec in
  let capture_index = Variable.reg spec ~width:7 in
  let drop = Variable.reg spec ~width:1 in
  let header = Array.init 4 ~f:(fun _ -> Variable.reg spec ~width:8) in
  let request_length = Variable.reg spec ~width:7 in
  let status = Variable.reg spec ~width:8 in
  let response_length = Variable.reg spec ~width:7 in
  let apply_index = Variable.reg spec ~width:6 in
  let init_index = Variable.reg spec ~width:cell_bits in
  let write_enable = Variable.wire ~default:gnd () in
  let frame_start = Variable.wire ~default:gnd () in
  let send = State_machine.create (module Send_fsm) spec in
  let midi_byte = Variable.wire ~default:(zero 8) () in
  let pending = ~:(send.is Idle) in
  (* the doorbell cells are port registers: the sender and the response path read them at
     independent times, and the register file has one address port. These registers are
     the one storage of the cells; the regfile copies of cells 0 to [go_cell] are written
     by the uniform walks but never read. *)
  let msg_store =
    Array.init Abi.Limits.max_msg_len ~f:(fun _ -> Variable.reg spec ~width:8)
  in
  let len_store = Variable.reg spec ~width:8 in
  (* the names put the machines and the write strobe into the waveform tests; the port
     already owns the name [state], thus the main machine shows as [fsm] *)
  let _ = sm.current -- "fsm" in
  let _ = send.current -- "send_fsm" in
  let _ = write_enable.value -- "write_enable" in
  let _ = pending -- "msg_pending" in
  (* the header fields, as views of the header registers. The registers hold through the
     whole transaction: [capture] is gated on [Receive], and the op echo of
     [response_byte] already depends on this *)
  let op = header.(0).value in
  let header_address = header.(2).value @: header.(1).value in
  let header_length = header.(3).value in
  (* the decoder *)
  let decoder =
    Cobs_decoder.create
      { Cobs_decoder.I.clock = i.clock
      ; clear = i.clear
      ; in_data = i.in_data
      ; in_valid = i.in_valid
      }
  in
  (* the payload buffer *)
  let capture = sm.is Receive &: decoder.out_valid &: ~:(drop.value) in
  let payload_byte =
    (multiport_memory
       buffer_size
       ~write_ports:
         [| { Write_port.write_clock = i.clock
            ; write_address = uresize capture_index.value ~width:buffer_address_bits
            ; write_enable = capture
            ; write_data = decoder.out_data
            }
         |]
       ~read_addresses:[| uresize (apply_index.value +:. 4) ~width:buffer_address_bits |]).(
    0)
  in
  (* the encoder, fed by a registered read of the response bytes *)
  let response_data = wire 8 in
  let encoder =
    Cobs_encoder.create
      { Cobs_encoder.I.clock = i.clock
      ; clear = i.clear
      ; frame_start = frame_start.value
      ; payload_length = response_length.value
      ; read_data = response_data
      ; hold = i.hold
      }
  in
  (* the register file; the low address bits select the cell, per the alignment check *)
  let target_cell = select header_address ~high:(cell_bits - 1) ~low:0 in
  let apply_target =
    target_cell +: select apply_index.value ~high:(cell_bits - 1) ~low:0
  in
  let response_cell =
    target_cell +: select (encoder.address -:. 2) ~high:(cell_bits - 1) ~low:0
  in
  let cell_address =
    mux2 (sm.is Init) init_index.value (mux2 (sm.is Apply) apply_target response_cell)
    -- "cell_address"
  in
  let cell_write_data =
    mux2
      (sm.is Init)
      (mux init_index.value (List.map ~f:(of_unsigned_int ~width:8) defaults))
      payload_byte
    -- "cell_write_data"
  in
  let regfile =
    Ctl_regfile.create
      { Ctl_regfile.I.clock = i.clock
      ; clear = i.clear
      ; write_enable = write_enable.value
      ; address = cell_address
      ; write_data = cell_write_data
      }
  in
  let msg_length = len_store.value in
  (* the doorbell cells and MSG_GO answer from the port registers; the plain cells answer
     from the register file *)
  let cell_read =
    mux2
      (response_cell <=:. go_cell)
      (mux
         response_cell
         [ msg_store.(0).value
         ; msg_store.(1).value
         ; msg_store.(2).value
         ; len_store.value
         ; uresize pending ~width:8
         ])
      regfile.read_data
  in
  (* response byte [j]: the op echo, the status, then the cells *)
  let response_byte j =
    mux2
      (j ==:. 0)
      (op |: of_unsigned_int ~width:8 0x80)
      (mux2 (j ==:. 1) status.value cell_read)
  in
  assign response_data (reg spec (response_byte encoder.address));
  (* the range check, in 17 bits so the top of the space cannot wrap *)
  let range_end = uresize header_address ~width:17 +: uresize header_length ~width:17 in
  let address_ok =
    header_address
    >=:. Abi.Reg.Ctl.base
    &: (range_end <=:. Abi.Reg.Ctl.base + Abi.Reg.Ctl.size)
  in
  let length_ok = header_length >=:. 1 &: (header_length <=:. Abi.Limits.max_data_len) in
  let is_read = op ==:. Abi.Op.read in
  let is_write = op ==:. Abi.Op.write in
  (* a frame with the wrong shape gets no response *)
  let structural_ok =
    mux2
      is_write
      (uresize header_length ~width:7 +:. 4 ==: request_length.value)
      (mux2 is_read (request_length.value ==:. 4) vdd)
  in
  let reset_frame = proc [ capture_index <--. 0; drop <-- gnd ] in
  let respond_with code =
    proc [ status <--. Abi.Status.to_code code; sm.set_next Respond ]
  in
  compile
    [ sm.switch
        [ ( Init
          , [ (* one default each cycle; the cells are not valid before the end *)
              write_enable <-- vdd
            ; init_index <-- init_index.value +:. 1
            ; when_ (init_index.value ==:. Abi.Reg.Ctl.size - 1) [ sm.set_next Receive ]
            ] )
        ; ( Receive
          , [ when_
                (decoder.out_valid &: ~:(drop.value))
                [ capture_index <-- capture_index.value +:. 1
                ; when_ (capture_index.value ==:. max_payload) [ drop <-- vdd ]
                ; proc
                    (List.init 4 ~f:(fun k ->
                       when_
                         (capture_index.value ==:. k)
                         [ header.(k) <-- decoder.out_data ]))
                ]
            ; when_ decoder.abort [ reset_frame ]
            ; when_
                decoder.frame_end
                [ if_
                    (~:(drop.value) &: (capture_index.value >=:. 4))
                    [ request_length <-- capture_index.value; sm.set_next Parse ]
                    [ reset_frame ]
                ]
            ] )
        ; ( Parse
          , [ reset_frame
            ; apply_index <--. 0
            ; if_
                ~:structural_ok
                [ sm.set_next Receive ]
                [ status <--. Abi.Status.to_code Abi.Status.Ok
                ; response_length <--. 2
                ; if_
                    ~:length_ok
                    [ respond_with Bad_length ]
                    [ if_
                        ~:(is_read |: is_write)
                        [ respond_with Bad_op ]
                        [ if_
                            ~:address_ok
                            [ respond_with Bad_address ]
                            [ if_
                                is_read
                                [ response_length <-- uresize header_length ~width:7 +:. 2
                                ; sm.set_next Respond
                                ]
                                [ sm.set_next Apply ]
                            ]
                        ]
                    ]
                ]
            ] )
        ; ( Apply
          , [ (* one cell each cycle, in the sequence of increasing addresses *)
              write_enable <-- vdd
            ; apply_index <-- apply_index.value +:. 1
            ; when_
                (apply_index.value +:. 1 ==: uresize header_length ~width:6)
                [ sm.set_next Respond ]
            ] )
        ; Respond, [ frame_start <-- vdd; sm.set_next Sending ]
        ; Sending, [ when_ ~:(encoder.busy) [ sm.set_next Receive ] ]
        ]
      (* the write decode of the doorbell cells: the same walks that fill the regfile —
         the [Init] defaults and the [Apply] bytes — fill the port registers *)
    ; when_
        write_enable.value
        [ proc
            (List.init Abi.Limits.max_msg_len ~f:(fun k ->
               when_
                 (cell_address ==:. msg_cell + k)
                 [ msg_store.(k) <-- cell_write_data ]))
        ; when_ (cell_address ==:. len_cell) [ len_store <-- cell_write_data ]
        ]
    ; send.switch
        [ ( Idle
          , [ (* the ring: an [Apply] write of a value with bit 0 = 1 to MSG_GO, with a
                 length in range. The ascending write order has already put MSG and
                 MSG_LEN of the same burst into the cells. *)
              when_
                (sm.is Apply
                 &: (cell_address ==:. go_cell)
                 &: lsb cell_write_data
                 &: (msg_length >=:. 1)
                 &: (msg_length <=:. Abi.Limits.max_msg_len))
                [ send.set_next Byte_0 ]
            ] )
          (* each [Byte_k]: offer the cell; the transmitter takes it when [midi_hold] is 0 *)
        ; ( Byte_0
          , [ midi_byte <-- msg_store.(0).value
            ; when_
                ~:(i.midi_hold)
                [ if_ (msg_length ==:. 1) [ send.set_next Idle ] [ send.set_next Byte_1 ]
                ]
            ] )
        ; ( Byte_1
          , [ midi_byte <-- msg_store.(1).value
            ; when_
                ~:(i.midi_hold)
                [ if_ (msg_length ==:. 2) [ send.set_next Idle ] [ send.set_next Byte_2 ]
                ]
            ] )
        ; ( Byte_2
          , [ midi_byte <-- msg_store.(2).value
            ; when_ ~:(i.midi_hold) [ send.set_next Idle ]
            ] )
        ]
    ];
  { O.out_data = encoder.data
  ; out_valid = encoder.valid
  ; midi_data = midi_byte.value
  ; midi_valid = pending
  ; state =
      mux2
        (sm.is Init)
        (of_unsigned_int ~width:2 (State.to_code State.Init))
        (mux2
           (sm.is Receive)
           (of_unsigned_int ~width:2 (State.to_code State.Ready))
           (of_unsigned_int ~width:2 (State.to_code State.Busy)))
  }
;;

let%expect_test "transactions against the register file" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inp.clear := Bits.gnd;
  (* the port loads the defaults, then reports ready *)
  let budget = ref 40 in
  while Bits.to_int_trunc !(out.state) <> State.to_code State.Ready && !budget > 0 do
    Cyclesim.cycle sim;
    budget := !budget - 1
  done;
  Stdio.printf
    "ready after init: %b\n"
    (Bits.to_int_trunc !(out.state) = State.to_code State.Ready);
  [%expect {| ready after init: true |}];
  let response = Buffer.create 64 in
  let complete = ref false in
  let cycle () =
    Cyclesim.cycle sim;
    if (not !complete) && Bits.to_bool !(out.out_valid)
    then (
      let byte = Bits.to_int_trunc !(out.out_data) in
      Buffer.add_char response (Char.of_int_exn byte);
      if byte = 0 then complete := true)
  in
  let transact frame =
    Buffer.clear response;
    complete := false;
    String.iter
      ~f:(fun b ->
        inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
        inp.in_valid := Bits.vdd;
        cycle ())
      (Bytes.to_string frame);
    inp.in_valid := Bits.gnd;
    let budget = ref 500 in
    while (not !complete) && !budget > 0 do
      cycle ();
      budget := !budget - 1
    done;
    if Buffer.length response = 0
    then Stdio.print_endline "no response"
    else (
      match Abi.decode_response (Buffer.contents_bytes response) with
      | Error e -> Stdio.printf "bad response: %s\n" e
      | Ok { op; status; data } ->
        let hex =
          data
          |> Bytes.to_list
          |> List.map ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c))
          |> String.concat ~sep:" "
        in
        Stdio.printf
          "op %d status %s%s\n"
          op
          (match status with
           | Abi.Status.Ok -> "ok"
           | Bad_op -> "bad-op"
           | Bad_address -> "bad-address"
           | Bad_length -> "bad-length")
          (if String.length hex = 0 then "" else " data " ^ hex))
  in
  (* the register file has its defaults at power-on *)
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.base; len = Abi.Reg.Ctl.size }));
  [%expect {| op 1 status ok data 00 00 00 00 00 2a 00 00 00 64 7d 00 fa 00 02 00 |}];
  (* write, then read back *)
  transact
    (Abi.encode_request
       (Write { addr = Abi.Reg.Ctl.velocity; data = Bytes.of_string "\x30" }));
  [%expect {| op 2 status ok |}];
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.velocity; len = 1 }));
  [%expect {| op 1 status ok data 30 |}];
  (* the one-shot doorbell write: MSG, MSG_LEN, MSG_GO ascending. [midi_hold] is 0, thus
     the doorbell within sends at once, and the read shows MSG_GO back at 0 *)
  transact
    (Abi.encode_request
       (Write { addr = Abi.Reg.Ctl.msg; data = Bytes.of_string "\x92\x3C\x64\x03\x01" }));
  [%expect {| op 2 status ok |}];
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.msg; len = 5 }));
  [%expect {| op 1 status ok data 92 3c 64 03 00 |}];
  (* a ring against a stalled transmitter: MSG_GO reads the wait state, and 0 again after
     the release *)
  inp.midi_hold := Bits.vdd;
  transact
    (Abi.encode_request
       (Write { addr = Abi.Reg.Ctl.msg_go; data = Bytes.of_string "\x01" }));
  [%expect {| op 2 status ok |}];
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.msg_go; len = 1 }));
  [%expect {| op 1 status ok data 01 |}];
  inp.midi_hold := Bits.gnd;
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.msg_go; len = 1 }));
  [%expect {| op 1 status ok data 00 |}];
  (* errors *)
  transact (Abi.encode_request (Read { addr = 0x0000; len = 1 }));
  [%expect {| op 1 status bad-address |}];
  transact (Cobs.encode (Bytes.of_string "\x07\x00\x00\x01"));
  [%expect {| op 7 status bad-op |}];
  transact (Cobs.encode (Bytes.of_string "\x01\xF0\xFF\x00"));
  [%expect {| op 1 status bad-length |}];
  (* a frame with the wrong shape gets no response, and the engine recovers *)
  transact (Cobs.encode (Bytes.of_string "\xAA\xBB"));
  [%expect {| no response |}];
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.channel; len = 1 }));
  [%expect {| op 1 status ok data 02 |}];
  (* the port ignores a frame that arrives while it sends: one response only, and the next
     request transacts normally *)
  Buffer.clear response;
  complete := false;
  let feed frame =
    String.iter
      ~f:(fun b ->
        inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
        inp.in_valid := Bits.vdd;
        cycle ())
      (Bytes.to_string frame);
    inp.in_valid := Bits.gnd
  in
  feed (Abi.encode_request (Read { addr = Abi.Reg.Ctl.velocity; len = 1 }));
  let budget = ref 200 in
  while Buffer.length response = 0 && !budget > 0 do
    cycle ();
    budget := !budget - 1
  done;
  (* the response has begun: the port is in Sending; inject a complete frame *)
  feed (Abi.encode_request (Read { addr = Abi.Reg.Ctl.channel; len = 1 }));
  let budget = ref 500 in
  while (not !complete) && !budget > 0 do
    cycle ();
    budget := !budget - 1
  done;
  (match Abi.decode_response (Buffer.contents_bytes response) with
   | Ok { op; status = Abi.Status.Ok; data } ->
     Stdio.printf
       "during-send response: op %d data %02x\n"
       op
       (Char.to_int (Bytes.get data 0))
   | _ -> Stdio.print_endline "bad response");
  Buffer.clear response;
  complete := false;
  for _ = 1 to 400 do
    cycle ()
  done;
  Stdio.printf "response to the injected frame: %d bytes\n" (Buffer.length response);
  [%expect
    {|
    during-send response: op 1 data 30
    response to the injected frame: 0 bytes
    |}];
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.channel; len = 1 }));
  [%expect {| op 1 status ok data 02 |}]
;;

let%expect_test "the waveform of a write tear" =
  (* a two-byte write to STEP_MS. [Apply] writes one cell each cycle, in the sequence of
     increasing addresses: the value is torn between the two cycles, and [state] holds
     Busy for the whole transaction. The window opens at the tail of the request frame.
     The fsm tags follow [Fsm.t]: Init Recv Parse Apply Rspnd Send. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  (* the init walk *)
  for _ = 1 to 16 do
    Cyclesim.cycle sim
  done;
  String.iter
    ~f:(fun b ->
      inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
      inp.in_valid := Bits.vdd;
      Cyclesim.cycle sim)
    (Bytes.to_string
       (Abi.encode_request
          (Write { addr = Abi.Reg.Ctl.step_ms; data = Bytes.of_string "\x11\x22" })));
  inp.in_valid := Bits.gnd;
  for _ = 1 to 8 do
    Cyclesim.cycle sim
  done;
  let rules =
    let signal name =
      Hardcaml_waveterm.Display_rule.port_name_is
        name
        ~wave_format:Wave_format.(Bit_or Hex)
    in
    [ signal "clock"
    ; signal "in_data"
    ; signal "in_valid"
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "fsm"
        ~wave_format:
          (Wave_format.Index [ "Init"; "Recv"; "Parse"; "Apply"; "Rspnd"; "Send" ])
    ; signal "write_enable"
    ; signal "cell_address"
    ; signal "cell_write_data"
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "state"
        ~wave_format:(Wave_format.Index [ "Init"; "Ready"; "Busy" ])
    ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:2
    ~start_cycle:22
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──│
    │               ││   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  │
    │               ││──────┬────────────────────────────────────────────│
    │in_data        ││ 22   │00                                          │
    │               ││──────┴────────────────────────────────────────────│
    │in_valid       ││────────────┐                                      │
    │               ││            └──────────────────────────────────────│
    │               ││────────────┬─────┬───────────┬─────┬──────────────│
    │fsm            ││ Recv       │Parse│Apply      │Rspnd│Send          │
    │               ││────────────┴─────┴───────────┴─────┴──────────────│
    │write_enable   ││                  ┌───────────┐                    │
    │               ││──────────────────┘           └────────────────────│
    │               ││──────────────────┬─────┬─────┬─────────────────┬──│
    │cell_address   ││ A                │C    │D    │A                │B │
    │               ││──────────────────┴─────┴─────┴─────────────────┴──│
    │               ││────────────────────────┬─────┬────────────────────│
    │cell_write_data││ 11                     │22   │00                  │
    │               ││────────────────────────┴─────┴────────────────────│
    │               ││────────────┬──────────────────────────────────────│
    │state          ││ Ready      │Busy                                  │
    │               ││────────────┴──────────────────────────────────────│
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the doorbell rules" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  (* a fake transmitter that counts every byte: after it takes one, it holds for [pace]
     cycles; [stall] holds the stream still for as long as a case needs *)
  let pace = 8 in
  let stall = ref false in
  let busy_left = ref 0 in
  let taken = Buffer.create 8 in
  let response = Buffer.create 64 in
  let complete = ref false in
  let cycle () =
    inp.midi_hold := if !stall || !busy_left > 0 then Bits.vdd else Bits.gnd;
    Cyclesim.cycle sim;
    if !busy_left > 0
    then busy_left := !busy_left - 1
    else if (not !stall) && Bits.to_bool !(out.midi_valid)
    then (
      Buffer.add_char taken (Char.of_int_exn (Bits.to_int_trunc !(out.midi_data)));
      busy_left := pace);
    if (not !complete) && Bits.to_bool !(out.out_valid)
    then (
      let byte = Bits.to_int_trunc !(out.out_data) in
      Buffer.add_char response (Char.of_int_exn byte);
      if byte = 0 then complete := true)
  in
  let transact frame =
    Buffer.clear response;
    complete := false;
    String.iter
      ~f:(fun b ->
        inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
        inp.in_valid := Bits.vdd;
        cycle ())
      (Bytes.to_string frame);
    inp.in_valid := Bits.gnd;
    let budget = ref 500 in
    while (not !complete) && !budget > 0 do
      cycle ();
      budget := !budget - 1
    done
  in
  let write addr data =
    transact (Abi.encode_request (Write { addr; data = Bytes.of_string data }))
  in
  let msg_go () =
    transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.msg_go; len = 1 }));
    match Abi.decode_response (Buffer.contents_bytes response) with
    | Ok { status = Abi.Status.Ok; data; _ } when Bytes.length data = 1 ->
      Char.to_int (Bytes.get data 0)
    | _ -> -1
  in
  let hex b =
    Bytes.to_list b
    |> List.map ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c))
    |> String.concat ~sep:" "
  in
  let show tag =
    Stdio.printf
      "%s: go %d, line [%s]\n"
      tag
      (msg_go ())
      (hex (Buffer.contents_bytes taken))
  in
  let drain () =
    let budget = ref 200 in
    while Buffer.length taken < Abi.Limits.max_msg_len && !budget > 0 do
      cycle ();
      budget := !budget - 1
    done;
    (* room for one more message: a wrongly queued ring would show here *)
    for _ = 1 to 5 * Abi.Limits.max_msg_len * pace do
      cycle ()
    done
  in
  inp.clear := Bits.vdd;
  Cyclesim.cycle sim;
  inp.clear := Bits.gnd;
  for _ = 1 to 20 do
    cycle ()
  done;
  (* the send bit is bit 0 alone: a burst with a GO byte of 00 loads the cells and does
     not ring, and 02 has bit 0 clear *)
  write Abi.Reg.Ctl.msg "\x92\x3C\x64\x03\x00";
  show "go byte 00";
  write Abi.Reg.Ctl.msg_go "\x02";
  show "go byte 02";
  (* MSG_LEN outside 1 to 3: the bell does not ring *)
  write Abi.Reg.Ctl.msg_len "\x00\x01";
  show "len 0";
  write Abi.Reg.Ctl.msg_len "\x04\x01";
  show "len 4";
  (* a ring while a message waits is ignored: hold the transmitter, ring twice, release —
     one message goes out *)
  stall := true;
  write Abi.Reg.Ctl.msg "\x92\x3C\x64\x03\x01";
  show "stalled ring";
  write Abi.Reg.Ctl.msg_go "\x01";
  show "ring while waiting";
  stall := false;
  drain ();
  show "released";
  (* the sender reads the live cells: change one MSG byte, ring the stored message *)
  Buffer.clear taken;
  write Abi.Reg.Ctl.msg "\x93";
  write Abi.Reg.Ctl.msg_go "\x01";
  drain ();
  show "live cells";
  [%expect
    {|
    go byte 00: go 0, line []
    go byte 02: go 0, line []
    len 0: go 0, line []
    len 4: go 0, line []
    stalled ring: go 1, line []
    ring while waiting: go 1, line []
    released: go 0, line [92 3c 64]
    live cells: go 0, line [93 3c 64]
    |}]
;;

let%expect_test "the waveform of a doorbell ring" =
  (* the one-shot burst rings the bell: the trigger fires on the [Apply] write of the
     MSG_GO cell, [send_fsm] walks one state for each byte, and MSG_GO reads [msg_pending]
     — 1 exactly while the message waits. The transmitter is free in this test, thus one
     byte goes each cycle. The tags are compact for the narrow cells: the fsm ones follow
     [Fsm.t], the send ones [Send_fsm.t]. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  (* the init walk *)
  for _ = 1 to 16 do
    Cyclesim.cycle sim
  done;
  String.iter
    ~f:(fun b ->
      inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
      inp.in_valid := Bits.vdd;
      Cyclesim.cycle sim)
    (Bytes.to_string
       (Abi.encode_request
          (Write { addr = Abi.Reg.Ctl.msg; data = Bytes.of_string "\x92\x3C\x64\x03\x01" })));
  inp.in_valid := Bits.gnd;
  for _ = 1 to 10 do
    Cyclesim.cycle sim
  done;
  let rules =
    let signal name =
      Hardcaml_waveterm.Display_rule.port_name_is
        name
        ~wave_format:Wave_format.(Bit_or Hex)
    in
    [ signal "clock"
    ; signal "in_valid"
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "fsm"
        ~wave_format:(Wave_format.Index [ "Init"; "Rcv"; "Prs"; "App"; "Rsp"; "Snd" ])
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "send_fsm"
        ~wave_format:(Wave_format.Index [ "Idl"; "B0"; "B1"; "B2" ])
    ; signal "msg_pending"
    ; signal "midi_data"
    ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:1
    ~start_cycle:26
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐│
    │               ││  └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └│
    │in_valid       ││────┐                                              │
    │               ││    └───────────────────────────────────────       │
    │               ││────┬───┬───────────────────┬───┬───────────       │
    │fsm            ││ Rcv│Prs│App                │Rsp│Snd               │
    │               ││────┴───┴───────────────────┴───┴───────────       │
    │               ││────────────────────────────┬───┬───┬───┬───       │
    │send_fsm       ││ Idl                        │B0 │B1 │B2 │Idl       │
    │               ││────────────────────────────┴───┴───┴───┴───       │
    │msg_pending    ││                            ┌───────────┐          │
    │               ││────────────────────────────┘           └───       │
    │               ││────────────────────────────┬───┬───┬───┬───       │
    │midi_data      ││ 00                         │92 │3C │64 │00        │
    │               ││────────────────────────────┴───┴───┴───┴───       │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
