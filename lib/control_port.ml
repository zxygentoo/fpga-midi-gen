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
    ; read_data : 'a [@bits 8]
    }
  [@@deriving hardcaml]
end

(* the two header sizes get short names: the engine uses them at each stage *)
let request_header_bytes = Control.Constants.request_header_bytes
let response_header_bytes = Control.Constants.response_header_bytes

(* the buffer holds the largest legal payload, and it discards a longer one; the buffer
   size and its address width follow *)
let max_payload = Control.Constants.max_payload_bytes
let buffer_address_bits = address_bits_for (max_payload + 1)
let buffer_size = 1 lsl buffer_address_bits

(* the width of a cursor over the data bytes of one burst *)
let index_bits = address_bits_for (Control.Constants.max_data_len + 1)

(* the low bits of a control address select the cell: the base must be aligned *)
let () = assert (Control.Reg.base % Control.Reg.size = 0)
let cell_bits = address_bits_for Control.Reg.size

module O = struct
  type 'a t =
    { out_data : 'a [@bits 8]
    ; out_valid : 'a
    ; write_enable : 'a
    ; write_address : 'a [@bits cell_bits]
    ; write_data : 'a [@bits 8]
    ; commit : 'a
    ; read_address : 'a [@bits cell_bits]
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

(* One transaction at a time: Receive buffers a frame, Parse judges it, Apply fills the
   shadow copy, Commit moves the whole burst into the cells, and Respond and Sending run
   the encoder. Commit is its own state, because the last Apply byte reaches the shadow
   only at the end of its cycle. *)
module Fsm = struct
  type t =
    | Receive
    (** buffers decoded bytes until [frame_end]. The first constructor encodes as 0, the
        value of the state register at power-on and at clear *)
    | Parse (** judges the header and chooses the response *)
    | Apply (** writes one shadow byte each cycle *)
    | Commit (** strobes [commit]: the burst applies at one time *)
    | Respond (** strobes [frame_start] to the encoder *)
    | Sending (** the encoder sends; back to [Receive] when it is done *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module Fsm) spec in
  let capture_index = Variable.reg spec ~width:7 in
  let drop = Variable.reg spec ~width:1 in
  let header = Array.init request_header_bytes ~f:(fun _ -> Variable.reg spec ~width:8) in
  let request_length = Variable.reg spec ~width:7 in
  let status = Variable.reg spec ~width:8 in
  let response_length = Variable.reg spec ~width:7 in
  let apply_index = Variable.reg spec ~width:index_bits in
  let frame_start = Variable.wire ~default:gnd () in
  (* the name puts the machine into the waveform tests; the other signals of interest are
     already ports *)
  let _ = sm.current -- "fsm" in
  (* the header fields, as views of the header registers. The registers hold through the
     whole transaction: [capture] is gated on [Receive], and the op echo of
     [response_byte] already depends on this *)
  let op = header.(0).value in
  let header_address = header.(1).value in
  let header_length = header.(2).value in
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
       ~read_addresses:
         [| uresize
              (apply_index.value +:. request_header_bytes)
              ~width:buffer_address_bits
         |]).(0)
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
  (* the cell ports; the low address bits select the cell, per the alignment check *)
  let target_cell = select header_address ~high:(cell_bits - 1) ~low:0 in
  let write_address =
    target_cell +: select apply_index.value ~high:(cell_bits - 1) ~low:0
  in
  let read_address =
    target_cell
    +: select (encoder.address -:. response_header_bytes) ~high:(cell_bits - 1) ~low:0
  in
  (* response byte [j]: the op echo, the status, then the cells *)
  let response_byte j =
    mux2
      (j ==:. 0)
      (op |: of_unsigned_int ~width:8 0x80)
      (mux2 (j ==:. 1) status.value i.read_data)
  in
  assign response_data (reg spec (response_byte encoder.address));
  (* the range check, in 9 bits so the top of the space cannot wrap *)
  let range_end = uresize header_address ~width:9 +: uresize header_length ~width:9 in
  let address_ok =
    header_address
    >=:. Control.Reg.base
    &: (range_end <=:. Control.Reg.base + Control.Reg.size)
  in
  let length_ok =
    header_length >=:. 1 &: (header_length <=:. Control.Constants.max_data_len)
  in
  let is_read = op ==:. Control.Op.read in
  let is_write = op ==:. Control.Op.write in
  (* a frame with the wrong shape gets no response *)
  let structural_ok =
    mux2
      is_write
      (uresize header_length ~width:7 +:. request_header_bytes ==: request_length.value)
      (mux2 is_read (request_length.value ==:. request_header_bytes) vdd)
  in
  let reset_frame = proc [ capture_index <--. 0; drop <-- gnd ] in
  let respond_with code =
    proc [ status <--. Control.Status.to_code code; sm.set_next Respond ]
  in
  (* the header checks of [Parse], in the priority order of the host control: the first
     check that fails gives the status, and [accept] runs when each one passes *)
  let rec reject_first checks ~accept =
    match checks with
    | [] -> accept
    | (bad, code) :: rest -> [ if_ bad [ respond_with code ] (reject_first rest ~accept) ]
  in
  compile
    [ sm.switch
        [ ( Receive
          , [ when_
                (decoder.out_valid &: ~:(drop.value))
                [ capture_index <-- capture_index.value +:. 1
                ; when_ (capture_index.value ==:. max_payload) [ drop <-- vdd ]
                ; proc
                    (List.init request_header_bytes ~f:(fun k ->
                       when_
                         (capture_index.value ==:. k)
                         [ header.(k) <-- decoder.out_data ]))
                ]
            ; when_ decoder.abort [ reset_frame ]
            ; when_
                decoder.frame_end
                [ if_
                    (~:(drop.value) &: (capture_index.value >=:. request_header_bytes))
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
                ([ status <--. Control.Status.to_code Control.Status.Ok
                 ; response_length <--. response_header_bytes
                 ]
                 @ reject_first
                     [ ~:length_ok, Control.Status.Bad_length
                     ; ~:(is_read |: is_write), Control.Status.Bad_op
                     ; ~:address_ok, Control.Status.Bad_address
                     ]
                     ~accept:
                       [ if_
                           is_read
                           [ response_length
                             <-- uresize header_length ~width:7 +:. response_header_bytes
                           ; sm.set_next Respond
                           ]
                           [ sm.set_next Apply ]
                       ])
            ] )
        ; ( Apply
          , [ (* one shadow byte each cycle, in the sequence of increasing addresses *)
              apply_index <-- apply_index.value +:. 1
            ; when_
                (apply_index.value +:. 1 ==: uresize header_length ~width:index_bits)
                [ sm.set_next Commit ]
            ] )
        ; Commit, [ sm.set_next Respond ]
        ; Respond, [ frame_start <-- vdd; sm.set_next Sending ]
        ; Sending, [ when_ ~:(encoder.busy) [ sm.set_next Receive ] ]
        ]
    ];
  { O.out_data = encoder.data
  ; out_valid = encoder.valid
  ; write_enable = sm.is Apply
  ; write_address
  ; write_data = payload_byte
  ; commit = sm.is Commit
  ; read_address
  ; busy = ~:(sm.is Receive)
  }
;;

(* The test harness: the port with the cells that answer it, as the top level wires them.
   [doorbell_ready] takes the test message, in place of the MIDI path. *)

module Harness_i = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; in_data : 'a [@bits 8]
    ; in_valid : 'a
    ; hold : 'a
    ; doorbell_ready : 'a
    }
  [@@deriving hardcaml]
end

module Harness_o = struct
  type 'a t =
    { out_data : 'a [@bits 8]
    ; out_valid : 'a
    ; busy : 'a
    ; doorbell : 'a Midi.Message.t
        (* the cell port, so that a waveform test can show the walk and the commit *)
    ; write_enable : 'a
    ; write_address : 'a [@bits cell_bits]
    ; write_data : 'a [@bits 8]
    ; commit : 'a
    }
  [@@deriving hardcaml]
end

let harness (h : _ Harness_i.t) : _ Harness_o.t =
  let read_data = wire 8 in
  let port =
    create
      { I.clock = h.clock
      ; clear = h.clear
      ; in_data = h.in_data
      ; in_valid = h.in_valid
      ; hold = h.hold
      ; read_data
      }
  in
  let regs =
    Control_regs.create
      { Control_regs.I.clock = h.clock
      ; clear = h.clear
      ; write_enable = port.write_enable
      ; write_address = port.write_address
      ; write_data = port.write_data
      ; commit = port.commit
      ; read_address = port.read_address
      ; run_toggle = gnd
      ; doorbell_ready = h.doorbell_ready
      }
  in
  assign read_data regs.read_data;
  { Harness_o.out_data = port.out_data
  ; out_valid = port.out_valid
  ; busy = port.busy
  ; doorbell = regs.doorbell
  ; write_enable = port.write_enable
  ; write_address = port.write_address
  ; write_data = port.write_data
  ; commit = port.commit
  }
;;

let sim_harness () =
  let module Sim = Cyclesim.With_interface (Harness_i) (Harness_o) in
  let sim = Sim.create harness in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let response = Buffer.create 64 in
  let complete = ref false in
  let cycle () =
    Cyclesim.cycle sim;
    if (not !complete) && Bits.to_bool !(out.out_valid)
    then (
      let byte = Bits.to_int_trunc !(out.out_data) in
      Buffer.add_char response (Char.of_int_exn byte);
      if Char.equal (Char.of_int_exn byte) Cobs.delimiter then complete := true)
  in
  let feed frame =
    String.iter
      ~f:(fun b ->
        inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
        inp.in_valid := Bits.vdd;
        cycle ())
      (Bytes.to_string frame);
    inp.in_valid := Bits.gnd
  in
  let run_until ~budget ready =
    let left = ref budget in
    while (not (ready ())) && !left > 0 do
      cycle ();
      Int.decr left
    done
  in
  let transact frame =
    Buffer.clear response;
    complete := false;
    feed frame;
    run_until ~budget:500 (fun () -> !complete);
    if Buffer.length response = 0
    then Stdio.print_endline "no response"
    else (
      match Control.decode_response (Buffer.contents_bytes response) with
      | Error e -> Stdio.printf "bad response: %s\n" e
      | Ok { op; status; data } ->
        let hex = Bytes_util.hex data in
        Stdio.printf
          "op %d status %s%s\n"
          op
          (Control.Status.to_string status)
          (if String.length hex = 0 then "" else " data " ^ hex))
  in
  sim, inp, out, response, complete, cycle, feed, run_until, transact
;;

let%expect_test "transactions against the cells" =
  let _sim, inp, _out, response, complete, cycle, feed, run_until, transact =
    sim_harness ()
  in
  (* the MIDI path takes the test message at once *)
  inp.doorbell_ready := Bits.vdd;
  (* the first request needs no start-up: the cells carry their defaults from the
     bitstream, thus there is no init walk to wait for *)
  transact
    (Control.encode_request (Read { addr = Control.Reg.base; len = Control.Reg.size }));
  [%expect {| op 1 status ok data 00 00 00 00 00 2a 00 00 00 64 7d 00 fa 00 02 00 |}];
  (* write, then read back *)
  transact
    (Control.encode_request
       (Write { addr = Control.Reg.velocity; data = Bytes.of_string "\x30" }));
  transact (Control.encode_request (Read { addr = Control.Reg.velocity; len = 1 }));
  [%expect {|
    op 2 status ok
    op 1 status ok data 30
    |}];
  (* the one-shot doorbell write: MIDI_MSG, MIDI_LEN, MIDI_GO ascending. The MIDI path is
     free, thus the message goes at once and MIDI_GO reads 0 again *)
  transact
    (Control.encode_request
       (Write
          { addr = Control.Reg.midi_msg; data = Bytes.of_string "\x92\x3C\x64\x03\x01" }));
  transact (Control.encode_request (Read { addr = Control.Reg.midi_msg; len = 5 }));
  [%expect {|
    op 2 status ok
    op 1 status ok data 92 3c 64 03 00
    |}];
  (* a ring against a MIDI path that cannot take it: MIDI_GO reads the wait state, and 0
     again after the release *)
  inp.doorbell_ready := Bits.gnd;
  transact
    (Control.encode_request
       (Write { addr = Control.Reg.midi_go; data = Bytes.of_string "\x01" }));
  transact (Control.encode_request (Read { addr = Control.Reg.midi_go; len = 1 }));
  inp.doorbell_ready := Bits.vdd;
  transact (Control.encode_request (Read { addr = Control.Reg.midi_go; len = 1 }));
  [%expect
    {|
    op 2 status ok
    op 1 status ok data 01
    op 1 status ok data 00
    |}];
  (* errors. The raw frames are the request payload: OP, ADDR, LEN. *)
  transact (Control.encode_request (Read { addr = Control.Reg.size; len = 1 }));
  transact (Cobs.encode (Bytes.of_string "\x07\x00\x01"));
  transact (Cobs.encode (Bytes.of_string "\x01\x00\xF0"));
  [%expect
    {|
    op 1 status bad-address
    op 7 status bad-op
    op 1 status bad-length
    |}];
  (* a frame with the wrong shape gets no response, and the engine recovers *)
  transact (Cobs.encode (Bytes.of_string "\xAA\xBB"));
  transact (Control.encode_request (Read { addr = Control.Reg.channel; len = 1 }));
  [%expect {|
    no response
    op 1 status ok data 02
    |}];
  (* the port ignores a frame that arrives while it sends: one response only, and the next
     request transacts normally *)
  Buffer.clear response;
  complete := false;
  feed (Control.encode_request (Read { addr = Control.Reg.velocity; len = 1 }));
  run_until ~budget:200 (fun () -> Buffer.length response > 0);
  feed (Control.encode_request (Read { addr = Control.Reg.channel; len = 1 }));
  run_until ~budget:500 (fun () -> !complete);
  (match Control.decode_response (Buffer.contents_bytes response) with
   | Ok { op; status = Control.Status.Ok; data } ->
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
  transact (Control.encode_request (Read { addr = Control.Reg.channel; len = 1 }));
  [%expect {| op 1 status ok data 02 |}]
;;

let%expect_test "the payload bound" =
  (* the discard rule of the host control: the port takes a payload of
     [Control.Constants.max_payload_bytes] and discards a longer one. Each payload byte is
     0x41, thus the frame decodes but its LEN field is out of range: an accepted frame
     answers Bad_length, and a discarded frame is silent. *)
  let try_payload n =
    let _sim, _inp, _out, _response, _complete, _cycle, _feed, _run_until, transact =
      sim_harness ()
    in
    Stdio.printf "payload %d: " n;
    transact (Cobs.encode (Bytes.make n 'A'))
  in
  try_payload (Control.Constants.max_payload_bytes - 1);
  try_payload Control.Constants.max_payload_bytes;
  try_payload (Control.Constants.max_payload_bytes + 1);
  [%expect
    {|
    payload 34: op 65 status bad-length
    payload 35: op 65 status bad-length
    payload 36: no response
    |}]
;;

let%expect_test "the waveform of a write and its commit" =
  (* a two-byte write to STEP_MS. [Apply] fills the shadow copy one byte in each cycle,
     and the [Commit] state moves the whole burst at one time. [busy] is the envelope of
     the transaction. The window opens at the tail of the request frame. The fsm tags
     follow [Fsm.t]: Recv Parse Apply Cmit Rspnd Send. *)
  let module Sim = Cyclesim.With_interface (Harness_i) (Harness_o) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all harness in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  String.iter
    ~f:(fun b ->
      inp.in_data := Bits.of_unsigned_int ~width:8 (Char.to_int b);
      inp.in_valid := Bits.vdd;
      Cyclesim.cycle sim)
    (Bytes.to_string
       (Control.encode_request
          (Write { addr = Control.Reg.step_ms; data = Bytes.of_string "\x11\x22" })));
  inp.in_valid := Bits.gnd;
  Cyclesim.cycle ~n:8 sim;
  let rules =
    let signals names =
      Hardcaml_waveterm.Display_rule.port_name_is_one_of
        names
        ~wave_format:Wave_format.(Bit_or Hex)
    in
    [ signals [ "clock"; "in_data"; "in_valid" ]
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "fsm"
        ~wave_format:
          (Wave_format.Index [ "Recv"; "Parse"; "Apply"; "Cmit"; "Rspnd"; "Send" ])
    ; signals [ "write_enable"; "write_address"; "write_data"; "commit"; "busy" ]
    ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:2
    ~start_cycle:6
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──│
    │               ││   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘  │
    │               ││───────────────────────────────────────────────────│
    │in_data        ││ 00                                                │
    │               ││───────────────────────────────────────────────────│
    │in_valid       ││──────┐                                            │
    │               ││      └────────────────────────────────────────────│
    │               ││──────┬─────┬───────────┬─────┬─────┬──────────────│
    │fsm            ││ Recv │Parse│Apply      │Cmit │Rspnd│Send          │
    │               ││──────┴─────┴───────────┴─────┴─────┴──────────────│
    │write_enable   ││            ┌───────────┐                          │
    │               ││────────────┘           └──────────────────────────│
    │               ││──────────────────┬─────┬──────────────────────────│
    │write_address  ││ C                │D    │E                         │
    │               ││──────────────────┴─────┴──────────────────────────│
    │               ││──────────────────┬─────┬──────────────────────────│
    │write_data     ││ 11               │22   │00                        │
    │               ││──────────────────┴─────┴──────────────────────────│
    │commit         ││                        ┌─────┐                    │
    │               ││────────────────────────┘     └────────────────────│
    │busy           ││      ┌────────────────────────────────────────────│
    │               ││──────┘                                            │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
