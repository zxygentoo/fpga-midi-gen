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
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { out_data : 'a [@bits 8]
    ; out_valid : 'a
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
  (* the names put the machine and the write strobe into the waveform tests; the port
     already owns the name [state], thus the machine shows as [fsm] *)
  let _ = sm.current -- "fsm" in
  let _ = write_enable.value -- "write_enable" in
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
  (* response byte [j]: the op echo, the status, then the cells *)
  let response_byte j =
    mux2
      (j ==:. 0)
      (op |: of_unsigned_int ~width:8 0x80)
      (mux2 (j ==:. 1) status.value regfile.read_data)
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
    ];
  { O.out_data = encoder.data
  ; out_valid = encoder.valid
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
  (* the one-shot doorbell write: MSG, MSG_LEN, MSG_GO ascending *)
  transact
    (Abi.encode_request
       (Write { addr = Abi.Reg.Ctl.msg; data = Bytes.of_string "\x92\x3C\x64\x03\x01" }));
  [%expect {| op 2 status ok |}];
  transact (Abi.encode_request (Read { addr = Abi.Reg.Ctl.msg; len = 5 }));
  [%expect {| op 1 status ok data 92 3c 64 03 01 |}];
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
