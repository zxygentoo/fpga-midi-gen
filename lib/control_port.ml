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
      (fun (address, width, value) ->
        List.init width (fun k -> address + k, (value lsr (8 * k)) land 0xff))
      fields
  in
  List.init Abi.Reg.Ctl.size (fun k ->
    Option.value ~default:0 (List.assoc_opt (Abi.Reg.Ctl.base + k) bytes))
;;

(* payload capacity: the header plus the largest write burst *)
let max_payload = 4 + Abi.Limits.max_data_len
let buffer_size = 64

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
  let header = Array.init 4 (fun _ -> Variable.reg spec ~width:8) in
  let request_length = Variable.reg spec ~width:7 in
  let target_address = Variable.reg spec ~width:16 in
  let data_length = Variable.reg spec ~width:8 in
  let status = Variable.reg spec ~width:8 in
  let response_length = Variable.reg spec ~width:7 in
  let apply_index = Variable.reg spec ~width:6 in
  let init_index = Variable.reg spec ~width:4 in
  let write_enable = Variable.wire ~default:gnd () in
  let frame_start = Variable.wire ~default:gnd () in
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
            ; write_address = uresize capture_index.value ~width:6
            ; write_enable = capture
            ; write_data = decoder.out_data
            }
         |]
       ~read_addresses:[| uresize (apply_index.value +:. 4) ~width:6 |]).(0)
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
  (* the register file *)
  let apply_target =
    select target_address.value ~high:3 ~low:0 +: select apply_index.value ~high:3 ~low:0
  in
  let response_cell =
    select target_address.value ~high:3 ~low:0
    +: select (encoder.address -:. 2) ~high:3 ~low:0
  in
  let regfile =
    Ctl_regfile.create
      { Ctl_regfile.I.clock = i.clock
      ; clear = i.clear
      ; write_enable = write_enable.value
      ; address =
          mux2
            (sm.is Init)
            init_index.value
            (mux2 (sm.is Apply) apply_target response_cell)
      ; write_data =
          mux2
            (sm.is Init)
            (mux init_index.value (List.map (of_unsigned_int ~width:8) defaults))
            payload_byte
      }
  in
  (* response byte [j]: the op echo, the status, then the cells *)
  let response_byte j =
    mux2
      (j ==:. 0)
      (header.(0).value |: of_unsigned_int ~width:8 0x80)
      (mux2 (j ==:. 1) status.value regfile.read_data)
  in
  assign response_data (reg spec (response_byte encoder.address));
  (* header fields *)
  let op = header.(0).value in
  let header_address = header.(2).value @: header.(1).value in
  let header_length = header.(3).value in
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
                    (List.init 4 (fun k ->
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
            ; target_address <-- header_address
            ; data_length <-- header_length
            ; apply_index <--. 0
            ; if_
                ~:structural_ok
                [ sm.set_next Receive ]
                [ status <--. Abi.Status.to_code Abi.Status.Ok
                ; response_length <--. 2
                ; if_
                    ~:length_ok
                    [ status <--. Abi.Status.to_code Abi.Status.Bad_length
                    ; sm.set_next Respond
                    ]
                    [ if_
                        ~:(is_read |: is_write)
                        [ status <--. Abi.Status.to_code Abi.Status.Bad_op
                        ; sm.set_next Respond
                        ]
                        [ if_
                            ~:address_ok
                            [ status <--. Abi.Status.to_code Abi.Status.Bad_address
                            ; sm.set_next Respond
                            ]
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
                (apply_index.value +:. 1 ==: uresize data_length.value ~width:6)
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
  Printf.printf
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
      Buffer.add_char response (Char.chr byte);
      if byte = 0 then complete := true)
  in
  let transact frame =
    Buffer.clear response;
    complete := false;
    Bytes.iter
      (fun b ->
        inp.in_data := Bits.of_unsigned_int ~width:8 (Char.code b);
        inp.in_valid := Bits.vdd;
        cycle ())
      frame;
    inp.in_valid := Bits.gnd;
    let budget = ref 500 in
    while (not !complete) && !budget > 0 do
      cycle ();
      budget := !budget - 1
    done;
    if Buffer.length response = 0
    then print_endline "no response"
    else (
      match Abi.decode_response (Buffer.to_bytes response) with
      | Error e -> Printf.printf "bad response: %s\n" e
      | Ok { op; status; data } ->
        let hex =
          data
          |> Bytes.to_seq
          |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
          |> List.of_seq
          |> String.concat " "
        in
        Printf.printf
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
  [%expect {| op 1 status ok data 02 |}]
;;
