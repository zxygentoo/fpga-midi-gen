open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; rx_data : 'a [@bits 8]
    ; rx_valid : 'a
    ; tx_busy : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { tx_data : 'a [@bits 8]
    ; tx_valid : 'a
    ; state : 'a [@bits 2]
    }
  [@@deriving hardcaml]
end

module State = struct
  let init = 0
  let ready = 1
  let busy = 2
end

(* The power-on values of the control cells, from the ABI constants. The bridge writes
   them into the register file in the init state. *)
let defaults =
  let d = Array.make Abi.Reg.Ctl.size 0 in
  let set_byte addr value = d.(addr - Abi.Reg.Ctl.base) <- value land 0xff in
  let set_bytes addr n value =
    for i = 0 to n - 1 do
      set_byte (addr + i) (value lsr (8 * i))
    done
  in
  set_byte Abi.Reg.Ctl.channel Abi.Default.channel;
  set_bytes Abi.Reg.Ctl.step_ms 2 Abi.Default.step_ms;
  set_bytes Abi.Reg.Ctl.gate_ms 2 Abi.Default.gate_ms;
  set_byte Abi.Reg.Ctl.velocity Abi.Default.velocity;
  set_bytes Abi.Reg.Ctl.seed 4 Abi.Default.seed;
  d
;;

(* payload capacity: the header plus the largest write burst *)
let max_payload = 4 + Abi.Limits.max_data_len
let buffer_size = 64

module Ctl_regfile = Regfile.Make (struct
    let size = Abi.Reg.Ctl.size
  end)

(* the FSM states; init is 0, the value of the state register at power-on *)
let s_init = 0
let s_receive = 1
let s_parse = 2
let s_apply = 3
let s_start = 4
let s_wait = 5

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let state = Variable.reg spec ~width:3 in
  let wr_idx = Variable.reg spec ~width:7 in
  let drop = Variable.reg spec ~width:1 in
  let hdr = Array.init 4 (fun _ -> Variable.reg spec ~width:8) in
  let plen = Variable.reg spec ~width:7 in
  let addr = Variable.reg spec ~width:16 in
  let len = Variable.reg spec ~width:8 in
  let status = Variable.reg spec ~width:8 in
  let resp_len = Variable.reg spec ~width:7 in
  let apply_idx = Variable.reg spec ~width:6 in
  let init_idx = Variable.reg spec ~width:4 in
  let rf_we = Variable.wire ~default:gnd () in
  let cobs_start = Variable.wire ~default:gnd () in
  let in_state k = state.value ==:. k in
  let goto k = state <--. k in
  (* the decoder *)
  let decoder =
    Cobs_decoder.create
      { Cobs_decoder.I.clock = i.clock
      ; clear = i.clear
      ; in_data = i.rx_data
      ; in_valid = i.rx_valid
      }
  in
  (* the payload buffer *)
  let capture = in_state s_receive &: decoder.out_valid &: ~:(drop.value) in
  let ram_q =
    (multiport_memory
       buffer_size
       ~write_ports:
         [| { Write_port.write_clock = i.clock
            ; write_address = uresize wr_idx.value ~width:6
            ; write_enable = capture
            ; write_data = decoder.out_data
            }
         |]
       ~read_addresses:[| uresize (apply_idx.value +:. 4) ~width:6 |]).(0)
  in
  (* the encoder, fed by a registered read of the response bytes *)
  let cobs_rd_data = wire 8 in
  let encoder =
    Cobs_encoder.create
      { Cobs_encoder.I.clock = i.clock
      ; clear = i.clear
      ; start = cobs_start.value
      ; length = resp_len.value
      ; rd_data = cobs_rd_data
      ; tx_busy = i.tx_busy
      }
  in
  (* the register file *)
  let apply_target =
    select addr.value ~high:3 ~low:0 +: select apply_idx.value ~high:3 ~low:0
  in
  let resp_cell_idx =
    select addr.value ~high:3 ~low:0 +: select (encoder.rd_addr -:. 2) ~high:3 ~low:0
  in
  let regfile =
    Ctl_regfile.create
      { Ctl_regfile.I.clock = i.clock
      ; clear = i.clear
      ; write_enable = rf_we.value
      ; address =
          mux2
            (in_state s_init)
            init_idx.value
            (mux2 (in_state s_apply) apply_target resp_cell_idx)
      ; write_data =
          mux2
            (in_state s_init)
            (mux
               init_idx.value
               (List.map (of_unsigned_int ~width:8) (Array.to_list defaults)))
            ram_q
      }
  in
  (* response byte [j]: the op echo, the status, then the cells *)
  let resp_byte j =
    mux2
      (j ==:. 0)
      (hdr.(0).value |: of_unsigned_int ~width:8 0x80)
      (mux2 (j ==:. 1) status.value regfile.read_data)
  in
  assign cobs_rd_data (reg spec (resp_byte encoder.rd_addr));
  (* header fields *)
  let h_op = hdr.(0).value in
  let h_addr = hdr.(2).value @: hdr.(1).value in
  let h_len = hdr.(3).value in
  (* the range check, in 17 bits so the top of the space cannot wrap *)
  let range_end = uresize h_addr ~width:17 +: uresize h_len ~width:17 in
  let addr_ok =
    h_addr >=:. Abi.Reg.Ctl.base &: (range_end <=:. Abi.Reg.Ctl.base + Abi.Reg.Ctl.size)
  in
  let len_ok = h_len >=:. 1 &: (h_len <=:. Abi.Limits.max_data_len) in
  let is_read = h_op ==:. Abi.Op.read in
  let is_write = h_op ==:. Abi.Op.write in
  (* a frame with the wrong shape gets no response *)
  let structural_ok =
    mux2
      is_write
      (uresize h_len ~width:7 +:. 4 ==: plen.value)
      (mux2 is_read (plen.value ==:. 4) vdd)
  in
  let reset_frame = proc [ wr_idx <--. 0; drop <-- gnd ] in
  compile
    [ when_
        (in_state s_init)
        [ (* one default each cycle; the cells are not valid before the end *)
          rf_we <-- vdd
        ; init_idx <-- init_idx.value +:. 1
        ; when_ (init_idx.value ==:. Abi.Reg.Ctl.size - 1) [ goto s_receive ]
        ]
    ; when_
        (in_state s_receive)
        [ when_
            (decoder.out_valid &: ~:(drop.value))
            [ wr_idx <-- wr_idx.value +:. 1
            ; when_ (wr_idx.value ==:. max_payload) [ drop <-- vdd ]
            ; proc
                (List.init 4 (fun k ->
                   when_ (wr_idx.value ==:. k) [ hdr.(k) <-- decoder.out_data ]))
            ]
        ; when_ decoder.abort [ reset_frame ]
        ; when_
            decoder.frame_end
            [ if_
                (~:(drop.value) &: (wr_idx.value >=:. 4))
                [ plen <-- wr_idx.value; goto s_parse ]
                [ reset_frame ]
            ]
        ]
    ; when_
        (in_state s_parse)
        [ reset_frame
        ; addr <-- h_addr
        ; len <-- h_len
        ; apply_idx <--. 0
        ; if_
            ~:structural_ok
            [ goto s_receive ]
            [ status <--. Abi.Status.to_code Abi.Status.Ok
            ; resp_len <--. 2
            ; if_
                ~:len_ok
                [ status <--. Abi.Status.to_code Abi.Status.Bad_length; goto s_start ]
                [ if_
                    ~:(is_read |: is_write)
                    [ status <--. Abi.Status.to_code Abi.Status.Bad_op; goto s_start ]
                    [ if_
                        ~:addr_ok
                        [ status <--. Abi.Status.to_code Abi.Status.Bad_address
                        ; goto s_start
                        ]
                        [ if_
                            is_read
                            [ resp_len <-- uresize h_len ~width:7 +:. 2; goto s_start ]
                            [ goto s_apply ]
                        ]
                    ]
                ]
            ]
        ]
    ; when_
        (in_state s_apply)
        [ (* one cell each cycle, in the sequence of increasing addresses *)
          rf_we <-- vdd
        ; apply_idx <-- apply_idx.value +:. 1
        ; when_ (apply_idx.value +:. 1 ==: uresize len.value ~width:6) [ goto s_start ]
        ]
    ; when_ (in_state s_start) [ cobs_start <-- vdd; goto s_wait ]
    ; when_ (in_state s_wait) [ when_ ~:(encoder.busy) [ goto s_receive ] ]
    ];
  { O.tx_data = encoder.tx_data
  ; tx_valid = encoder.tx_valid
  ; state =
      mux2
        (in_state s_init)
        (of_unsigned_int ~width:2 State.init)
        (mux2
           (in_state s_receive)
           (of_unsigned_int ~width:2 State.ready)
           (of_unsigned_int ~width:2 State.busy))
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
  (* the bridge loads the defaults, then reports ready *)
  let budget = ref 40 in
  while Bits.to_int_trunc !(out.state) <> State.ready && !budget > 0 do
    Cyclesim.cycle sim;
    budget := !budget - 1
  done;
  Printf.printf "ready after init: %b\n" (Bits.to_int_trunc !(out.state) = State.ready);
  [%expect {| ready after init: true |}];
  let response = Buffer.create 64 in
  let complete = ref false in
  let cycle () =
    Cyclesim.cycle sim;
    if (not !complete) && Bits.to_bool !(out.tx_valid)
    then (
      let byte = Bits.to_int_trunc !(out.tx_data) in
      Buffer.add_char response (Char.chr byte);
      if byte = 0 then complete := true)
  in
  let transact frame =
    Buffer.clear response;
    complete := false;
    Bytes.iter
      (fun b ->
        inp.rx_data := Bits.of_unsigned_int ~width:8 (Char.code b);
        inp.rx_valid := Bits.vdd;
        cycle ())
      frame;
    inp.rx_valid := Bits.gnd;
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
