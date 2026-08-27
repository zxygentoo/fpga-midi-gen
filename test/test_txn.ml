(* Integration test: host-control transactions through the board top level at the real
   UART divisors. The test drives the RsRx waveform bit by bit with frames from
   [Control_frame.encode_request], samples RsTx and the MIDI line JD[0] on every cycle,
   decodes the waveforms with a software receiver, and parses the responses with
   [Control_frame.decode_response].

   The MIDI line is here as the silence it must keep: RUN rests at 0 through the whole
   test, thus JD must stay at its no-current level and carry no byte. The bytes of a run
   are not observable at this level for a useful price — era six draws a whole canvas
   before its first note, which is millions of cycles of the real board clock — thus
   [Midi_out] proves the line format and the socket chain tests prove the message stream. *)

open Base

let cpb = Mgen_nexys4.Top.host_clocks_per_bit
let midi_cpb = Mgen_nexys4.Top.midi_clocks_per_bit

(* the two lines run at different baud rates, thus each call names its own divisor *)
let decode_uart wave cpb =
  Bytes.to_string (Mgen_board.Uart_rx.For_test.decode_line wave ~clocks_per_bit:cpb)
;;

let hex s = Mgen_core.Bytes_util.hex (Bytes.of_string s)

let () =
  let open Hardcaml in
  (* The model seat takes an elaboration of drawn weights at the smallest geometry the era
     admits: the control path does not read the weights, and a test must not read a
     checkpoint that git ignores. The shape sizes the counters, the ROMs and the lanes,
     thus one group of one lane over the shortest canvas elaborates in a test. P stays at
     the era's 48 — the seat registers of the opening are the corpus's. *)
  let model = Mgen_diffusion.Quantized.Model.For_test.(init config ~seed:11) in
  let e = Mgen_diffusion.Elaboration.create model ~steps:4 ~lanes:1 ~walk:2 in
  let sim = Cyclesim.create (Mgen_nexys4.Top.create ~e ()) in
  let rxd = Cyclesim.in_port sim "RsRx" in
  let rstn = Cyclesim.in_port sim "btnCpuReset" in
  let sw = Cyclesim.in_port sim "sw" in
  let txd = Cyclesim.out_port sim "RsTx" in
  let jd = Cyclesim.out_port sim "JD" in
  let tx_wave = Buffer.create (1024 * 1024) in
  let jd_wave = Buffer.create (1024 * 1024) in
  let cycle () =
    Buffer.add_char tx_wave (if Bits.to_bool !txd then '1' else '0');
    Buffer.add_char jd_wave (if Bits.to_int_trunc !jd land 1 = 1 then '1' else '0');
    Cyclesim.cycle sim
  in
  let level b n =
    rxd := if b then Bits.vdd else Bits.gnd;
    for _ = 1 to n do
      cycle ()
    done
  in
  let send_frame frame =
    String.iter
      ~f:(fun c ->
        let byte = Char.to_int c in
        level false cpb;
        for i = 0 to 7 do
          level ((byte lsr i) land 1 = 1) cpb
        done;
        level true cpb)
      (Bytes.to_string frame)
  in
  (* The slide switches are the second writer of SEED, and this is the top-level statement
     of that rule: the panel holds a value, thus the section read answers with the
     switches and not with a stored default. *)
  sw := Bits.of_unsigned_int ~width:16 0xBEEF;
  (* reset, then idle *)
  rstn := Bits.gnd;
  level true 4;
  rstn := Bits.vdd;
  level true (2 * cpb);
  (* write VELOCITY, then read it back *)
  send_frame
    (Mgen_board.Control_frame.encode_request
       (Write
          { addr = Mgen_core.Control_intf.Reg.velocity; data = Bytes.of_string "\x42" }));
  level true (60 * cpb);
  send_frame
    (Mgen_board.Control_frame.encode_request
       (Read { addr = Mgen_core.Control_intf.Reg.velocity; len = 1 }));
  level true (80 * cpb);
  (* one read of the whole section, which is what [board_tool dump] sends *)
  let read_section () =
    send_frame
      (Mgen_board.Control_frame.encode_request
         (Read
            { addr = Mgen_core.Control_intf.Reg.base
            ; len = Mgen_core.Control_intf.Reg.size
            }))
  in
  read_section ();
  level true (200 * cpb);
  (* a switch moves: the panel writes the cell again, thus the read that follows states
     the new value and the rule holds while the board stands *)
  sw := Bits.of_unsigned_int ~width:16 0xBEE0;
  level true (2 * cpb);
  read_section ();
  level true (200 * cpb);
  (* split the response byte stream at the frame delimiters and parse *)
  let frames =
    String.split (decode_uart (Buffer.contents tx_wave) cpb) ~on:'\000'
    |> List.filter ~f:(fun s -> String.length s > 0)
    |> List.map ~f:(fun s -> Bytes.of_string (s ^ "\000"))
  in
  let show frame =
    match Mgen_board.Control_frame.decode_response frame with
    | Error e -> Stdio.printf "bad response: %s\n" e
    | Ok { op; status; data } ->
      Stdio.printf
        "op %d ok %b data %s\n"
        op
        (match status with
         | Mgen_core.Control_intf.Status.Ok -> true
         | _ -> false)
        (hex (Bytes.to_string data))
  in
  List.iter ~f:show frames;
  assert (List.length frames = 4);
  (* The two section reads carry the seed of the panel in their first four bytes, in the
     little-endian order of the cells: the switches at the power-on, and the switches
     after the move. No frame carries a stored default, because the panel writes SEED a
     few cycles after the power-on and the fastest transaction is thousands of cycles. *)
  let seed_bytes frame =
    match Mgen_board.Control_frame.decode_response frame with
    | Ok { data; _ } -> String.prefix (Bytes.to_string data) 4
    | Error e -> failwith e
  in
  assert (String.equal (seed_bytes (List.nth_exn frames 2)) "\xef\xbe\x00\x00");
  assert (String.equal (seed_bytes (List.nth_exn frames 3)) "\xe0\xbe\x00\x00");
  (* RUN rests at 0, thus the line must carry nothing and rest at its no-current level *)
  let midi = decode_uart (Buffer.contents jd_wave) midi_cpb in
  Stdio.printf "midi %s\n" (hex midi);
  assert (String.is_empty midi);
  (* the first sample is the port before the simulator has computed it one time, thus the
     idle level starts at the sample behind it *)
  assert (
    String.for_all (String.drop_prefix (Buffer.contents jd_wave) 1) ~f:(Char.equal '1'));
  (* the seven pins with no connection stay at 1 beside it *)
  assert (Bits.to_int_trunc !jd = 0xFF);
  Stdio.print_endline "txn ok"
;;
