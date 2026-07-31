(* Integration test: host-control transactions through the board top level at the real
   UART divisors. The test drives the RsRx waveform bit by bit with frames from
   [Control.encode_request], samples RsTx and the MIDI line JD[0] on every cycle, decodes
   the waveforms with a software receiver, and parses the responses with
   [Control.decode_response]. The doorbell write must put the test message on the MIDI
   line at 31250 baud. *)

open Base

let cpb = Mgen.Top.host_clocks_per_bit
let midi_cpb = Mgen.Top.midi_clocks_per_bit

(* recover the bytes of an 8N1 waveform sampled one character per cycle *)
let decode_uart wave cpb =
  let n = String.length wave in
  let bit i = if i < n && Char.equal wave.[i] '1' then 1 else 0 in
  let received = Buffer.create 16 in
  let i = ref 1 in
  while !i < n do
    if bit (!i - 1) = 1 && bit !i = 0
    then (
      let center k = !i + (cpb / 2) + (k * cpb) in
      let byte = ref 0 in
      for k = 1 to 8 do
        byte := !byte lor (bit (center k) lsl (k - 1))
      done;
      if bit (center 0) = 0 && bit (center 9) = 1
      then Buffer.add_char received (Char.of_int_exn !byte);
      i := !i + (10 * cpb))
    else Int.incr i
  done;
  Buffer.contents received
;;

let hex s = Mgen.Bytes_util.hex (Bytes.of_string s)

let () =
  let open Hardcaml in
  let sim = Cyclesim.create (Mgen.Top.create ()) in
  let rxd = Cyclesim.in_port sim "RsRx" in
  let rstn = Cyclesim.in_port sim "btnCpuReset" in
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
  (* reset, then idle *)
  rstn := Bits.gnd;
  level true 4;
  rstn := Bits.vdd;
  level true (2 * cpb);
  (* write VELOCITY, then read it back *)
  send_frame
    (Mgen.Control.encode_request
       (Write { addr = Mgen.Control.Reg.velocity; data = Bytes.of_string "\x42" }));
  level true (60 * cpb);
  send_frame
    (Mgen.Control.encode_request (Read { addr = Mgen.Control.Reg.velocity; len = 1 }));
  level true (80 * cpb);
  (* the one-shot doorbell write: the message must appear on the MIDI line *)
  send_frame
    (Mgen.Control.encode_request
       (Write
          { addr = Mgen.Control.Reg.midi_msg
          ; data = Bytes.of_string "\x92\x3C\x64\x03\x01"
          }));
  level true (40 * midi_cpb);
  (* split the response byte stream at the frame delimiters and parse *)
  let frames =
    String.split (decode_uart (Buffer.contents tx_wave) cpb) ~on:'\000'
    |> List.filter ~f:(fun s -> String.length s > 0)
    |> List.map ~f:(fun s -> Bytes.of_string (s ^ "\000"))
  in
  let show frame =
    match Mgen.Control.decode_response frame with
    | Error e -> Stdio.printf "bad response: %s\n" e
    | Ok { op; status; data } ->
      Stdio.printf
        "op %d ok %b data %s\n"
        op
        (match status with
         | Mgen.Control.Status.Ok -> true
         | _ -> false)
        (hex (Bytes.to_string data))
  in
  List.iter ~f:show frames;
  assert (List.length frames = 3);
  let midi = decode_uart (Buffer.contents jd_wave) midi_cpb in
  Stdio.printf "midi %s\n" (hex midi);
  assert (String.equal midi "\x92\x3C\x64");
  Stdio.print_endline "txn ok"
;;
