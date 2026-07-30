(* Integration test: ABI transactions through the board top level at the real UART
   divisor. The test drives the RsRx waveform bit by bit with frames from
   [Abi.encode_request], samples RsTx on every cycle, decodes the waveform with a software
   receiver, and parses the responses with [Abi.decode_response]. *)

let cpb = Mgen.Top.host_clocks_per_bit

let () =
  let open Hardcaml in
  let sim = Cyclesim.create (Mgen.Top.create ()) in
  let rxd = Cyclesim.in_port sim "RsRx" in
  let rstn = Cyclesim.in_port sim "btnCpuReset" in
  let txd = Cyclesim.out_port sim "RsTx" in
  let tx_wave = Buffer.create (1024 * 1024) in
  let cycle () =
    Buffer.add_char tx_wave (if Bits.to_bool !txd then '1' else '0');
    Cyclesim.cycle sim
  in
  let level b n =
    rxd := if b then Bits.vdd else Bits.gnd;
    for _ = 1 to n do
      cycle ()
    done
  in
  let send_frame frame =
    Bytes.iter
      (fun c ->
        let byte = Char.code c in
        level false cpb;
        for i = 0 to 7 do
          level ((byte lsr i) land 1 = 1) cpb
        done;
        level true cpb)
      frame
  in
  (* reset, then idle *)
  rstn := Bits.gnd;
  level true 4;
  rstn := Bits.vdd;
  level true (2 * cpb);
  (* write VELOCITY, then read it back *)
  send_frame
    (Mgen.Abi.encode_request
       (Write { addr = Mgen.Abi.Reg.Ctl.velocity; data = Bytes.of_string "\x42" }));
  level true (60 * cpb);
  send_frame
    (Mgen.Abi.encode_request (Read { addr = Mgen.Abi.Reg.Ctl.velocity; len = 1 }));
  level true (80 * cpb);
  (* recover the response bytes from the sampled RsTx waveform *)
  let wave = Buffer.contents tx_wave in
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
      then Buffer.add_char received (Char.chr !byte);
      i := !i + (10 * cpb))
    else incr i
  done;
  (* split the byte stream at the frame delimiters and parse *)
  let frames =
    String.split_on_char '\000' (Buffer.contents received)
    |> List.filter (fun s -> String.length s > 0)
    |> List.map (fun s -> Bytes.of_string (s ^ "\000"))
  in
  let show frame =
    match Mgen.Abi.decode_response frame with
    | Error e -> Printf.printf "bad response: %s\n" e
    | Ok { op; status; data } ->
      Printf.printf
        "op %d ok %b data %s\n"
        op
        (match status with
         | Mgen.Abi.Status.Ok -> true
         | _ -> false)
        (data
         |> Bytes.to_seq
         |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
         |> List.of_seq
         |> String.concat " ")
  in
  List.iter show frames;
  assert (List.length frames = 2);
  print_endline "txn ok"
;;
