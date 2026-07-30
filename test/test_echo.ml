(* Integration test: the board top level echoes bytes through the host UART at the real
   divisor. The test drives the RsRx waveform bit by bit, samples RsTx on every cycle, and
   decodes the result with a software receiver. *)

let cpb = Mgen.Top.host_clocks_per_bit

let () =
  let open Hardcaml in
  let sim = Cyclesim.create (Mgen.Top.create ()) in
  let rxd = Cyclesim.in_port sim "RsRx" in
  let rstn = Cyclesim.in_port sim "btnCpuReset" in
  let txd = Cyclesim.out_port sim "RsTx" in
  let tx_wave = Buffer.create (64 * 1024) in
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
  (* reset, then idle *)
  rstn := Bits.gnd;
  level true 4;
  rstn := Bits.vdd;
  level true (2 * cpb);
  (* send the bytes back to back *)
  let message = "abc" in
  String.iter
    (fun c ->
      let byte = Char.code c in
      level false cpb;
      for i = 0 to 7 do
        level ((byte lsr i) land 1 = 1) cpb
      done;
      level true cpb)
    message;
  (* drain: the echo of the last byte needs the buffer and a frame *)
  level true (15 * cpb);
  (* decode the sampled RsTx waveform *)
  let wave = Buffer.contents tx_wave in
  let n = String.length wave in
  let bit i = if i < n && Char.equal wave.[i] '1' then 1 else 0 in
  let decoded = Buffer.create 8 in
  let i = ref 1 in
  while !i < n do
    if bit (!i - 1) = 1 && bit !i = 0
    then (
      (* a falling edge: sample the frame at the bit centers *)
      let center k = !i + (cpb / 2) + (k * cpb) in
      let byte = ref 0 in
      for k = 1 to 8 do
        byte := !byte lor (bit (center k) lsl (k - 1))
      done;
      if bit (center 0) = 0 && bit (center 9) = 1
      then Buffer.add_char decoded (Char.chr !byte);
      i := !i + (10 * cpb))
    else incr i
  done;
  let decoded = Buffer.contents decoded in
  assert (String.equal decoded message);
  Printf.printf "echo ok: %s\n" decoded
;;
