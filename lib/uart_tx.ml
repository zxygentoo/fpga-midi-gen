open Hardcaml
open Signal

type t =
  { txd : Signal.t
  ; busy : Signal.t
  }

let create ~clocks_per_bit ~clock ~clear ~data ~valid =
  let spec = Reg_spec.create ~clock ~clear () in
  let open Always in
  (* start bit, 8 data bits, stop bit; the lsb goes out first *)
  let shift = Variable.reg spec ~width:10 in
  let count = Variable.reg spec ~width:4 in
  let baud = Variable.reg spec ~width:(address_bits_for clocks_per_bit) in
  let busy = count.value <>:. 0 in
  let bit_end = baud.value ==:. clocks_per_bit - 1 in
  compile
    [ when_
        busy
        [ if_
            bit_end
            [ baud <--. 0
            ; count <-- count.value -:. 1
            ; shift <-- concat_msb [ vdd; select shift.value ~high:9 ~low:1 ]
            ]
            [ baud <-- baud.value +:. 1 ]
        ]
    ; when_
        (valid &: ~:busy)
        [ shift <-- concat_msb [ vdd; data; gnd ]; count <--. 10; baud <--. 0 ]
    ];
  { txd = mux2 busy (select shift.value ~high:0 ~low:0) vdd; busy }
;;

let%expect_test "one frame at 4 clocks per bit" =
  let clocks_per_bit = 4 in
  let circuit =
    let t =
      create
        ~clocks_per_bit
        ~clock:(input "clock" 1)
        ~clear:(input "clear" 1)
        ~data:(input "data" 8)
        ~valid:(input "valid" 1)
    in
    Circuit.create_exn ~name:"uart_tx" [ output "txd" t.txd; output "busy" t.busy ]
  in
  let sim = Cyclesim.create circuit in
  let data = Cyclesim.in_port sim "data" in
  let valid = Cyclesim.in_port sim "valid" in
  let txd = Cyclesim.out_port sim "txd" in
  let busy = Cyclesim.out_port sim "busy" in
  data := Bits.of_unsigned_int ~width:8 0x55;
  valid := Bits.vdd;
  Cyclesim.cycle sim;
  valid := Bits.gnd;
  (* 10 bits of 4 cycles each, then 4 cycles of idle *)
  let wave = Buffer.create 44 in
  for _ = 1 to 44 do
    Buffer.add_string wave (if Bits.to_bool !txd then "1" else "0");
    Cyclesim.cycle sim
  done;
  Printf.printf "txd  %s\n" (Buffer.contents wave);
  Printf.printf "busy %b\n" (Bits.to_bool !busy);
  [%expect {|
    txd  00001111000011110000111100001111000011111111
    busy false
    |}]
;;
