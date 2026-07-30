open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; data : 'a [@bits 8]
    ; valid : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { txd : 'a
    ; busy : 'a
    }
  [@@deriving hardcaml]
end

let create ~clocks_per_bit (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
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
        (i.valid &: ~:busy)
        [ shift <-- concat_msb [ vdd; i.data; gnd ]; count <--. 10; baud <--. 0 ]
    ];
  { O.txd = mux2 busy (select shift.value ~high:0 ~low:0) vdd; busy }
;;

let%expect_test "one frame at 4 clocks per bit" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_bit:4) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  inp.data := Bits.of_unsigned_int ~width:8 0x55;
  inp.valid := Bits.vdd;
  Cyclesim.cycle sim;
  inp.valid := Bits.gnd;
  (* 10 bits of 4 cycles each, then 4 cycles of idle *)
  let wave = Buffer.create 44 in
  for _ = 1 to 44 do
    Buffer.add_string wave (if Bits.to_bool !(out.txd) then "1" else "0");
    Cyclesim.cycle sim
  done;
  Printf.printf "txd  %s\n" (Buffer.contents wave);
  Printf.printf "busy %b\n" (Bits.to_bool !(out.busy));
  [%expect {|
    txd  00001111000011110000111100001111000011111111
    busy false
    |}]
;;
