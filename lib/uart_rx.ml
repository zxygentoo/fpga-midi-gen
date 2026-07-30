open Hardcaml
open Signal

type t =
  { data : Signal.t
  ; valid : Signal.t
  }

let create ~clocks_per_bit ~clock ~clear ~rxd =
  let spec = Reg_spec.create ~clock ~clear () in
  let rxd = reg spec (reg spec rxd) in
  let baud_width = address_bits_for clocks_per_bit in
  let open Always in
  let busy = Variable.reg spec ~width:1 in
  let baud = Variable.reg spec ~width:baud_width in
  (* 0: the start bit; 1 to 8: the data bits; 9: the stop bit *)
  let count = Variable.reg spec ~width:4 in
  let shift = Variable.reg spec ~width:8 in
  let valid = Variable.wire ~default:gnd () in
  (* the first sample point is the center of the start bit *)
  let target =
    mux2
      (count.value ==:. 0)
      (of_unsigned_int ~width:baud_width ((clocks_per_bit / 2) - 1))
      (of_unsigned_int ~width:baud_width (clocks_per_bit - 1))
  in
  compile
    [ if_
        busy.value
        [ if_
            (baud.value ==: target)
            [ baud <--. 0
            ; if_
                (count.value ==:. 0)
                [ if_ rxd [ busy <-- gnd ] [ count <--. 1 ] ]
                [ if_
                    (count.value ==:. 9)
                    [ busy <-- gnd; when_ rxd [ valid <-- vdd ] ]
                    [ shift <-- concat_msb [ rxd; select shift.value ~high:7 ~low:1 ]
                    ; count <-- count.value +:. 1
                    ]
                ]
            ]
            [ baud <-- baud.value +:. 1 ]
        ]
        [ when_ ~:rxd [ busy <-- vdd; baud <--. 0; count <--. 0 ] ]
    ];
  { data = shift.value; valid = valid.value }
;;

let%expect_test "frames at 4 clocks per bit" =
  let clocks_per_bit = 4 in
  let circuit =
    let t =
      create
        ~clocks_per_bit
        ~clock:(input "clock" 1)
        ~clear:(input "clear" 1)
        ~rxd:(input "rxd" 1)
    in
    Circuit.create_exn ~name:"uart_rx" [ output "data" t.data; output "valid" t.valid ]
  in
  let sim = Cyclesim.create circuit in
  let rxd = Cyclesim.in_port sim "rxd" in
  let data = Cyclesim.out_port sim "data" in
  let valid = Cyclesim.out_port sim "valid" in
  rxd := Bits.vdd;
  Cyclesim.cycle ~n:8 sim;
  let send_frame ~stop byte =
    let level b = rxd := if b then Bits.vdd else Bits.gnd in
    let bit b =
      level b;
      for _ = 1 to clocks_per_bit do
        Cyclesim.cycle sim;
        if Bits.to_bool !valid then Printf.printf "byte %02x\n" (Bits.to_int_trunc !data)
      done
    in
    bit false;
    for i = 0 to 7 do
      bit ((byte lsr i) land 1 = 1)
    done;
    bit stop;
    (* idle so the synchronizer and the state settle *)
    level true;
    for _ = 1 to 8 do
      Cyclesim.cycle sim;
      if Bits.to_bool !valid then Printf.printf "byte %02x\n" (Bits.to_int_trunc !data)
    done
  in
  send_frame ~stop:true 0xa3;
  send_frame ~stop:true 0x00;
  [%expect {|
    byte a3
    byte 00
    |}];
  (* a frame with a bad stop bit is silent *)
  send_frame ~stop:false 0x5a;
  [%expect {| |}]
;;
