open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; switches : 'a [@bits 16]
    ; seed : 'a [@bits 32]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { seed_write : 'a
    ; seed_value : 'a [@bits 32]
    ; digit : 'a [@bits 8]
    ; segment : 'a [@bits 7]
    }
  [@@deriving hardcaml]
end

(* The display is one set of segment wires and one anode for each digit, thus the block
   lights one digit at a time and the eye joins them. Below about 60 Hz the display
   flickers, and a scan that is too fast leaves the light of the digit before on the digit
   after. A slice of the free counter gives the sequence: one digit stands for 2^16
   cycles, which is 655 us at 100 MHz, and the eight digits give a scan of 5.24 ms at 191
   Hz. The Digilent reference asks for 1 ms to 16 ms. *)
let digit_clocks_log2 = 16
let digits = 8
let counter_bits = digit_clocks_log2 + Int.ceil_log2 digits

(* The seven segments of each hexadecimal digit, as the bits g f e d c b a, where 1 is a
   lamp that is lit. *)
let lamps =
  [ 0b0111111 (* 0 *)
  ; 0b0000110 (* 1 *)
  ; 0b1011011 (* 2 *)
  ; 0b1001111 (* 3 *)
  ; 0b1100110 (* 4 *)
  ; 0b1101101 (* 5 *)
  ; 0b1111101 (* 6 *)
  ; 0b0000111 (* 7 *)
  ; 0b1111111 (* 8 *)
  ; 0b1101111 (* 9 *)
  ; 0b1110111 (* A *)
  ; 0b1111100 (* b *)
  ; 0b0111001 (* C *)
  ; 0b1011110 (* d *)
  ; 0b1111001 (* E *)
  ; 0b1110001 (* F *)
  ]
;;

(* the segment pins of a nibble; they are active low, thus the table is inverted *)
let decode nibble = ~:(mux nibble (List.map lamps ~f:(of_unsigned_int ~width:7)))

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  (* the switches are asynchronous to the clock, and a slide switch bounces *)
  let synchronised = reg spec (reg spec i.switches) in
  (* The value that the change detector read last time. It has no power-on value, thus it
     is 0 and a panel that is not at zero writes the cell three cycles after the power-on:
     the design needs no first-time flag and no init walk. A panel at zero writes nothing
     and the cell keeps 0, which is the value of the panel. An enable on this register
     would be a no-op, because it holds [synchronised] whenever the two agree. *)
  let previous = reg spec synchronised in
  let scan = reg_fb spec ~width:counter_bits ~f:(fun count -> count +:. 1) in
  let index = select scan ~high:(counter_bits - 1) ~low:digit_clocks_log2 in
  let nibbles = split_lsb ~part_width:4 i.seed in
  assert (List.length nibbles = digits);
  { O.seed_write = synchronised <>: previous
  ; (* the synchronised value and not the pins: the strobe states that the synchroniser
       moved, thus the cell must take what it moved to *)
    seed_value = uresize synchronised ~width:32
  ; digit = ~:(binary_to_onehot index)
  ; segment = decode (mux index nibbles)
  }
;;

let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  sim, Cyclesim.inputs sim, Cyclesim.outputs sim
;;

let%expect_test "the panel writes the cell at the power-on" =
  (* the synchroniser and the value read last have no power-on value, thus the switches
     write the cell three cycles after the power-on and no first-time flag is necessary *)
  let sim, inp, out = harness () in
  inp.switches := Bits.of_unsigned_int ~width:16 0x1234;
  List.iter (List.range 0 5) ~f:(fun cycle ->
    Cyclesim.cycle sim;
    Stdio.printf
      "%d  seed_write %d  seed_value %08x\n"
      cycle
      (Bits.to_int_trunc !(out.seed_write))
      (Bits.to_int_trunc !(out.seed_value)));
  [%expect
    {|
    0  seed_write 0  seed_value 00000000
    1  seed_write 1  seed_value 00001234
    2  seed_write 0  seed_value 00001234
    3  seed_write 0  seed_value 00001234
    4  seed_write 0  seed_value 00001234
    |}]
;;

(* drive the panel at [value] for [cycles] cycles, and give the strobes of that time *)
let hold (sim, (inp : _ I.t), (out : _ O.t)) value ~cycles =
  inp.switches := Bits.of_unsigned_int ~width:16 value;
  List.count (List.range 0 cycles) ~f:(fun _ ->
    Cyclesim.cycle sim;
    Bits.to_bool !(out.seed_write))
;;

let%expect_test "the switches at zero write nothing" =
  (* the rest position of the panel is 0 and the cell is 0, thus nothing fires and the
     cell holds the value of the panel: the power-on case and the moved case give one
     result *)
  let h = harness () in
  Stdio.printf "strobes %d\n" (hold h 0x0000 ~cycles:8);
  [%expect {| strobes 0 |}]
;;

let%expect_test "one move writes one time" =
  let h = harness () in
  let _, _, (out : _ O.t) = h in
  let show tag strobes =
    Stdio.printf
      "%-16s strobes %d  seed_value %08x\n"
      tag
      strobes
      (Bits.to_int_trunc !(out.seed_value))
  in
  show "the power-on" (hold h 0x1234 ~cycles:5);
  show "one switch" (hold h 0x1235 ~cycles:5);
  show "and it stands" (hold h 0x1235 ~cycles:5);
  [%expect
    {|
    the power-on     strobes 1  seed_value 00001234
    one switch       strobes 1  seed_value 00001235
    and it stands    strobes 0  seed_value 00001235
    |}]
;;

let%expect_test "a bounce writes each edge and settles on the last" =
  (* A slide switch bounces, thus one move can give more than one write. Each write is a
     whole seed and the last one stands, thus the bounce needs no debounce: the panel ends
     at the value a person set. *)
  let h = harness () in
  let _, _, (out : _ O.t) = h in
  let rest = hold h 0x0000 ~cycles:4 in
  let bounce =
    List.fold [ 0x0001; 0x0000; 0x0001; 0x0000 ] ~init:0 ~f:(fun strobes value ->
      strobes + hold h value ~cycles:1)
  in
  let settled = hold h 0x0001 ~cycles:5 in
  Stdio.printf
    "strobes %d  seed_value %08x\n"
    (rest + bounce + settled)
    (Bits.to_int_trunc !(out.seed_value));
  [%expect {| strobes 5  seed_value 00000001 |}]
;;

(* One digit, drawn from its segment pins: the three lines of the lamps that are lit. The
   letters a to g are the bits 0 to 6, and a pin is active low. *)
let draw segment =
  let bar letter mark =
    let bit = Char.to_int letter - Char.to_int 'a' in
    if (Bits.to_int_trunc segment lsr bit) land 1 = 0 then mark else " "
  in
  [ " " ^ bar 'a' "_" ^ " "
  ; bar 'f' "|" ^ bar 'g' "_" ^ bar 'b' "|"
  ; bar 'e' "|" ^ bar 'd' "_" ^ bar 'c' "|"
  ]
;;

let%expect_test "the sixteen digits" =
  (* the scan index stands at 0 for the first 65536 cycles, thus digit 0 shows the low
     nibble of the seed and one cycle of each value draws the whole table *)
  let sim, inp, out = harness () in
  let rows = Array.init 3 ~f:(fun _ -> Buffer.create 80) in
  List.iter (List.range 0 16) ~f:(fun nibble ->
    inp.seed := Bits.of_unsigned_int ~width:32 nibble;
    Cyclesim.cycle sim;
    List.iteri (draw !(out.segment)) ~f:(fun row line ->
      Buffer.add_string rows.(row) (line ^ " ")));
  Array.iter rows ~f:(fun row -> Stdio.print_endline (Buffer.contents row));
  [%expect
    {|
     _       _   _       _   _   _   _   _   _       _       _   _
    | |   |  _|  _| |_| |_  |_    | |_| |_| |_| |_  |    _| |_  |_
    |_|   | |_   _|   |  _| |_|   | |_|  _| | | |_| |_  |_| |_  |
    |}]
;;

let%expect_test "the waveform of one scan" =
  (* One whole scan at the rate of the board, with the seed at 0xF2A10000. One character
     is 8192 cycles, thus one digit stands for eight characters and the picture is 5.24
     ms.

     [digit] is one hot and active low, and it walks from the digit at the right (FE) to
     the digit at the left (7F). [segment] holds 40 — the lamps of "0" — through the four
     digits of the low half, and then it draws 1, A, 2 and F as 79, 08, 24 and 0E. Thus
     the display states F2A10000 from the left, which is the value of the cell. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  inp.seed := Bits.of_unsigned_int ~width:32 0xF2A10000;
  Cyclesim.cycle ~n:(1 lsl counter_bits) sim;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Hex)
          [ "digit"; "segment" ]
      ]
    ~show_digest:false
    ~wave_width:(-8192)
    ~display_width:92
    waves;
  [%expect
    {|
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────────────────┐
    │                  ││────────┬───────┬───────┬───────┬───────┬───────┬───────┬───────      │
    │digit             ││ FE     │FD     │FB     │F7     │EF     │DF     │BF     │7F           │
    │                  ││────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────      │
    │                  ││────────────────────────────────┬───────┬───────┬───────┬───────      │
    │segment           ││ 40                             │79     │08     │24     │0E           │
    │                  ││────────────────────────────────┴───────┴───────┴───────┴───────      │
    └──────────────────┘└──────────────────────────────────────────────────────────────────────┘
    |}]
;;
