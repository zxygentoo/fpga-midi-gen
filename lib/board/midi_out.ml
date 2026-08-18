open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; message : 'a Midi.Rtl.Message.t
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { serial : 'a
    ; ready : 'a
    }
  [@@deriving hardcaml]
end

(* [Idle] takes a message; [Send] gives one byte to the transmitter in each free cycle;
   [Drain] holds [ready] at 0 until the last stop bit is on the line *)
module State = struct
  type t =
    | Idle
    (** the first constructor encodes as 0, the value of the state register at power-on
        and at clear *)
    | Send
    | Drain
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let index_bits = address_bits_for Midi.max_message_bytes

let create ~clocks_per_bit (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let data = Variable.reg spec ~width:(Midi.max_message_bytes * 8) in
  let len = Variable.reg spec ~width:index_bits in
  let index = Variable.reg spec ~width:index_bits in
  (* the transmitter answers in the same cycle, thus the wire breaks the order *)
  let tx_busy = wire 1 in
  let ready = sm.is Idle in
  (* the names put the machine, the cursor and the envelope into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = index.value -- "index" in
  let _ = ready -- "ready" in
  let byte =
    mux
      index.value
      (List.init Midi.max_message_bytes ~f:(fun k ->
         select data.value ~high:((k * 8) + 7) ~low:(k * 8)))
  in
  (* the transmitter is free, thus it takes the byte in this cycle *)
  let load = sm.is Send &: ~:tx_busy in
  (* the walk ends at the length, and also at the last byte. Thus a length outside 1 to
     [Midi.max_message_bytes] cannot hold the block. *)
  let last =
    index.value +:. 1 ==: len.value |: (index.value ==:. Midi.max_message_bytes - 1)
  in
  compile
    [ sm.switch
        [ ( Idle
          , [ when_
                i.message.valid
                [ data <-- i.message.data
                ; len <-- uresize i.message.len ~width:index_bits
                ; index <--. 0
                ; sm.set_next Send
                ]
            ] )
        ; ( Send
          , [ when_
                ~:tx_busy
                [ index <-- index.value +:. 1; when_ last [ sm.set_next Drain ] ]
            ] )
          (* the transmitter takes one cycle to show [busy], thus the first cycle of
             [Drain] already sees it *)
        ; Drain, [ when_ ~:tx_busy [ sm.set_next Idle ] ]
        ]
    ];
  let tx =
    Uart_tx.create
      ~clocks_per_bit
      { Uart_tx.I.clock = i.clock; clear = i.clear; data = byte; valid = load }
  in
  assign tx_busy tx.busy;
  { O.serial = tx.serial; ready }
;;

let clocks_per_bit = 4

let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_bit) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let wave = Buffer.create 512 in
  let cycle () =
    Cyclesim.cycle sim;
    Buffer.add_char wave (if Bits.to_bool !(out.serial) then '1' else '0')
  in
  let offer bytes =
    let value = List.foldi bytes ~init:0 ~f:(fun k acc b -> acc lor (b lsl (8 * k))) in
    inp.message.data := Bits.of_unsigned_int ~width:(Midi.max_message_bytes * 8) value;
    inp.message.len := Bits.of_unsigned_int ~width:8 (List.length bytes);
    inp.message.valid := Bits.vdd
  in
  let idle n =
    inp.message.valid := Bits.gnd;
    for _ = 1 to n do
      cycle ()
    done
  in
  sim, inp, out, wave, cycle, offer, idle
;;

let%expect_test "the messages on the line" =
  let _sim, _inp, out, wave, cycle, offer, idle = harness () in
  let send bytes =
    offer bytes;
    cycle ();
    idle (14 * clocks_per_bit * List.length bytes)
  in
  idle 4;
  send [ 0x92; 0x3C; 0x64 ];
  send [ 0xB2; 0x4A ];
  send [ 0xF8 ];
  Stdio.printf
    "line [%s]\nready at the end: %b\n"
    (Bytes_util.hex (Uart_rx.For_test.decode_line (Buffer.contents wave) ~clocks_per_bit))
    (Bits.to_bool !(out.ready));
  [%expect {|
    line [92 3c 64 b2 4a f8]
    ready at the end: true
    |}]
;;

let%expect_test "a length outside the range cannot hold the block" =
  let _sim, _inp, out, wave, cycle, offer, idle = harness () in
  idle 4;
  (* the message states a length of 0: the walk still ends at the last byte *)
  offer [];
  cycle ();
  idle (14 * clocks_per_bit * Midi.max_message_bytes);
  Stdio.printf
    "bytes out %d, ready %b\n"
    (Bytes.length (Uart_rx.For_test.decode_line (Buffer.contents wave) ~clocks_per_bit))
    (Bits.to_bool !(out.ready));
  [%expect {| bytes out 3, ready true |}]
;;

let%expect_test "the waveform of the byte walk" =
  (* a message of two bytes at 2 clocks per bit. [ready] falls at the transfer and it
     rises again after the last stop bit, thus it is the envelope of the send. [index]
     steps at each load, and the transmitter waits no cycle between the two bytes. The
     step after the last load gives an [index] of 2 for a message of two bytes, and no
     byte comes from it. The state tags follow [State.t]: Idl Snd Drn. *)
  let clocks_per_bit = 2 in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all (create ~clocks_per_bit) in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  inp.message.data := Bits.of_unsigned_int ~width:(Midi.max_message_bytes * 8) 0x4AB2;
  inp.message.len := Bits.of_unsigned_int ~width:8 2;
  inp.message.valid := Bits.vdd;
  Cyclesim.cycle sim;
  inp.message.valid := Bits.gnd;
  Cyclesim.cycle ~n:46 sim;
  let rules =
    [ Hardcaml_waveterm.Display_rule.port_name_is
        "message$valid"
        ~wave_format:Wave_format.(Bit_or Hex)
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "state"
        ~wave_format:(Wave_format.Index [ "Idl"; "Snd"; "Drn" ])
    ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "index"; "ready"; "serial" ]
    ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:(-1)
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │message$valid  ││─┐                                                 │
    │               ││ └─────────────────────────────────────────────    │
    │               ││─┬─────────────────────┬────────────────────┬──    │
    │state          ││ │Snd                  │Drn                 │I.    │
    │               ││─┴─────────────────────┴────────────────────┴──    │
    │               ││──┬────────────────────┬───────────────────────    │
    │index          ││ 0│1                   │2                          │
    │               ││──┴────────────────────┴───────────────────────    │
    │ready          ││─┐                                          ┌──    │
    │               ││ └──────────────────────────────────────────┘      │
    │serial         ││──┐   ┌─┐   ┌───┐ ┌────┐   ┌─┐ ┌─┐   ┌─┐ ┌─────    │
    │               ││  └───┘ └───┘   └─┘    └───┘ └─┘ └───┘ └─┘         │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
