open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; params : 'a Control_regs.Params.t
    ; source : 'a Source_intf.O.t
    ; midi_ready : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { midi : 'a Midi.Rtl.Message.t
    ; source_rewind : 'a
    ; source_step : 'a
    ; source_ready : 'a
    }
  [@@deriving hardcaml]
end

(* [Idle] waits for the run bit. [Loading] waits while the source goes to its origin.
   [Take] waits for a note of the source, or for its [idle] that ends the step. [SendOff]
   and [SendOn] hold one message each for the merge, and the [SendOff] path is the voice
   that already holds a note. [Wait] counts the milliseconds of the step. [GateOff] closes
   the highest voice at the gate. [StopScan] and [StopOff] walk the seats and close each
   open one when the run stops. *)
module State = struct
  type t =
    | Idle
    | Loading
    | Take
    | SendOff
    | SendOn
    | Wait
    | GateOff
    | StopScan
    | StopOff
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~clocks_per_ms (i : _ I.t) : _ O.t =
  assert (clocks_per_ms >= 2);
  let voices = Source_intf.voices in
  let top = voices - 1 in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let prescaler = Variable.reg spec ~width:(Int.ceil_log2 clocks_per_ms) in
  let ms = Variable.reg spec ~width:16 in
  let step_len = Variable.reg spec ~width:16 in
  let gate_len = Variable.reg spec ~width:16 in
  (* one open-note register for each voice: the note and the channel of its Note On *)
  let open_flag = Array.init voices ~f:(fun _ -> Variable.reg spec ~width:1) in
  let open_note = Array.init voices ~f:(fun _ -> Variable.reg spec ~width:8) in
  let open_channel = Array.init voices ~f:(fun _ -> Variable.reg spec ~width:4) in
  let seat = Variable.reg spec ~width:(address_bits_for voices) in
  let msg_data = Variable.reg spec ~width:(Midi.max_message_bytes * 8) in
  let msg_valid = Variable.reg spec ~width:1 in
  let source_rewind = Variable.wire ~default:gnd () in
  let source_step = Variable.wire ~default:gnd () in
  let source_ready = Variable.wire ~default:gnd () in
  (* the names put the engine into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = ms.value -- "ms" in
  let _ = seat.value -- "seat" in
  let run_bit = i.params.run in
  let tick = prescaler.value ==:. clocks_per_ms - 1 in
  let transfer = msg_valid.value &: i.midi_ready in
  (* a sampled STEP_MS of 0 counts as 1: the boundary must always come *)
  let step_sample = mux2 (i.params.step_ms ==:. 0) (one 16) i.params.step_ms in
  let at_step = ms.value >=: step_len.value in
  (* the gate closes the highest voice, and no other *)
  let at_gate =
    open_flag.(top).value
    &: (gate_len.value <: step_len.value)
    &: (ms.value >=: gate_len.value)
  in
  let values regs = List.map (Array.to_list regs) ~f:(fun (v : Variable.t) -> v.value) in
  (* the voice of the note that the source offers, and the seat that the walk of the stop
     is at *)
  let incoming = i.source.note.voice in
  let incoming_open = mux incoming (values open_flag) in
  let seat_open = mux seat.value (values open_flag) in
  let last_seat = seat.value ==:. voices - 1 in
  (* the messages; [Midi] holds the byte layout *)
  let on_data =
    Midi.Rtl.note_on_data
      ~channel:i.params.channel
      ~pitch:i.source.note.pitch
      ~velocity:i.params.velocity
  in
  (* the Note Off of the voice that [index] names, from its own open-note registers *)
  let off_at index =
    Midi.Rtl.note_off_data
      ~channel:(mux index (values open_channel))
      ~pitch:(mux index (values open_note))
  in
  let off_incoming = off_at incoming in
  let off_seat = off_at seat.value in
  let off_top =
    Midi.Rtl.note_off_data ~channel:open_channel.(top).value ~pitch:open_note.(top).value
  in
  let at_seat f = proc (List.init voices ~f:(fun k -> when_ (seat.value ==:. k) (f k))) in
  (* the scan ends at the last seat and otherwise goes on to the next one *)
  let next_seat =
    [ seat <-- seat.value +:. 1
    ; if_ last_seat [ sm.set_next Idle ] [ sm.set_next StopScan ]
    ]
  in
  compile
    [ (* the clock of the run: the start resets it, a stall does not pause it *)
      if_
        (sm.is Idle)
        [ prescaler <--. 0; ms <--. 0 ]
        [ if_
            tick
            [ prescaler <--. 0; ms <-- ms.value +:. 1 ]
            [ prescaler <-- prescaler.value +:. 1 ]
        ]
    ; sm.switch
        [ Idle, [ when_ run_bit [ source_rewind <-- vdd; sm.set_next Loading ] ]
        ; ( Loading
          , [ when_
                i.source.idle
                [ step_len <-- step_sample
                ; gate_len <-- i.params.gate_ms
                ; ms <--. 0
                ; source_step <-- vdd
                ; sm.set_next Take
                ]
            ] )
        ; ( Take
          , [ if_
                i.source.valid
                [ seat <-- incoming
                ; if_
                    incoming_open
                    [ msg_data <-- off_incoming; msg_valid <-- vdd; sm.set_next SendOff ]
                    [ msg_data <-- on_data; msg_valid <-- vdd; sm.set_next SendOn ]
                ]
                [ when_ i.source.idle [ sm.set_next Wait ] ]
            ] )
        ; ( SendOff
          , [ when_
                transfer
                [ at_seat (fun k -> [ open_flag.(k) <-- gnd ])
                ; msg_data <-- on_data
                ; msg_valid <-- vdd
                ; sm.set_next SendOn
                ]
            ] )
        ; ( SendOn
          , [ when_
                transfer
                [ at_seat (fun k ->
                    [ open_flag.(k) <-- vdd
                    ; open_note.(k) <-- select msg_data.value ~high:15 ~low:8
                    ; open_channel.(k) <-- sel_bottom msg_data.value ~width:4
                    ])
                ; msg_valid <-- gnd
                ; (* the source holds the note until here, thus both messages read a
                     stable pitch *)
                  source_ready <-- vdd
                ; sm.set_next Take
                ]
            ] )
        ; ( Wait
          , [ if_
                at_step
                [ step_len <-- step_sample
                ; gate_len <-- i.params.gate_ms
                ; ms <--. 0
                ; if_
                    run_bit
                    [ source_step <-- vdd; sm.set_next Take ]
                    [ seat <--. 0; sm.set_next StopScan ]
                ]
                [ when_
                    at_gate
                    [ msg_data <-- off_top; msg_valid <-- vdd; sm.set_next GateOff ]
                ]
            ] )
        ; ( GateOff
          , [ when_
                transfer
                [ open_flag.(top) <-- gnd; msg_valid <-- gnd; sm.set_next Wait ]
            ] )
        ; ( StopScan
          , [ if_
                seat_open
                [ msg_data <-- off_seat; msg_valid <-- vdd; sm.set_next StopOff ]
                next_seat
            ] )
        ; ( StopOff
          , [ when_
                transfer
                ([ at_seat (fun k -> [ open_flag.(k) <-- gnd ]); msg_valid <-- gnd ]
                 @ next_seat)
            ] )
        ]
    ];
  { O.midi =
      { Midi.Rtl.Message.data = msg_data.value
      ; len = of_unsigned_int ~width:8 Midi.max_message_bytes
      ; valid = msg_valid.value
      }
  ; source_rewind = source_rewind.value
  ; source_step = source_step.value
  ; source_ready = source_ready.value
  }
;;

let clocks_per_ms = 4

(* The harness stubs the note source. It answers each [source_step] with [program] — the
   voice and the pitch of every note that speaks at a step — and it gives them one at a
   time, each held until [source_ready]. Voice 0 sounds 24, voice 1 sounds 30 and voice 2
   sounds 3c, thus the log names the voice of each message; the highest voice counts up
   from 48. The strobes are read before the edge. The log shows the transfers as
   [t=cycle bytes], with t counted from the start of the run. *)
let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_ms) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.midi_ready := Bits.vdd;
  inp.source.idle := Bits.vdd;
  let time = ref 0 in
  let step_index = ref 0 in
  let program = ref [ 0, 0x24; 1, 0x30; 2, 0x3c; 3, 0x48 ] in
  let pending = ref [] in
  let offer () =
    match !pending with
    | [] ->
      inp.source.valid := Bits.gnd;
      inp.source.idle := Bits.vdd
    | (voice, pitch) :: _ ->
      inp.source.valid := Bits.vdd;
      inp.source.idle := Bits.gnd;
      inp.source.note.voice
      := Bits.of_unsigned_int ~width:(Bits.width !(inp.source.note.voice)) voice;
      inp.source.note.pitch := Bits.of_unsigned_int ~width:8 pitch
  in
  let cycle () =
    offer ();
    Cyclesim.cycle sim;
    if Bits.to_bool !(out.source_step)
    then (
      (* the highest voice moves at each step, thus the log is easy to read *)
      pending
      := List.map !program ~f:(fun (voice, pitch) ->
           voice, if voice = Source_intf.voices - 1 then pitch + !step_index else pitch);
      Int.incr step_index);
    if Bits.to_bool !(out.source_ready) then pending := List.tl_exn !pending;
    if Bits.to_bool !(out.source_rewind) then Stdio.printf "t=%03d rewind\n" !time;
    if Bits.to_bool !(out.midi.valid)
    then (
      let data = Bits.to_int_trunc !(out.midi.data) in
      Stdio.printf
        "t=%03d %02x %02x %02x\n"
        !time
        (data land 0xff)
        ((data lsr 8) land 0xff)
        ((data lsr 16) land 0xff));
    Int.incr time
  in
  let set field value = field := Bits.of_unsigned_int ~width:(Bits.width !field) value in
  sim, inp, cycle, set, program
;;

let%expect_test "the voices speak in the order of the source, and the gate closes the \
                 highest"
  =
  let _sim, inp, cycle, set, _program = harness () in
  set inp.params.step_ms 8;
  set inp.params.gate_ms 4;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  set inp.params.run 1;
  for _ = 1 to 70 do
    cycle ()
  done;
  set inp.params.run 0;
  for _ = 1 to 60 do
    cycle ()
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 24 64
    t=005 92 30 64
    t=007 92 3c 64
    t=009 92 48 64
    t=018 82 48 40
    t=035 82 24 40
    t=036 92 24 64
    t=038 82 30 40
    t=039 92 30 64
    t=041 82 3c 40
    t=042 92 3c 64
    t=044 92 49 64
    t=050 82 49 40
    t=067 82 24 40
    t=068 92 24 64
    t=070 82 30 40
    t=071 92 30 64
    t=073 82 3c 40
    t=074 92 3c 64
    t=076 92 4a 64
    t=082 82 4a 40
    t=099 82 24 40
    t=101 82 30 40
    t=103 82 3c 40
    |}]
;;

let%expect_test "a voice that the source does not name stays silent and holds its note" =
  let _sim, inp, cycle, set, program = harness () in
  set inp.params.step_ms 8;
  set inp.params.gate_ms 4;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  (* the lowest voice and the highest voice speak; the two middle ones never do *)
  program := [ 0, 0x24; 3, 0x48 ];
  set inp.params.run 1;
  for _ = 1 to 70 do
    cycle ()
  done;
  set inp.params.run 0;
  for _ = 1 to 60 do
    cycle ()
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 24 64
    t=005 92 48 64
    t=018 82 48 40
    t=035 82 24 40
    t=036 92 24 64
    t=038 92 49 64
    t=050 82 49 40
    t=067 82 24 40
    t=068 92 24 64
    t=070 92 4a 64
    t=082 82 4a 40
    t=099 82 24 40
    |}]
;;

let%expect_test "the stop closes each open voice, from the lowest" =
  let _sim, inp, cycle, set, _program = harness () in
  set inp.params.step_ms 8;
  (* the gate is not less than the step, thus it never comes and the highest voice also
     holds its note at the stop *)
  set inp.params.gate_ms 20;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  set inp.params.run 1;
  for _ = 1 to 40 do
    cycle ()
  done;
  (* the channel changes inside the run: each Note Off keeps the channel of its Note On *)
  set inp.params.channel 5;
  set inp.params.run 0;
  for _ = 1 to 60 do
    cycle ()
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 24 64
    t=005 92 30 64
    t=007 92 3c 64
    t=009 92 48 64
    t=035 82 24 40
    t=036 92 24 64
    t=038 82 30 40
    t=039 92 30 64
    t=041 82 3c 40
    t=042 95 3c 64
    t=044 82 48 40
    t=045 95 49 64
    t=067 82 24 40
    t=069 82 30 40
    t=071 85 3c 40
    t=073 85 49 40
    |}]
;;

let%expect_test "STEP_MS applies at the next step" =
  let _sim, inp, cycle, set, program = harness () in
  set inp.params.step_ms 3;
  set inp.params.gate_ms 1;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  (* one voice only, thus the timing is easy to read *)
  program := [ 3, 0x48 ];
  set inp.params.run 1;
  for _ = 1 to 6 do
    cycle ()
  done;
  set inp.params.step_ms 6;
  for _ = 1 to 46 do
    cycle ()
  done;
  set inp.params.run 0;
  for _ = 1 to 30 do
    cycle ()
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 48 64
    t=006 82 48 40
    t=015 92 49 64
    t=018 82 49 40
    t=039 92 4a 64
    t=042 82 4a 40
    |}]
;;
