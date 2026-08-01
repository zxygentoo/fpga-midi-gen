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
    { midi : 'a Midi.Message.t
    ; source_rewind : 'a
    ; source_step : 'a
    }
  [@@deriving hardcaml]
end

(* [Idle] waits for the run bit. [Loading] waits while the source goes to its origin.
   [Fetch] waits for the note of the step. [SendOff] and [SendOn] hold one message each
   for the merge; the boundary path goes through [SendOff] only in the legato case. [Wait]
   counts the milliseconds of the step. [GateOff] closes the note at the gate, and
   [StopOff] closes it when the run stops. *)
module State = struct
  type t =
    | Idle
    | Loading
    | Fetch
    | SendOff
    | SendOn
    | Wait
    | GateOff
    | StopOff
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~clocks_per_ms (i : _ I.t) : _ O.t =
  assert (clocks_per_ms >= 2);
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let prescaler = Variable.reg spec ~width:(Int.ceil_log2 clocks_per_ms) in
  let ms = Variable.reg spec ~width:16 in
  let step_len = Variable.reg spec ~width:16 in
  let gate_len = Variable.reg spec ~width:16 in
  let open_flag = Variable.reg spec ~width:1 in
  let open_note = Variable.reg spec ~width:8 in
  let open_channel = Variable.reg spec ~width:4 in
  let msg_data = Variable.reg spec ~width:(Midi.max_message_bytes * 8) in
  let msg_valid = Variable.reg spec ~width:1 in
  let source_rewind = Variable.wire ~default:gnd () in
  let source_step = Variable.wire ~default:gnd () in
  (* the names put the engine into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = ms.value -- "ms" in
  let _ = open_flag.value -- "note_open" in
  let run_bit = lsb i.params.run in
  let tick = prescaler.value ==:. clocks_per_ms - 1 in
  let transfer = msg_valid.value &: i.midi_ready in
  (* a sampled STEP_MS of 0 counts as 1: the boundary must always come *)
  let step_sample = mux2 (i.params.step_ms ==:. 0) (one 16) i.params.step_ms in
  let at_step = ms.value >=: step_len.value in
  let at_gate =
    open_flag.value &: (gate_len.value <: step_len.value) &: (ms.value >=: gate_len.value)
  in
  (* the message bytes; the status low nibble carries the channel *)
  let on_data =
    concat_lsb
      [ concat_msb
          [ of_unsigned_int ~width:4 (Midi.note_on lsr 4)
          ; sel_bottom i.params.channel ~width:4
          ]
      ; i.source.note
      ; i.params.velocity
      ]
  in
  let off_data =
    concat_lsb
      [ concat_msb [ of_unsigned_int ~width:4 (Midi.note_off lsr 4); open_channel.value ]
      ; open_note.value
      ; of_unsigned_int ~width:8 Midi.release_velocity
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
                i.source.ready
                [ step_len <-- step_sample
                ; gate_len <-- i.params.gate_ms
                ; ms <--. 0
                ; source_step <-- vdd
                ; sm.set_next Fetch
                ]
            ] )
        ; ( Fetch
          , [ when_
                i.source.valid
                [ if_
                    open_flag.value
                    [ msg_data <-- off_data; msg_valid <-- vdd; sm.set_next SendOff ]
                    [ msg_data <-- on_data; msg_valid <-- vdd; sm.set_next SendOn ]
                ]
            ] )
        ; ( SendOff
          , [ when_
                transfer
                [ open_flag <-- gnd
                ; msg_data <-- on_data
                ; msg_valid <-- vdd
                ; sm.set_next SendOn
                ]
            ] )
        ; ( SendOn
          , [ when_
                transfer
                [ open_flag <-- vdd
                ; open_note <-- select msg_data.value ~high:15 ~low:8
                ; open_channel <-- sel_bottom msg_data.value ~width:4
                ; msg_valid <-- gnd
                ; sm.set_next Wait
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
                    [ source_step <-- vdd; sm.set_next Fetch ]
                    [ if_
                        open_flag.value
                        [ msg_data <-- off_data; msg_valid <-- vdd; sm.set_next StopOff ]
                        [ sm.set_next Idle ]
                    ]
                ]
                [ when_
                    at_gate
                    [ msg_data <-- off_data; msg_valid <-- vdd; sm.set_next GateOff ]
                ]
            ] )
        ; ( GateOff
          , [ when_ transfer [ open_flag <-- gnd; msg_valid <-- gnd; sm.set_next Wait ] ]
          )
        ; ( StopOff
          , [ when_ transfer [ open_flag <-- gnd; msg_valid <-- gnd; sm.set_next Idle ] ]
          )
        ]
    ];
  { O.midi =
      { Midi.Message.data = msg_data.value
      ; len = of_unsigned_int ~width:8 Midi.max_message_bytes
      ; valid = msg_valid.value
      }
  ; source_rewind = source_rewind.value
  ; source_step = source_step.value
  }
;;

let clocks_per_ms = 4

(* The harness stubs the note source: [source.ready] is always 1, and a [source_step]
   strobe answers with [source.valid] one cycle later, with a note that counts up from 60
   at each step. The strobes are read before the edge. The log shows the transfers as
   [t=cycle bytes], with t counted from the start of the run. *)
let harness () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_ms) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.source.ready := Bits.vdd;
  inp.midi_ready := Bits.vdd;
  let time = ref 0 in
  let next_note = ref 60 in
  let pending = ref false in
  let cycle log =
    inp.source.valid := Bits.gnd;
    if !pending
    then (
      inp.source.note := Bits.of_unsigned_int ~width:8 !next_note;
      inp.source.valid := Bits.vdd;
      Int.incr next_note;
      pending := false);
    Cyclesim.cycle sim;
    if Bits.to_bool !(out.source_step) then pending := true;
    if Bits.to_bool !(out.source_rewind) then Stdio.printf "t=%03d rewind\n" !time;
    if Bits.to_bool !(out.midi.valid) && not (Option.is_none log)
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
  sim, inp, cycle, set
;;

let%expect_test "a staccato run: the gate closes each note, the stop is silent" =
  let _sim, inp, cycle, set = harness () in
  set inp.params.step_ms 3;
  set inp.params.gate_ms 1;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  (* the run bit goes 0 during the third step: that step completes, and the stop at the
     next boundary is silent because the gate already closed the note *)
  set inp.params.run 1;
  for _ = 1 to 26 do
    cycle (Some ())
  done;
  set inp.params.run 0;
  for _ = 1 to 20 do
    cycle (Some ())
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 3c 64
    t=006 82 3c 40
    t=015 92 3d 64
    t=018 82 3d 40
    t=027 92 3e 64
    t=030 82 3e 40
    |}]
;;

let%expect_test "a legato run: the off goes immediately before the next on" =
  let _sim, inp, cycle, set = harness () in
  set inp.params.step_ms 3;
  set inp.params.gate_ms 5;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  set inp.params.run 1;
  (* the channel changes after the first note; the off of that note keeps channel 2, and
     the next on takes channel 5 *)
  for _ = 1 to 8 do
    cycle (Some ())
  done;
  set inp.params.channel 5;
  for _ = 1 to 18 do
    cycle (Some ())
  done;
  (* the stop with an open note sends its off *)
  set inp.params.run 0;
  for _ = 1 to 20 do
    cycle (Some ())
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 3c 64
    t=015 82 3c 40
    t=016 95 3d 64
    t=027 85 3d 40
    t=028 95 3e 64
    t=038 85 3e 40
    |}]
;;

let%expect_test "STEP_MS applies at the next step" =
  let _sim, inp, cycle, set = harness () in
  set inp.params.step_ms 3;
  set inp.params.gate_ms 1;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  set inp.params.run 1;
  (* the write lands inside step 1, thus the step that begins at the next boundary already
     has the new length: the gap between the second and the third on is 24 cycles, which
     is 6 ms *)
  for _ = 1 to 6 do
    cycle (Some ())
  done;
  set inp.params.step_ms 6;
  for _ = 1 to 46 do
    cycle (Some ())
  done;
  set inp.params.run 0;
  for _ = 1 to 30 do
    cycle (Some ())
  done;
  [%expect
    {|
    t=000 rewind
    t=003 92 3c 64
    t=006 82 3c 40
    t=015 92 3d 64
    t=018 82 3d 40
    t=039 92 3e 64
    t=042 82 3e 40
    |}]
;;
