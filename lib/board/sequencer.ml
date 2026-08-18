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
    }
  [@@deriving hardcaml]
end

(* [Idle] waits for the run bit. [WaitRewind] waits while the source goes to its origin.
   [TakeFrame] takes the frame of the step. Then the decode, which is two walks over the
   same choice — the lowest pitch that must move: [Release] offers each Note Off and
   [Strike] each Note On, and the [Send] state beside each one holds its message for the
   merge. [WaitStep] counts the milliseconds of the step.

   The stop takes no states of its own: it puts the silent frame in and enters the release
   walk, thus every held pitch closes by the one rule that closes any other. *)
module State = struct
  type t =
    | Idle
    | WaitRewind
    | TakeFrame
    | Release
    | ReleaseSend
    | Strike
    | StrikeSend
    | WaitStep
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~clocks_per_ms (i : _ I.t) : _ O.t =
  assert (clocks_per_ms >= 2);
  let voices = Frame.voices in
  let pitch_bits = Frame.code_bits - 1 in
  let slot_bits = address_bits_for voices in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let prescaler = Variable.reg spec ~width:(Int.ceil_log2 clocks_per_ms) in
  let ms = Variable.reg spec ~width:16 in
  let step_len = Variable.reg spec ~width:16 in
  (* The set of pitches that sound, as [voices] slots: a frame asks for four pitches at
     the most, thus four slots hold any set the walk can reach. A slot keeps the channel
     of its Note On beside the note, thus the Note Off takes the pair that opened it. *)
  let held_flag = Array.init voices ~f:(fun _ -> Variable.reg spec ~width:1) in
  let held_note = Array.init voices ~f:(fun _ -> Variable.reg spec ~width:8) in
  let held_channel = Array.init voices ~f:(fun _ -> Variable.reg spec ~width:4) in
  (* the frame under decode, and the slot the offered message belongs to *)
  let frame = Variable.reg spec ~width:(Frame.code_bits * voices) in
  let slot = Variable.reg spec ~width:slot_bits in
  (* 1 while the walk closes the run: the release walk is the same, and this states where
     it goes when the frame is empty *)
  let stopping = Variable.reg spec ~width:1 in
  let msg_data = Variable.reg spec ~width:(Midi.max_message_bytes * 8) in
  let msg_valid = Variable.reg spec ~width:1 in
  let source_rewind = Variable.wire ~default:gnd () in
  let source_step = Variable.wire ~default:gnd () in
  (* the names put the engine into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = ms.value -- "ms" in
  let _ = frame.value -- "frame" in
  let run_bit = i.params.run in
  let tick = prescaler.value ==:. clocks_per_ms - 1 in
  let transfer = msg_valid.value &: i.midi_ready in
  (* a sampled STEP_MS of 0 counts as 1: the boundary must always come *)
  let step_sample = mux2 (i.params.step_ms ==:. 0) (one 16) i.params.step_ms in
  let at_step = ms.value >=: step_len.value in
  let any signals = List.fold signals ~init:gnd ~f:( |: ) in
  (* the voice code of seat [s] of the frame, and what it says *)
  let code s =
    select frame.value ~high:((Frame.code_bits * (s + 1)) - 1) ~low:(Frame.code_bits * s)
  in
  let sounds s = msb (code s) in
  let asks s = uresize (sel_bottom (code s) ~width:pitch_bits) ~width:8 in
  (* the two membership tests the decode rests on: the frame asks for a pitch, and a slot
     holds one. Both walk four comparators, because both sets hold four members at the
     most. *)
  let asked pitch = any (List.init voices ~f:(fun s -> sounds s &: (asks s ==: pitch))) in
  let holds pitch =
    any
      (List.init voices ~f:(fun k ->
         held_flag.(k).value &: (held_note.(k).value ==: pitch)))
  in
  (* [lowest] is the one choice of both walks: of the [voices] candidates, the one with
     the lowest pitch. The walks differ only in what makes a candidate, thus the ascending
     order of [Frame.events_of_frames] has one definition here.

     The reduction is BALANCED and not a fold. A fold puts the compares in series — each
     one waits for the running best of the one above it — and that chain, from the frame
     register through the message register, was the critical path of the first six-layer
     build of the era: 17 levels and 1.93 ns over the period. A tree of the same compares
     is two levels deep at four voices. It chooses the same candidate: [better] takes the
     right one only on a strictly lower pitch, thus a tie keeps the left one, which is the
     lower index in either shape. *)
  let better (found, pitch, index) (b_found, b_pitch, b_index) =
    let takes = b_found &: (~:found |: (b_pitch <: pitch)) in
    found |: b_found, mux2 takes b_pitch pitch, mux2 takes b_index index
  in
  let lowest ~candidate ~pitch =
    List.init voices ~f:(fun k ->
      candidate k, pitch k, of_unsigned_int ~width:slot_bits k)
    |> List.reduce_balanced_exn ~f:better
  in
  (* a release: a slot that sounds a pitch the frame does not ask for *)
  let release_any, (_ : Signal.t), release_slot =
    lowest
      ~candidate:(fun k -> held_flag.(k).value &: ~:(asked held_note.(k).value))
      ~pitch:(fun k -> held_note.(k).value)
  in
  (* a strike: a seat that asks for a pitch no slot holds. A unison names one pitch twice
     and the walk strikes it one time, because the first strike puts it in a slot. *)
  let strike_any, strike_pitch, (_ : Signal.t) =
    lowest ~candidate:(fun s -> sounds s &: ~:(holds (asks s))) ~pitch:asks
  in
  (* The slot a strike takes: the lowest free one. One always stands free, because the
     releases run first — the held set is then inside the asked set, which holds four
     pitches at the most, and each strike adds one of the pitches it does not hold yet. *)
  let free_slot =
    List.fold
      (List.rev (List.range 0 voices))
      ~init:(zero slot_bits)
      ~f:(fun below k ->
        mux2 ~:(held_flag.(k).value) (of_unsigned_int ~width:slot_bits k) below)
  in
  let values regs = List.map (Array.to_list regs) ~f:(fun (v : Variable.t) -> v.value) in
  let on_data =
    Midi.Rtl.note_on_data
      ~channel:i.params.channel
      ~pitch:strike_pitch
      ~velocity:i.params.velocity
  in
  (* the Note Off of the slot the walk named, from its own held pair *)
  let off_data =
    Midi.Rtl.note_off_data
      ~channel:(mux release_slot (values held_channel))
      ~pitch:(mux release_slot (values held_note))
  in
  (* A switch and not a chain of guards. [Always] compiles a statement list into one mux
     level for each statement that writes a target, thus [voices] guards put [voices]
     muxes in series on every held register; a switch whose matches are constants compiles
     into one parallel case. The same rule holds the op dispatch of [Transformer.Source]. *)
  let at_slot f =
    switch
      slot.value
      (List.init voices ~f:(fun k -> of_unsigned_int ~width:slot_bits k, f k))
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
        [ Idle, [ when_ run_bit [ source_rewind <-- vdd; sm.set_next WaitRewind ] ]
        ; ( WaitRewind
          , [ when_
                i.source.idle
                [ step_len <-- step_sample
                ; ms <--. 0
                ; source_step <-- vdd
                ; sm.set_next TakeFrame
                ]
            ] )
        ; ( TakeFrame
          , [ when_ i.source.valid [ frame <-- i.source.frame; sm.set_next Release ] ] )
        ; ( Release
          , [ if_
                release_any
                [ msg_data <-- off_data
                ; msg_valid <-- vdd
                ; slot <-- release_slot
                ; sm.set_next ReleaseSend
                ]
                [ sm.set_next Strike ]
            ] )
        ; ( ReleaseSend
          , [ when_
                transfer
                [ at_slot (fun k -> [ held_flag.(k) <-- gnd ])
                ; msg_valid <-- gnd
                ; sm.set_next Release
                ]
            ] )
        ; ( Strike
          , [ if_
                strike_any
                [ msg_data <-- on_data
                ; msg_valid <-- vdd
                ; slot <-- free_slot
                ; sm.set_next StrikeSend
                ]
                [ if_
                    stopping.value
                    [ stopping <-- gnd; sm.set_next Idle ]
                    [ sm.set_next WaitStep ]
                ]
            ] )
        ; ( StrikeSend
          , [ when_
                transfer
                [ (* the pair comes back out of the message, thus the Note Off of this
                     note cannot disagree with its Note On in a byte *)
                  at_slot (fun k ->
                    [ held_flag.(k) <-- vdd
                    ; held_note.(k) <-- select msg_data.value ~high:15 ~low:8
                    ; held_channel.(k) <-- sel_bottom msg_data.value ~width:4
                    ])
                ; msg_valid <-- gnd
                ; sm.set_next Strike
                ]
            ] )
        ; ( WaitStep
          , (* the source holds the boundary until its draw is done; the stop does not
               wait for it, because it takes no frame *)
            [ when_
                (at_step &: (i.source.idle |: ~:run_bit))
                [ step_len <-- step_sample
                ; ms <--. 0
                ; if_
                    run_bit
                    [ source_step <-- vdd; sm.set_next TakeFrame ]
                    [ frame <--. Frame.silent; stopping <-- vdd; sm.set_next Release ]
                ]
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
  }
;;

let clocks_per_ms = 4

(* The harness stubs the note source. It answers each [source_step] with the next frame of
   [program]: [valid] rises in the cycle after the strobe, as a source that has drawn
   already answers it, and [idle] stays low for a few cycles behind it — the draw of the
   next frame. A frame is stated as its pitches, seat 0 first, and a negative pitch is a
   rest. The strobes are read before the edge, and the log shows the transfers as
   [t=cycle bytes] with [t] counted from the start of the run. *)
type stub =
  { inp : Bits.t ref I.t
  ; cycle : unit -> unit (** one cycle, and the log of what crossed in it *)
  ; set : Bits.t ref -> int -> unit
  ; program : int list list ref
  (** the frames to serve, as pitches; the walk cycles them *)
  ; served : int list ref (** the frames the stub gave, newest first *)
  ; messages : (int * int * int) list ref (** the messages that crossed, newest first *)
  }

let harness ?(log = true) () =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~clocks_per_ms) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  inp.midi_ready := Bits.vdd;
  inp.source.idle := Bits.vdd;
  let time = ref 0 in
  let step_index = ref 0 in
  let program = ref [ [ 0x24; 0x30; 0x3c; 0x48 ] ] in
  let served = ref [] in
  let messages = ref [] in
  let frame = ref Frame.silent in
  let answering = ref false in
  let busy = ref 0 in
  let cycle () =
    inp.source.frame
    := Bits.of_unsigned_int ~width:(Frame.code_bits * Frame.voices) !frame;
    inp.source.valid := if !answering then Bits.vdd else Bits.gnd;
    inp.source.idle := if !busy = 0 then Bits.vdd else Bits.gnd;
    Cyclesim.cycle sim;
    answering := false;
    if !busy > 0 then Int.decr busy;
    if Bits.to_bool !(out.source_step)
    then (
      let pitches =
        List.nth !program (!step_index % List.length !program) |> Option.value ~default:[]
      in
      frame := Frame.of_codes (List.map pitches ~f:Frame.code_of_pitch);
      served := !frame :: !served;
      Int.incr step_index;
      answering := true;
      busy := 4);
    if Bits.to_bool !(out.source_rewind)
    then (
      if log then Stdio.printf "t=%03d rewind\n" !time;
      step_index := 0;
      served := []);
    if Bits.to_bool !(out.midi.valid)
    then (
      let data = Bits.to_int_trunc !(out.midi.data) in
      let status = data land 0xff
      and note = (data lsr 8) land 0xff
      and velocity = (data lsr 16) land 0xff in
      messages := (status, note, velocity) :: !messages;
      if log then Stdio.printf "t=%03d %02x %02x %02x\n" !time status note velocity);
    Int.incr time
  in
  let set field value = field := Bits.of_unsigned_int ~width:(Bits.width !field) value in
  { inp; cycle; set; program; served; messages }
;;

let%expect_test "a chord opens, and a new frame releases before it strikes" =
  let { inp; cycle; set; program; _ } = harness () in
  set inp.params.step_ms 8;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  (* one chord, then a chord that shares the bass: the bass holds and nothing closes it *)
  program := [ [ 0x24; 0x30; 0x3c; 0x48 ]; [ 0x24; 0x32; 0x3e; 0x4a ] ];
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
    t=005 92 24 64
    t=007 92 30 64
    t=009 92 3c 64
    t=011 92 48 64
    t=036 82 30 40
    t=038 82 3c 40
    t=040 82 48 40
    t=043 92 32 64
    t=045 92 3e 64
    t=047 92 4a 64
    t=068 82 32 40
    t=070 82 3e 40
    t=072 82 4a 40
    t=075 92 30 64
    t=077 92 3c 64
    t=079 92 48 64
    t=099 82 24 40
    t=101 82 30 40
    t=103 82 3c 40
    t=105 82 48 40
    |}]
;;

let%expect_test "two voices that exchange pitches send nothing, and a unison sounds one \
                 note"
  =
  let { inp; cycle; set; program; _ } = harness () in
  set inp.params.step_ms 8;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  (* The two cases a seat walk cannot state. The exchange: the set does not move, thus no
     message crosses. The unison: two seats name one pitch, and one Note On sounds it. *)
  program := [ [ 0x30; 0x3c; -1; -1 ]; [ 0x3c; 0x30; -1; -1 ]; [ 0x30; 0x30; -1; -1 ] ];
  set inp.params.run 1;
  for _ = 1 to 100 do
    cycle ()
  done;
  set inp.params.run 0;
  for _ = 1 to 60 do
    cycle ()
  done;
  [%expect
    {|
    t=000 rewind
    t=005 92 30 64
    t=007 92 3c 64
    t=068 82 3c 40
    t=101 92 3c 64
    t=131 82 30 40
    t=133 82 3c 40
    |}]
;;

let%expect_test "the stop is a silent frame, and each Note Off keeps the channel of its \
                 Note On"
  =
  let { inp; cycle; set; program; _ } = harness () in
  set inp.params.step_ms 8;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  program := [ [ 0x24; 0x30; 0x3c; 0x48 ] ];
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
    t=005 92 24 64
    t=007 92 30 64
    t=009 92 3c 64
    t=011 92 48 64
    t=067 82 24 40
    t=069 82 30 40
    t=071 82 3c 40
    t=073 82 48 40
    |}]
;;

let%expect_test "a rest in every seat closes the sonority and the walk goes on" =
  let { inp; cycle; set; program; _ } = harness () in
  set inp.params.step_ms 8;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  program := [ [ 0x30; 0x3c; -1; -1 ]; [ -1; -1; -1; -1 ]; [ 0x3c; -1; -1; -1 ] ];
  set inp.params.run 1;
  for _ = 1 to 100 do
    cycle ()
  done;
  set inp.params.run 0;
  for _ = 1 to 60 do
    cycle ()
  done;
  [%expect
    {|
    t=000 rewind
    t=005 92 30 64
    t=007 92 3c 64
    t=036 82 30 40
    t=038 82 3c 40
    t=069 92 3c 64
    t=101 92 30 64
    t=131 82 30 40
    t=133 82 3c 40
    |}]
;;

let%expect_test "STEP_MS applies at the next step" =
  let { inp; cycle; set; program; _ } = harness () in
  set inp.params.step_ms 3;
  set inp.params.velocity 100;
  set inp.params.channel 2;
  (* one voice only, thus the timing is easy to read *)
  program := [ [ 0x48; -1; -1; -1 ]; [ 0x49; -1; -1; -1 ] ];
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
    t=005 92 48 64
    t=016 82 48 40
    t=019 92 49 64
    t=040 82 49 40
    t=043 92 48 64
    t=063 82 48 40
    |}]
;;

(* The decode against its rule, over drawn frames. The expect tests above state the cases
   a reader must see; this states the rule itself, over frames that no one chose: the
   messages of the circuit are the messages of [Frame.events_of_frames] over the frames
   the stub served and the silent frame of the stop, byte for byte and in order.

   The three properties of [Frame] follow from that agreement and are not tested again
   here: the rule holds them by construction, and a circuit that equals the rule holds
   them too. *)
let%expect_test "the messages of the circuit are the events of the frames" =
  let pitches = QCheck.Gen.(oneof [ return (-1); int_range 36 81 ]) in
  let frame = QCheck.Gen.(list_size (return Frame.voices) pitches) in
  let run frames =
    let stub = harness ~log:false () in
    stub.set stub.inp.params.step_ms 8;
    stub.set stub.inp.params.velocity 100;
    stub.set stub.inp.params.channel 2;
    stub.program := frames;
    stub.set stub.inp.params.run 1;
    for _ = 1 to 32 * List.length frames do
      stub.cycle ()
    done;
    stub.set stub.inp.params.run 0;
    for _ = 1 to 80 do
      stub.cycle ()
    done;
    let played = List.rev !(stub.served) in
    let reference =
      Frame.events_of_frames (Array.of_list (played @ [ Frame.silent ]))
      |> List.concat
      |> List.map ~f:(function
        | Frame.Event.On note -> 0x92, note, 100
        | Frame.Event.Off note -> 0x82, note, 0x40)
    in
    [%compare.equal: (int * int * int) list] (List.rev !(stub.messages)) reference
  in
  let disagree =
    List.count (List.range 0 24) ~f:(fun seed ->
      not
        (run (QCheck.Gen.generate ~rand:(Stdlib.Random.State.make [| seed |]) ~n:6 frame)))
  in
  Stdio.printf "24 runs of 6 drawn frames, %d disagree\n" disagree;
  [%expect {| 24 runs of 6 drawn frames, 0 disagree |}]
;;
