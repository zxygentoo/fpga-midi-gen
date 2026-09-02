(* The scheduler — see scheduler.mli, and docs/diffusion_rtl.md, "Phase II: the locked
   design", for the design.

   IT MOVES MEMORY AND TIME AND NEVER A VALUE. Every byte of a sheet is the [Generator]'s
   and every message is the [Sequencer]'s; what this unit adds is WHICH sheet stands, WHEN
   it plays and WHAT sounds between two of them. Therefore no gate of the era's twin moves
   with this file, and the one thing a reader must hold is the walk below.

   THE PING-PONG IS TWO MEMORIES THAT ALREADY EXIST. Gibbs rewrites the sheet in place,
   thus the playing sheet must be its own copy: the generator's sheet registers are the
   ping and the frame store here is the pong. The copy crosses through the transfer face —
   T strobes, one frame each — thus the store holds FRAMES and this unit carries no
   vocabulary and no class width. *)

open Core
open Hardcaml
open Signal
module Placement = Mgen_nn.Placement
module I = Source_intf.I
module O = Source_intf.O

module State = struct
  type t =
    | Rest (* the power-on rest: [idle] 1, and a [step] answers the silent frame *)
    | Open (* the run opens: wait a stale walk out, then strobe the generator's [start] *)
    | Draw (* the generator walks the sheet the store wants *)
    | Copy (* T transfer strobes into the frame store *)
    | Play (* the sheet, then the gap; [idle] 1 and the socket answers from the store *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~e ~gap ~seed ~generator (i : _ I.t) : _ O.t =
  let steps = e.Elaboration.steps in
  (* THE FIRST GAP STEP IS THE DRAIN. At gap 0 the decode would carry held pitches across
     the boundary where the software drains whole and strikes again, and the two streams
     would part. *)
  if gap < 1
  then invalid_argf "the gap is %d steps and the drain takes the first one" gap ();
  let frame_bits = Frame.code_bits * Frame.voices in
  (* one rewind-to-rewind period of the performance: the sheet and the silence behind it *)
  let period = steps + gap in
  let step_bits = address_bits_for steps in
  let play_bits = address_bits_for period in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  (* the store's own output register: the datapath holds no clear, as the walk's does not
     — what is real is what the strobes mark *)
  let dspec = Reg_spec.create ~clock:i.clock () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  (* the generator seats a machine of its own and the engine under it seats a third, thus
     this one carries its owner's name and not the bare [state] all three would answer to *)
  let _ = sm.current -- "scheduler_state" in
  (* THE SEED OF THE SHEET THE GENERATOR IS DRAWING. A rewind latches the panel view into
     it and DRAW steps it by one, ahead of the start that hands it over, thus it reads one
     ahead of the sheet the store plays and sheet k of a run is [(S + k) mod 2^32]. *)
  let seed_reg = Variable.reg spec ~width:(width seed) in
  (* the step of the period the socket answers next: under [steps] the store, at or above
     it the gap *)
  let play_step = Variable.reg spec ~width:play_bits in
  let copy_step = Variable.reg spec ~width:step_bits in
  (* the store's sheet has played whole, thus the next step 0 owes a fresh copy *)
  let spent = Variable.reg spec ~width:1 in
  (* the boundary took a step strobe and still owes its answer *)
  let owes = Variable.reg spec ~width:1 in
  (* the answer is a strobe one cycle behind the step, as the transfer face is *)
  let answer = Variable.reg spec ~width:1 in
  let answer_silent = Variable.reg spec ~width:1 in
  let gen_start = Variable.wire ~default:gnd () in
  let gen_step = Variable.wire ~default:gnd () in
  let store_write = Variable.wire ~default:gnd () in
  let read_enable = Variable.wire ~default:gnd () in
  let _ = seed_reg.value -- "seed_reg" in
  let _ = play_step.value -- "play_step" in
  let _ = copy_step.value -- "copy_step" in
  let _ = gen_step.value -- "gen_step" in
  let _ = store_write.value -- "store_write" in
  (* THE SCHEDULER OWNS THE SEED WIRE: [Source] applies nothing, thus the succession has
     one home and a caller cannot state a second rule for it. *)
  let (g : _ Generator.O.t) =
    generator
      ~seed:seed_reg.value
      { Generator.I.clock = i.clock
      ; clear = i.clear
      ; start = gen_start.value
      ; step = gen_step.value
      }
  in
  let _ = g.valid -- "gen_valid" in
  let _ = g.idle -- "gen_idle" in
  (* The frame store: [T] words of one frame, one write port and one registered read.

     THE REGISTER GOES ON THE OUTPUT AND NOT ON THE ADDRESS. A block RAM reads on the
     clock and has no asynchronous port at all, thus a memory whose RTL read is
     combinational cannot be one however the address arrives: with the register in front,
     Vivado answered `Infeasible attribute ram_style = "block"` and put the whole store in
     LUTRAM at 88 LUTs. Behind the memory it is the primitive's own output register, the
     enable is the read strobe — the word stands while [valid] is 1 and the gap needs no
     read at all — and the answer is one cycle behind the strobe either way. *)
  let store =
    (multiport_memory
       ~attributes:[ Placement.block_ram ]
       steps
       ~write_ports:
         [| { Write_port.write_clock = i.clock
            ; write_address = copy_step.value
            ; write_enable = store_write.value
            ; write_data = g.frame
            }
         |]
       ~read_addresses:[| sel_bottom play_step.value ~width:step_bits |]).(0)
    |> reg dspec ~enable:read_enable.value
  in
  let in_gap = play_step.value >=:. steps in
  (* the run opens the same way from the rest and from a restart inside PLAY *)
  let open_run = [ seed_reg <-- seed; sm.set_next Open ] in
  compile
    [ answer <-- gnd
    ; sm.switch
        [ ( Rest
          , [ if_
                i.rewind
                open_run
                (* totality: a step at rest is silence, and no walk reaches this *)
                [ when_ i.step [ answer <-- vdd; answer_silent <-- vdd ] ]
            ] )
        ; ( Open
          , (* A START INSIDE A WALK IS WHAT THE GENERATOR'S CONTRACT REFUSES. A restart
               may find it inside its lookahead walk, thus the stale walk is waited out
               and the generator gains no abort port for a gesture this rare. *)
            [ when_
                g.idle
                [ gen_start <-- vdd
                ; play_step <--. 0
                ; spent <-- gnd
                ; owes <-- gnd
                ; sm.set_next Draw
                ]
            ] )
        ; ( Draw
          , (* the seed steps HERE and not at the strobe below: the generator reads the
               wire in the cycle the strobe stands, thus the register must already hold
               what the next start hands over *)
            [ when_
                g.idle
                [ copy_step <--. 0
                ; seed_reg <-- seed_reg.value +:. 1
                ; gen_step <-- vdd
                ; sm.set_next Copy
                ]
            ] )
        ; ( Copy
          , [ when_
                g.valid
                [ store_write <-- vdd
                ; if_
                    (copy_step.value ==:. steps - 1)
                    [ (* the sheet stands in the store, and the generator opens the next
                         one over its own *)
                      gen_start <-- vdd
                    ; spent <-- gnd
                    ; sm.set_next Play
                    ]
                    [ copy_step <-- copy_step.value +:. 1; gen_step <-- vdd ]
                ]
            ] )
        ; ( Play
          , [ if_
                i.rewind
                open_run
                [ when_
                    (i.step |: owes.value)
                    [ if_
                        spent.value
                        [ (* THE BOUNDARY. The step is taken and its answer waits on the
                             copy: [idle] falls, the sequencer stands in its own take, and
                             the step stretches by the microseconds the copy costs. *)
                          owes <-- vdd
                        ; sm.set_next Draw
                        ]
                        [ owes <-- gnd
                        ; answer <-- vdd
                        ; answer_silent <-- in_gap
                        ; read_enable <-- ~:in_gap
                        ; if_
                            (play_step.value ==:. period - 1)
                            [ play_step <--. 0; spent <-- vdd ]
                            [ play_step <-- play_step.value +:. 1 ]
                        ]
                    ]
                ]
            ] )
        ]
    ];
  { O.frame = mux2 answer_silent.value (zero frame_bits) store
  ; valid = answer.value
  ; idle = sm.is Rest |: sm.is Play
  }
;;

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* THE PAIR AND NOT A STUB. A stub generator would answer the transfer face in its own
   time and never hold [idle] low for a walk, thus the one thing this unit exists to
   arrange — a draw that outlives a sheet — would go untested. The shape is tiny and the
   walk is one pass, thus a real generator costs a few thousand cycles here. *)
module Bench = struct
  module Sim = Cyclesim.With_interface (I) (O)

  type t =
    { rewind : unit -> unit
    (** strobe [rewind] and cycle until [idle] rises: one draw and one copy *)
    ; play : unit -> int (** strobe [step] and give the frame it answers *)
    ; strobe_rewind : unit -> unit (** strobe [rewind] and give the machine one cycle *)
    ; cycle : unit -> unit
    ; cycles : unit -> int
    ; idle : unit -> bool
    ; seed_reg : unit -> int
    ; waves : Hardcaml_waveterm.Waveform.t option
    }

  (* A BUDGET AND NOT A WAIT: a machine that stalls must fail the gate and not hang it.
     One rewind covers a whole walk, the copy behind it and the slack of a tiny shape. *)
  let budget e = (4 * Elaboration.cell_walk_cycles e) + (e.Elaboration.walk * 40_000)

  let harness ?(trace = false) ~e ~gap ~seed () =
    let sim =
      Sim.create
        ~config:Harness.long_bench
        (create
           ~e
           ~gap
           ~seed:(of_unsigned_int ~width:32 seed)
           ~generator:(Generator.create ~e))
    in
    let waves, sim = Cyclesim.Waveform.create_if ~enabled:trace sim in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs ~clock_edge:Before sim in
    let seed_reg = Harness.node sim "seed_reg" in
    let cycles = ref 0 in
    let cycle () =
      Cyclesim.cycle sim;
      Int.incr cycles
    in
    let idle () = Bits.to_bool !(out.idle) in
    let wait_for what ~until =
      let left = ref (budget e) in
      while (not (until ())) && !left > 0 do
        cycle ();
        Int.decr left
      done;
      if !left <= 0 then failwithf "%s did not come inside %d cycles" what (budget e) ()
    in
    let strobe_rewind () =
      inp.rewind := Bits.vdd;
      cycle ();
      inp.rewind := Bits.gnd
    in
    let rewind () =
      strobe_rewind ();
      (* the machine leaves the rest on the edge behind the strobe, thus the wait cycles
         before it reads [idle] *)
      cycle ();
      wait_for "the first sheet" ~until:idle
    in
    let play () =
      inp.step := Bits.vdd;
      cycle ();
      inp.step := Bits.gnd;
      wait_for "the step" ~until:(fun () -> Bits.to_bool !(out.valid));
      Bits.to_int_trunc !(out.frame)
    in
    { rewind
    ; play
    ; strobe_rewind
    ; cycle
    ; cycles = (fun () -> !cycles)
    ; idle
    ; seed_reg = (fun () -> Cyclesim.Node.to_int seed_reg)
    ; waves
    }
  ;;

  (* the shape every test below mounts: four layers of eight channels, one pass, and a
     sheet of three steps. Nothing here reads a width — what a test reads is the ORDER of
     the sheets and the silence between them. *)
  let tiny ~steps =
    Elaboration.create
      (Model.For_test.drawn ~layers:4 ~width:8 ~seed:11)
      ~steps
      ~lanes:2
      ~walk:1
  ;;
end

let%expect_test "the boundary: the drain, the gap, the copy, and step 0 of the next sheet"
  =
  (* THE PICTURE OF THE ONE THING THIS UNIT DOES, in two windows. A sheet of three steps
     and a gap of two, thus the whole period is five.

     WINDOW ONE is the end of a period. Steps 0 to 2 answer from the store; step 3 is THE
     DRAIN, whose silent frame closes every pitch the sheet held in ascending order —
     [midi.play]'s own sorted tail; step 4 is the rest of the gap and says nothing. The
     step behind it is THE BOUNDARY: the strobe is taken, [scheduler_state] leaves PLAY
     for DRAW, and [idle] falls while the answer waits.

     WINDOW TWO is that answer. THE BENCH PLAYS BACK TO BACK AND THE BOARD DOES NOT: here
     the boundary waits a whole walk out, where at the elected rung the draw ends nine
     seconds early and the wait is zero. What the window shows is what the board does in 3
     us — [gen_idle] rises, the copy strobes the transfer face and [store_write] takes
     each frame at [copy_step], [seed_reg] steps, the generator opens the next sheet, and
     the answer of step 0 comes off the fresh store. *)
  let e = Bench.tiny ~steps:3 in
  let h = Bench.harness ~trace:true ~e ~gap:2 ~seed:7 () in
  h.rewind ();
  let frames = List.map (List.range 0 5) ~f:(fun (_ : int) -> h.play ()) in
  printf
    "the first period: %s\n"
    (String.concat ~sep:" " (List.map frames ~f:(sprintf "%08x")));
  let boundary = h.cycles () in
  printf "step 0 of the next sheet: %08x\n" (h.play ());
  let answered = h.cycles () in
  let window ~start_cycle =
    Hardcaml_waveterm.Waveform.expect
      ~display_rules:
        [ Hardcaml_waveterm.Display_rule.port_name_is
            "scheduler_state"
            ~wave_format:(Wave_format.Index [ "Rst"; "Opn"; "Drw"; "Cpy"; "Ply" ])
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Unsigned_int
            [ "play_step"; "copy_step"; "seed_reg" ]
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Bit
            [ "step"
            ; "gen_idle"
            ; "gen_step"
            ; "gen_valid"
            ; "store_write"
            ; "valid"
            ; "idle"
            ]
        ; Hardcaml_waveterm.Display_rule.port_name_is "frame" ~wave_format:Wave_format.Hex
        ]
      ~show_digest:false
      ~wave_width:0
      ~display_width:80
      ~start_cycle
      (Option.value_exn h.waves ~message:"a traced run gives a waveform")
  in
  window ~start_cycle:(boundary - 8);
  window ~start_cycle:(answered - 14);
  [%expect
    {|
    the first period: b3bdaea8 a5b6aaca bba9b3c9 00000000 00000000
    step 0 of the next sheet: a6c7bfa9
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │                  ││──────────────────┬───────────────────────────────────────│
    │scheduler_state   ││ Ply              │Drw                                    │
    │                  ││──────────────────┴───────────────────────────────────────│
    │                  ││──┬───┬───┬───┬───────────────────────────────────────────│
    │play_step         ││ 1│2  │3  │4  │0                                          │
    │                  ││──┴───┴───┴───┴───────────────────────────────────────────│
    │                  ││──────────────────────────────────────────────────────────│
    │copy_step         ││ 2                                                        │
    │                  ││──────────────────────────────────────────────────────────│
    │                  ││──────────────────────────────────────────────────────────│
    │seed_reg          ││ 8                                                        │
    │                  ││──────────────────────────────────────────────────────────│
    │step              ││──┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐                                       │
    │                  ││  └─┘ └─┘ └─┘ └─┘ └───────────────────────────────────────│
    │gen_idle          ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │gen_step          ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │gen_valid         ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │store_write       ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │valid             ││  ┌─┐ ┌─┐ ┌─┐ ┌─┐                                         │
    │                  ││──┘ └─┘ └─┘ └─┘ └─────────────────────────────────────────│
    │idle              ││──────────────────┐                                       │
    │                  ││                  └───────────────────────────────────────│
    │                  ││──┬───┬───┬───────────────────────────────────────────────│
    │frame             ││ .│A5.│BB.│00000000                                       │
    │                  ││──┴───┴───┴───────────────────────────────────────────────│
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │                  ││──────────────────┬─────┬───                              │
    │scheduler_state   ││ Drw              │Cpy  │Ply                              │
    │                  ││──────────────────┴─────┴───                              │
    │                  ││──────────────────────────┬─                              │
    │play_step         ││ 0                        │1                              │
    │                  ││──────────────────────────┴─                              │
    │                  ││──────────────────┬─┬─┬─────                              │
    │copy_step         ││ 2                │0│1│2                                  │
    │                  ││──────────────────┴─┴─┴─────                              │
    │                  ││──────────────────┬─────────                              │
    │seed_reg          ││ 8                │9                                      │
    │                  ││──────────────────┴─────────                              │
    │step              ││                                                          │
    │                  ││────────────────────────────                              │
    │gen_idle          ││                ┌───────┐                                 │
    │                  ││────────────────┘       └───                              │
    │gen_step          ││                ┌─────┐                                   │
    │                  ││────────────────┘     └─────                              │
    │gen_valid         ││                  ┌─────┐                                 │
    │                  ││──────────────────┘     └───                              │
    │store_write       ││                  ┌─────┐                                 │
    │                  ││──────────────────┘     └───                              │
    │valid             ││                          ┌─                              │
    │                  ││──────────────────────────┘                               │
    │idle              ││                        ┌───                              │
    │                  ││────────────────────────┘                                 │
    │                  ││──────────────────────────┬─                              │
    │frame             ││ 00000000                 │.                              │
    │                  ││──────────────────────────┴─                              │
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the succession: sheet k of the run is the generator's sheet at S plus k" =
  (* THE HANDOFF, STATED BY THE MACHINE AND CHECKED AGAINST THE DRAW ITSELF. The reference
     is one [Generator] played alone at each seed of the succession, thus what this reads
     is the SCHEDULER'S arrangement and never the network: the sheets are the same sheets,
     in the order the seeds name, with the drain between them.

     [seed_reg] is the seed of the sheet the generator is drawing, thus it reads one ahead
     of the sheet the store plays — the draw is always the sheet to come. *)
  let e = Bench.tiny ~steps:3 in
  let seed = 1000 in
  let h = Bench.harness ~e ~gap:1 ~seed () in
  let drawn at =
    let g = Generator.For_test.Bench.harness ~e ~seed:(seed + at) () in
    g.start ();
    List.map (List.range 0 3) ~f:(fun (_ : int) -> g.play ())
  in
  h.rewind ();
  let sheet at =
    let frames = List.map (List.range 0 3) ~f:(fun (_ : int) -> h.play ()) in
    let drain = h.play () in
    printf
      "sheet %d is the draw at seed %d: %b, the drain is silence: %b, seed_reg %d\n"
      at
      (seed + at)
      (List.equal Int.equal frames (drawn at))
      (drain = Frame.silent)
      (h.seed_reg ())
  in
  sheet 0;
  sheet 1;
  sheet 2;
  [%expect
    {|
    sheet 0 is the draw at seed 1000: true, the drain is silence: true, seed_reg 1001
    sheet 1 is the draw at seed 1001: true, the drain is silence: true, seed_reg 1002
    sheet 2 is the draw at seed 1002: true, the drain is silence: true, seed_reg 1003
    |}]
;;

let%expect_test "a rewind inside PLAY waits the stale walk out and re-anchors the seed" =
  (* THE RESTART. The generator is inside its lookahead walk when the rewind arrives, and
     a [start] inside a walk is what its contract refuses. The scheduler drops [idle] at
     once — the sequencer's own [WaitRewind] then holds — waits the stale walk out, and
     draws the panel seed again. What the gate reads is that [idle] really falls, that the
     seed is the panel's and not the succession's, and that the sheet is the run's first
     one again. *)
  let e = Bench.tiny ~steps:3 in
  let h = Bench.harness ~e ~gap:1 ~seed:1000 () in
  h.rewind ();
  let first = List.map (List.range 0 3) ~f:(fun (_ : int) -> h.play ()) in
  h.strobe_rewind ();
  h.cycle ();
  printf "idle inside the restart: %b\n" (h.idle ());
  let left = ref (Bench.budget e) in
  while (not (h.idle ())) && !left > 0 do
    h.cycle ();
    Int.decr left
  done;
  let again = List.map (List.range 0 3) ~f:(fun (_ : int) -> h.play ()) in
  printf
    "the sheet repeats: %b, and seed_reg is the panel's plus one: %d\n"
    (List.equal Int.equal first again)
    (h.seed_reg ());
  [%expect
    {|
    idle inside the restart: false
    the sheet repeats: true, and seed_reg is the panel's plus one: 1001
    |}]
;;

let%expect_test "the rest answers a step with silence, and a gap of 0 is refused" =
  (* TOTALITY AND THE FLOOR. Nothing in the sequencer's walk strobes a step before a
     rewind, thus the first line states a behaviour and not a case; the second is the
     drain's own floor. *)
  let e = Bench.tiny ~steps:3 in
  let h = Bench.harness ~e ~gap:1 ~seed:7 () in
  printf "idle at the rest: %b, and a step there is %08x\n" (h.idle ()) (h.play ());
  printf
    "a gap of 0: %s\n"
    (match Bench.harness ~e ~gap:0 ~seed:7 () with
     | (_ : Bench.t) -> "taken"
     | exception Invalid_argument message -> message);
  [%expect
    {|
    idle at the rest: true, and a step there is 00000000
    a gap of 0: the gap is 0 steps and the drain takes the first one
    |}]
;;
