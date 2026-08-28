(* The walk — see source.mli for the contract and docs/diffusion_rtl.md, "The walk" and
   "The seam to the sequencer", for the design. What stands here is the WHY of each rule.

   THE RISK OF THIS UNIT IS ORDER AND NOT TIMING. What it adds of its own is serial
   machinery at 2.7 percent of a pass, which the walk bench below measures: three cycles
   for each uniform, one cycle for each standing cell, and one draw for each hidden one.
   Nothing here is near the critical path. What is near the whole piece is the CONSUMPTION
   ORDER — one generator, one order, and a walk that takes one uniform out of place draws
   another sheet with no local symptom.

   TWO FRAMES, TWO CYCLES APART, AS THE ENGINE HAS. The LEAD frame of a cell walk steps
   the generator; the NOW frame — the lead through two registers — is where the cell is
   written, because the third byte of a uniform lands one cycle behind its step and the
   shift register states the whole 24 one cycle behind that. The step rides the LEAD
   frame, thus the two cycles of tail write without drawing and the generator moves
   exactly three times for each cell. *)

open Core
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

(* the activation format of the twin: what a logit column carries in each row *)
let activation_bits = Model.activation_bits

(* one uniform is three bytes of the generator, high byte first *)
let uniform_bits = Prng.uniform_bits
let byte_bits = Prng.byte_bits

(* the ticks of one cell of a cell walk: three steps of the generator *)
let cell_ticks = Prng.uniform_bytes

(* The service of one hidden cell, before the draw: three steps, the cycle their last byte
   lands, and the cycle the whole uniform stands and starts the draw. The draw reads
   [uniform] after its total and states no cycle for it, thus the walk hands over a value
   that has already stopped moving rather than one that stops moving behind it. *)
let uniform_ticks = 5

module State = struct
  type t =
    | Idle (* the rest, and the PLAY phase: the score face answers [step] *)
    | Open (* the opening: one class for each cell *)
    | Mask (* the mask of one pass: one bit for each cell *)
    | Serve
      (* the forward runs; the walk rides [step_ready], and stands here for the whole of a
         service *)
    | Take (* the offered step is taken *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

(* THE SERVICE OF ONE OFFERED STEP, a machine of its own. It walks the four seats of the
   step the walk's count names: a standing seat costs one cycle, a hidden one its uniform
   and its draw. It starts on the offer while the walk stands in [Serve], and ITS EXIT IS
   COMBINATIONAL — the walk watches the last seat retire and moves to [Take] on the very
   edge the service rests — thus the cut of this machine out of the walk's own adds no
   cycle anywhere, which the walk bench holds.

   THE EXCLUSIVITY IS THE CONSUMPTION ORDER'S. One generator, one order: the cell walks
   step it from the walk's own arms, the service from [Uniform]. In one machine no two
   phases could stand at once by construction; across two, the same holds because the
   service runs only while the walk stands parked in [Serve], and this comment is where
   that invariant is stated. *)
module Service = struct
  type t =
    | Idle (* no step stands open *)
    | Seat (* the level stands: read the mask bit of one seat *)
    | Uniform (* the three steps of a redraw's uniform *)
    | Redraw (* the draw runs; the cycle it ends writes the class *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(e : Elaboration.t) ~seed (i : _ I.t) : _ O.t =
  let module Engine =
    Forward.Make (struct
      let e = e
    end)
  in
  let module Cells =
    Sheet.Make (struct
      let steps = e.steps
      let rows = e.rows
    end)
  in
  let module Drawer =
    Draw.Make (struct
      let classes = e.rows
    end)
  in
  let steps = e.steps in
  let rows = e.rows in
  let voices = Frame.voices in
  (* THE SHEET MUST HOLD WHAT THE OPENING DRAWS. A probe geometry may narrow P — the
     elaboration takes [rows] so that the twin can follow later — and the registers of the
     seats are the corpus's, thus a narrow P states a class no cell can hold. It is a
     refusal and not a clamp: a clamped opening is another walk. *)
  if Array.length e.openings <> voices
  then
    invalid_argf
      "the elaboration opens %d seats and a step holds %d"
      (Array.length e.openings)
      voices
      ();
  Array.iteri e.openings ~f:(fun seat { Model.low; width } ->
    if low + width > rows
    then
      invalid_argf
        "seat %d opens inside the classes %d to %d and a sheet of %d rows cannot hold it"
        seat
        low
        (low + width - 1)
        rows
        ());
  let step_bits = address_bits_for steps in
  let seat_bits = address_bits_for voices in
  let class_bits = address_bits_for rows in
  let pass_bits = address_bits_for e.walk in
  let tick_bits = address_bits_for (uniform_ticks + 1) in
  let frame_bits = Frame.code_bits * voices in
  (* A COUNTER HOLDS ITS COUNT AND NOT ONLY ITS LAST INDEX: the opening multiplies by the
     width of a register, thus the field holds [width] itself. *)
  let width_bits =
    address_bits_for
      (1 + Array.fold e.openings ~init:1 ~f:(fun widest o -> max widest o.width))
  in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  (* the datapath holds no clear: what is real is what the strobes mark *)
  let dspec = Reg_spec.create ~clock:i.clock () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let srv = State_machine.create (module Service) spec in
  (* Each state carries its own name, and NOT the [state] the engine already answers to:
     three machines stand in one simulation, thus the walk bench would probe whichever of
     them the name mangler reached first. *)
  let _ = sm.current -- "walk_state" in
  let _ = srv.current -- "service_state" in
  (* the LEAD frame of a cell walk: the cell whose three bytes are being drawn *)
  let lead_step = Variable.reg spec ~width:step_bits in
  let lead_seat = Variable.reg spec ~width:seat_bits in
  let lead_tick = Variable.reg spec ~width:(address_bits_for cell_ticks) in
  let lead_running = Variable.reg spec ~width:1 in
  let u = Variable.reg dspec ~width:uniform_bits in
  let pass = Variable.reg spec ~width:pass_bits in
  (* THE OFFER ORDER IS COUNTED AND NOT TAGGED, which is the seam's own rule: the head
     offers the steps in order and each one exactly one time, thus the walk's count of its
     own acknowledgements IS the step it names at the cell port. *)
  let served = Variable.reg spec ~width:step_bits in
  let seat = Variable.reg spec ~width:seat_bits in
  let tick = Variable.reg spec ~width:tick_bits in
  let play_step = Variable.reg spec ~width:step_bits in
  (* the sheet is played one time: past the last step the frame is four zero bytes *)
  let spent = Variable.reg spec ~width:1 in
  let held = Variable.reg spec ~width:frame_bits in
  let valid = Variable.reg spec ~width:1 in
  let prng_step = Variable.wire ~default:gnd () in
  let draw_start = Variable.wire ~default:gnd () in
  let step_taken = Variable.wire ~default:gnd () in
  let forward_start = Variable.wire ~default:gnd () in
  let idle = sm.is Idle in
  (* ---------------------------------------------------------------- *)
  (* the generator, and the uniform it assembles *)
  (* ---------------------------------------------------------------- *)
  let prng =
    Prng.Rtl.create
      { Prng.Rtl.I.clock = i.clock
      ; clear = i.clear
      ; load = i.rewind &: idle
      ; seed
      ; step = prng_step.value
      }
  in
  let prng_byte = sel_bottom prng.value ~width:byte_bits in
  let _ = prng_step.value -- "prng_step" in
  let _ = step_taken.value -- "step_taken" in
  let _ = u.value -- "uniform" in
  (* ONE CAPTURE RULE FOR EVERY PHASE: a step states its byte in the cycle that follows
     it, thus the shift register takes a byte exactly when the cycle before stepped. *)
  let take_byte = reg spec prng_step.value in
  (* THE UNIFORM RIDES REPLICAS TO ITS ARITHMETIC — the broadcast round's rule, learned
     again on the cut's builds. One [u] register fed the opening multiply, the mask
     compare and the draw's threshold at a fanout near sixty, and two placements in a row
     could not carry the cone: the first met setup only by phys_opt adjusting CLOCK SKEW
     inside the shift register (Physopt 32-703), and the board answered with a DIFFERENT
     sheet at a fixed seed on every run — hold met by a picosecond at the fast corner is
     not met. A replica loads the same next value on the same edge, thus it IS [u] cycle
     for cycle and no gate can tell them apart; what it buys is a register the placer can
     put beside each consumer's own arithmetic. *)
  let next_u =
    mux2
      take_byte
      (sel_bottom u.value ~width:(uniform_bits - byte_bits) @: prng_byte)
      u.value
  in
  let uniform_replica name = Column_array.replica (reg dspec next_u) -- name in
  let u_open = uniform_replica "uniform_open" in
  let u_mask = uniform_replica "uniform_mask" in
  let u_draw = uniform_replica "uniform_draw" in
  (* ---------------------------------------------------------------- *)
  (* the two frames of a cell walk *)
  (* ---------------------------------------------------------------- *)
  let lead_word =
    concat_lsb [ lead_running.value; lead_tick.value; lead_seat.value; lead_step.value ]
  in
  let now_word = reg spec (reg spec lead_word) in
  let tick_width = width lead_tick.value in
  (* EACH FIELD IS TAKEN WHERE THE ONE BEFORE IT ENDED, and [lead_word] above states the
     order: an offset written as a cumulative sum has to be moved by hand when a field is
     added, and nothing below this frame would say that one was not. *)
  let field ~low ~width = select now_word ~high:(low + width - 1) ~low, low + width in
  let now_valid, low = field ~low:0 ~width:1 in
  let now_tick, low = field ~low ~width:tick_width in
  let now_seat, low = field ~low ~width:seat_bits in
  let now_step, (_ : int) = field ~low ~width:step_bits in
  let now_writes = now_valid &: (now_tick ==:. cell_ticks - 1) in
  let phase_done =
    now_writes &: (now_seat ==:. voices - 1) &: (now_step ==:. steps - 1)
  in
  (* ---------------------------------------------------------------- *)
  (* the units the walk drives *)
  (* ---------------------------------------------------------------- *)
  let plane_column = wire (rows * activation_bits) in
  let forward =
    Engine.create
      { Engine.I.clock = i.clock
      ; clear = i.clear
      ; start = forward_start.value
      ; plane_column
      ; logit_seat = seat.value
      ; step_taken = step_taken.value
      }
  in
  let _ = forward.step_ready -- "step_ready" in
  let _ = forward.busy -- "forward_busy" in
  let drawer =
    Drawer.create
      ~temper:e.temper
      { Drawer.I.clock = i.clock
      ; clear = i.clear
      ; start = draw_start.value
      ; logits = forward.logits
      ; uniform = u_draw
      }
  in
  let _ = drawer.busy -- "draw_busy" in
  (* THE ANNEAL ENTRY IS READ THROUGH ERA FOUR'S RULE — the address register before the
     memory and the data register behind it — thus it stands two cycles after the pass
     counter moves, and the first mask write of a phase cannot want it before its fourth
     cycle. No attribute states a memory kind: one entry each pass is a read the tools may
     hold in whatever they have spare. *)
  let alpha =
    let size = Array.length e.alpha_rom in
    (multiport_memory
       ~initialize_to:e.alpha_rom
       size
       ~write_ports:
         [| { Write_port.write_clock = i.clock
            ; write_address = zero (address_bits_for size)
            ; write_enable = gnd
            ; write_data = zero Elaboration.alpha_bits
            }
         |]
       ~read_addresses:[| reg spec pass.value |]).(0)
    |> reg spec
  in
  (* ---------------------------------------------------------------- *)
  (* the opening: one multiply, no divide, and no DSP *)
  (* ---------------------------------------------------------------- *)
  let of_seat ~width f =
    mux
      now_seat
      (List.map (Array.to_list e.openings) ~f:(fun o -> of_unsigned_int ~width (f o)))
  in
  let opened_class =
    let low = of_seat ~width:class_bits (fun o -> o.Model.low) in
    let register = of_seat ~width:width_bits (fun o -> o.Model.width) in
    (* the array owns the DSPs, thus this product is pinned like every other one outside
       it *)
    let product = Column_array.no_dsp (u_open *: register) in
    let index = select product ~high:(uniform_bits + width_bits - 1) ~low:uniform_bits in
    low +: uresize index ~width:class_bits
  in
  (* ---------------------------------------------------------------- *)
  (* THE CELL PORT: one address, three users, and no contention BY STATE. The names are
     the per-phase gate's contract. *)
  (* ---------------------------------------------------------------- *)
  let walking = sm.is Open |: sm.is Mask in
  let draw_lands = srv.is Redraw &: ~:(drawer.busy) in
  let cell_step = mux2 walking now_step served.value -- "cell_step" in
  let cell_seat = mux2 walking now_seat seat.value -- "cell_seat" in
  let write_class = (sm.is Open &: now_writes |: draw_lands) -- "write_class" in
  let cell_class = mux2 (sm.is Open) opened_class drawer.drawn -- "cell_class" in
  let write_mask = (sm.is Mask &: now_writes) -- "write_mask" in
  let cell_hidden = (u_mask <: alpha) -- "cell_hidden" in
  let sheet =
    Cells.create
      { Cells.I.clock = i.clock
      ; cell_step
      ; cell_seat
      ; write_class
      ; cell_class
      ; write_mask
      ; cell_hidden
      ; plane_step = forward.plane_step
      ; plane = forward.plane
      ; score_step = play_step.value
      }
  in
  Signal.assign plane_column sheet.plane_column;
  let _ = sheet.hidden -- "hidden" in
  (* ---------------------------------------------------------------- *)
  (* the machine *)
  (* ---------------------------------------------------------------- *)
  let enter_walk =
    [ lead_step <--. 0; lead_seat <--. 0; lead_tick <--. 0; lead_running <-- vdd ]
  in
  (* the last seat retires — the exit [next_seat] takes to [Service.Idle], stated
     combinationally so that the walk moves to [Take] on the same edge and no cycle is
     added at the seam *)
  let service_done =
    srv.is Seat
    &: ~:(sheet.hidden)
    |: (srv.is Redraw &: ~:(drawer.busy))
    &: (seat.value ==:. voices - 1)
  in
  let next_seat =
    [ if_
        (seat.value ==:. voices - 1)
        [ srv.set_next Idle ]
        [ seat <-- seat.value +:. 1; srv.set_next Seat ]
    ]
  in
  compile
    [ (* the answer is a strobe of one cycle *)
      valid <-- gnd
    ; when_
        take_byte
        [ u <-- sel_bottom u.value ~width:(uniform_bits - byte_bits) @: prng_byte ]
    ; (* THE LEAD FRAME. It runs in the cell walks alone, thus this block is inert
         everywhere else and the generator cannot take a step no phase asked for. *)
      when_
        lead_running.value
        [ prng_step <-- vdd
        ; if_
            (lead_tick.value ==:. cell_ticks - 1)
            [ lead_tick <--. 0
            ; if_
                (lead_seat.value ==:. voices - 1)
                [ lead_seat <--. 0
                ; if_
                    (lead_step.value ==:. steps - 1)
                    [ lead_running <-- gnd ]
                    [ lead_step <-- lead_step.value +:. 1 ]
                ]
                [ lead_seat <-- lead_seat.value +:. 1 ]
            ]
            [ lead_tick <-- lead_tick.value +:. 1 ]
        ]
    ; sm.switch
        [ ( State.Idle
          , [ if_
                i.rewind
                ([ pass <--. 0; play_step <--. 0; spent <-- gnd ]
                 @ enter_walk
                 @ [ sm.set_next Open ])
                [ (* PLAY. The score face answers combinationally from the cells, thus one
                     register holds the answer and the walk keeps no copy. *)
                  when_
                    i.step
                    [ held <-- mux2 spent.value (zero frame_bits) sheet.frame
                    ; valid <-- vdd
                    ; if_
                        (play_step.value ==:. steps - 1)
                        [ spent <-- vdd ]
                        [ play_step <-- play_step.value +:. 1 ]
                    ]
                ]
            ] )
        ; Open, [ when_ phase_done (enter_walk @ [ sm.set_next Mask ]) ]
        ; ( Mask
          , [ when_ phase_done [ forward_start <-- vdd; served <--. 0; sm.set_next Serve ]
            ] )
        ; ( Serve
          , [ if_
                forward.step_ready
                [ (* the service walks the offered step; the walk takes it on the very
                     edge the last seat retires *)
                  when_ service_done [ sm.set_next Take ]
                ]
                [ (* the pass ends where the engine ends: [busy] covers the head's waits,
                     thus the walk holds one wire and counts nothing *)
                  when_
                    ~:(forward.busy)
                    [ if_
                        (pass.value ==:. e.walk - 1)
                        [ sm.set_next Idle ]
                        ([ pass <-- pass.value +:. 1 ] @ enter_walk @ [ sm.set_next Mask ])
                    ]
                ]
            ] )
        ; Take, [ step_taken <-- vdd; served <-- served.value +:. 1; sm.set_next Serve ]
        ]
    ; srv.switch
        [ ( Service.Idle
          , [ when_ (sm.is Serve &: forward.step_ready) [ seat <--. 0; srv.set_next Seat ]
            ] )
          (* a standing cell costs this one cycle and nothing more *)
        ; Seat, [ if_ sheet.hidden [ tick <--. 0; srv.set_next Uniform ] next_seat ]
        ; ( Uniform
          , [ tick <-- tick.value +:. 1
            ; when_ (tick.value <:. cell_ticks) [ prng_step <-- vdd ]
            ; when_
                (tick.value ==:. uniform_ticks - 1)
                [ draw_start <-- vdd; srv.set_next Redraw ]
            ] )
          (* [drawn] is whole in the cycle [busy] falls, thus the class writes in it *)
        ; Redraw, [ when_ ~:(drawer.busy) next_seat ]
        ]
    ];
  { O.frame = held.value; valid = valid.value; idle }
;;

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* INSTRUMENT 2, AND ITS NAMES ARE A CONTRACT. The gate probes the cell port by name —
   [cell_step], [cell_seat], [write_class], [cell_class], [write_mask], [cell_hidden] —
   thus a rename cannot silently blind it. It is the ONE port all three phases share, and
   a walk whose masks stand one pass out of phase, or whose draws take a uniform out of
   order, writes the wrong thing THERE and nowhere else: the finished sheet alone would
   pass such a walk, which is era five's lesson for the third time. *)
module Bench = struct
  module Sim = Cyclesim.With_interface (I) (O)

  (* one write of the cell port, as the probe sees it *)
  type write =
    { mask : bool (** the mask face, and not the class face *)
    ; step : int
    ; seat : int
    ; value : int (** the class a face wrote, or the mask bit *)
    }

  (* The walk, driven. [rewind] runs one whole walk from the rest to the rest and clears
     the write log behind it; [play] strobes one step and gives the frame it answers. *)
  type t =
    { rewind : unit -> unit
    ; play : unit -> int
    ; writes : unit -> write list
    ; spent : State.t -> int
    (** the cycles the last walk stood in one of its own states. ONE CYCLE, ONE OWNER: a
        cycle the service is out of its rest is the service's and not the [Serve] the walk
        parks in, thus [spent Serve] reads the engine alone, exactly as it did when the
        two machines were one. *)
    ; cycles : unit -> int
    ; entered : State.t -> int option (** the first cycle of the last walk in one state *)
    ; service_spent : Service.t -> int
    (** the cycles the last walk's service stood in one of ITS states, under the same
        one-owner rule *)
    ; service_entered : Service.t -> int option
    ; waves : Hardcaml_waveterm.Waveform.t option
    }

  let harness ?(trace = false) ~e ~seed () =
    let module Drawer =
      Draw.Make (struct
        let classes = e.Elaboration.rows
      end)
    in
    (* THE TRACE IS THE NAMED SIGNALS AND NOT EVERY SIGNAL. A walk is thousands of cycles
       over an array of [rows] by [lanes] lanes, thus a full trace would cost what the
       machine could show and not what it computes. *)
    let sim =
      Sim.create
        ~config:(Cyclesim.Config.trace `All_named)
        (create ~e ~seed:(of_unsigned_int ~width:32 seed))
    in
    let waves, sim = Cyclesim.Waveform.create_if ~enabled:trace sim in
    let inp = Cyclesim.inputs sim in
    (* the traced nodes answer with the cycle that has just run, thus the ports are read
       on the same edge *)
    let out = Cyclesim.outputs ~clock_edge:Before sim in
    let node = Harness.node sim in
    let cell_step = node "cell_step" in
    let cell_seat = node "cell_seat" in
    let write_class = node "write_class" in
    let cell_class = node "cell_class" in
    let write_mask = node "write_mask" in
    let cell_hidden = node "cell_hidden" in
    let state = node "walk_state" in
    let service = node "service_state" in
    let spent = Harness.Tally.create (module State) in
    let service_spent = Harness.Tally.create (module Service) in
    let writes = ref [] in
    let cycles = ref 0 in
    (* THE STATE REGISTER READS ONE CYCLE AHEAD. A combinational node answers with the
       cycle that has just run; a register answers with the value the edge has just put
       into it, which is the state of the cycle to come. The bench therefore carries the
       reading one cycle, and the writes it counts beside it stay in step — a bench that
       did not would name every span in the wrong place and no gate would say so. *)
    let at_rest = Harness.Tally.encoded spent Idle in
    let serving_rest = Harness.Tally.encoded service_spent Service.Idle in
    let standing = ref at_rest in
    let serving = ref serving_rest in
    let cycle () =
      Cyclesim.cycle sim;
      let at = !standing in
      let sat = !serving in
      standing := Cyclesim.Node.to_int state;
      serving := Cyclesim.Node.to_int service;
      (* ONE CYCLE, ONE OWNER: a cycle the service is out of its rest is the service's and
         not the [Serve] the walk parks in *)
      if sat = serving_rest
      then Harness.Tally.count spent ~encoded:at ~cycle:!cycles
      else Harness.Tally.count service_spent ~encoded:sat ~cycle:!cycles;
      Int.incr cycles;
      let take ~mask value =
        writes
        := { mask
           ; step = Cyclesim.Node.to_int cell_step
           ; seat = Cyclesim.Node.to_int cell_seat
           ; value = Cyclesim.Node.to_int value
           }
           :: !writes
      in
      if Cyclesim.Node.to_int write_class = 1 then take ~mask:false cell_class;
      if Cyclesim.Node.to_int write_mask = 1 then take ~mask:true cell_hidden
    in
    (* A BUDGET AND NOT A WAIT: a machine that stalls must fail the gate and not hang it.
       The walk's own cost model states the pass, and the draws it cannot state are
       counted at their worst — every cell hidden. *)
    let budget =
      let cells = e.steps * Frame.voices in
      (e.walk * (Elaboration.pass_cycles e + (cells * (Drawer.busy_cycles + 16))))
      + Elaboration.cell_walk_cycles e
      (* the engine's own overheads, which the model does not count: nine cycles of
         preamble at each layer and the head's wait of about P + 9 at each step *)
      + (e.walk * e.steps * 128)
      + 4096
    in
    let rewind () =
      Harness.Tally.clear spent;
      Harness.Tally.clear service_spent;
      writes := [];
      cycles := 0;
      standing := at_rest;
      serving := serving_rest;
      inp.rewind := Bits.vdd;
      cycle ();
      inp.rewind := Bits.gnd;
      (* the walk leaves the rest on the edge behind the strobe, thus the wait cycles
         before it reads [idle] *)
      cycle ();
      let left = ref budget in
      while (not (Bits.to_bool !(out.idle))) && !left > 0 do
        cycle ();
        Int.decr left
      done;
      if !left <= 0 then failwithf "the walk did not finish inside %d cycles" budget ()
    in
    let play () =
      inp.step := Bits.vdd;
      cycle ();
      inp.step := Bits.gnd;
      let left = ref 8 in
      while (not (Bits.to_bool !(out.valid))) && !left > 0 do
        cycle ();
        Int.decr left
      done;
      if !left <= 0 then failwith "the step was not answered";
      Bits.to_int_trunc !(out.frame)
    in
    { rewind
    ; play
    ; writes = (fun () -> List.rev !writes)
    ; spent = Harness.Tally.spent spent
    ; cycles = (fun () -> !cycles)
    ; entered = Harness.Tally.entered spent
    ; service_spent = Harness.Tally.spent service_spent
    ; service_entered = Harness.Tally.entered service_spent
    ; waves
    }
  ;;
end

let%expect_test "the service of one step: the level, a standing seat, a hidden one" =
  (* THE SERVICE AS A PICTURE, at the era's own P and a sheet of three steps. The seat
     registers are the corpus's, thus this unit refuses a narrower P and the picture
     cannot shrink the draw the way the engine's picture shrinks the dwell: what a window
     holds is the ORDER of the service, and the draw's 155 cycles stand between the two.

     WINDOW ONE is the level rising and the two seats behind it. [step_ready] stands, and
     the walk reads [hidden] at seat 0 for ONE CYCLE: that seat stands, thus it costs the
     cycle and nothing more. Seat 1 hides, and the five cycles of a uniform open — three
     steps of the generator, then the cycle its last byte lands in and the cycle the whole
     24 stands in, which is the cycle that starts the draw. [draw_busy] rises behind it.

     WINDOW TWO is the close of the step. The draw of seat 3 ends; [write_class] puts the
     class it drew into the cell the port names, in the very cycle [draw_busy] falls; the
     walk strobes [step_taken]; and the level falls on the edge behind that strobe, thus
     the engine opens the next step. *)
  let model = Model.For_test.drawn ~layers:4 ~width:8 ~seed:1 in
  let e = Elaboration.create model ~steps:3 ~lanes:2 ~walk:1 in
  let h = Bench.harness ~trace:true ~e ~seed:5 () in
  h.rewind ();
  let waves = Option.value_exn h.waves ~message:"a traced run gives a waveform" in
  let served =
    Option.value_exn (h.service_entered Seat) ~message:"the walk served a step"
  in
  let taken = Option.value_exn (h.entered Take) ~message:"the walk took a step" in
  let window ~start_cycle =
    Hardcaml_waveterm.Waveform.expect
      ~display_rules:
        [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Bit
            [ "step_ready"; "hidden" ]
        ; Hardcaml_waveterm.Display_rule.port_name_is
            "walk_state"
            ~wave_format:(Wave_format.Index [ "Idl"; "Opn"; "Msk"; "Srv"; "Tak" ])
        ; Hardcaml_waveterm.Display_rule.port_name_is
            "service_state"
            ~wave_format:(Wave_format.Index [ "Idl"; "Sea"; "Uni"; "Red" ])
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Unsigned_int
            [ "cell_seat" ]
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Bit
            [ "prng_step" ]
        ; Hardcaml_waveterm.Display_rule.port_name_is
            "uniform"
            ~wave_format:Wave_format.Hex
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Bit
            [ "draw_busy"; "write_class" ]
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Unsigned_int
            [ "cell_class" ]
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Bit
            [ "step_taken" ]
        ]
      ~show_digest:false
      ~wave_width:0
      ~display_width:80
      ~start_cycle
      waves
  in
  printf "the level rose at cycle %d and the step was taken at %d\n" (served - 1) taken;
  window ~start_cycle:(served - 2);
  window ~start_cycle:(taken - 8);
  [%expect
    {|
    the level rose at cycle 3022 and the step was taken at 3510
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │step_ready        ││  ┌───────────────────────────────────────────────────────│
    │                  ││──┘                                                       │
    │hidden            ││      ┌───────────────────────────────────────────────────│
    │                  ││──────┘                                                   │
    │                  ││──────────────────────────────────────────────────────────│
    │walk_state        ││ Srv                                                      │
    │                  ││──────────────────────────────────────────────────────────│
    │                  ││────┬───┬─────────┬───────────────────────────────────────│
    │service_state     ││ Idl│Sea│Uni      │Red                                    │
    │                  ││────┴───┴─────────┴───────────────────────────────────────│
    │                  ││──────┬───────────────────────────────────────────────────│
    │cell_seat         ││ 0    │1                                                  │
    │                  ││──────┴───────────────────────────────────────────────────│
    │prng_step         ││        ┌─────┐                                           │
    │                  ││────────┘     └───────────────────────────────────────────│
    │                  ││────────────┬─┬─┬─────────────────────────────────────────│
    │uniform           ││ 868D6F     │.│.│C2F25D                                   │
    │                  ││────────────┴─┴─┴─────────────────────────────────────────│
    │draw_busy         ││                  ┌───────────────────────────────────────│
    │                  ││──────────────────┘                                       │
    │write_class       ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │                  ││──────────────────────────────────────────────────────────│
    │cell_class        ││ 0                                                        │
    │                  ││──────────────────────────────────────────────────────────│
    │step_taken        ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │step_ready        ││──────────────────┐                                       │
    │                  ││                  └───────────────────────────────────────│
    │hidden            ││──────────────────────────────────────────────────────────│
    │                  ││                                                          │
    │                  ││────────────────┬─┬───────────────────────────────────────│
    │walk_state        ││ Srv            │.│Srv                                    │
    │                  ││────────────────┴─┴───────────────────────────────────────│
    │                  ││────────────────┬─────────────────────────────────────────│
    │service_state     ││ Red            │Idl                                      │
    │                  ││────────────────┴─────────────────────────────────────────│
    │                  ││──────────────────────────────────────────────────────────│
    │cell_seat         ││ 3                                                        │
    │                  ││──────────────────────────────────────────────────────────│
    │prng_step         ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │                  ││──────────────────────────────────────────────────────────│
    │uniform           ││ 5C1232                                                   │
    │                  ││──────────────────────────────────────────────────────────│
    │draw_busy         ││──────────────┐                                           │
    │                  ││              └───────────────────────────────────────────│
    │write_class       ││              ┌─┐                                         │
    │                  ││──────────────┘ └─────────────────────────────────────────│
    │                  ││──────────────────────────────────────────────────────────│
    │cell_class        ││ 17                                                       │
    │                  ││──────────────────────────────────────────────────────────│
    │step_taken        ││                ┌─┐                                       │
    │                  ││────────────────┘ └───────────────────────────────────────│
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "where a pass spends its cycles, against the cost model" =
  (* INSTRUMENT 4'S LAST HOLE. [Elaboration] prices the engine and the cell walks; S3's
     bench measured the engine against that price. What no number covered until here is
     the machine around it — the opening, the masks and the SERVICE — thus the cost model
     of the document and the walk could part with nothing saying so.

     The measurement is at a shape a test can run and the CLAIM is about the elected rung,
     thus the bench measures the CONSTANTS and states the rung: a standing cell costs one
     cycle, a hidden cell costs the same one and then its uniform and its draw, and both
     numbers are P 48's — the test shape and the rung hold the same P, thus the constant
     carries with no scaling.

     The expected hidden cells of the rung come from the anneal table itself: a pass hides
     [alpha] of its cells, and [alpha_rom] holds the thresholds on the generator's grid.

     WHAT THE NUMBERS SAY.

     - **A cell walk costs its uniforms and two cycles more**, and the two are the write
       frame's lag: the model counts the three steps of each cell, and the machine also
       has to write the last of them.
     - **The engine inside the walk is the engine S3 measured, and the walk adds nothing
       to it.** S3's cycle bench reads 10 200 cycles for one forward at this very shape,
       with its [step_taken] tied to [step_ready]; the walk reads 10 199 in [Serve], and
       the one cycle is the tie's — it answers the level in the cycle the level rises and
       the walk leaves for [Seat] in it. Every cycle of the service is therefore the
       walk's own, and none of it is the engine waiting for something new.
     - **The service is about three percent of a pass at the rung**, which is the claim of
       the design chapter, measured. It is what Phase II's overlap would buy back beside
       the head's wait; Phase I spends it. *)
  let cells_of steps = steps * Frame.voices in
  let model = Model.For_test.drawn ~layers:6 ~width:8 ~seed:1 in
  let steps = 6 in
  let walk = 3 in
  let seed = 1 in
  let e = Elaboration.create model ~steps ~lanes:2 ~walk in
  let module Drawer =
    Draw.Make (struct
      let classes = e.rows
    end)
  in
  let h = Bench.harness ~e ~seed () in
  h.rewind ();
  let spent = h.spent in
  let served = h.service_spent in
  (* THE HIDDEN CELLS COME OUT OF THE MACHINE'S OWN COUNTER, as they do in the rung-1
     measurement below: the service takes one uniform for each cell it redraws and nothing
     else takes one there, thus its uniform cycles divided by the ticks of a uniform ARE
     the redraws. WHETHER the machine hid the right cells is not this gate's question — it
     is the walk gate's, in [jax/tests/test_rtl.py] — and this one prices the cycles. *)
  let hidden = served Uniform / uniform_ticks in
  let service = served Seat + served Uniform + served Redraw + spent Take in
  (* what one cell of the service costs, measured: the seat read that every cell pays, and
     the uniform and the draw that only a hidden one does *)
  let hidden_cell = 1 + ((served Uniform + served Redraw) / hidden) in
  printf
    "H 8, G 2, two pairs, T %d, N %d, seed %d: %d cycles, %d hidden cells of %d\n"
    steps
    walk
    seed
    (h.cycles ())
    hidden
    (walk * cells_of steps);
  printf
    "  the opening %d cycles against the model's cell walk %d\n"
    (spent Open)
    (Elaboration.cell_walk_cycles e);
  printf
    "  the masks %d, thus %d a pass against the same %d\n"
    (spent Mask)
    (spent Mask / walk)
    (Elaboration.cell_walk_cycles e);
  printf
    "  the engine %d, thus %d a pass against forward_cycles %d\n"
    (spent Serve)
    (spent Serve / walk)
    (Elaboration.forward_cycles e);
  printf
    "  the service %d: %d seat reads, %d uniform, %d draw, %d acknowledgements\n"
    service
    (served Seat)
    (served Uniform)
    (served Redraw)
    (spent Take);
  printf
    "  a standing cell 1 cycle, a hidden cell %d — the draw states %d of them\n"
    hidden_cell
    Drawer.busy_cycles;
  (* THE CLAIM, AT THE RUNG THE COST MODEL STATES. *)
  let rung =
    Elaboration.create
      (Model.For_test.drawn ~layers:16 ~width:16 ~seed:1)
      ~steps:128
      ~lanes:4
      ~walk:512
  in
  let rung_cells = cells_of rung.steps in
  let hides pass =
    Hardcaml.Bits.to_unsigned_int rung.alpha_rom.(pass) * rung_cells / (1 lsl 24)
  in
  let rung_hidden = List.sum (module Int) (List.range 0 rung.walk) ~f:hides in
  let rung_service =
    (rung.walk * rung_cells) + (rung.walk * rung.steps) + (rung_hidden * (hidden_cell - 1))
  in
  let pass =
    Elaboration.forward_cycles rung
    + Elaboration.cell_walk_cycles rung
    + (rung_service / rung.walk)
  in
  printf
    "the rung T %d, N %d, P %d, G %d: %d hidden cells over the walk, %d a pass\n"
    rung.steps
    rung.walk
    rung.rows
    rung.lanes
    rung_hidden
    (rung_hidden / rung.walk);
  printf
    "  a pass %d cycles: the engine %d, the mask %d, the service %d — the service is \
     %.1f percent\n"
    pass
    (Elaboration.forward_cycles rung)
    (Elaboration.cell_walk_cycles rung)
    (rung_service / rung.walk)
    (100.0 *. Float.of_int (rung_service / rung.walk) /. Float.of_int pass);
  [%expect
    {|
    H 8, G 2, two pairs, T 6, N 3, seed 1: 37393 cycles, 42 hidden cells of 72
      the opening 74 cycles against the model's cell walk 72
      the masks 222, thus 74 a pass against the same 72
      the engine 30243, thus 10081 a pass against forward_cycles 9696
      the service 6852: 72 seat reads, 210 uniform, 6552 draw, 18 acknowledgements
      a standing cell 1 cycle, a hidden cell 162 — the draw states 155 of them
    the rung T 128, N 512, P 48, G 4: 99604 hidden cells over the walk, 194 a pass
      a pass 1121416 cycles: the engine 1087920, the mask 1536, the service 31960 — the service is 2.8 percent
    |}]
;;

let%expect_test "the cycles of one pass at rung 1, measured" =
  (* RUNG 1'S SHAPE, RUN. A cycle count is data-independent in the forward and
     seed-dependent only in the service — the mask alone decides how many cells a pass
     redraws — thus DRAWN WEIGHTS AT A RUNG SHAPE MEASURE THE PASS EXACTLY, and no
     checkpoint enters a test. [Params.init] is licensed for that and for nothing else.

     IT STAYS AT RUNG 1 AFTER THE RUNG-2 ELECTION, deliberately: l64 is the same machine
     at a longer table — same geometry, same counters, same widths — thus rung 1 already
     answers what no small shape can, that every width, every address and every counter
     holds at T 128 and H 16, and l64 would buy four times the runtime and no new
     structure. IT COSTS ABOUT HALF A MINUTE, which is what one pass of this shape IS:
     1.17 M cycles at about 38 000 a second in Cyclesim.

     WHAT THE NUMBERS SAY. The walk bench above extrapolates the MEAN pass of the walk;
     this measures PASS 0, the HOTTEST. The anneal opens at alpha 0.9, thus pass 0 redraws
     about nine cells in ten where the mean pass redraws four in ten: the engine and the
     cell walks are the same in both, and the service is the whole of the difference. The
     playback window holds against the mean and not against this one. *)
  let model = Model.For_test.drawn ~layers:16 ~width:16 ~seed:1 in
  let e = Elaboration.create model ~steps:128 ~lanes:4 ~walk:1 in
  let cells = e.steps * Frame.voices in
  let h = Bench.harness ~e ~seed:42 () in
  h.rewind ();
  let spent = h.spent in
  let served = h.service_spent in
  let service = served Seat + served Uniform + served Redraw + spent Take in
  let hidden = served Uniform / uniform_ticks in
  printf
    "the opening and pass 0 at T %d, H %d, G %d, P %d: %d cycles\n"
    e.steps
    e.store_channels
    e.lanes
    e.rows
    (h.cycles ());
  printf
    "  the opening %d and the mask %d, against the model's cell walk %d for each\n"
    (spent Open)
    (spent Mask)
    (Elaboration.cell_walk_cycles e);
  printf
    "  the engine %d against forward_cycles %d, thus %d for the preambles and the head\n"
    (spent Serve)
    (Elaboration.forward_cycles e)
    (spent Serve - Elaboration.forward_cycles e);
  printf
    "  the service %d: %d cells redrawn of %d, at %d cycles each\n"
    service
    hidden
    cells
    (1 + ((served Uniform + served Redraw) / hidden));
  [%expect
    {|
    the opening and pass 0 at T 128, H 16, G 4, P 48: 1175273 cycles
      the opening 1538 and the mask 1538, against the model's cell walk 1536 for each
      the engine 1095885 against forward_cycles 1087920, thus 7965 for the preambles and the head
      the service 76310: 470 cells redrawn of 512, at 162 cycles each
    |}]
;;

(* ==================================================================== *)
(* The export of the RTL gate *)
(* ==================================================================== *)

module For_test = struct
  (* THE BENCH, NARROWED TO WHAT THE DRIVER OF THE RTL GATE READS. The expect tests above
     read the cycle tallies, the state encodings and the waveforms beside these three; a
     driver outside this library reads the walk alone, thus nothing else leaves. The
     driver is [bin/gate_diffusion.ml] and the gate is [jax/tests/test_rtl.py], where the
     ORACLE is the JAX twin: this side runs the circuit and states what it did, and
     nothing here states what it should have done. *)
  module Bench = struct
    type write = Bench.write =
      { mask : bool
      ; step : int
      ; seat : int
      ; value : int
      }

    type t =
      { rewind : unit -> unit
      ; play : unit -> int
      ; writes : unit -> write list
      }

    let harness ~e ~seed () =
      let bench = Bench.harness ~e ~seed () in
      { rewind = bench.rewind; play = bench.play; writes = bench.writes }
    ;;
  end
end
