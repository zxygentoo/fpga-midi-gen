open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

(* [Idle] takes [rewind] or [step]; [Draw] advances the PRNG one time; [Grab] captures the
   draw into one row; [Note] captures the note of each voice and the mask of the voices
   that speak; [Report] gives those notes to the sequencer, one at a time. A rewind walk
   ends in [Idle] with no report, a step walk ends through [Note]. *)
module State = struct
  type t =
    | Idle
    | Draw
    | Grab
    | Note
    | Report
  [@@deriving compare ~localize, enumerate, sexp_of]
end

(* the state of one voice: its seat number, the register that holds its pitch — which is
   also the pitch of the step before — the flag that it still owes a report, and the note
   and the speak decision of this step *)
type voice_state =
  { voice : int
  ; note : Always.Variable.t
  ; owed : Always.Variable.t
  ; value : Signal.t
  ; speaks : Signal.t
  }

let create ~(model : Pink.t) ~seed (i : _ I.t) : _ O.t =
  assert (List.length model.voices >= 1 && List.length model.voices <= Source_intf.voices);
  (* the voices take the rows in order: the head of the list takes the first rows, thus it
     re-rolls at every step and it is the fastest voice *)
  let groups =
    List.folding_map model.voices ~init:0 ~f:(fun start (v : Pink.Voice.t) ->
      start + v.params.rows, (start, v))
  in
  let rows = Pink.total_rows model.voices in
  assert (rows >= 2);
  let count_bits = rows - 1 in
  let index_bits = address_bits_for rows in
  let target_bits = Int.ceil_log2 (rows + 1) in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let row = Array.init rows ~f:(fun _ -> Variable.reg spec ~width:8) in
  let count = Variable.reg spec ~width:count_bits in
  let index = Variable.reg spec ~width:index_bits in
  let target = Variable.reg spec ~width:target_bits in
  (* 1 while the walk ends with an answer; a rewind walk keeps it at 0 *)
  let emit = Variable.reg spec ~width:1 in
  (* 1 until the first step of a run answers: every voice speaks at that step *)
  let first_step = Variable.reg spec ~width:1 in
  let voice_bits = Signal.address_bits_for Source_intf.voices in
  let idle = sm.is Idle in
  (* the names put the walk into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = index.value -- "index" in
  let _ = count.value -- "step_count" in
  (* the PRNG advances in each [Draw] cycle, and [Grab] reads the new state one cycle
     later *)
  let prng =
    Prng.Rtl.create
      { Prng.Rtl.I.clock = i.clock
      ; clear = i.clear
      ; load = i.rewind &: idle
      ; seed
      ; step = sm.is Draw
      }
  in
  let draw = sel_bottom prng.value ~width:8 in
  (* the rows that one step re-rolls: the lowest set bit of the new count names the due
     rows, and a count of 0 — the wrap — re-rolls all of them, which is the clamp of the
     reference *)
  let next_count = count.value +:. 1 in
  let reroll_count =
    List.fold_right
      (List.range 0 (rows - 1))
      ~init:(of_unsigned_int ~width:target_bits rows)
      ~f:(fun k rest ->
        mux2
          (select next_count ~high:k ~low:k)
          (of_unsigned_int ~width:target_bits (k + 1))
          rest)
  in
  let last = uresize index.value ~width:target_bits +:. 1 ==: target.value in
  (* the note of one voice: the sum of its own rows, mapped with its own constants. The
     conditions of the elaboration make the mapping shifts, adds and one constant multiply
     — no divider. *)
  let mapped (params : Pink.Params.t) ~start =
    let { Pink.Params.rows = n; root; degrees; _ } = params in
    let offsets = Pink.degree_offsets ~scale:model.scale params in
    let low, window = Pink.window params in
    assert (Int.is_pow2 window);
    assert (degrees >= 2);
    List.iter offsets ~f:(fun offset ->
      assert (root + offset >= 0 && root + offset <= 127));
    let sum_bits = Int.ceil_log2 ((n * 255) + 1) in
    let x_bits = Int.floor_log2 window in
    (* the width that holds the value [degrees], and not the width of an index below it: a
       count that is a power of two needs one bit more *)
    let degree_bits = Int.ceil_log2 (degrees + 1) in
    let sum =
      List.fold
        (List.range start (start + n))
        ~init:(zero sum_bits)
        ~f:(fun acc r -> acc +: uresize row.(r).value ~width:sum_bits)
    in
    let x =
      mux2
        (sum <:. low)
        (zero x_bits)
        (mux2
           (sum >=:. low + window)
           (of_unsigned_int ~width:x_bits (window - 1))
           (uresize (sum -:. low) ~width:x_bits))
    in
    let product = x *: of_unsigned_int ~width:degree_bits degrees in
    let degree = select product ~high:(x_bits + degree_bits - 1) ~low:x_bits in
    let offset = mux degree (List.map offsets ~f:(of_unsigned_int ~width:8)) in
    of_unsigned_int ~width:8 root +: offset
  in
  (* the voices in row order, one register pair for each: its note, and 1 while it still
     owes a report. The voice number counts down while the list counts up, thus a source
     with fewer voices takes the high numbers — the melody seats. *)
  let by_row =
    List.mapi groups ~f:(fun position (start, (v : Pink.Voice.t)) ->
      let note = Variable.reg spec ~width:8 in
      let owed = Variable.reg spec ~width:1 in
      let value = mapped v.params ~start in
      (* the walk re-rolled a row of this group *)
      let rerolled = if start = 0 then vdd else target.value >:. start in
      (* a voice with no re-strike speaks only when its pitch moves. The comparison reads
         the note register one cycle before it takes the new value, thus it costs no
         register. *)
      let speaks = if v.restrike then rerolled else rerolled &: (value <>: note.value) in
      { voice = Source_intf.voices - 1 - position
      ; note
      ; owed
      ; value
      ; speaks = first_step.value |: speaks
      })
  in
  (* the report goes from the lowest voice upward — the order of the wire, and the order
     of the reference *)
  let by_voice = List.rev by_row in
  let pending =
    List.fold by_voice ~init:gnd ~f:(fun acc (a : voice_state) -> acc |: a.owed.value)
  in
  (* the lowest voice that still owes a report, and its note *)
  let selected_voice, selected_pitch =
    List.fold_right
      by_voice
      ~init:(zero voice_bits, zero 8)
      ~f:(fun (a : voice_state) (rest_voice, rest_pitch) ->
        ( mux2 a.owed.value (of_unsigned_int ~width:voice_bits a.voice) rest_voice
        , mux2 a.owed.value a.note.value rest_pitch ))
  in
  (* 1 when a voice other than the selected one still owes a report *)
  let others =
    List.fold by_voice ~init:gnd ~f:(fun acc (a : voice_state) ->
      acc |: (a.owed.value &: ~:(selected_voice ==:. a.voice)))
  in
  let valid = sm.is Report &: pending in
  compile
    [ sm.switch
        [ ( Idle
          , [ if_
                i.rewind
                [ count <--. 0
                ; index <--. 0
                ; target <--. rows
                ; emit <-- gnd
                ; first_step <-- vdd
                ; sm.set_next Draw
                ]
                [ when_
                    i.step
                    [ count <-- next_count
                    ; index <--. 0
                    ; target <-- reroll_count
                    ; emit <-- vdd
                    ; sm.set_next Draw
                    ]
                ]
            ] )
        ; Draw, [ sm.set_next Grab ]
        ; ( Grab
          , [ proc
                (List.init rows ~f:(fun r ->
                   when_ (index.value ==:. r) [ row.(r) <-- draw ]))
            ; index <-- index.value +:. 1
            ; when_ last [ if_ emit.value [ sm.set_next Note ] [ sm.set_next Idle ] ]
            ; when_ ~:last [ sm.set_next Draw ]
            ] )
        ; ( Note
          , List.concat_map by_row ~f:(fun (a : voice_state) ->
              [ a.note <-- a.value; a.owed <-- a.speaks ])
            @ [ first_step <-- gnd; sm.set_next Report ] )
        ; ( Report
          , [ if_
                pending
                [ when_
                    i.ready
                    [ proc
                        (List.map by_voice ~f:(fun (a : voice_state) ->
                           when_ (selected_voice ==:. a.voice) [ a.owed <-- gnd ]))
                    ; when_ ~:others [ sm.set_next Idle ]
                    ]
                ]
                [ sm.set_next Idle ]
            ] )
        ]
    ];
  { O.note = { Source_intf.Note.voice = selected_voice; pitch = selected_pitch }
  ; valid
  ; idle
  }
;;

(* Drive [rewind] and a run of [step] pulses, and give the report of each step back: the
   voice and the pitch of every note that speaks, from the lowest voice upward. The sink
   is always ready, thus each note transfers in the cycle that offers it. The seed is an
   elaboration constant of the test circuit, as the closure carries it in the top level. *)
let harness ~model ~seed =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 seed)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let rows = Pink.total_rows model.Pink.voices in
  inp.ready := Bits.vdd;
  (* Cycle until [ends] reports the end of the walk; the budget catches a stall. The
     source is still in [Idle] in the cycle that takes the command, thus the wait must
     cycle before it reads [idle]. *)
  let run_until ends =
    let budget = ref ((4 * rows) + (4 * Source_intf.voices) + 8) in
    let finished = ref false in
    while (not !finished) && !budget > 0 do
      Cyclesim.cycle sim;
      finished := ends ();
      Int.decr budget
    done;
    assert !finished
  in
  let rewind () =
    inp.rewind := Bits.vdd;
    Cyclesim.cycle sim;
    inp.rewind := Bits.gnd;
    run_until (fun () -> Bits.to_bool !(out.idle))
  in
  let step () =
    inp.step := Bits.vdd;
    Cyclesim.cycle sim;
    inp.step := Bits.gnd;
    let notes = ref [] in
    run_until (fun () ->
      if Bits.to_bool !(out.valid)
      then (
        notes
        := (Bits.to_int_trunc !(out.note.voice), Bits.to_int_trunc !(out.note.pitch))
           :: !notes;
        false)
      else Bits.to_bool !(out.idle));
    List.rev !notes
  in
  rewind, step
;;

(* [List.init] applies [f] in the reverse index order, thus it cannot collect from a
   simulation; the fold steps in the true order *)
let collect n step =
  List.rev (List.fold (List.range 0 n) ~init:[] ~f:(fun acc _ -> step () :: acc))
;;

(* the pitch of a voice changes only at a step where it speaks, thus the reports rebuild
   the note of every voice and the rebuild must equal the reference *)
let rebuild reports =
  let held = Array.create ~len:Source_intf.voices 0 in
  List.map reports ~f:(fun report ->
    List.iter report ~f:(fun (voice, pitch) -> held.(voice) <- pitch);
    Array.to_list held)
;;

let%expect_test "the four voices agree with the reference, note for note" =
  let seed = Control.Default.seed in
  let rewind, step = harness ~model:Pink.default ~seed in
  rewind ();
  let circuit = rebuild (collect 200 step) in
  let reference =
    let model = ref (Pink.create ~model:Pink.default ~seed) in
    collect 200 (fun () ->
      let model', states = Pink.next_step !model in
      model := model';
      List.map states ~f:(fun (s : Pink.state) -> s.note))
  in
  Stdio.print_s ([%sexp_of: int list] (List.hd_exn circuit));
  Stdio.printf "200 steps agree: %b\n" ([%compare.equal: int list list] circuit reference);
  [%expect {|
    (45 57 62 88)
    200 steps agree: true
    |}]
;;

let%expect_test "the report is the decomposition" =
  let rewind, step = harness ~model:Pink.default ~seed:Control.Default.seed in
  rewind ();
  let steps = collect 128 step in
  Stdio.printf "step  the notes that speak, from the lowest voice\n";
  List.iteri (List.take steps 16) ~f:(fun k report ->
    Stdio.printf
      "%4d %s\n"
      (k + 1)
      (String.concat
         (List.map report ~f:(fun (voice, pitch) -> Printf.sprintf "  %d:%3d" voice pitch))));
  let spoke voice =
    List.count steps ~f:(fun report -> List.exists report ~f:(fun (v, _) -> v = voice))
  in
  Stdio.printf
    "in 128 steps: bass %d, tenor %d, alto %d, soprano %d\n"
    (spoke 0)
    (spoke 1)
    (spoke 2)
    (spoke 3);
  [%expect
    {|
    step  the notes that speak, from the lowest voice
       1   0: 45  1: 57  2: 62  3: 88
       2   3: 84
       3   3: 84
       4   2: 67  3: 88
       5   3: 93
       6   3: 91
       7   3: 93
       8   2: 62  3: 74
       9   3: 76
      10   3: 86
      11   3: 93
      12   2: 64  3: 88
      13   3: 93
      14   3: 93
      15   3: 93
      16   2: 60  3: 74
    in 128 steps: bass 3, tenor 8, alto 33, soprano 128
    |}]
;;

let%expect_test "the rewind repeats the sequence, and a new seed changes it" =
  let rewind, step = harness ~model:Pink.default ~seed:7 in
  rewind ();
  let first = collect 8 step in
  rewind ();
  let again = collect 8 step in
  let rewind, step = harness ~model:Pink.default ~seed:1234 in
  rewind ();
  let other = collect 8 step in
  Stdio.printf
    "same seed repeats: %b\nnew seed differs: %b\n"
    ([%compare.equal: (int * int) list list] first again)
    (not ([%compare.equal: (int * int) list list] first other));
  [%expect {|
    same seed repeats: true
    new seed differs: true
    |}]
;;

let%expect_test "the waveform of one step, and the source holds a note while the sink \
                 waits"
  =
  (* The rewind walk fills the eight rows, two cycles for each draw. The first [step] is
     step 1: one due Draw-Grab pair, then [Note], and then the report of the four voices
     from the lowest. The sink holds [ready] at 0 for some cycles, and the source holds
     the note and [valid] until the transfer. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create
      ~config:Cyclesim.Config.trace_all
      (create ~model:Pink.default ~seed:(of_unsigned_int ~width:32 Control.Default.seed))
  in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  inp.rewind := Bits.vdd;
  Cyclesim.cycle sim;
  inp.rewind := Bits.gnd;
  Cyclesim.cycle ~n:17 sim;
  inp.step := Bits.vdd;
  Cyclesim.cycle sim;
  inp.step := Bits.gnd;
  (* the sink is not ready for three cycles *)
  Cyclesim.cycle ~n:5 sim;
  inp.ready := Bits.vdd;
  Cyclesim.cycle ~n:6 sim;
  let rules =
    [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "rewind"; "step" ]
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "state"
        ~wave_format:(Wave_format.Index [ "Idl"; "Drw"; "Grb"; "Not"; "Rep" ])
    ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "note$voice"; "note$pitch"; "valid"; "ready"; "idle" ]
    ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:0
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │rewind         ││──┐                                                │
    │               ││  └────────────────────────────────────────────────│
    │step           ││                                    ┌─┐            │
    │               ││────────────────────────────────────┘ └────────────│
    │               ││──┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┬─┬─┬─┬──────│
    │state          ││ .│.│.│.│.│.│.│.│.│.│.│.│.│.│.│.│.│Idl│.│.│.│Rep   │
    │               ││──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┴─┴─┴─┴──────│
    │               ││──────────────────────────────────────────────────┬│
    │note$voice     ││ 0                                                ││
    │               ││──────────────────────────────────────────────────┴│
    │               ││────────────────────────────────────────────┬─────┬│
    │note$pitch     ││ 00                                         │2D   ││
    │               ││────────────────────────────────────────────┴─────┴│
    │valid          ││                                            ┌──────│
    │               ││────────────────────────────────────────────┘      │
    │ready          ││                                                ┌──│
    │               ││────────────────────────────────────────────────┘  │
    │idle           ││──┐                               ┌───┐            │
    │               ││  └───────────────────────────────┘   └────────────│
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
