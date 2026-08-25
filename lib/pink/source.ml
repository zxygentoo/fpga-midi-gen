open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

(* [Idle] takes [rewind] or [step]; [Draw] advances the PRNG one time; [Grab] captures the
   draw into one row; [Frame] packs the note of each voice into the frame and answers the
   step. A rewind walk ends in [Idle] with no answer, a step walk ends through [Frame]. *)
module State = struct
  type t =
    | Idle
    | Draw
    | Grab
    | Frame
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(model : Pink.t) ~seed (i : _ I.t) : _ O.t =
  assert (List.length model.voices >= 1 && List.length model.voices <= Frame.voices);
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
  (* the frame of the step, and the strobe that answers [step] with it *)
  let held = Variable.reg spec ~width:(Frame.code_bits * Frame.voices) in
  let valid = Variable.reg spec ~width:1 in
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
  (* The frame of the step, seat by seat. The row order puts the highest voice first and
     seat 0 is the lowest, thus the list turns around; a model with fewer voices leaves
     the low seats silent and takes the melody seats. A voice of this model never rests,
     thus each code it fills carries the sounding flag.

     No voice needs a register of its own. A frame states which pitch a voice holds and
     not that the voice struck it, thus the pitch of the step before decides nothing, and
     the note of each voice is combinational from the rows. *)
  let seats =
    Array.create ~len:Frame.voices (of_unsigned_int ~width:8 Frame.silent_code)
  in
  List.iteri groups ~f:(fun position (start, (v : Pink.Voice.t)) ->
    let sounds = of_unsigned_int ~width:1 1 in
    seats.(Frame.voices - 1 - position)
    <- concat_msb
         [ sounds; sel_bottom (mapped v.params ~start) ~width:(Frame.code_bits - 1) ]);
  let frame_value = concat_lsb (Array.to_list seats) in
  compile
    [ (* the answer is a strobe of one cycle *)
      valid <-- gnd
    ; sm.switch
        [ ( Idle
          , [ if_
                i.rewind
                [ count <--. 0
                ; index <--. 0
                ; target <--. rows
                ; emit <-- gnd
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
            ; when_ last [ if_ emit.value [ sm.set_next Frame ] [ sm.set_next Idle ] ]
            ; when_ ~:last [ sm.set_next Draw ]
            ] )
          (* the rows hold the values of this step, thus the frame stands and one cycle
             latches it beside the strobe that answers *)
        ; Frame, [ held <-- frame_value; valid <-- vdd; sm.set_next Idle ]
        ]
    ];
  { O.frame = held.value; valid = valid.value; idle }
;;

(* Drive [rewind] and a run of [step] pulses, and give the frame that each step states.
   There is no [ready]: the sequencer strobes and waits, thus a step ends at the strobe of
   [valid]. The seed is an elaboration constant of the test circuit, as the closure
   carries it in the top level. *)
let harness ~model ~seed =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create (create ~model ~seed:(of_unsigned_int ~width:32 seed)) in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs ~clock_edge:Before sim in
  let rows = Pink.total_rows model.Pink.voices in
  (* Cycle until [ends] reports the end of the walk; the budget catches a stall. The
     source is still in [Idle] in the cycle that takes the command, thus the wait must
     cycle before it reads [idle]. *)
  let run_until ends =
    let budget = ref ((4 * rows) + 8) in
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
    run_until (fun () -> Bits.to_bool !(out.valid));
    Bits.to_int_trunc !(out.frame)
  in
  rewind, step
;;

(* [List.init] applies [f] in the reverse index order, thus it cannot collect from a
   simulation; the fold steps in the true order *)
let collect n step =
  List.rev (List.fold (List.range 0 n) ~init:[] ~f:(fun acc _ -> step () :: acc))
;;

let%expect_test "the frames agree with the reference, step for step" =
  let seed = Control_intf.Default.seed in
  let rewind, step = harness ~model:Pink.default ~seed in
  rewind ();
  let circuit = collect 200 step in
  let reference =
    List.folding_map
      (List.range 0 200)
      ~init:(Pink.create ~model:Pink.default ~seed)
      ~f:(fun walk (_ : int) -> Pink.next_frame walk)
  in
  Stdio.print_s ([%sexp_of: int list] (Frame.pitches (List.hd_exn circuit)));
  Stdio.printf "200 steps agree: %b\n" ([%compare.equal: int list] circuit reference);
  [%expect {|
    (45 57 62 88)
    200 steps agree: true
    |}]
;;

(* The decomposition is the rhythm, and the frame states it as movement and not as
   articulation: a seat whose pitch does not move states the same code again, thus the
   note sustains. The counts below are the steps at which each seat MOVES, and the period
   of the group is what they measure — 1, 4, 16 and 64 steps. *)
let%expect_test "the frames are the decomposition" =
  let rewind, step = harness ~model:Pink.default ~seed:Control_intf.Default.seed in
  rewind ();
  let frames = collect 128 step in
  Stdio.printf "step  the pitch of each seat, the bass first\n";
  List.iteri (List.take frames 16) ~f:(fun k frame ->
    Stdio.printf
      "%4d %s\n"
      (k + 1)
      (String.concat
         (List.map (Frame.codes frame) ~f:(fun code ->
            match Frame.pitch_of_code code with
            | None -> "    -"
            | Some pitch -> Printf.sprintf "  %3d" pitch))));
  let moves seat =
    List.folding_map frames ~init:(-1) ~f:(fun held frame ->
      let code = List.nth_exn (Frame.codes frame) seat in
      code, if code = held then 0 else 1)
    |> List.sum (module Int) ~f:Fn.id
  in
  Stdio.printf
    "in 128 steps the seats move: bass %d, tenor %d, alto %d, soprano %d\n"
    (moves 0)
    (moves 1)
    (moves 2)
    (moves 3);
  [%expect
    {|
    step  the pitch of each seat, the bass first
       1    45   57   62   88
       2    45   57   62   84
       3    45   57   62   84
       4    45   57   67   88
       5    45   57   67   93
       6    45   57   67   91
       7    45   57   67   93
       8    45   57   62   74
       9    45   57   62   76
      10    45   57   62   86
      11    45   57   62   93
      12    45   57   64   88
      13    45   57   64   93
      14    45   57   64   93
      15    45   57   64   93
      16    45   57   60   74
    in 128 steps the seats move: bass 3, tenor 8, alto 22, soprano 103
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
    ([%compare.equal: int list] first again)
    (not ([%compare.equal: int list] first other));
  [%expect {|
    same seed repeats: true
    new seed differs: true
    |}]
;;

let%expect_test "the waveform of one step" =
  (* The rewind walk fills the eight rows, two cycles for each draw, and it ends in [Idle]
     with no answer. The first [step] is step 1: one due Draw-Grab pair, then the frame
     state, and [valid] answers one cycle later beside [idle]. There is no [ready] and no
     report walk — one strobe carries the whole step. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create
      ~config:Cyclesim.Config.trace_all
      (create
         ~model:Pink.default
         ~seed:(of_unsigned_int ~width:32 Control_intf.Default.seed))
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
  Cyclesim.cycle ~n:6 sim;
  let rules =
    [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "rewind"; "step" ]
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "state"
        ~wave_format:(Wave_format.Index [ "Idl"; "Drw"; "Grb"; "Frm" ])
    ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "frame"; "valid"; "idle" ]
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
    │               ││  └─────────────────────────────────────────────── │
    │step           ││                                    ┌─┐            │
    │               ││────────────────────────────────────┘ └─────────── │
    │               ││──┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┬─┬─┬─┬───── │
    │state          ││ .│.│.│.│.│.│.│.│.│.│.│.│.│.│.│.│.│Idl│.│.│.│Idl   │
    │               ││──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┴─┴─┴─┴───── │
    │               ││────────────────────────────────────────────┬───── │
    │frame          ││ 00000000                                   │D8BE. │
    │               ││────────────────────────────────────────────┴───── │
    │valid          ││                                            ┌─┐    │
    │               ││────────────────────────────────────────────┘ └─── │
    │idle           ││──┐                               ┌───┐     ┌───── │
    │               ││  └───────────────────────────────┘   └─────┘      │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
