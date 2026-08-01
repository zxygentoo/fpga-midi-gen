open Base
open Hardcaml
open Signal
module I = Source_intf.I
module O = Source_intf.O

(* [Idle] takes [rewind] or [step]; [Draw] advances the PRNG one time; [Grab] captures the
   draw into one row; [Note] captures the mapped note. A rewind walk ends in [Idle] with
   no note, a step walk ends through [Note]. *)
module State = struct
  type t =
    | Idle
    | Draw
    | Grab
    | Note
  [@@deriving compare ~localize, enumerate, sexp_of]
end

let create ~(params : Pink.Params.t) ~seed (i : _ I.t) : _ O.t =
  let { Pink.Params.rows; root; degrees; scale = _; stretch } = params in
  let offsets = Pink.degree_offsets params in
  let window = rows * 256 / stretch in
  let low = ((rows * 256) - window) / 2 in
  (* the conditions that make the mapping shifts and adds *)
  assert (rows >= 2);
  assert (Int.is_pow2 window);
  List.iter offsets ~f:(fun offset -> assert (root + offset >= 0 && root + offset <= 127));
  let count_bits = rows - 1 in
  let index_bits = address_bits_for rows in
  let target_bits = Int.ceil_log2 (rows + 1) in
  let sum_bits = Int.ceil_log2 ((rows * 255) + 1) in
  let x_bits = Int.floor_log2 window in
  let degree_bits = Int.ceil_log2 degrees in
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let sm = State_machine.create (module State) spec in
  let row = Array.init rows ~f:(fun _ -> Variable.reg spec ~width:8) in
  let count = Variable.reg spec ~width:count_bits in
  let index = Variable.reg spec ~width:index_bits in
  let target = Variable.reg spec ~width:target_bits in
  (* 1 while the walk ends with a note; a load walk keeps it at 0 *)
  let emit = Variable.reg spec ~width:1 in
  let note = Variable.reg spec ~width:8 in
  let valid = Variable.reg spec ~width:1 in
  let ready = sm.is Idle in
  (* the names put the walk into the waveform tests *)
  let _ = sm.current -- "state" in
  let _ = index.value -- "index" in
  let _ = count.value -- "step_count" in
  (* the PRNG advances in each [Draw] cycle, and [Grab] reads the new state one cycle
     later *)
  let prng =
    Prng.create
      { Prng.I.clock = i.clock
      ; clear = i.clear
      ; load = i.rewind &: ready
      ; seed
      ; step = sm.is Draw
      }
  in
  let draw = sel_bottom prng.value ~width:8 in
  (* the rows that one step re-rolls: the lowest set bit of the new count names the due
     rows, and a count of 0 — the wrap — re-rolls all of them, which is the clamp of the
     reference *)
  let next_count = count.value +:. 1 in
  let due =
    let rec go k =
      if k = rows - 1
      then of_unsigned_int ~width:target_bits rows
      else
        mux2
          (select next_count ~high:k ~low:k)
          (of_unsigned_int ~width:target_bits (k + 1))
          (go (k + 1))
    in
    go 0
  in
  let last = uresize index.value ~width:target_bits +:. 1 ==: target.value in
  let sum =
    List.fold (Array.to_list row) ~init:(zero sum_bits) ~f:(fun acc r ->
      acc +: uresize r.value ~width:sum_bits)
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
  let mapped = of_unsigned_int ~width:8 root +: offset in
  compile
    [ valid <-- gnd
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
                    ; target <-- due
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
        ; Note, [ note <-- mapped; valid <-- vdd; sm.set_next Idle ]
        ]
    ];
  { O.note = note.value; valid = valid.value; ready }
;;

(* drive [rewind] and a run of [step] pulses, and give each note back; the seed is an
   elaboration constant of the test circuit, as the closure carries it in the top level *)
let harness ~seed =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create (create ~params:Pink.Params.default ~seed:(of_unsigned_int ~width:32 seed))
  in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let wait_ready () =
    let budget = ref (4 * 8 * Pink.Params.default.rows) in
    while (not (Bits.to_bool !(out.ready))) && !budget > 0 do
      Cyclesim.cycle sim;
      Int.decr budget
    done;
    assert (Bits.to_bool !(out.ready))
  in
  let rewind () =
    inp.rewind := Bits.vdd;
    Cyclesim.cycle sim;
    inp.rewind := Bits.gnd;
    wait_ready ()
  in
  let step () =
    inp.step := Bits.vdd;
    Cyclesim.cycle sim;
    inp.step := Bits.gnd;
    let budget = ref (4 * 8 * Pink.Params.default.rows) in
    let result = ref None in
    while Option.is_none !result && !budget > 0 do
      Cyclesim.cycle sim;
      if Bits.to_bool !(out.valid) then result := Some (Bits.to_int_trunc !(out.note));
      Int.decr budget
    done;
    Option.value_exn !result
  in
  rewind, step
;;

(* [List.init] applies [f] in the reverse index order, thus it cannot collect from a
   simulation; the fold steps in the true order *)
let collect n step =
  List.rev (List.fold (List.range 0 n) ~init:[] ~f:(fun acc _ -> step () :: acc))
;;

let%expect_test "the stream comparison against the reference" =
  let seed = Control.Default.seed in
  let rewind, step = harness ~seed in
  rewind ();
  let circuit = collect 200 step in
  let reference =
    Pink.notes Pink.Params.default ~seed
    |> (fun sequence -> Sequence.take sequence 200)
    |> Sequence.to_list
  in
  Stdio.print_s ([%sexp_of: int list] (List.take circuit 16));
  Stdio.printf "200 notes agree: %b\n" ([%compare.equal: int list] circuit reference);
  [%expect
    {|
    (86 84 84 91 93 93 93 81 84 86 91 88 91 93 93 79)
    200 notes agree: true
    |}]
;;

let%expect_test "the rewind repeats the sequence, and a new seed changes it" =
  let rewind, step = harness ~seed:7 in
  rewind ();
  let first = collect 8 step in
  rewind ();
  let again = collect 8 step in
  let rewind, step = harness ~seed:1234 in
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

let%expect_test "the waveform of one step walk" =
  (* the rewind walk fills the eight rows, two cycles for each draw. The first [step] is
     step 1: one due Draw-Grab pair, then [Note] and the [valid] strobe one cycle later,
     with the note held. *)
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim =
    Sim.create
      ~config:Cyclesim.Config.trace_all
      (create
         ~params:Pink.Params.default
         ~seed:(of_unsigned_int ~width:32 Control.Default.seed))
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
  Cyclesim.cycle ~n:5 sim;
  let rules =
    [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "rewind"; "step" ]
    ; Hardcaml_waveterm.Display_rule.port_name_is
        "state"
        ~wave_format:(Wave_format.Index [ "Idl"; "Drw"; "Grb"; "Not" ])
    ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "index"; "step_count"; "note"; "valid"; "ready" ]
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
    │               ││  └─────────────────────────────────────────────   │
    │step           ││                                    ┌─┐            │
    │               ││────────────────────────────────────┘ └─────────   │
    │               ││──┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┬─┬─┬─┬───   │
    │state          ││ .│.│.│.│.│.│.│.│.│.│.│.│.│.│.│.│.│Idl│.│.│.│Idl   │
    │               ││──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┴─┴─┴─┴───   │
    │               ││──────┬───┬───┬───┬───┬───┬───┬───┬───────┬─────   │
    │index          ││ 0    │1  │2  │3  │4  │5  │6  │7  │0      │1       │
    │               ││──────┴───┴───┴───┴───┴───┴───┴───┴───────┴─────   │
    │               ││──────────────────────────────────────┬─────────   │
    │step_count     ││ 00                                   │01          │
    │               ││──────────────────────────────────────┴─────────   │
    │               ││────────────────────────────────────────────┬───   │
    │note           ││ 00                                         │56    │
    │               ││────────────────────────────────────────────┴───   │
    │valid          ││                                            ┌─┐    │
    │               ││────────────────────────────────────────────┘ └─   │
    │ready          ││──┐                               ┌───┐     ┌───   │
    │               ││  └───────────────────────────────┘   └─────┘      │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
