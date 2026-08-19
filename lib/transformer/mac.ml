(* The MAC and its walk — see mac.mli for the contract. The datapath registers have no
   clear: the tags decide what is real, and a stale value that no tag marks can touch
   nothing. The control registers clear. *)

open Base
open Hardcaml
open Signal

let read_latency = 2
let depth = read_latency + 2

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; go : 'a
    ; inner : 'a [@bits 9]
    ; outer : 'a [@bits 9]
    ; hold : 'a
    ; a : 'a [@bits 25]
    ; b : 'a [@bits 18]
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t =
    { ii : 'a [@bits 9]
    ; oo : 'a [@bits 9]
    ; product : 'a [@bits 43]
    ; sum : 'a [@bits 48]
    ; row_done : 'a
    ; row : 'a [@bits 9]
    ; done_ : 'a
    }
  [@@deriving hardcaml]
end

let create (i : _ I.t) : _ O.t =
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let dspec = Reg_spec.create ~clock:i.clock () in
  let run = ~:(i.hold) in
  (* the free-running data pipe *)
  let opa = reg dspec ~enable:run i.a in
  let opb = reg dspec ~enable:run i.b in
  let product = reg dspec ~enable:run (opa *+ opb) in
  let product48 = sresize product ~width:48 in
  let open Always in
  let ii = Variable.reg spec ~width:9 in
  let oo = Variable.reg spec ~width:9 in
  let row = Variable.reg spec ~width:9 in
  (* one flag, because the retire side needs none of its own: the tags carry the terms
     still in flight. [go] raises it and the last term's issue lowers it. *)
  let issuing = Variable.reg spec ~width:1 in
  let last_in = ii.value ==: i.inner -:. 1 in
  (* the tags follow a term from its address to its retirement *)
  let tag_in = issuing.value @: (ii.value ==:. 0) @: last_in in
  let tags = pipeline spec ~enable:run ~n:depth tag_in in
  let valid_r = msb tags in
  let first_r = select tags ~high:1 ~low:1 in
  let last_r = lsb tags in
  let retire = valid_r &: run in
  let row_done = retire &: last_r in
  let done_ = row_done &: (row.value ==: i.outer -:. 1) in
  (* the row's first term loads the accumulator; [sum] is whole in the retire cycle *)
  let acc, sum =
    reg_fb_and_next dspec ~enable:retire ~width:48 ~f:(fun acc ->
      mux2 first_r product48 (acc +: product48))
  in
  ignore (acc : Signal.t);
  compile
    [ when_
        run
        [ when_
            (issuing.value &: ~:(i.go))
            [ if_
                last_in
                [ ii <--. 0
                ; if_
                    (oo.value ==: i.outer -:. 1)
                    [ issuing <-- gnd ]
                    [ oo <-- oo.value +:. 1 ]
                ]
                [ ii <-- ii.value +:. 1 ]
            ]
        ; when_ row_done [ row <-- row.value +:. 1 ]
        ]
    ; (* last, thus its resets win when a walk starts in the old walk's [done_] cycle —
         the chain convention. A command also lands under hold: the walk is over when the
         caller starts one. *)
      when_ i.go [ ii <--. 0; oo <--. 0; row <--. 0; issuing <-- vdd ]
    ];
  { O.ii = ii.value; oo = oo.value; product; sum; row_done; row = row.value; done_ }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

module Sim = Cyclesim.With_interface (I) (O)

(* the operand of a term is a function of its indexes; the testbench serves them at the
   read latency, as the memories do *)
let a_of ii oo = ((7 * ii) - (3 * oo) + 2) % 200
let b_of ii oo = ((5 * ii) + (11 * oo) - 8) % 100

let oracle ~inner ~outer =
  List.init outer ~f:(fun oo ->
    List.init inner ~f:(fun ii -> a_of ii oo * b_of ii oo) |> List.fold ~init:0 ~f:( + ))
;;

(* One walk on a given simulator, giving back the (row, sum) of every [row_done] pulse.

   The feed models the memories: the operands of the term the counters named arrive
   [read_latency] cycles later, and the feed freezes under hold as the read registers do.
   [go] pulses in the observation cycle of the last walk's [done_] — the chain convention
   of the source. The simulator is a parameter so that the waveform gate can trace one. *)
let driver (sim : Sim.t) =
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let feed = Queue.create () in
  for _ = 1 to read_latency + 1 do
    Queue.enqueue feed (0, 0)
  done;
  let step ~hold =
    inp.hold := if hold then Bits.vdd else Bits.gnd;
    if not hold
    then (
      let ai, ao = Queue.dequeue_exn feed in
      inp.a := Bits.of_signed_int ~width:25 (a_of ai ao);
      inp.b := Bits.of_signed_int ~width:18 (b_of ai ao));
    Cyclesim.cycle sim;
    if not hold
    then Queue.enqueue feed (Bits.to_int_trunc !(out.ii), Bits.to_int_trunc !(out.oo))
  in
  fun ~inner ~outer ~holds ->
    inp.inner := Bits.of_unsigned_int ~width:9 inner;
    inp.outer := Bits.of_unsigned_int ~width:9 outer;
    let sums = ref [] in
    inp.go := Bits.vdd;
    step ~hold:false;
    inp.go := Bits.gnd;
    let t = ref 0 in
    (* the budget is a runaway guard and never a bound on a real walk: a walk that hits it
       gives short rows, and the checks below call that a fault *)
    while (not (Bits.to_bool !(out.done_))) && !t < 5000 do
      step ~hold:(List.mem holds !t ~equal:Int.equal);
      if Bits.to_bool !(out.row_done)
      then sums := (Bits.to_int_trunc !(out.row), Bits.to_signed_int !(out.sum)) :: !sums;
      Int.incr t
    done;
    List.rev !sums
;;

let harness () = driver (Sim.create create)

(* one row for each of [outer], in order, and each sum the reference sum *)
let walk_is_exact sums ~inner ~outer =
  List.length sums = outer
  && List.for_all2_exn sums (oracle ~inner ~outer) ~f:(fun (_, s) e -> s = e)
  && List.for_alli sums ~f:(fun k (r, (_ : int)) -> r = k)
;;

let%expect_test "the walk sums its rows, as the reference does" =
  let walk = harness () in
  let report ~inner ~outer ~holds =
    let sums = walk ~inner ~outer ~holds in
    Stdio.printf
      "inner %d outer %d holds %b: %d pulses, rows in order and sums exact: %b\n"
      inner
      outer
      (not (List.is_empty holds))
      (List.length sums)
      (walk_is_exact sums ~inner ~outer)
  in
  report ~inner:4 ~outer:3 ~holds:[];
  report ~inner:1 ~outer:5 ~holds:[];
  report ~inner:16 ~outer:2 ~holds:[];
  report ~inner:3 ~outer:4 ~holds:[ 3; 4; 7 ];
  (* back to back: the next go lands in the cycle after the last done_ observation *)
  report ~inner:2 ~outer:2 ~holds:[];
  [%expect
    {|
    inner 4 outer 3 holds false: 3 pulses, rows in order and sums exact: true
    inner 1 outer 5 holds false: 5 pulses, rows in order and sums exact: true
    inner 16 outer 2 holds false: 2 pulses, rows in order and sums exact: true
    inner 3 outer 4 holds true: 4 pulses, rows in order and sums exact: true
    inner 2 outer 2 holds false: 2 pulses, rows in order and sums exact: true
    |}]
;;

let%expect_test "the fuzz: every shape sums its rows in order, under holds anywhere" =
  let walk = harness () in
  (* The seed is fixed, thus the gate is the same gate on every machine, and the report is
     a verdict and never the drawn shapes. The walks run back to back on one simulator, as
     the source drives them, thus a walk that left state behind would show in the next. *)
  let state = Random.State.make [| 20260819 |] in
  let drawn (_ : int) =
    let inner = Random.State.int_incl state 1 24 in
    let outer = Random.State.int_incl state 1 8 in
    (* a hold may land on any cycle of the walk, thus the freeze falls inside the pipe as
       often as it falls between two rows — the case the tags must survive *)
    let held_now (_ : int) = Random.State.int_incl state 0 4 = 0 in
    let span = (inner * outer) + (4 * depth) in
    inner, outer, List.filter (List.range 0 span) ~f:held_now
  in
  (* inner 1 is the row that is its own first and last term, thus one tag carries both
     flags. The draw stops at 24; 256 is the widest row the board really walks — the [w2]
     matvec of the feed-forward, whose inner walk is 4 d at d 64 — thus the accumulator is
     exercised at the length it runs on the board and not only at the length a draw
     affords. *)
  let edges = [ 1, 1, []; 1, 8, []; 24, 1, []; 256, 1, []; 1, 4, [ 0; 1; 2; 3; 4; 5 ] ] in
  let cases = edges @ List.map (List.range 0 60) ~f:drawn in
  let fault (inner, outer, holds) =
    if walk_is_exact (walk ~inner ~outer ~holds) ~inner ~outer
    then None
    else Some (inner, outer, List.length holds)
  in
  (match List.filter_map cases ~f:fault with
   | [] ->
     Stdio.printf
       "%d walks back to back: every row in order, every sum the reference sum\n"
       (List.length cases)
   | (inner, outer, held) :: (_ : (int * int * int) list) ->
     Stdio.printf "inner %d outer %d under %d holds went wrong\n" inner outer held);
  [%expect {| 65 walks back to back: every row in order, every sum the reference sum |}]
;;

let%expect_test "the waveform of a walk: the counters issue, the tags retire" =
  (* Three terms in each of two rows. [go] clears the counters and issue begins on the
     next cycle; [ii] walks the row and [oo] the rows. A tag travels beside each term,
     thus the retire side lags the issue side by the pipe depth: [row_done] pulses [depth]
     cycles after the row's last term issued, [sum] is whole only in that cycle, and
     [done_] coincides with the last [row_done]. Nothing stands between the two rows — the
     second row's terms issue while the first row's are still in flight. *)
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let sums = driver sim ~inner:3 ~outer:2 ~holds:[] in
  (* the walk stops in the cycle [done_] rises; a few cycles after it bring that cycle
     into the picture and show the counters at rest *)
  for _ = 1 to 4 do
    Cyclesim.cycle sim
  done;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Unsigned_int)
          [ "clock"; "go"; "ii"; "oo"; "row_done"; "row"; "sum"; "done_" ]
      ]
    ~show_digest:false
    ~wave_width:0
    waves;
  Stdio.printf
    "the rows retired as %s, and the reference gives %s\n"
    (List.map sums ~f:(fun ((_ : int), s) -> Int.to_string s) |> String.concat ~sep:" ")
    (List.map (oracle ~inner:3 ~outer:2) ~f:Int.to_string |> String.concat ~sep:" ");
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌│
    │               ││ └┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘│
    │go             ││──┐                                                │
    │               ││  └─────────────────────────                       │
    │               ││────┬─┬─┬─┬─┬─┬─────────────                       │
    │ii             ││ 0  │1│2│0│1│2│0                                   │
    │               ││────┴─┴─┴─┴─┴─┴─────────────                       │
    │               ││────────┬───────────────────                       │
    │oo             ││ 0      │1                                         │
    │               ││────────┴───────────────────                       │
    │row_done       ││              ┌─┐   ┌─┐                            │
    │               ││──────────────┘ └───┘ └─────                       │
    │               ││────────────────┬─────┬─────                       │
    │row            ││ 0              │1    │2                           │
    │               ││────────────────┴─────┴─────                       │
    │               ││────┬───────┬─┬─┬─┬─┬─┬─────                       │
    │sum            ││ 0  │184    │.│.│.│.│.│597                         │
    │               ││────┴───────┴─┴─┴─┴─┴─┴─────                       │
    │done_          ││                    ┌─┐                            │
    │               ││────────────────────┘ └─────                       │
    └───────────────┘└───────────────────────────────────────────────────┘
    the rows retired as 1089 814, and the reference gives 1089 814
    |}]
;;

let%expect_test "the waveform of a hold: the walk freezes, tags and all" =
  (* The same walk with [hold] raised for three cycles in the middle of the first row.
     Every counter, every tag and the pipe stand still, thus the picture is the walk above
     with three cycles inserted and nothing else moved. The caller freezes its memory read
     registers with this same signal, or the data and the tags fall out of step. *)
  let sim = Sim.create ~config:Cyclesim.Config.trace_all create in
  let waves, sim = Cyclesim.Waveform.create sim in
  let sums = driver sim ~inner:3 ~outer:2 ~holds:[ 2; 3; 4 ] in
  for _ = 1 to 4 do
    Cyclesim.cycle sim
  done;
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.(Bit_or Unsigned_int)
          [ "clock"; "hold"; "ii"; "oo"; "row_done"; "sum"; "done_" ]
      ]
    ~show_digest:false
    ~wave_width:0
    waves;
  Stdio.printf
    "the rows retired as %s, the same sums the walk above gave\n"
    (List.map sums ~f:(fun ((_ : int), s) -> Int.to_string s) |> String.concat ~sep:" ");
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌┐┌│
    │               ││ └┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘└┘│
    │hold           ││      ┌─────┐                                      │
    │               ││──────┘     └─────────────────────                 │
    │               ││────┬─┬───────┬─┬─┬─┬─────────────                 │
    │ii             ││ 0  │1│2      │0│1│2│0                             │
    │               ││────┴─┴───────┴─┴─┴─┴─────────────                 │
    │               ││──────────────┬───────────────────                 │
    │oo             ││ 0            │1                                   │
    │               ││──────────────┴───────────────────                 │
    │row_done       ││                    ┌─┐   ┌─┐                      │
    │               ││────────────────────┘ └───┘ └─────                 │
    │               ││────┬─────────────┬─┬─┬─┬─┬─┬─────                 │
    │sum            ││ 0  │184          │.│.│.│.│.│597                   │
    │               ││────┴─────────────┴─┴─┴─┴─┴─┴─────                 │
    │done_          ││                          ┌─┐                      │
    │               ││──────────────────────────┘ └─────                 │
    └───────────────┘└───────────────────────────────────────────────────┘
    the rows retired as 1089 814, the same sums the walk above gave
    |}]
;;

let%expect_test "the product tap holds the bespoke cadence: operands, one wait, product" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  let product a b =
    inp.a := Bits.of_signed_int ~width:25 a;
    inp.b := Bits.of_signed_int ~width:18 b;
    Cyclesim.cycle sim;
    Cyclesim.cycle sim;
    Bits.to_signed_int !(out.product)
  in
  List.iter
    [ 3, 4; -100, 7; 12345, -678; 0, 999 ]
    ~f:(fun (a, b) -> Stdio.printf "%d * %d = %d\n" a b (product a b));
  [%expect
    {|
    3 * 4 = 12
    -100 * 7 = -700
    12345 * -678 = -8369910
    0 * 999 = 0
    |}]
;;
