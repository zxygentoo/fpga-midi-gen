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
  let running = Variable.reg spec ~width:1 in
  let issued_all = Variable.reg spec ~width:1 in
  let issuing = running.value &: ~:(issued_all.value) in
  let last_in = ii.value ==: i.inner -:. 1 in
  (* the tags follow a term from its address to its retirement *)
  let tag_in = issuing @: (ii.value ==:. 0) @: last_in in
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
            (issuing &: ~:(i.go))
            [ if_
                last_in
                [ ii <--. 0
                ; if_
                    (oo.value ==: i.outer -:. 1)
                    [ issued_all <-- vdd ]
                    [ oo <-- oo.value +:. 1 ]
                ]
                [ ii <-- ii.value +:. 1 ]
            ]
        ; when_ row_done [ row <-- row.value +:. 1 ]
        ; when_ done_ [ running <-- gnd ]
        ]
    ; (* last, thus its resets win when a walk starts in the old walk's [done_] cycle —
         the chain convention. A command also lands under hold: the walk is over when the
         caller starts one. *)
      when_ i.go [ ii <--. 0; oo <--. 0; row <--. 0; running <-- vdd; issued_all <-- gnd ]
    ];
  { O.ii = ii.value; oo = oo.value; product; sum; row_done; row = row.value; done_ }
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "the walk sums its rows, as the reference does" =
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create create in
  let inp = Cyclesim.inputs sim in
  let out = Cyclesim.outputs sim in
  (* the operand of a term is a function of its indexes; the testbench serves them at the
     read latency, as the memories do *)
  let a_of ii oo = ((7 * ii) - (3 * oo) + 2) % 200 in
  let b_of ii oo = ((5 * ii) + (11 * oo) - 8) % 100 in
  let oracle inner outer =
    List.init outer ~f:(fun oo ->
      List.init inner ~f:(fun ii -> a_of ii oo * b_of ii oo) |> List.fold ~init:0 ~f:( + ))
  in
  let signed_of width bits =
    let v = Bits.to_int_trunc bits in
    if v land (1 lsl (width - 1)) <> 0 then v - (1 lsl width) else v
  in
  (* the feed models the memories: the operands of the term the counters named arrive
     [read_latency] cycles later, and the feed freezes under hold as the read registers
     do. [go] pulses in the observation cycle of the last walk's [done_] — the chain
     convention of the source. *)
  let feed = Queue.create () in
  Queue.enqueue feed (0, 0);
  Queue.enqueue feed (0, 0);
  Queue.enqueue feed (0, 0);
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
  let walk ~inner ~outer ~holds =
    inp.inner := Bits.of_unsigned_int ~width:9 inner;
    inp.outer := Bits.of_unsigned_int ~width:9 outer;
    let sums = ref [] in
    let pulses = ref 0 in
    let sample () =
      if Bits.to_bool !(out.row_done)
      then (
        Int.incr pulses;
        sums := (Bits.to_int_trunc !(out.row), signed_of 48 !(out.sum)) :: !sums)
    in
    inp.go := Bits.vdd;
    step ~hold:false;
    inp.go := Bits.gnd;
    let t = ref 0 in
    while (not (Bits.to_bool !(out.done_))) && !t < 500 do
      step ~hold:(List.mem holds !t ~equal:Int.equal);
      sample ();
      Int.incr t
    done;
    let sums = List.rev !sums in
    let expected = oracle inner outer in
    let ok =
      List.for_all2_exn sums expected ~f:(fun (_, s) e -> s = e)
      && List.for_alli sums ~f:(fun k (r, _) -> r = k)
    in
    Stdio.printf
      "inner %d outer %d holds %b: %d pulses, rows in order and sums exact: %b\n"
      inner
      outer
      (not (List.is_empty holds))
      !pulses
      ok
  in
  walk ~inner:4 ~outer:3 ~holds:[];
  walk ~inner:1 ~outer:5 ~holds:[];
  walk ~inner:16 ~outer:2 ~holds:[];
  walk ~inner:3 ~outer:4 ~holds:[ 3; 4; 7 ];
  (* back to back: the next go lands in the cycle after the last done_ observation *)
  walk ~inner:2 ~outer:2 ~holds:[];
  [%expect
    {|
    inner 4 outer 3 holds false: 3 pulses, rows in order and sums exact: true
    inner 1 outer 5 holds false: 5 pulses, rows in order and sums exact: true
    inner 16 outer 2 holds false: 2 pulses, rows in order and sums exact: true
    inner 3 outer 4 holds true: 4 pulses, rows in order and sums exact: true
    inner 2 outer 2 holds false: 2 pulses, rows in order and sums exact: true
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
    let v = Bits.to_int_trunc !(out.product) in
    if v land (1 lsl 42) <> 0 then v - (1 lsl 43) else v
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
