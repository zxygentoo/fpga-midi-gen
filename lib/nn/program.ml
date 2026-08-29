(* L3, the compiler of a step-frame program — see program.mli for the contract.

   NOTHING HERE DECLARES A REGISTER OR A WIRE, for the reason [Sampler] states: Hardcaml
   names an unnamed signal by the order of its creation, thus a declaration moved into
   this module would rename every signal after it. The era declares [pc], [seat], its
   state machine and the rest, and passes them in.

   FOR THE SAME REASON [idle] IS A THUNK. The era's [sm.set_next Idle] makes a constant
   when it is called, and the era called it inside [chain_done] — after the head's entry
   was built. A caller that passed an already-built list would make that constant earlier
   and move every name after it. *)
open Base
open Hardcaml
open Signal
open Always

type 'op t =
  { chain : 'op list
  ; forward : 'op list
  }

module State = struct
  type t =
    | Idle
    | Run
  [@@deriving compare ~localize, enumerate, sexp_of]
end

type compiled =
  { forward_entry : Always.t list
  ; run_body : Always.t list
  }

let at (counter : Variable.t) k = of_unsigned_int ~width:(width counter.value) k

let chain_over (counter : Variable.t) bodies =
  let last = List.length bodies - 1 in
  switch
    counter.value
    (List.mapi bodies ~f:(fun k body ->
       at counter k, if k = last then body else (counter <--. k + 1) :: body))
;;

let case_over (counter : Variable.t) bodies =
  switch counter.value (List.mapi bodies ~f:(fun k body -> at counter k, body))
;;

let compile ~build ~(pc : Variable.t) ~(seat : Variable.t) ~idle ~forward_done prog =
  let forward_length = List.length prog.forward in
  (* op [k]'s finish is op [k+1]'s entry and the pc move; the last op of a program takes
     the final actions instead *)
  let rec link index final = function
    | [] -> final, []
    | op :: rest ->
      let next_entry, tail = link (index + 1) final rest in
      let entry, body = build op ~finish:next_entry in
      entry @ [ pc <--. index ], (index, body) :: tail
  in
  let chain_head = List.hd_exn prog.chain in
  let chain_head_entry = fst (build chain_head ~finish:[]) @ [ pc <--. forward_length ] in
  let chain_done =
    [ if_ (seat.value ==:. 0) (idle ()) ([ seat <-- seat.value -:. 1 ] @ chain_head_entry)
    ]
  in
  let chain_entry, chain_bodies = link forward_length chain_done prog.chain in
  (* the chain opens at the soprano; the twin draws in the same order *)
  let enter_chain = [ seat <--. Frame.voices - 1 ] @ chain_entry in
  (* the era's own last actions of a forward, bound HERE and not passed as an argument
     expression: OCaml does not state the order it evaluates arguments in, and this is
     where the era made the signals inside them *)
  let done_ = forward_done ~enter_chain in
  let forward_entry, forward_bodies = link 0 done_ prog.forward in
  (* one parallel case, not a chain of guards: see the L3 note of an era's module comment *)
  let run_body =
    [ switch
        pc.value
        (List.map (forward_bodies @ chain_bodies) ~f:(fun (index, body) ->
           at pc index, body))
    ]
  in
  { forward_entry; run_body }
;;
