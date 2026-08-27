(* The bench fixture — see harness.mli. *)

open Core
open Hardcaml

let set port value = port := Bits.of_unsigned_int ~width:(Bits.width !port) value

let pack values ~width =
  Bits.concat_lsb (List.map (Array.to_list values) ~f:(Bits.of_signed_int ~width))
;;

let unpack word ~width =
  Bits.split_lsb ~part_width:width word |> List.map ~f:Bits.to_signed_int |> Array.of_list
;;

let node sim name =
  Option.value_exn (Cyclesim.lookup_node_or_reg_by_name sim name) ~message:name
;;

module type Enumerated = sig
  type t

  val all : t list
  val compare : t -> t -> int
end

module Tally = struct
  type 'state t =
    { index : 'state -> int
    ; spent : int array
    ; entered : int option array
    }

  let create (type state) (module State : Enumerated with type t = state) =
    let index which =
      fst (List.findi_exn State.all ~f:(fun (_ : int) s -> State.compare s which = 0))
    in
    let count = List.length State.all in
    { index; spent = Array.create ~len:count 0; entered = Array.create ~len:count None }
  ;;

  let encoded t which = t.index which

  let count t ~encoded ~cycle =
    t.spent.(encoded) <- t.spent.(encoded) + 1;
    if Option.is_none t.entered.(encoded) then t.entered.(encoded) <- Some cycle
  ;;

  let clear t =
    Array.fill t.spent ~pos:0 ~len:(Array.length t.spent) 0;
    Array.fill t.entered ~pos:0 ~len:(Array.length t.entered) None
  ;;

  let spent t which = t.spent.(t.index which)
  let entered t which = t.entered.(t.index which)
end
