(* The checkpoint seam — see checkpoint.mli for the contract. *)

open Base

type tensor = (float, Nx.float32_elt) Nx.t

let numel shape = Array.fold shape ~init:1 ~f:( * )

(* the message of the [Invalid_argument] that a rule raises, and ["no raise"] when the
   rule accepts: a gate that pins a message needs no exception handler of its own *)
let refusal f =
  match f () with
  | () -> "no raise"
  | exception Invalid_argument message -> message
;;

(* The checkpoint as the JAX trainers write it: the tensors named "0" upward, in the flat
   order of the era's [Params_data.to_list]. The gate makes the file itself, thus it reads
   no file that git ignores, and [f] holds the whole life of the file: it goes when [f]
   gives and when [f] raises. *)
let with_checkpoint tensors ~f =
  let path = Stdlib.Filename.temp_file "mgen_checkpoint" ".safetensors" in
  Exn.protect
    ~f:(fun () ->
      Nx_io.save_safetensors
        path
        (List.mapi tensors ~f:(fun index tensor -> Int.to_string index, Nx_io.P tensor));
      f path)
    ~finally:(fun () -> Stdlib.Sys.remove path)
;;

(* A reader of the checkpoint names the file in its refusal, and the file of a gate is a
   temporary one, thus the name must leave the message before an expected block holds it. *)
let scrubbed_refusal ~path f =
  String.substr_replace_all (refusal f) ~pattern:path ~with_:"<file>"
;;
