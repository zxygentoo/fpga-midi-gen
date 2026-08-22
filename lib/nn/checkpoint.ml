(* The checkpoint seam — see checkpoint.mli for the contract. *)

open Base

type tensor = (float, Nx.float32_elt) Nx.t

let numel shape = Array.fold shape ~init:1 ~f:( * )

let refusal f =
  match f () with
  | () -> "no raise"
  | exception Invalid_argument message -> message
;;

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

let scrubbed_refusal ~path f =
  String.substr_replace_all (refusal f) ~pattern:path ~with_:"<file>"
;;
