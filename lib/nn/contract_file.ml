(* The reader of a contract file — see contract_file.mli for the contract. Two facts of
   [Nx_io] shape the whole of it, and they are stated in the interface. *)
open Core

type t =
  { path : string
  ; archive : (string, Nx_io.packed) Stdlib.Hashtbl.t
  }

let open_ path = { path; archive = Nx_io.load_safetensors path }

let packed { path; archive } name =
  match Stdlib.Hashtbl.find_opt archive name with
  | None -> invalid_argf "%s has no tensor %s" path name ()
  | Some packed -> packed
;;

let values t name =
  Array.map (Nx.to_array (Nx_io.to_typed Nx.int32 (packed t name))) ~f:Int32.to_int_exn
;;

let only t name =
  match values t name with
  | [| value |] -> value
  | row ->
    invalid_argf "%s: %s holds %d values, not one" t.path name (Array.length row) ()
;;

let shape t name = Nx_io.packed_shape (packed t name)

let scale t name =
  match values t name with
  | [| q_value; q |] -> { Quantized.Constants.q_value; q }
  | row ->
    invalid_argf "%s: %s holds %d values, not two" t.path name (Array.length row) ()
;;

let tensor_count t ~beside = Stdlib.Hashtbl.length t.archive - beside
let path t = t.path
