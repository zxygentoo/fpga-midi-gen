(** The checkpoint seam: the naming rule of the safetensors file the JAX trainers write,
    as one writer for every gate that crosses it.

    A checkpoint holds tensors named "0" upward, in the flat order of the era's
    [Params_data.to_list]. The order itself is the era's — the parameter structures of the
    two eras are different types — but the naming rule, the temporary file of a gate and
    the scrubbing of its name are one thing, and they stand here. *)

(** every tensor of the float models is float32 *)
type tensor = (float, Nx.float32_elt) Nx.t

(** [numel shape] is the element count of a shape. The quantization of each era sizes its
    flat tensors from its own shape table, and the product is the step between the two. *)
val numel : int array -> int

(** [refusal f] is the message of the [Invalid_argument] that [f] raises, and ["no raise"]
    when [f] raises nothing: a gate that pins a message needs no exception handler of its
    own. *)
val refusal : (unit -> unit) -> string

(** [with_checkpoint tensors ~f] writes [tensors] to a temporary safetensors file under
    the naming rule — the file the JAX trainers write — and gives [f] its path. The file
    goes when [f] gives and when [f] raises, thus no gate reads a file that git ignores. *)
val with_checkpoint : tensor list -> f:(string -> 'a) -> 'a

(** [scrubbed_refusal ~path f] is [refusal f] with [path] replaced by ["<file>"]: a reader
    of the checkpoint names the file in its refusal, and the file of a gate is a temporary
    one, thus the name must leave the message before an expected block holds it. *)
val scrubbed_refusal : path:string -> (unit -> unit) -> string
