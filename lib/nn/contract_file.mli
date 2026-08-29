(** The reader of a CONTRACT FILE: the safetensors archive each era's quantizer writes
    above the seam, and the only thing that crosses that seam for a build.

    The quantization happens above the seam, one time, thus nothing here quantizes: it
    takes the tensors, the exponents, the shape numbers and the scales as they stand, and
    each era's [check_shape] holds the rules its consumers assume.

    TWO FACTS OF [Nx_io] SHAPE EVERY READER, and they are stated here so that no era
    states them again:

    - EVERY TENSOR IS INT32, the int8 image included, because [Nx_io.load_safetensors]
      SKIPS every dtype it does not hold. An int8 tensor would arrive as a hole and the
      model would refuse for the wrong reason. The values are the int8 image all the same.
    - EVERY SCALAR TRAVELS AS A NAMED TENSOR, because [Nx_io] gives no access to
      [__metadata__]. A shape number, an exponent and a temper are each a tensor of one or
      two values, and [tensor_count] discounts them.

    Every function raises [Invalid_argument], naming the file and the tensor, when the
    archive does not hold what the caller asked for. A quantizer and this reader are two
    statements of one layout, and what holds them together is each era's netlist gate. *)

type t

(** [open_ path] reads the archive. It raises when the file is not a safetensors archive. *)
val open_ : string -> t

(** the path the archive came from — what an era names in a complaint of its own *)
val path : t -> string

(** [values t name] is one tensor, flat, in the row-major order the file carries *)
val values : t -> string -> int array

(** [only t name] is the single value of a scalar tensor. It raises when the tensor holds
    any other count. *)
val only : t -> string -> int

(** [shape t name] is the axes of one tensor, which is how an era reads a width it did not
    state as a number of its own *)
val shape : t -> string -> int array

(** [scale t name] is a two-value tensor as the (q_value, q) pair the circuits multiply by
    — the temper is the one every era carries *)
val scale : t -> string -> Quantized.Constants.scale

(** [tensor_count t ~beside] is how many WEIGHT tensors the archive holds: everything in
    it, less the [beside] scalars that travel next to them. *)
val tensor_count : t -> beside:int -> int
