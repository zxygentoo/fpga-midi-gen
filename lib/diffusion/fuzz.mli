(** What the gates of era six share: the draws they take, and the two moves that put a
    drawn value on a port and read it back.

    Every random thing in this repository comes from [Prng] and the seed is an input, thus
    a fuzz that finds a fault finds it again. The draws carry the generator's state as
    [Prng] does, and a caller sequences them by threading it. The packing stands beside
    them because a gate draws its values and then packs them, and the two halves of that
    one move want one home. *)

(** [draw state ~limit] is one value in [-limit, limit] and the state behind it. *)
val draw : Prng.state -> limit:int -> Prng.state * int

(** [draw_between state ~low ~high] is one value in [low, high], both ends included. *)
val draw_between : Prng.state -> low:int -> high:int -> Prng.state * int

(** [draw_array state ~len ~limit] is [len] values of [draw], in the order the generator
    states them. *)
val draw_array : Prng.state -> len:int -> limit:int -> Prng.state * int array

(** [pack values ~width] is the values as one word, value 0 in the low bits and each one
    signed at [width] bits. It is the shape every wide port of the era takes — a column of
    rows, a row of lanes, a cell of classes — thus a gate states the port and never the
    slicing. *)
val pack : int array -> width:int -> Hardcaml.Bits.t

(** [set port value] writes [value] into a simulation port at the width the shape gave it,
    thus a bench states a number and never a width. *)
val set : Hardcaml.Bits.t ref -> int -> unit
