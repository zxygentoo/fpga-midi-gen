(** The draws a gate takes: pseudo-random values from the generator of the project.

    Every random thing in this repository comes from [Prng] and the seed is an input, thus
    a fuzz that finds a fault finds it again. These are the three shapes of draw the gates
    of era six want; they carry the generator's state as [Prng] does, and a caller
    sequences them by threading it. *)

(** [draw state ~limit] is one value in [-limit, limit] and the state behind it. *)
val draw : Prng.state -> limit:int -> Prng.state * int

(** [draw_between state ~low ~high] is one value in [low, high], both ends included. *)
val draw_between : Prng.state -> low:int -> high:int -> Prng.state * int

(** [draw_array state ~len ~limit] is [len] values of [draw], in the order the generator
    states them. *)
val draw_array : Prng.state -> len:int -> limit:int -> Prng.state * int array
