(** The sampling policy: the tempered draw of one class, its bounds, and the numbers the
    ear elected. Both eras' float references and integer twins draw through this module,
    thus the two eras and the four layers of each are comparable pick for pick and the ear
    moves a number in one place.

    The draw pipeline is: temper the raw logits against their peak, refuse each class
    whose tempered weight falls under the min-p share of the peak's, and take the class
    whose running total passes the uniform. The peak always survives the filter, thus a
    draw always exists. *)

(** [tempered raw ~temperature] is the tempered weight of each class against the peak: the
    peak weighs one, thus [min_p] is a share of the peak and one compare holds the filter. *)
val tempered : float array -> temperature:float -> float array

(** [above_min_p weights ~min_p] refuses each weight under [min_p]; zero turns the filter
    off. The weights are shares of the peak, as [tempered] gives them. *)
val above_min_p : float array -> min_p:float -> float array

(** [pick weights ~uniform] is the class whose running total passes [uniform] times the
    total. One function owns both sums, thus the walk always ends on a class that holds
    weight and no fallback exists. *)
val pick : float array -> uniform:float -> int

(** [draw_class raw ~temperature ~min_p ~uniform] is the draw of one seat as one function:
    the tempered weights, the min-p floor, and the pick. The bounds of [check_policy] hold
    here; this function does not check them. *)
val draw_class : float array -> temperature:float -> min_p:float -> uniform:float -> int

(** [check_policy ~temperature ~min_p] raises [Invalid_argument] when [temperature] is 0
    or less, or when [min_p] falls outside 0 up to 1. Every sampler states these bounds
    through this one function, thus one message answers each. *)
val check_policy : temperature:float -> min_p:float -> unit

(** The draw the ear elected on 2026-08-18, over a sweep of temperature 0.7 to 1.3 against
    min-p 0.0039 to 0.15: 1.0 and 0.05. The audition tools state them again, because no
    constant crosses the language seam. *)
val elected_temperature : float

val elected_min_p : float
