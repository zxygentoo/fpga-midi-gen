(** The texture of a walk over time: the instrument of step 2 of the plan of
    docs/transformer_model.md.

    The endless walk of the era before the packing decayed. The texture left the corpus in
    minutes and settled at a quarter of the onset rate, with no note shorter than a
    quarter note. A number over a whole draw cannot see that, because the good opening
    hides the bad end. Therefore this instrument cuts the walk into windows and reports
    each one, and the question becomes readable: do the windows hold their first values,
    or do they walk away from them?

    Read the numbers against the same three over the canonical packed stream, and never
    against one seed: the spread over seeds is wide, and the design document asks for
    twelve. *)

(** the measurements of one window *)
type window =
  { onsets : float (** the ON events for each step *)
  ; single_on : float
  (** the share of steps that hold exactly one ON. The corpus of the era gives 0.11 over
      its own walk. *)
  ; median_duration : float
  (** the median steps a note sounds, over the notes that START in this window, thus each
      note counts one time and in one window. It is [nan] when no note of the window
      closes. *)
  ; under_a_quarter : float
  (** the share of those notes that sound fewer than four steps. The corpus median is a
      quarter note already, thus the median alone cannot see the decay of the era — "no
      note shorter than a quarter note". This share can: it goes to zero there, and the
      corpus holds it near a fifth. *)
  }

(** [windows music ~span] cuts [music] into blocks of [span] steps and measures each one.
    One element of [music] is one step: its events, without the [End], as
    [Transformer.sample] gives them. The last block is short when [span] does not divide
    the walk, and its shares still divide by its own step count. *)
val windows : Token.t list list -> span:int -> window list

(** [steps_of_frames frames] is the walk a packed frame stream holds: the events of each
    step, released before struck. It reads a stream of [Jsb] into the shape this
    instrument uses, thus the corpus measures with the same code as a drawn walk.

    This is the decode of docs/transformer_model.md, and the sequencer takes it at the
    step that builds the circuit. It lives here while this instrument is its one reader. *)
val steps_of_frames : int array -> Token.t list list
