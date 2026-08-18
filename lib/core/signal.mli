(** The POSIX signals a host tool answers.

    A signal is the one thing that arrives from outside a loop, thus the rules for
    answering one stand together here and a tool states none of them itself. *)

(** The request to stop playing: SIGINT and SIGTERM ask a play loop to leave by its
    ordinary road.

    Ctrl-C must not leave a chord ringing on the synthesizer, and an exception cannot
    carry that rule here. Measured on 5.2.0+ox, 2026-08-18: a [Sys.Break] raised out of
    the blocking sleep of a step is NOT caught by an enclosing handler. It reaches the top
    level with the notes still sounding, thus [Stdlib.Sys.catch_break] with a [try] around
    the loop drains nothing.

    The handler therefore states the exit code and nothing else. It touches no note: a
    handler must be portable, thus it holds neither the player nor the set of sounding
    pitches, and it has no business writing to the wire while a step may be halfway
    through a message. The loop reads the request at its next step and leaves the way it
    leaves at its last one, where the drain stands. The wait is one step at the most.

    Every player of the project takes this road, thus the measurement above stands one
    time and a change to the way out changes one file. *)
type stop_play

(** [watch_stop_play ()] installs the handlers for SIGINT and SIGTERM and gives the
    request they write. Call it one time, before the loop. *)
val watch_stop_play : unit -> stop_play

(** [stop_requested t] is true once a signal has arrived *)
val stop_requested : stop_play -> bool

(** [stop_code t] is the exit code the signal names — 130 for SIGINT and 143 for SIGTERM,
    which are the codes a shell reports for them — and 0 while none has arrived. *)
val stop_code : stop_play -> int

(** [exit_if_stopped t] leaves the program with [stop_code t] when a signal has arrived,
    and gives back unit when none has. A player calls it after its drain, thus the notes
    are already released when the process ends. *)
val exit_if_stopped : stop_play -> unit
