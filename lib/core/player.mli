(** The player: the shared half of the era players of [bin/].

    A player of an era draws a piece with the model of that era, and then does what every
    player does — print the steps, and with [-play] send them to the synthesizer. That
    second half is one loop and a few flags, and it stands here one time.

    THE STEP LINE IS A GATE CONTRACT. [step_line] prints the characters that
    [jax/midi.py]'s [step_line] prints, thus a walk of a player and a walk of its JAX twin
    compare with [diff]. That comparison is Gate C of [jax/tests/test_parity.py] for the
    canvas era, and the walk gate of the two eras before it. One character of drift breaks
    three gates, thus this format is not a formatting choice of a player.

    THE LOOP HAS ONE POINT OF VARIATION: the velocity of an onset. That is where the fade
    of the canvas era lives, and it is the only difference the three players had. Velocity
    is a fact of the onset — the synthesizer makes a control change audible only on the
    next note — thus a gesture on velocity reaches the notes that BEGIN in its window, and
    [Midi.fading] holds that rule and the measurement behind it.

    ORCHESTRATION ABOVE ONE PIECE STAYS IN THE PLAYER. A batch of seeds, a rest between
    two pieces, a banner over each: these are the playback behavior of the board of era
    six, and that prototype keeps one home in its own player. This module plays one piece
    on one open device.

    [bin/play_pink.ml] DOES NOT TAKE THIS MODULE, AND THAT IS THE DESIGN — DO NOT MIGRATE
    IT. Era one's player has no piece to play: it streams a walk with no end, frame by
    frame, and leaves by sending one silent frame. It prints nothing, thus it shares no
    step line; it holds no music, thus it shares no loop. What it carries that reads like
    a copy of [ranged] and [sleep_ms] IS a copy, and it stays one. The three players that
    draw a piece first — the transformer, the state space and the canvas — are the callers
    this module has, and the ones it is for. *)

(** The rawmidi device of the synthesizer, as the host names it. The JAX audition takes
    the same default, thus one path names the synthesizer on both sides of the seam. *)
val default_device : string

(** [ranged name address] is the argument check of a flag that carries the value of the
    control cell at [address]: the range of the register is the range of the flag. A value
    outside it prints [name] with the bounds and leaves with 2.

    A cell that the table gives no range takes any value that fits its width — SEED is
    such a cell, because the slide switches can set every one of its 32 bits and the board
    accepts them all — thus the width is the range where the table states none. The check
    is therefore total over the register table and raises on no cell in it.

    The check is on the argument, thus it stands before the tool opens the device. A value
    that passes it can raise out of no library — [Prng.create] for a seed,
    [Char.of_int_exn] for a velocity above a byte — and can make no wrong status byte, and
    neither of those is a diagnostic for a person at a command line.

    It stays public as the general form of the two checks below, for a player that gives
    another control cell a flag. *)
val ranged : string -> int -> int Core.Command.Arg_type.t

(** [channel_arg] is [ranged] over [Control_intf.Reg.channel]: MIDI channel 0 to 15. *)
val channel_arg : int Core.Command.Arg_type.t

(** [velocity_arg] is [ranged] over [Control_intf.Reg.velocity]: 1 to 127. Velocity 0 is a
    Note Off on the wire, thus the register refuses it and so does the flag. *)
val velocity_arg : int Core.Command.Arg_type.t

(** [step_ms_arg] refuses a step under 1 ms. STEP_MS has no range in the register table,
    because the circuit counts a 0 as a 1; a player of the host has no such rule and says
    so instead of playing a piece with no time in it. *)
val step_ms_arg : int Core.Command.Arg_type.t

(** [sleep_ms ms] waits [ms] milliseconds. The wait of one step of a piece. *)
val sleep_ms : int -> unit

(** [step_line ~step events] is the line of one step: [step], right-aligned in three
    columns, then two spaces, then the events. An event is ["on:P"] or ["off:P"] and one
    space joins two of them; a step that moves nothing gives ["-"]. There is no newline.

    THIS IS THE GATE CONTRACT of the module comment. Do not change these characters. *)
val step_line : step:int -> Frame.Event.t list -> string

(** [print_step ~step events] prints [step_line] and a newline, and flushes. The flush is
    the reason a player calls this and not [printf]: a walk gate reads the lines from a
    pipe, and a player that a stop signal ends must not lose the steps it played. *)
val print_step : step:int -> Frame.Event.t list -> unit

(** [open_or_die path] opens the rawmidi device at [path] for writing. On a system error
    it prints ["cannot open PATH: REASON"] and leaves with 1.

    A player that plays several pieces opens the device one time and holds the descriptor
    across them, thus the open stands apart from [play]. *)
val open_or_die : string -> Core_unix.File_descr.t

(** [play_piece fd stopped music ~step_ms ~channel ~velocity_at] plays one piece: for each
    step it prints the step line, sends the note-offs and note-ons of that step, and waits
    [step_ms]. [velocity_at ~step] gives the velocity of a note-on of that step, and it is
    read one time for each step.

    [stopped] ends the piece: the loop reads the request at the top of a step and leaves
    by the drain, thus the wait is one step at the most. The drain releases every pitch
    that still sounds — a piece ends with its last chord ringing, as a chorale does, and
    Ctrl-C must not leave one on the synthesizer. The drain runs on the way out of an
    exception too. It does not leave the program: the caller of the last piece calls
    [Signal.exit_if_stopped], thus a batch of pieces stops as one walk and not as its
    first piece.

    [fd] and [stopped] are arguments and not private state, because a player of several
    pieces opens one device and watches for one stop across all of them. *)
val play_piece
  :  Core_unix.File_descr.t
  -> Signal.stop_play
  -> Frame.Event.t list list
  -> step_ms:int
  -> channel:int
  -> velocity_at:(step:int -> int)
  -> unit

(** [play music ~device ~step_ms ~channel ~velocity] is the whole of a one-piece player's
    [-play] path: it opens [device], watches for a stop, plays [music] at a constant
    velocity, and leaves with the code of the signal if one arrived.

    A player of one piece needs nothing else. A player of several pieces builds the same
    path from [open_or_die], [Signal.watch_stop_play], [play_piece] and
    [Signal.exit_if_stopped], and holds its own rests and banners between the calls. *)
val play
  :  Frame.Event.t list list
  -> device:string
  -> step_ms:int
  -> channel:int
  -> velocity:int
  -> unit
