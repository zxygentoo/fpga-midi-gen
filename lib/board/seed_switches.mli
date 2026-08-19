(** The seed panel: the 16 slide switches write SEED, and the eight seven-segment digits
    show the cell.

    The switches and the digits are one feature, thus they are one block: the switches
    state a seed and the digits state the seed the board holds. A person who sets a value
    on 16 switches and cannot see what the board took is setting it blind, because the
    switches state their own position and not the position the board took. Therefore the
    display reads the SEED cell and never the switches, and it shows all 32 bits: the host
    writes the same cell, thus a readout of the low half alone could report a value that
    is not the seed.

    The block holds no seed of its own. It gives a strobe and a value, and [Control_regs]
    takes them as it takes the toggle of the run button: SEED has two writers, the host
    and the panel, and the last one wins. Therefore a host write stands until a switch
    moves, and a switch move applies at once whether the board plays or rests. The seed
    rule does not change — the model reads SEED at the run start, thus a move inside a run
    applies to the run after it.

    Power-on needs no first-time flag. The synchroniser and the value read last have no
    power-on value, thus they are 0: a panel that is not at zero disagrees with 0 and
    writes the cell three cycles after the power-on, and a panel at zero writes nothing
    and leaves the cell at 0, which is the value of the panel. A clear behaves as a
    power-on.

    The block reads the SEED cell and nothing else, thus it takes no part in the seed rule
    and a fault in the display cannot change a note. *)

open Hardcaml

module I : sig
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; switches : 'a
    (** the slide switches, straight from the pins and asynchronous to the clock *)
    ; seed : 'a
    (** the live view of the SEED cell: what the display shows, and not what the switches
        say *)
    }
  [@@deriving hardcaml]
end

module O : sig
  type 'a t =
    { seed_write : 'a
    (** a strobe: the synchronised switches moved, thus [seed_value] must go into the cell *)
    ; seed_value : 'a
    (** the seed the panel states: the switches in the low 16 bits and zero above them. It
        is 32 bits and not 16, because [Control_regs] must not know how wide the panel is. *)
    ; digit : 'a
    (** the digit that is lit; one hot and active low. The board calls this wire the
        anode; the interface states what it selects. *)
    ; segment : 'a (** the segments a to g of that digit; active low *)
    }
  [@@deriving hardcaml]
end

(** [create] takes no parameter: the scan period is a slice of a counter and the clock is
    the board clock, thus there is no divisor to pass. The display shows one digit at a
    time, because the eight digits share their segment wires; one digit stands for 655 us
    and the scan is 5.24 ms, which is inside the 1 ms to 16 ms of the Digilent reference.

    The decimal point is not an output. It is a shared cathode like the segments, thus it
    is a pin this design must drive, and its value is 1 for ever: the top level ties it
    off as it ties off [JD[7:1]]. The block would own it if it ever lit it. *)
val create : Signal.t I.t -> Signal.t O.t
