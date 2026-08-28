(** The canvas: the piece as it stands, the mask over it, and the two faces that read
    them.

    Part of L4. One canvas is [steps] steps of [Frame.voices] seats, each cell a class of
    the vocabulary, beside one bit for each cell that says whether the pass has hidden it.
    It is small — a few thousand bits — thus it is registers and it answers every port at
    once.

    **THREE FACES, ONE MEMORY, AND THAT IS WHY THIS INTERFACE IS THE WIDEST OF THE
    UNITS.** The walk writes cells and reads their mask bits; the stem reads a plane
    column; the sequencer reads a frame. The two readers do not touch the classes as
    classes — they each decode them — thus a unit for each face would have to take the
    canvas out through a port and decode it outside. The faces stand together because the
    memory does.

    The two decodes are the rules of the era, and this unit restates neither:

    - **A plane column is the software half of this unit, [For_test.plane_activations].**
      The [Frame.voices] class planes carry the activation one at the row the cell holds
      and zero elsewhere, and they carry nothing at all where the cell is hidden; the
      [Frame.voices] mask planes carry the activation one at EVERY row where the cell is
      hidden, and zero where it stands. The one is [Quantized.activation_q]'s, thus the
      format has one home.
    - **A frame is [Vocab.Rtl]'s.** Each seat's class becomes its voice code and the four
      pack with seat 0 in the low byte, which is [Frame]'s rule and the sequencer's.

    What a caller must know:

    - **EVERY FACE IS COMBINATIONAL FROM THE REGISTERS.** [hidden], [plane_column] and
      [frame] all answer in the cycle their addresses stand, and no stage stands between a
      register and any of them. The stem stores nothing, thus this unit is its input
      tensor: the caller's window is the register that holds a plane column, as it holds a
      column of a store.
    - **THE CALLER OWNS THE EDGE.** A step before 0 or past [steps] - 1 is the zero
      column, and this unit does not state it: [plane_step] names a step that exists. The
      walk muxes the zero column once, for the store and for the planes alike.
    - **The cell port is one address and two writes.** The opening and a redraw write a
      class; a mask draw writes a bit; the three stand in different phases of a pass, thus
      one [cell_step] and [cell_seat] name the cell for all of them and for the mask bit
      the walk reads back.
    - **The unit takes no clear, because the memories take no clear.** The opening writes
      every class and a mask draw writes every bit before anything reads either, thus a
      cleared canvas would be a canvas no walk ever sees. The board's reset runs an
      opening.
    - **The score port is deliberately its own face.** Phase II gives the playing canvas
      its own copy — a walk rewrites this one in place — and that doubling is a change
      inside this unit when the port already stands apart. *)

open Hardcaml

(** The shape one instantiation is built for. The seats are [Frame.voices] and never a
    parameter: four is a fact of the synthesizer and not of a model. *)
module type Shape = sig
  (** T: the steps of one canvas *)
  val steps : int

  (** P: the classes of a cell, and the rows of a plane column *)
  val rows : int
end

module Make (Shape : Shape) : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; (* the walk's face: one cell, named once, written two ways *)
        cell_step : 'a (** the step of the cell the walk names *)
      ; cell_seat : 'a (** the seat of it *)
      ; write_class : 'a
      (** a strobe: the named cell takes [cell_class] — the opening, and every redraw *)
      ; cell_class : 'a
      ; write_mask : 'a
      (** a strobe: the named cell takes [cell_hidden] — the mask draw of a pass *)
      ; cell_hidden : 'a
      ; (* the stem's face *)
        plane_step : 'a (** the step whose plane column the stem reads; it exists *)
      ; plane : 'a
      (** which plane: under [Frame.voices] a class plane of that seat, at or above it the
          mask plane of the seat below by [Frame.voices] *)
      ; (* the score's face *)
        score_step : 'a (** the step whose frame the sequencer reads *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { hidden : 'a
      (** whether the pass has hidden the cell that [cell_step] and [cell_seat] name. The
          draw reads it to know which cells it must redraw. *)
      ; plane_column : 'a
      (** the plane the stem reads, as one column: [rows] activations, row 0 in the low
          bits *)
      ; frame : 'a
      (** the frame of [score_step]: the voice code of each seat, seat 0 in the low byte,
          which is what the socket carries *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block. It takes no model and no constant: a canvas holds classes,
      and what a class means is [Vocab]'s. *)
  val create : Signal.t I.t -> Signal.t O.t
end

(** The software half of the stem's decode, exported for the gates that must build the
    stem's input: [Forward]'s cycle bench and [bin/gate_diffusion.ml]. The plane face of
    the circuit must equal it, which is what this unit's own gate holds. *)
module For_test : sig
  (** [plane_activations classes hidden ~steps] is the stem's input tensor: the
      [steps; rows; 2 * Frame.voices] activations that the class planes and the mask
      planes carry over one masked canvas. A standing cell carries the one of the
      activation format at the row it holds and nothing elsewhere; a hidden cell carries
      it at EVERY row of its mask plane and nothing in its class plane. *)
  val plane_activations : int array array -> bool array array -> steps:int -> int array

  (** [plane_column x ~step ~plane] is one column of that tensor: the [rows] activations
      of one step and one plane, row 0 first. The index rule itself is
      [Diffusion.tensor_column]'s — every tensor of the era reads as
      [steps; rows; channels] — thus what this states is the plane count alone. *)
  val plane_column : int array -> step:int -> plane:int -> int array
end
