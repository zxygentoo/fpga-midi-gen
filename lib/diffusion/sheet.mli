(** The sheet: the piece as it stands, the mask over it, and the two faces that read them.

    One sheet is [steps] steps of [Frame.voices] seats, each cell a class of the
    vocabulary, beside one bit for each cell that says whether the pass has hidden it. It
    is a few thousand bits, thus it is registers and it answers every port at once.

    THREE FACES, ONE MEMORY. The walk writes cells and reads their mask bits; the stem
    reads a plane column; the sequencer reads a frame. The two readers each DECODE the
    classes rather than take them as classes, thus a unit for each face would have to take
    the sheet out through a port and decode it outside. The faces stand together because
    the memory does — which is why this interface is the widest of the units.

    The two decodes are the rules of the era, and this unit restates neither: a plane
    column is [For_test.plane_activations], and a frame is [Vocab.Rtl]'s.

    What a caller must know:

    - **EVERY FACE IS COMBINATIONAL FROM THE REGISTERS.** [hidden], [plane_column] and
      [frame] all answer in the cycle their addresses stand. The stem stores nothing, thus
      this unit is its input tensor.
    - **THE CALLER OWNS THE EDGE.** A step before 0 or past [steps] - 1 is the zero
      column, and this unit does not state it: [plane_step] names a step that exists. The
      walk muxes the zero column once, for the store and for the planes alike.
    - **The cell port is one address and two writes.** A class write and a mask write
      stand in different phases of a pass, thus one [cell_step] and [cell_seat] name the
      cell for both and for the mask bit the walk reads back.
    - **The unit takes no clear, because the memories take no clear.** The opening writes
      every class and a mask draw writes every bit before anything reads either. The
      board's reset runs an opening.
    - **The score port is deliberately its own face**, so that Phase II can give the
      playing sheet its own copy inside this unit. *)

open Hardcaml

(** The shape one instantiation is built for. The seats are [Frame.voices] and never a
    parameter: four is a fact of the synthesizer. *)
module type Shape = sig
  (** T: the steps of one sheet *)
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
      (** whether the pass has hidden the cell [cell_step] and [cell_seat] name; the draw
          reads it to know which cells it must redraw *)
      ; plane_column : 'a
      (** the plane the stem reads, as one column: [rows] activations, row 0 in the low
          bits *)
      ; frame : 'a
      (** the frame of [score_step]: the voice code of each seat, seat 0 in the low byte,
          which is what the socket carries *)
      }
    [@@deriving hardcaml]
  end

  (** [create i] is the block. It takes no model and no constant: what a class means is
      [Vocab]'s. *)
  val create : Signal.t I.t -> Signal.t O.t
end

(** The software half of the stem's decode, for the gates that must build the stem's
    input. The plane face of the circuit must equal it, which is what this unit's gate
    holds. *)
module For_test : sig
  (** [plane_activations classes hidden ~steps ~rows] is the stem's input tensor,
      [steps; rows; Model.planes]. A standing cell carries the one of the activation
      format at the row it holds and nothing elsewhere; a hidden cell carries it at EVERY
      row of its mask plane and nothing in its class plane.

      [rows] IS P AND NOT [Model.rows]: the circuit takes its own P from [Shape], thus a
      bench at a narrow P compares against the tensor that P really writes. *)
  val plane_activations
    :  int array array
    -> bool array array
    -> steps:int
    -> rows:int
    -> int array

  (** [plane_column x ~step ~plane ~rows] is one column of that tensor: the [rows]
      activations of one step and one plane, row 0 first. Every tensor the circuit writes
      reads as [steps; rows; channels], thus this decode states the plane count alone. *)
  val plane_column : int array -> step:int -> plane:int -> rows:int -> int array

  (** [stem_input ~steps ~rows ~walk ~seed] is the sheet and the mask the stream gate
      feeds the engine: one uniform for each cell in [Model.cell_order], the class
      [floor (u * rows)], then the mask of pass 0 at
      [Model.anneal_threshold ~step:0 ~walk].

      THE STREAM GATE'S INPUT IS DATA AND NOT A WALK. Nothing downstream cares that the
      sheet is a chorale's opening — it cares that every class fits the column and that
      the twin and the circuit read the same sheet — thus the classes are drawn over P and
      NOT inside [Model.seat_openings]. A seat register reaches class 46, and at a P under
      47 [plane_activations] writes such a class into the next step's region of the tensor
      or past its end; a class drawn over [rows] fits P by construction, at every P a gate
      can ask for. The walk's own gate draws the real opening, because a walk is what it
      tests. *)
  val stem_input
    :  steps:int
    -> rows:int
    -> walk:int
    -> seed:int
    -> int array array * bool array array
end
