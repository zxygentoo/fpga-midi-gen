(* The sheet — see sheet.mli.

   THE DECODE STANDS AFTER THE MUX AND NEVER BEFORE IT: a face names one step, thus it
   costs one [steps]-way mux of the packed cells and then one decode of the cell it found.
   A sheet held as decoded planes would be [steps] decodes and a mux of the whole plane
   width, for the same answer. *)

open Core
open Hardcaml
open Signal

module type Shape = sig
  val steps : int
  val rows : int
end

(* the activation format of the twin: what a plane column carries in each row *)
let activation_bits = Model.activation_bits

(* ==================================================================== *)
(* The stem's decode, in software *)
(* ==================================================================== *)

(* The plane face of the circuit must equal this decode, thus the two stand together. *)

let voices = Model.voices
let planes = Model.planes

(* the one of the activation format, which a hot row of a plane carries *)
let activation_one = 1 lsl Model.activation_q

(* [rows] is P and not [Model.rows]: the circuit takes its own P from [Shape], thus the
   software half takes it too. *)
let plane_activations sheet hidden ~steps ~rows =
  let x = Array.create ~len:(steps * rows * planes) 0 in
  for step = 0 to steps - 1 do
    for voice = 0 to voices - 1 do
      if hidden.(step).(voice)
      then
        for row = 0 to rows - 1 do
          x.((((step * rows) + row) * planes) + voices + voice) <- activation_one
        done
      else x.((((step * rows) + sheet.(step).(voice)) * planes) + voice) <- activation_one
    done
  done;
  x
;;

(* Every tensor the circuit writes reads as [steps; rows; channels] — the stem's planes, a
   layer's output, the head's logits — thus the index rule stands here one time. *)
let tensor_column x ~step ~channel ~channels ~rows =
  Array.init rows ~f:(fun row -> x.((((step * rows) + row) * channels) + channel))
;;

let plane_column x ~step ~plane ~rows =
  tensor_column x ~step ~channel:plane ~channels:planes ~rows
;;

module For_test = struct
  let plane_activations = plane_activations
  let plane_column = plane_column
end

module Make (Shape : Shape) = struct
  let steps = Shape.steps
  let rows = Shape.rows
  let step_bits = address_bits_for steps
  let seat_bits = address_bits_for voices
  let class_bits = address_bits_for rows
  let plane_bits = address_bits_for planes

  (* one cell as one word: the mask bit above the class *)
  let cell_bits = class_bits + 1

  module I = struct
    type 'a t =
      { clock : 'a
      ; cell_step : 'a [@bits step_bits]
      ; cell_seat : 'a [@bits seat_bits]
      ; write_class : 'a
      ; cell_class : 'a [@bits class_bits]
      ; write_mask : 'a
      ; cell_hidden : 'a
      ; plane_step : 'a [@bits step_bits]
      ; plane : 'a [@bits plane_bits]
      ; score_step : 'a [@bits step_bits]
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { hidden : 'a
      ; plane_column : 'a [@bits rows * activation_bits]
      ; frame : 'a [@bits voices * Frame.code_bits]
      }
    [@@deriving hardcaml]
  end

  let class_of_cell cell = sel_bottom cell ~width:class_bits
  let hidden_of_cell cell = msb cell

  let seat_of_step word seat =
    select word ~high:((seat * cell_bits) + cell_bits - 1) ~low:(seat * cell_bits)
  ;;

  let create (i : _ I.t) : _ O.t =
    (* no clear: the opening writes every class and a mask draw every bit before anything
       reads either, and the board's reset runs an opening *)
    let spec = Reg_spec.create ~clock:i.clock () in
    (* ONE DECODE FOR THE WHOLE SHEET: a cell's write enable is one AND of two bits, and
       not a comparator at each of the [steps] by [voices] registers *)
    let step_named = binary_to_onehot i.cell_step in
    let seat_named = binary_to_onehot i.cell_seat in
    let cell ~step ~seat =
      let named = bit step_named ~pos:step &: bit seat_named ~pos:seat in
      reg spec ~enable:(i.write_mask &: named) i.cell_hidden
      @: reg spec ~enable:(i.write_class &: named) i.cell_class
    in
    let sheet =
      List.init steps ~f:(fun step ->
        concat_lsb (List.init voices ~f:(fun seat -> cell ~step ~seat)))
    in
    let cell_at ~step_address ~seat_address =
      let word = mux step_address sheet in
      mux seat_address (List.init voices ~f:(seat_of_step word))
    in
    (* the walk's face: one bit, and no decode *)
    let hidden =
      hidden_of_cell (cell_at ~step_address:i.cell_step ~seat_address:i.cell_seat)
    in
    (* the stem's face. [Frame.voices] is four, thus a plane index IS the face bit above
       the seat and neither costs a subtract. *)
    let read =
      cell_at
        ~step_address:i.plane_step
        ~seat_address:(sel_bottom i.plane ~width:seat_bits)
    in
    let hides = hidden_of_cell read in
    (* the rows the plane carries: the class plane holds the row the cell stands at and
       NOTHING AT ALL where the cell hides; the mask plane holds every row where it hides *)
    let hot =
      mux2
        (i.plane >=:. voices)
        (repeat hides ~count:rows)
        (sel_bottom (binary_to_onehot (class_of_cell read)) ~width:rows
         &: repeat ~:hides ~count:rows)
    in
    (* a cell of the masked roll is 0 or one, exact, thus a hot row is the one of
       [Model.activation_q] *)
    let activation_of_row row =
      concat_msb
        [ zero (activation_bits - Model.activation_q - 1)
        ; bit hot ~pos:row
        ; zero Model.activation_q
        ]
    in
    (* the score's face. The map is [Vocab]'s and the packing is [Frame]'s. *)
    let scored = mux i.score_step sheet in
    let code_of_seat seat =
      Vocab.Rtl.code_of_class (class_of_cell (seat_of_step scored seat))
    in
    { O.hidden
    ; plane_column = concat_lsb (List.init rows ~f:activation_of_row)
    ; frame = concat_lsb (List.init voices ~f:code_of_seat)
    }
  ;;
end

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* THE REFERENCE IS THE ERA'S OWN RULES, CALLED: [Model.opening_sheet],
   [Model.hidden_cells], [plane_activations] and [Model.frames_of_sheet]. The gate
   compares the circuit against the functions themselves and never a second reading of
   them. *)
module Bench (Shape : Shape) = struct
  module Sheet = Make (Shape)
  module Sim = Cyclesim.With_interface (Sheet.I) (Sheet.O)

  let steps = Shape.steps
  let rows = Shape.rows

  (* What one read cycle names: one cell for the walk, one column for the stem and one
     step for the score. ONE CYCLE ANSWERS ALL THREE, thus the bench always asks all three
     and a face that answered under another face's address would show. *)
  type read =
    { cell : int * int (** the step and the seat the walk names *)
    ; column : int * int (** the step and the plane the stem reads *)
    ; score : int (** the step whose frame the sequencer reads *)
    }

  type answer =
    { hidden : bool
    ; activations : int array (** the plane column, row 0 first *)
    ; frame : int
    }

  (* the sheet the three writes of a pass leave: a redraw lands on the cells the mask hid
     and nowhere else *)
  let redrawn ~opening ~hidden ~redraw =
    Array.mapi opening ~f:(fun step seats ->
      Array.mapi seats ~f:(fun seat held ->
        if hidden.(step).(seat) then redraw.(step).(seat) else held))
  ;;

  (* A simulation whose sheet holds ONE PASS: the opening's classes, then the mask bits,
     then the classes of the cells the mask hid — the three writes in the order the walk
     takes them, over [Model.cell_order] inside each. A bench that wrote the classes one
     time would never see a class write land beside a mask bit that must not move. *)
  let filled ?(trace = false) ~opening ~hidden ~redraw () =
    let sim =
      Sim.create
        ~config:(if trace then Cyclesim.Config.trace_all else Cyclesim.Config.default)
        Sheet.create
    in
    let waves, sim = Cyclesim.Waveform.create_if ~enabled:trace sim in
    let inp = Cyclesim.inputs sim in
    let name (step, seat) =
      Harness.set inp.cell_step step;
      Harness.set inp.cell_seat seat
    in
    let write_class classes (step, seat) =
      name (step, seat);
      Harness.set inp.cell_class classes.(step).(seat);
      Cyclesim.cycle sim
    in
    let write_mask (step, seat) =
      name (step, seat);
      Harness.set inp.cell_hidden (if hidden.(step).(seat) then 1 else 0);
      Cyclesim.cycle sim
    in
    let order = Model.cell_order ~steps in
    let hides (step, seat) = hidden.(step).(seat) in
    inp.write_class := Bits.vdd;
    List.iter order ~f:(write_class opening);
    inp.write_class := Bits.gnd;
    inp.write_mask := Bits.vdd;
    List.iter order ~f:write_mask;
    inp.write_mask := Bits.gnd;
    inp.write_class := Bits.vdd;
    List.iter (List.filter order ~f:hides) ~f:(write_class redraw);
    inp.write_class := Bits.gnd;
    waves, sim
  ;;

  (* the three faces at the three addresses, in one cycle; the strobes stand low, thus the
     sheet does not move under a read *)
  let ask (sim : Sim.t) { cell = step, seat; column = plane_step, plane; score } =
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs sim in
    Harness.set inp.cell_step step;
    Harness.set inp.cell_seat seat;
    Harness.set inp.plane_step plane_step;
    Harness.set inp.plane plane;
    Harness.set inp.score_step score;
    Cyclesim.cycle sim;
    { hidden = Bits.to_bool !(out.hidden)
    ; activations = Harness.unpack !(out.plane_column) ~width:activation_bits
    ; frame = Bits.to_unsigned_int !(out.frame)
    }
  ;;

  type disagreements =
    { masks : int
    ; columns : int
    ; frames : int
    }

  (* the pass into the memory, then the three faces swept TOGETHER: one read cycle names a
     cell, a plane column and a step that have nothing to do with each other, and the
     sweep visits every address of every face *)
  let check ~opening ~hidden ~redraw =
    let sim = snd (filled ~opening ~hidden ~redraw ()) in
    let sheet = redrawn ~opening ~hidden ~redraw in
    let stem = plane_activations sheet hidden ~steps ~rows in
    let frames = Model.frames_of_sheet sheet in
    let cells = Array.of_list (Model.cell_order ~steps) in
    let disagrees agrees = if agrees then 0 else 1 in
    let take counts at =
      let ((step, seat) as cell) = cells.(at % Array.length cells) in
      let plane_step = at / planes in
      let plane = at % planes in
      let score = at % steps in
      let answer = ask sim { cell; column = plane_step, plane; score } in
      { masks = counts.masks + disagrees (Bool.equal answer.hidden hidden.(step).(seat))
      ; columns =
          counts.columns
          + disagrees
              (Array.equal
                 Int.equal
                 answer.activations
                 (plane_column stem ~step:plane_step ~plane ~rows))
      ; frames = counts.frames + disagrees (answer.frame = frames.(score))
      }
    in
    List.fold
      (List.range 0 (steps * planes))
      ~init:{ masks = 0; columns = 0; frames = 0 }
      ~f:take
  ;;
end

let%expect_test "the three faces answer the sheet one pass of the walk writes" =
  (* Every write is the walk's own, thus the memory holds what the machine will put in it
     and not a drawn pattern. The passes stand at the ends of the schedule: pass 0 hides
     nearly every cell and the last hides nearly none, thus both faces meet hidden cells
     and standing ones. *)
  let case ~steps ~pass ~walk ~seed =
    let module B =
      Bench (struct
        let steps = steps
        let rows = Model.rows
      end)
    in
    let state, opening = Model.opening_sheet (Prng.create_folded ~seed) ~steps in
    let threshold = Model.anneal_threshold ~step:pass ~walk in
    let state, hidden = Model.hidden_cells state ~steps ~threshold in
    (* the redraw is a second opening: a class inside each seat's register, the range a
       draw of the model states as well *)
    let (_ : Prng.state), redraw = Model.opening_sheet state ~steps in
    let hides = Array.sum (module Int) hidden ~f:(Array.count ~f:Fn.id) in
    let { B.masks; columns; frames } = B.check ~opening ~hidden ~redraw in
    printf
      "T %d, seed %d, pass %d of %d: %d of %d cells hidden and redrawn — %d masks, %d \
       columns, %d frames disagree\n"
      steps
      seed
      pass
      walk
      hides
      (steps * Frame.voices)
      masks
      columns
      frames
  in
  case ~steps:5 ~pass:0 ~walk:8 ~seed:1;
  case ~steps:5 ~pass:7 ~walk:8 ~seed:1;
  case ~steps:9 ~pass:3 ~walk:8 ~seed:7;
  case ~steps:2 ~pass:5 ~walk:16 ~seed:1234;
  [%expect
    {|
    T 5, seed 1, pass 0 of 8: 19 of 20 cells hidden and redrawn — 0 masks, 0 columns, 0 frames disagree
    T 5, seed 1, pass 7 of 8: 5 of 20 cells hidden and redrawn — 0 masks, 0 columns, 0 frames disagree
    T 9, seed 7, pass 3 of 8: 17 of 36 cells hidden and redrawn — 0 masks, 0 columns, 0 frames disagree
    T 2, seed 1234, pass 5 of 16: 1 of 8 cells hidden and redrawn — 0 masks, 0 columns, 0 frames disagree
    |}]
;;

let%expect_test "the waveform of one pass and the three faces" =
  (* Two steps of six classes and one whole pass. The picture is the WRITES AND THE FACES;
     the text below it is what each read said, because a plane column does not fit a wave
     column.

     [hidden] stands through the redraw's class writes, which an opening-only fill could
     never show. Reads 1 and 2 are the same hidden cell on its two planes: THE REDRAW
     LANDED IN IT — the frame carries class 5 at seat 2 — and its class plane is still
     empty while its mask plane is whole, which is the decode and not the memory. *)
  let module B =
    Bench (struct
      let steps = 2
      let rows = 6
    end)
  in
  let opening = [| [| 1; 2; 3; 4 |]; [| 0; 5; 2; 1 |] |] in
  let hidden = [| [| false; false; true; false |]; [| true; false; false; false |] |] in
  let redraw = [| [| 0; 0; 5; 0 |]; [| 3; 0; 0; 0 |] |] in
  let waves, sim = B.filled ~trace:true ~opening ~hidden ~redraw () in
  let reads =
    [ { B.cell = 0, 2; column = 0, 2; score = 0 }
    ; { B.cell = 0, 2; column = 0, 6; score = 0 }
    ; { B.cell = 0, 0; column = 0, 0; score = 0 }
    ; { B.cell = 1, 0; column = 1, 4; score = 1 }
    ; { B.cell = 1, 1; column = 1, 1; score = 1 }
    ; { B.cell = 0, 3; column = 1, 0; score = 0 }
    ]
  in
  let answers = List.map reads ~f:(B.ask sim) in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.Bit
          [ "write_class"; "write_mask"; "hidden" ]
      ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.Unsigned_int
          [ "cell_step"; "cell_seat"; "plane_step"; "plane"; "score_step" ]
      ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.Hex
          [ "frame" ]
      ]
    ~show_digest:false
    ~wave_width:0
    ~display_width:74
    (Option.value_exn waves ~message:"a traced fill gives a waveform");
  let show ({ B.cell = step, seat; column = plane_step, plane; score }, answer) =
    printf
      "cell (%d,%d) hidden %b | plane (%d,%d) %s | score %d frame %08x\n"
      step
      seat
      answer.B.hidden
      plane_step
      plane
      (String.concat
         ~sep:" "
         (List.map (Array.to_list answer.B.activations) ~f:Int.to_string))
      score
      answer.B.frame
  in
  List.iter (List.zip_exn reads answers) ~f:show;
  [%expect
    {|
    ┌Signals─────────┐┌Waves─────────────────────────────────────────────────┐
    │write_class     ││────────────────┐               ┌───┐                 │
    │                ││                └───────────────┘   └───────────      │
    │write_mask      ││                ┌───────────────┐                     │
    │                ││────────────────┘               └───────────────      │
    │hidden          ││                                ┌───────┐ ┌─┐         │
    │                ││────────────────────────────────┘       └─┘ └───      │
    │                ││────────┬───────┬───────┬───────┬─┬─┬─────┬───┬─      │
    │cell_step       ││ 0      │1      │0      │1      │0│1│0    │1  │0      │
    │                ││────────┴───────┴───────┴───────┴─┴─┴─────┴───┴─      │
    │                ││──┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┬───┬─┬─      │
    │cell_seat       ││ 0│1│2│3│0│1│2│3│0│1│2│3│0│1│2│3│2│0│2  │0  │1│3      │
    │                ││──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┴───┴─┴─      │
    │                ││──────────────────────────────────────────┬─────      │
    │plane_step      ││ 0                                        │1          │
    │                ││──────────────────────────────────────────┴─────      │
    │                ││────────────────────────────────────┬─┬─┬─┬─┬─┬─      │
    │plane           ││ 0                                  │2│6│0│4│1│0      │
    │                ││────────────────────────────────────┴─┴─┴─┴─┴─┴─      │
    │                ││──────────────────────────────────────────┬───┬─      │
    │score_step      ││ 0                                        │1  │0      │
    │                ││──────────────────────────────────────────┴───┴─      │
    │                ││──┬─┬─┬─┬─────────────────────────┬───────┬───┬─      │
    │frame           ││ .│.│.│.│A7A6A5A4                 │A7A8A5.│A4.│.      │
    │                ││──┴─┴─┴─┴─────────────────────────┴───────┴───┴─      │
    └────────────────┘└──────────────────────────────────────────────────────┘
    cell (0,2) hidden true | plane (0,2) 0 0 0 0 0 0 | score 0 frame a7a8a5a4
    cell (0,2) hidden true | plane (0,6) 64 64 64 64 64 64 | score 0 frame a7a8a5a4
    cell (0,0) hidden false | plane (0,0) 0 64 0 0 0 0 | score 0 frame a7a8a5a4
    cell (1,0) hidden true | plane (1,4) 64 64 64 64 64 64 | score 1 frame a4a5a8a6
    cell (1,1) hidden false | plane (1,1) 0 0 0 0 0 64 | score 1 frame a4a5a8a6
    cell (0,3) hidden false | plane (1,0) 0 0 0 0 0 0 | score 0 frame a7a8a5a4
    |}]
;;
