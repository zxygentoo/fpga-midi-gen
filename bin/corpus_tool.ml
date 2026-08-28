(* corpus_tool: the corpus side of the JAX seam. The export subcommand writes the packed
   corpus as one safetensors file: for each split, a set of streams — the frames, the
   rolling coordinates and the stream index. The seam carries data, never rules: the
   packing, the rotation and the shifts all come from the committed OCaml, and the JAX
   side only reads arrays.

   The pieces subcommand writes the other corpus, the one the sheet of docs/diffusion.md
   reads: whole pieces on the grid the caller names, one row for each, with the true
   length and the legal shift range beside the cells. A stream has no pieces and a sheet
   holds nothing else, thus these are two files and not two views of one.

   There are no masks. No frame is illegal, thus nothing guards a draw and nothing needs a
   walked state; the anchors left with them. *)

open Core
module Frame = Mgen_core.Frame
module Jsb = Mgen_corpus.Jsb

(* one int32 tensor of any rank, from its data in row-major order *)
let i32 ~shape data =
  let flat index =
    Array.foldi index ~init:0 ~f:(fun axis flat at -> (flat * shape.(axis)) + at)
  in
  Nx.init Nx.int32 shape (fun index -> Int32.of_int_exn data.(flat index))
;;

(* A frame is one word of four voice codes, and a code with its flag set does not fit the
   sign of an int32 when it stands in the top byte. Therefore the seam carries the seats
   as four columns, seat 0 first, and the word is rebuilt by whoever wants one. *)
let tensors_of_split streams =
  let all = Array.of_list streams in
  let total = Array.sum (module Int) all ~f:(fun s -> Array.length s.Jsb.frames) in
  let frames = Array.create ~len:(total * Jsb.voices) 0 in
  let positions = Array.create ~len:total 0 in
  let index = Array.create ~len:(Array.length all * 2) 0 in
  (* one stream into the flat tensors, and the offset the next one opens at *)
  let write_stream row offset (stream : Jsb.stream) =
    let length = Array.length stream.frames in
    Array.iteri stream.frames ~f:(fun step frame ->
      List.iteri (Frame.codes frame) ~f:(fun seat code ->
        frames.(((offset + step) * Jsb.voices) + seat) <- code));
    Array.blit ~src:stream.positions ~src_pos:0 ~dst:positions ~dst_pos:offset ~len:length;
    index.(2 * row) <- offset;
    index.((2 * row) + 1) <- length;
    offset + length
  in
  let (_ : int) = Array.foldi all ~init:0 ~f:write_stream in
  [ "frames", Nx_io.P (i32 ~shape:[| total; Jsb.voices |] frames)
  ; "positions", Nx_io.P (i32 ~shape:[| total |] positions)
  ; "index", Nx_io.P (i32 ~shape:[| Array.length all; 2 |] index)
  ]
;;

(* The sheet of docs/diffusion.md: whole pieces on one grid, one row for each.

   The cells stay as the corpus states them — the pitch that a voice sings, or -1 for a
   rest — because the class map of a model is the JAX side's own, as it is for the frames.
   Seat 0 is the bass, as the packed stream states it: the file gives the soprano first,
   thus the step turns around.

   A row is as wide as the widest piece and the tail of a short one is silence, thus the
   seam carries the true length beside the cells. The legal shifts travel as their two
   ends, because the policy of [Jsb] gives one contiguous ascending range and a trainer
   that transposes draws inside it. *)
let tensors_of_pieces ~steps pieces =
  let count = List.length pieces in
  let rest = -1 in
  let cells = Array.create ~len:(count * steps * Jsb.voices) rest in
  let lengths = Array.create ~len:count 0 in
  let shifts = Array.create ~len:(count * 2) 0 in
  List.iteri pieces ~f:(fun row { Jsb.cells = piece; legal_shifts } ->
    lengths.(row) <- Array.length piece;
    Array.iteri piece ~f:(fun step step_cells ->
      List.iteri (List.rev step_cells) ~f:(fun seat pitch ->
        cells.((((row * steps) + step) * Jsb.voices) + seat) <- pitch));
    shifts.(2 * row) <- List.hd_exn legal_shifts;
    shifts.((2 * row) + 1) <- List.last_exn legal_shifts);
  [ "cells", Nx_io.P (i32 ~shape:[| count; steps; Jsb.voices |] cells)
  ; "lengths", Nx_io.P (i32 ~shape:[| count |] lengths)
  ; "shifts", Nx_io.P (i32 ~shape:[| count; 2 |] shifts)
  ]
;;

let step_counts pieces =
  List.map pieces ~f:(fun { Jsb.cells; legal_shifts = _ } -> Array.length cells)
;;

let median counts =
  let sorted = List.sort counts ~compare:Int.compare in
  List.nth_exn sorted (List.length sorted / 2)
;;

(* The range a list of pitches covers, for the report. The vocabulary of the model is
   sized to this and to nothing else — 48 classes cover 36 to 81 with one spare — thus
   each export prints it and the number is checked and not assumed. *)
let pitch_range pitches =
  match List.min_elt pitches ~compare:Int.compare with
  | None -> "silent"
  | Some low ->
    sprintf "%d to %d" low (List.max_elt pitches ~compare:Int.compare |> Option.value_exn)
;;

(* The transposition policy holds each voice inside its observed range, thus a stream
   cannot widen the corpus range. [Frame.pitches] holds a pitch one time for each frame,
   which the range does not feel: a unison moves neither end. *)
let stream_pitches streams =
  List.concat_map streams ~f:(fun s -> Array.to_list s.Jsb.frames)
  |> List.concat_map ~f:Frame.pitches
;;

let piece_pitches pieces =
  List.concat_map pieces ~f:(fun { Jsb.cells; legal_shifts = _ } ->
    Array.to_list cells |> List.concat |> List.filter ~f:(fun pitch -> pitch >= 0))
;;

let export ~corpus ~out ~streams ~seed =
  let data = Jsb.load ~path:corpus in
  let splits = [ "train", data.train; "valid", data.valid; "test", data.test ] in
  (* one lane for the whole export, thus the streams of a split do not move when another
     split changes *)
  let random_state = Random.State.make [| seed |] in
  let entries =
    List.concat_map splits ~f:(fun (name, chorales) ->
      let packed = Jsb.streams chorales ~count:streams ~random_state in
      let tensors = tensors_of_split packed in
      let steps = List.sum (module Int) packed ~f:(fun s -> Array.length s.Jsb.frames) in
      printf
        "%-5s  %3d pieces  %3d streams  %7d steps  pitches %s\n%!"
        name
        (List.length chorales)
        (List.length packed)
        steps
        (pitch_range (stream_pitches packed));
      List.map tensors ~f:(fun (tensor, data) -> name ^ "/" ^ tensor, data))
  in
  Core_unix.mkdir_p (Filename.dirname out);
  Nx_io.save_safetensors ~overwrite:true out entries;
  printf "wrote %s\n%!" out
;;

let export_command =
  Command.basic
    ~summary:"write the packed corpus for the JAX trainer: one set of streams per split"
    (let%map_open.Command corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the voice-separated corpus file"
     and out =
       flag
         "-out"
         (optional_with_default "jax/_data/frames.safetensors" string)
         ~doc:"PATH the safetensors file to write"
     and streams =
       flag
         "-streams"
         (optional_with_default 8 int)
         ~doc:
           "N the streams of each split. Stream zero is the canonical one, and each of \
            the others draws a fresh order and a fresh shift for each piece. A piece has \
            7.4 legal shifts at the mean."
     and seed =
       flag "-seed" (optional_with_default 0 int) ~doc:"N the seed of the draw"
     in
     fun () -> export ~corpus ~out ~streams ~seed)
;;

(* The pieces of one grid, and the width of the row that holds them.

   With no -steps the corpus states the width: the longest piece of the whole export, thus
   no piece is dropped and every split reads one sheet shape. A caller that wants a
   narrower sheet states it, and the export says how many pieces it then dropped. The crop
   of the trainer is the JAX side's own — a sheet of 640 steps holds a crop of 128
   wherever the draw puts it. *)
let export_pieces ~corpus ~out ~grid ~steps =
  let data = Jsb.load ~path:corpus in
  let splits =
    [ "train", data.train; "valid", data.valid; "test", data.test ]
    |> List.map ~f:(fun (name, chorales) ->
      name, List.map chorales ~f:(Jsb.on_grid ~every:grid))
  in
  let width =
    match steps with
    | Some steps -> steps
    | None ->
      List.concat_map splits ~f:(fun (_, pieces) -> step_counts pieces)
      |> List.max_elt ~compare:Int.compare
      |> Option.value_exn
  in
  let entries =
    List.concat_map splits ~f:(fun (name, pieces) ->
      let kept, dropped =
        List.partition_tf pieces ~f:(fun { Jsb.cells; legal_shifts = _ } ->
          Array.length cells <= width)
      in
      let counts = step_counts kept in
      printf
        "%-5s  %3d pieces  %d dropped  median %3d  longest %3d  pitches %s\n%!"
        name
        (List.length kept)
        (List.length dropped)
        (median counts)
        (List.max_elt counts ~compare:Int.compare |> Option.value_exn)
        (pitch_range (piece_pitches kept));
      List.map (tensors_of_pieces ~steps:width kept) ~f:(fun (tensor, data) ->
        name ^ "/" ^ tensor, data))
  in
  Core_unix.mkdir_p (Filename.dirname out);
  Nx_io.save_safetensors ~overwrite:true out entries;
  printf "wrote %s: %d steps to a row\n%!" out width
;;

let pieces_command =
  Command.basic
    ~summary:"write the whole pieces for the sheet trainer: one grid, one row each"
    (let%map_open.Command corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the voice-separated corpus file"
     and out =
       flag
         "-out"
         (optional_with_default "jax/_data/pieces.safetensors" string)
         ~doc:"PATH the safetensors file to write"
     and grid =
       flag
         "-grid"
         (optional_with_default 1 int)
         ~doc:
           "N the sixteenth steps of one exported step. 1 is the sixteenth grid of the \
            corpus and 2 is the eighth grid, which halves the sheet and loses the onsets \
            that stand on an odd sixteenth."
     and steps =
       flag
         "-steps"
         (optional int)
         ~doc:
           "N the steps of one row. With no flag the longest piece states it and nothing \
            is dropped; a piece that needs more than N is dropped, and the export says \
            how many."
     in
     fun () -> export_pieces ~corpus ~out ~grid ~steps)
;;

let command =
  Command.group
    ~summary:"the corpus side of the JAX seam"
    [ "export", export_command; "pieces", pieces_command ]
;;

let () = Command_unix.run command
