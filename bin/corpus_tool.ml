(* corpus_tool: the corpus side of the JAX seam. The export subcommand writes the packed
   corpus as one safetensors file: for each split, a set of streams — the frames, the
   rolling coordinates and the stream index. The seam carries data, never rules: the
   packing, the rotation and the shifts all come from the committed OCaml, and the JAX
   side only reads arrays.

   There are no masks. No frame is illegal, thus nothing guards a draw and nothing needs a
   walked state; the anchors left with them. *)

open Core
module Jsb = Mgen_corpus.Jsb

(* A frame is one word of four voice codes, and a code with its flag set does not fit the
   sign of an int32 when it stands in the top byte. Therefore the seam carries the seats
   as four columns, seat 0 first, and the word is rebuilt by whoever wants one. *)
let tensors_of_split streams =
  let all = Array.of_list streams in
  let total = Array.sum (module Int) all ~f:(fun s -> Array.length s.Jsb.frames) in
  let frames = Array.create ~len:(total * Jsb.voices) 0 in
  let positions = Array.create ~len:total 0 in
  let index = Array.create ~len:(Array.length all * 2) 0 in
  let offset = ref 0 in
  Array.iteri all ~f:(fun row (stream : Jsb.stream) ->
    let length = Array.length stream.frames in
    Array.iteri stream.frames ~f:(fun step frame ->
      for seat = 0 to Jsb.voices - 1 do
        frames.(((!offset + step) * Jsb.voices) + seat)
        <- (frame lsr (8 * seat)) land 0xFF
      done);
    Array.blit
      ~src:stream.positions
      ~src_pos:0
      ~dst:positions
      ~dst_pos:!offset
      ~len:length;
    index.(2 * row) <- !offset;
    index.((2 * row) + 1) <- length;
    offset := !offset + length);
  let i32_vector data =
    Nx.init Nx.int32 [| Array.length data |] (fun i -> Int32.of_int_exn data.(i.(0)))
  in
  let i32_matrix ~cols data =
    Nx.init
      Nx.int32
      [| Array.length data / cols; cols |]
      (fun i -> Int32.of_int_exn data.((i.(0) * cols) + i.(1)))
  in
  [ "frames", Nx_io.P (i32_matrix ~cols:Jsb.voices frames)
  ; "positions", Nx_io.P (i32_vector positions)
  ; "index", Nx_io.P (i32_matrix ~cols:2 index)
  ]
;;

(* The pitches a split really states. The vocabulary of the model is sized to this and to
   nothing else — 48 classes cover 36 to 81 with one spare — thus the export prints it and
   the number is checked and not assumed. The transposition policy holds each voice inside
   its observed range, thus a stream cannot widen it. *)
let pitch_range streams =
  let sounding = List.concat_map streams ~f:(fun s -> Array.to_list s.Jsb.frames) in
  let pitches =
    List.concat_map sounding ~f:(fun frame ->
      List.init Jsb.voices ~f:(fun seat -> (frame lsr (8 * seat)) land 0xFF)
      |> List.filter ~f:(fun code -> code land 0x80 <> 0)
      |> List.map ~f:(fun code -> code land 0x7F))
  in
  match List.min_elt pitches ~compare:Int.compare with
  | None -> "silent"
  | Some low ->
    sprintf "%d to %d" low (List.max_elt pitches ~compare:Int.compare |> Option.value_exn)
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
        (pitch_range packed);
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

let command =
  Command.group ~summary:"the corpus side of the JAX seam" [ "export", export_command ]
;;

let () = Command_unix.run command
