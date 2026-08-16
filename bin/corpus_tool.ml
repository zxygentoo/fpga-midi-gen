(* corpus_tool: the corpus side of the JAX seam. The export subcommand writes the packed
   corpus as one safetensors file: for each split, a set of streams — the codes, the
   rolling coordinates, the bitpacked legality masks and the stream index. The seam
   carries data, never rules: the packing, the rotation, the shifts and the masks all come
   from the committed OCaml, and the JAX side only reads arrays. *)

open Core
module Evaluation = Mgen_transformer.Evaluation
module Jsb = Mgen_corpus.Jsb

let tensors_of_split streams =
  let all = Array.of_list streams in
  let total = Array.sum (module Int) all ~f:(fun s -> Array.length s.Jsb.codes) in
  let codes = Array.create ~len:total 0 in
  let positions = Array.create ~len:total 0 in
  let packed = Array.create ~len:(total * Evaluation.words_per_mask) 0 in
  let index = Array.create ~len:(Array.length all * 2) 0 in
  let offset = ref 0 in
  Array.iteri all ~f:(fun row (stream : Jsb.stream) ->
    let length = Array.length stream.codes in
    Array.blit ~src:stream.codes ~src_pos:0 ~dst:codes ~dst_pos:!offset ~len:length;
    Array.blit
      ~src:stream.positions
      ~src_pos:0
      ~dst:positions
      ~dst_pos:!offset
      ~len:length;
    Array.iteri (Evaluation.masks_after stream.codes) ~f:(fun i mask ->
      let words = Evaluation.mask_words mask in
      Array.blit
        ~src:words
        ~src_pos:0
        ~dst:packed
        ~dst_pos:((!offset + i) * Evaluation.words_per_mask)
        ~len:Evaluation.words_per_mask);
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
  [ "codes", Nx_io.P (i32_vector codes)
  ; "positions", Nx_io.P (i32_vector positions)
  ; "masks", Nx_io.P (i32_matrix ~cols:Evaluation.words_per_mask packed)
  ; "index", Nx_io.P (i32_matrix ~cols:2 index)
  ]
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
      let tokens = List.sum (module Int) packed ~f:(fun s -> Array.length s.Jsb.codes) in
      printf
        "%-5s  %3d pieces  %3d streams  %7d tokens\n%!"
        name
        (List.length chorales)
        (List.length packed)
        tokens;
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
         (optional_with_default "jax/_data/corpus.safetensors" string)
         ~doc:"PATH the safetensors file to write"
     and streams =
       flag
         "-streams"
         (optional_with_default 8 int)
         ~doc:
           "N the streams of each split. Stream zero is the canonical one, and each of \
            the others draws a fresh order and a fresh shift for each piece. A piece has \
            7.4 legal shifts at the mean, thus eight streams hold the token count of the \
            corpus of the era before the packing."
     and seed =
       flag "-seed" (optional_with_default 0 int) ~doc:"N the seed of the draw"
     in
     fun () -> export ~corpus ~out ~streams ~seed)
;;

let command =
  Command.group ~summary:"the corpus side of the JAX seam" [ "export", export_command ]
;;

let () = Command_unix.run command
