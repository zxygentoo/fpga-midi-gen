(* corpus_tool: the corpus side of the JAX seam. The export subcommand writes the encoded
   corpus as one safetensors file: for each split, every legal transposition of every
   piece — the codes, the phases, the bitpacked legality masks and the variant index. The
   seam carries data, never rules: the walk, the metre, the shifts and the masks all come
   from the committed OCaml, and the JAX side only reads arrays. *)

open Core
module Evaluation = Mgen.Evaluation
module Jsb = Mgen.Jsb
module Token = Mgen.Token

(* one encoded (piece, shift) variant of a split *)
type variant =
  { piece : int
  ; shift : int
  ; codes : int array
  ; phases : int array
  ; masks : bool array array
  }

let variants_of_split chorales =
  List.concat_mapi chorales ~f:(fun piece chorale ->
    List.map chorale.Jsb.legal_shifts ~f:(fun shift ->
      let ~codes, ~phases = Jsb.encode (Jsb.transpose ~by:shift chorale) in
      { piece; shift; codes; phases; masks = Evaluation.masks_after codes }))
;;

let mask_words = Token.vocab / 32

let tensors_of_split variants =
  let total = List.sum (module Int) variants ~f:(fun { codes; _ } -> Array.length codes) in
  let all = Array.of_list variants in
  let codes = Array.create ~len:total 0 in
  let phases = Array.create ~len:total 0 in
  let packed = Array.create ~len:(total * mask_words) 0 in
  let index = Array.create ~len:(Array.length all * 4) 0 in
  let offset = ref 0 in
  Array.iteri all ~f:(fun row { piece; shift; codes = c; phases = p; masks } ->
    let length = Array.length c in
    Array.blit ~src:c ~src_pos:0 ~dst:codes ~dst_pos:!offset ~len:length;
    Array.blit ~src:p ~src_pos:0 ~dst:phases ~dst_pos:!offset ~len:length;
    Array.iteri masks ~f:(fun i mask ->
      let words = Evaluation.mask_words mask in
      Array.blit
        ~src:words
        ~src_pos:0
        ~dst:packed
        ~dst_pos:((!offset + i) * mask_words)
        ~len:mask_words);
    index.(4 * row) <- piece;
    index.((4 * row) + 1) <- shift;
    index.((4 * row) + 2) <- !offset;
    index.((4 * row) + 3) <- length;
    offset := !offset + length);
  let i32_vector data =
    Nx.init Nx.int32 [| Array.length data |] (fun i -> Int32.of_int_exn data.(i.(0)))
  in
  let i32_matrix ~cols data =
    Nx.init Nx.int32 [| Array.length data / cols; cols |] (fun i ->
      Int32.of_int_exn data.((i.(0) * cols) + i.(1)))
  in
  [ "codes", Nx_io.P (i32_vector codes)
  ; "phases", Nx_io.P (i32_vector phases)
  ; "masks", Nx_io.P (i32_matrix ~cols:mask_words packed)
  ; "index", Nx_io.P (i32_matrix ~cols:4 index)
  ]
;;

let export ~corpus ~out =
  let data = Jsb.load ~path:corpus in
  let splits = [ "train", data.train; "valid", data.valid; "test", data.test ] in
  let entries =
    List.concat_map splits ~f:(fun (name, chorales) ->
      let variants = variants_of_split chorales in
      let tensors = tensors_of_split variants in
      let tokens = List.sum (module Int) variants ~f:(fun v -> Array.length v.codes) in
      printf
        "%-5s  %3d pieces  %4d variants  %7d tokens\n%!"
        name
        (List.length chorales)
        (List.length variants)
        tokens;
      List.map tensors ~f:(fun (tensor, packed) -> name ^ "/" ^ tensor, packed))
  in
  Core_unix.mkdir_p (Filename.dirname out);
  Nx_io.save_safetensors ~overwrite:true out entries;
  printf "wrote %s\n%!" out
;;

let export_command =
  Command.basic
    ~summary:"write the encoded corpus for the JAX trainer: every legal transposition"
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
     in
     fun () -> export ~corpus ~out)
;;

let command =
  Command.group ~summary:"the corpus side of the JAX seam" [ "export", export_command ]
;;

let () = Command_unix.run command
