(* checkpoint_tool: operations on one checkpoint. The eval subcommand is the referee of
   the election run standalone: the fixed valid windows of Evaluation. With -out it writes
   the gate file of the JAX seam — the batch and the loss — thus the parity gates compare
   against the same rows the election reads. The config flags must equal the flags of the
   training run; the checkpoint holds only tensors. *)

open Core
module Evaluation = Mgen.Evaluation
module Jsb = Mgen.Jsb
module Token = Mgen.Token
module Transformer = Mgen.Transformer

let mask_words = Token.vocab / 32

let gate_entries rows ~loss =
  let codes, phases, masks = Evaluation.batch_of_rows rows in
  let batch = Array.length codes in
  let need = Array.length codes.(0) in
  let context = need - 1 in
  let packed = Array.map masks ~f:(Array.map ~f:Evaluation.mask_words) in
  let i32 shape read = Nx.init Nx.int32 shape (fun i -> Int32.of_int_exn (read i)) in
  [ "codes", Nx_io.P (i32 [| batch; need |] (fun i -> codes.(i.(0)).(i.(1))))
  ; "phases", Nx_io.P (i32 [| batch; context |] (fun i -> phases.(i.(0)).(i.(1))))
  ; ( "masks"
    , Nx_io.P
        (i32 [| batch; context; mask_words |] (fun i -> packed.(i.(0)).(i.(1)).(i.(2)))) )
  ; "loss", Nx_io.P (Nx.init Nx.float32 [| 1 |] (fun (_ : int array) -> loss))
  ]
;;

let eval ~checkpoint ~corpus ~heads ~context ~slope_span ~rows ~batch ~out =
  let config = Transformer.Config.of_checkpoint checkpoint ~heads ~context ~slope_span in
  let like = Transformer.Params.to_ptree (Transformer.Params.draw config ~seed:0) in
  let tree = Kaun.Checkpoint.load checkpoint ~like in
  let params = Transformer.Params.of_ptree config tree in
  let data = Jsb.load ~path:corpus in
  let eval_rows = Evaluation.rows data.valid ~context ~limit:rows in
  let loss = Evaluation.loss config params eval_rows ~batch in
  printf "%d valid rows  loss %.4f\n%!" (List.length eval_rows) loss;
  Option.iter out ~f:(fun path ->
    Core_unix.mkdir_p (Filename.dirname path);
    Nx_io.save_safetensors ~overwrite:true path (gate_entries eval_rows ~loss);
    printf "wrote %s\n%!" path)
;;

let eval_command =
  Command.basic
    ~summary:"the election's own eval of one checkpoint; -out writes the gate file"
    (let%map_open.Command checkpoint =
       flag "-ckpt" (required string) ~doc:"PATH the checkpoint"
     and corpus =
       flag
         "-corpus"
         (optional_with_default Jsb.default_path string)
         ~doc:"PATH the voice-separated corpus file"
     and slope_span =
       flag
         "-alibi-span"
         (optional_with_default Transformer.Config.(baseline.slope_span) int)
         ~doc:"N the ALiBi exponent span; it must equal the span of the training run"
     and heads =
       flag
         "-heads"
         (optional_with_default Transformer.Config.(baseline.heads) int)
         ~doc:"N the heads"
     and context =
       flag
         "-context"
         (optional_with_default Transformer.Config.(baseline.context) int)
         ~doc:"N the window, in tokens"
     and rows =
       flag "-rows" (optional_with_default 16 int) ~doc:"N the widest gate batch, in rows"
     and batch = flag "-batch" (optional_with_default 16 int) ~doc:"N the compute batch"
     and out = flag "-out" (optional string) ~doc:"PATH write the gate file here" in
     fun () -> eval ~checkpoint ~corpus ~heads ~context ~slope_span ~rows ~batch ~out)
;;

let command =
  Command.group ~summary:"operations on one checkpoint" [ "eval", eval_command ]
;;

let () = Command_unix.run command
