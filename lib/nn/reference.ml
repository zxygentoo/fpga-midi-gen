(* The float rules both era references share — see reference.mli for the contract. Keep
   the exact arithmetic: the gates pin float values. *)

open Base

type tensor = Checkpoint.tensor

(* the table of one seat: the seat tensor holds the four in one, seat 0 first *)
let seat_table ~seats seat = Nx.contiguous (Nx.get [ seat ] seats)

(* float32, because the block goes into a matmul *)
let one_hot_rows ~num_classes rows =
  let batch = Array.length rows in
  let length = Array.length rows.(0) in
  let codes =
    Nx.init Nx.int32 [| batch; length |] (fun index ->
      Int32.of_int_exn rows.(index.(0)).(index.(1)))
  in
  Nx.astype Nx.float32 (Nx.one_hot ~num_classes codes)
;;

(* the table lookup as one-hot times table: small, and one definition serves every table *)
let embed_rows table ~num_classes rows = Nx.matmul (one_hot_rows ~num_classes rows) table

let seat_classes frames =
  let classes =
    Array.map frames ~f:(Array.map ~f:(fun frame -> Vocab.classes_of_frame frame))
  in
  Array.init Frame.voices ~f:(fun seat ->
    Array.map classes ~f:(Array.map ~f:(fun row -> List.nth_exn row seat)))
;;

(* RMSNorm with no scale: the trainers of both eras fold the scale away *)
let rms_norm x =
  let axis = Array.length (Nx.shape x) - 1 in
  let mean_square = Nx.mean (Nx.square x) ~axes:[ axis ] ~keepdims:true in
  Nx.mul x (Nx.rsqrt (Nx.add_s mean_square 1e-6))
;;

let softmax x ~axis =
  let weights = Nx.exp (Nx.sub x (Nx.max x ~axes:[ axis ] ~keepdims:true)) in
  Nx.div weights (Nx.sum weights ~axes:[ axis ] ~keepdims:true)
;;

(* the ALiBi slope of one head, negative: the head subtracts it times the distance *)
let alibi_slope ~span ~heads ~head =
  let rank = head + 1 in
  Float.(-(2.0 ** (-of_int span * of_int rank / of_int heads)))
;;

let feed_forward ~w1 ~w2 y = Nx.matmul (Nx.relu (Nx.matmul y w1)) w2

(* The input of one step: the four seat rows sum, and the bar phase adds to them.

   A shared table with a voice tag cannot work here, and the reason is arithmetic and not
   capacity. Every step carries all four seats, thus the sum of four tags is the same
   vector at every position — a bias, which carries nothing — and what remains is
   symmetric in the four classes. A soprano on 72 over a bass on 48 would give the vector
   of a soprano on 48 under a bass on 72, and the voices would be thrown away on the way
   in. *)
let embedding ~seats ~phase ~classes ~phases =
  let base = embed_rows phase ~num_classes:Jsb.bar_steps phases in
  Array.foldi classes ~init:base ~f:(fun seat h rows ->
    Nx.add h (embed_rows (seat_table ~seats seat) ~num_classes:Vocab.classes rows))
;;

(* The seats of the chain, soprano first: the head draws seat 3 and then walks down.

   The order keeps the one decision the ear accepted in era three — the top voice is
   chosen first, and it conditions on no voice under it, as the music is written. *)
let chain_seats = List.rev (List.range 0 Frame.voices)

(* The chained head. Each seat reads the stream that the seats above it have written:

   {v
     h3 = h                   logits(seat 3) = E[3] . rms(h3)
     h2 = h3 + E[3][c3]       logits(seat 2) = E[2] . rms(h2)
     h1 = h2 + E[2][c2]       logits(seat 1) = E[1] . rms(h1)
     h0 = h1 + E[1][c1]       logits(seat 0) = E[0] . rms(h0)
   v}

   [drawn] holds the classes the chain conditions on — the true frame in training, where
   the four heads then run in one pass with no sampling. Only seats 3, 2 and 1 are read.

   Four heads that drew in parallel would make the voices conditionally independent, and a
   chord is a joint choice: measured on era four, that costs 0.3157 nats for each step.
   The chain removes the cost for no parameters at all and three adds of a vector. *)
let seat_logits ~seats h ~drawn =
  let (_ : tensor), rows =
    List.fold_map chain_seats ~init:h ~f:(fun stream seat ->
      let table = seat_table ~seats seat in
      let raw = Nx.matmul (rms_norm stream) (Nx.transpose table) in
      let stream =
        if seat = 0
        then stream
        else Nx.add stream (embed_rows table ~num_classes:Vocab.classes drawn.(seat))
      in
      stream, (seat, raw))
  in
  rows
;;

(* the negative log likelihood of one seat: [raw] is [batch; length; classes] *)
let class_nll raw labels =
  let axis = 2 in
  let shifted = Nx.sub raw (Nx.max raw ~axes:[ axis ] ~keepdims:true) in
  let total = Nx.log (Nx.sum (Nx.exp shifted) ~axes:[ axis ] ~keepdims:true) in
  let hot = one_hot_rows ~num_classes:Vocab.classes labels in
  Nx.neg (Nx.sum (Nx.mul (Nx.sub shifted total) hot) ~axes:[ axis ])
;;

(* the inputs and the targets of one window: [context] positions state [context] targets,
   and the last frame of the window is a target alone *)
let inputs rows =
  Array.map rows ~f:(fun row -> Array.subo row ~len:(Array.length row - 1))
;;

let targets rows = Array.map rows ~f:(fun row -> Array.subo row ~pos:1)

let loss ~seats ~hidden ~windows =
  let frames = Array.of_list_map windows ~f:(fun (w : Jsb.stream) -> w.frames) in
  let positions = Array.of_list_map windows ~f:(fun (w : Jsb.stream) -> w.positions) in
  if Array.is_empty frames then invalid_arg "the loss takes one window or more";
  (* the bar phase is the low four bits of the rolling coordinate; the high four were the
     window position, which the ear dropped and the corpus still carries *)
  let phases =
    Array.map (inputs positions) ~f:(Array.map ~f:(fun at -> at % Jsb.bar_steps))
  in
  let classes = seat_classes (inputs frames) in
  let labels = seat_classes (targets frames) in
  let h = hidden ~classes ~phases in
  let nll =
    List.fold (seat_logits ~seats h ~drawn:labels) ~init:None ~f:(fun total (seat, raw) ->
      let seat_nll = class_nll raw labels.(seat) in
      Some (Option.value_map total ~default:seat_nll ~f:(Nx.add seat_nll)))
  in
  (* the sum over the seats is the loss of one step, and the mean is over the steps: a
     mean over the predictions would divide by a count that changes with the encoding *)
  Nx.item [] (Nx.mean (Option.value_exn nll))
;;

(* the row of a table as a stream of one position, thus the chain can add it *)
let table_row table index ~d = Nx.reshape [| 1; d |] (Nx.get [ index ] table)

let draw_frame ~seats ~d ~temperature ~min_p ~rng stream =
  let (rng, (_ : tensor)), classes =
    List.fold_map chain_seats ~init:(rng, stream) ~f:(fun (rng, stream) seat ->
      let table = seat_table ~seats seat in
      let raw = Nx.to_array (Nx.matmul (rms_norm stream) (Nx.transpose table)) in
      let rng, uniform = Prng.run Prng.uniform rng in
      let index = Policy.draw_class raw ~temperature ~min_p ~uniform in
      let stream =
        if seat = 0 then stream else Nx.add stream (table_row table index ~d)
      in
      (rng, stream), index)
  in
  (* the chain runs from the soprano down, and a frame reads from seat 0 up *)
  rng, List.rev classes
;;
