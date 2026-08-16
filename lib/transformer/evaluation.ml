open Core

type row =
  { codes : int array
  ; phases : int array
  ; progress : int array
  ; masks : bool array array
  }

type batch =
  { codes : int array array
  ; phases : int array array
  ; progress : int array array
  ; masks : bool array array array
  }

let walk state codes =
  Array.fold codes ~init:state ~f:(fun state code ->
    Sounding_state.step state (Token.of_code code))
;;

let masks_from state codes =
  let (_ : Sounding_state.t), masks =
    Array.fold_map codes ~init:state ~f:(fun state code ->
      let state = Sounding_state.step state (Token.of_code code) in
      state, Sounding_state.legal_mask state)
  in
  masks
;;

let masks_after codes = masks_from Sounding_state.silence codes
let words_per_mask = Token.vocab / 32

let mask_words mask =
  Array.init words_per_mask ~f:(fun word ->
    let value =
      List.range 0 32
      |> List.fold ~init:0 ~f:(fun acc bit ->
        if mask.((32 * word) + bit) then acc lor (1 lsl bit) else acc)
    in
    (* the top bit wraps into the int32 sign *)
    if value > 0x7FFFFFFF then value - 0x100000000 else value)
;;

let batch_of_rows rows =
  let rows = Array.of_list rows in
  { codes = Array.map rows ~f:(fun (row : row) -> row.codes)
  ; phases = Array.map rows ~f:(fun (row : row) -> row.phases)
  ; progress = Array.map rows ~f:(fun (row : row) -> row.progress)
  ; masks = Array.map rows ~f:(fun (row : row) -> row.masks)
  }
;;

(* The last anchor at or below an offset. A stream is too long to hold a mask for each of
   its tokens, thus a row walks [Sounding_state] to its start; the seam before a piece
   releases every sounding pitch, thus the walk begins at silence there. Before the first
   piece the stream is silent, and offset zero serves. *)
let anchor_at (stream : Jsb.stream) offset =
  let nearer anchor start = if start <= offset then start else anchor in
  Array.fold stream.anchors ~init:0 ~f:nearer
;;

let row (stream : Jsb.stream) ~start ~context =
  let anchor = anchor_at stream start in
  let state =
    walk Sounding_state.silence (Array.sub stream.codes ~pos:anchor ~len:(start - anchor))
  in
  let coordinate pos = stream.positions.(start + pos) in
  ({ codes = Array.sub stream.codes ~pos:start ~len:(context + 1)
   ; phases = Array.init context ~f:(fun pos -> coordinate pos % Jsb.bar_steps)
   ; progress = Array.init context ~f:(fun pos -> coordinate pos / Jsb.bar_steps)
   ; masks = masks_from state (Array.sub stream.codes ~pos:start ~len:context)
   }
   : row)
;;

let rows stream ~context ~limit =
  let need = context + 1 in
  let length = Array.length stream.Jsb.codes in
  if length < need
  then []
  else (
    let windows = min limit (((length - need) / context) + 1) in
    List.init windows ~f:(fun window -> row stream ~start:(window * context) ~context))
;;

let loss (config : Transformer.Config.t) params rows ~batch =
  let total, count =
    List.fold
      (List.chunks_of rows ~length:batch)
      ~init:(0.0, 0)
      ~f:(fun (total, count) chunk ->
        let stacked = batch_of_rows chunk in
        (* the rows of the referee are whole windows of a piece: none is padded, thus
           every position weighs one *)
        let weights =
          Array.map stacked.phases ~f:(fun row -> Array.map row ~f:(fun (_ : int) -> 1.0))
        in
        let dropout = Transformer.Dropout.none in
        let value =
          Nx.item
            []
            (Transformer.loss
               config
               params
               ~codes:stacked.codes
               ~phases:stacked.phases
               ~progress:stacked.progress
               ~masks:stacked.masks
               ~weights
               ~dropout)
        in
        total +. (value *. Float.of_int (List.length chunk)), count + List.length chunk)
  in
  total /. Float.of_int count
;;
