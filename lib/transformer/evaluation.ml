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

let masks_after codes =
  let (_ : Sounding_state.t), masks =
    Array.fold_map codes ~init:Sounding_state.silence ~f:(fun state code ->
      let state = Sounding_state.step state (Token.of_code code) in
      state, Sounding_state.legal_mask state)
  in
  masks
;;

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

let rows chorales ~context ~limit =
  let rows =
    List.concat_map chorales ~f:(fun chorale ->
      let ~codes, ~phases, ~progress = Jsb.encode chorale in
      let need = context + 1 in
      let length = Array.length codes in
      if length < need
      then []
      else (
        let masks = masks_after codes in
        List.init
          (((length - need) / context) + 1)
          ~f:(fun window ->
            let start = min (window * context) (length - need) in
            ({ codes = Array.sub codes ~pos:start ~len:need
             ; phases = Array.sub phases ~pos:start ~len:context
             ; progress = Array.sub progress ~pos:start ~len:context
             ; masks = Array.sub masks ~pos:start ~len:context
             }
             : row))))
  in
  List.take rows limit
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
