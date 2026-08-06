open Core

type row = int array * int array * bool array array

let masks_after codes =
  let (_ : Sounding_state.t), masks =
    Array.fold_map codes ~init:Sounding_state.silence ~f:(fun state code ->
      let state = Sounding_state.step state (Token.of_byte code) in
      state, Sounding_state.legal_mask state)
  in
  masks
;;

let mask_words mask =
  Array.init (Token.vocab / 32) ~f:(fun word ->
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
  ( Array.map rows ~f:(fun (codes, _, _) -> codes)
  , Array.map rows ~f:(fun (_, phases, _) -> phases)
  , Array.map rows ~f:(fun (_, _, masks) -> masks) )
;;

let rows chorales ~context ~limit =
  let rows =
    List.concat_map chorales ~f:(fun chorale ->
      let ~codes, ~phases = Jsb.encode chorale in
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
            ( Array.sub codes ~pos:start ~len:need
            , Array.sub phases ~pos:start ~len:context
            , Array.sub masks ~pos:start ~len:context ))))
  in
  List.take rows limit
;;

let loss config params rows ~batch ~masked =
  let total, count =
    List.fold
      (List.chunks_of rows ~length:batch)
      ~init:(0.0, 0)
      ~f:(fun (total, count) chunk ->
        let codes, phases, masks = batch_of_rows chunk in
        let value =
          Nx.item
            []
            (if masked
             then Transformer.masked_loss config params ~codes ~phases ~masks
             else Transformer.loss config params ~codes ~phases)
        in
        total +. (value *. Float.of_int (List.length chunk)), count + List.length chunk)
  in
  total /. Float.of_int count
;;
