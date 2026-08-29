open Core
module Nn_quantized = Mgen_nn.Quantized
module Contract_file = Mgen_nn.Contract_file

module Constants = struct
  (* the scores and the logits are Q12 as well, and stay wide; no constant names them *)
  include Nn_quantized.Constants

  (* the query, the keys, the values and the context: Q12 in int16. It is a name of its
     own because the rings store these rows and the ring is where the format is a design
     choice, not an accident of the datapath. *)
  let kv_q = 12
  let score_shift ~head_d = score_shift ~row_q:kv_q ~head_d
end

module Params_data = struct
  type 'a t =
    { seats : 'a
    ; phase : 'a
    ; layers : 'a layer array
    }

  and 'a layer =
    { wq : 'a
    ; wk : 'a
    ; wv : 'a
    ; wo : 'a
    ; w1 : 'a
    ; w2 : 'a
    }

  let layer_to_list { wq; wk; wv; wo; w1; w2 } = [ wq; wk; wv; wo; w1; w2 ]
  let per_layer = 6

  let to_list { seats; phase; layers } =
    seats :: phase :: List.concat_map (Array.to_list layers) ~f:layer_to_list
  ;;

  let of_list ~layers items =
    let wanted = 2 + (per_layer * layers) in
    if List.length items <> wanted
    then invalid_argf "%d tensors, not %d" (List.length items) wanted ();
    match items with
    | seats :: phase :: rest ->
      let rest = Array.of_list rest in
      { seats
      ; phase
      ; layers =
          Array.init layers ~f:(fun at ->
            let base = per_layer * at in
            { wq = rest.(base)
            ; wk = rest.(base + 1)
            ; wv = rest.(base + 2)
            ; wo = rest.(base + 3)
            ; w1 = rest.(base + 4)
            ; w2 = rest.(base + 5)
            })
      }
    | _ -> assert false
  ;;
end

type quantized = Nn_quantized.quantized =
  { q : int array
  ; e : int
  }

type t =
  { d : int
  ; heads : int
  ; context : int
  ; slope_span : int
  ; params : quantized Params_data.t
  ; temper : Constants.scale
  ; min_weight : int
  }

let layers t = Array.length t.params.layers

(* The shapes of the tensors in the flat order, from the shape numbers. The contract file
   states its own shapes, thus the reader does not need this; the drawn model of a test
   does, and it is the one statement of what a tensor of this era holds. *)
let shapes ~d ~layers =
  let layer_shapes =
    [ [| d; d |]; [| d; d |]; [| d; d |]; [| d; d |]; [| d; 4 * d |]; [| 4 * d; d |] ]
  in
  let tables = [ [| Frame.voices; Vocab.classes; d |]; [| Jsb.bar_steps; d |] ] in
  tables @ List.concat (List.init layers ~f:(fun (_ : int) -> layer_shapes))
;;

let sizes ~d ~layers = List.map (shapes ~d ~layers) ~f:(Array.fold ~init:1 ~f:( * ))

(* The arithmetic of the circuit is shifts, thus the shape obeys the shift rules. The
   record is open — a contract file and a drawn model both build one — thus a model that
   no constructor here made can break a rule, and both consumers of a broken model would
   be silently wrong. *)
let check_shape t =
  let { Params_data.seats; phase; layers = tensors } = t.params in
  let refuse rule wrong = if wrong then invalid_argf "%s" rule () in
  refuse "d is not a power of two" (not (Int.is_pow2 t.d));
  refuse "the context is not a power of two" (not (Int.is_pow2 t.context));
  if t.heads <= 0 || t.d % t.heads <> 0
  then invalid_argf "%d heads do not divide d %d" t.heads t.d ();
  refuse
    "the head width is not a power of four, thus no shift is its square root"
    (Int.floor_log2 (t.d / t.heads) % 2 <> 0);
  refuse "a model with no layer" (Array.length tensors = 0);
  (* the seat rows and the phase row add row for row — the Embed op of the circuit walks
     them as one tensor — thus one exponent covers both. The four seat tables share it for
     the same reason: they stand in one tensor. *)
  if phase.e <> seats.e
  then
    invalid_argf
      "the phase table reads exponent %d and the seat tables %d"
      phase.e
      seats.e
      ();
  let holds name tensor size =
    if Array.length tensor <> size
    then invalid_argf "%s holds %d weights, not %d" name (Array.length tensor) size ()
  in
  holds "the seat tables" seats.q (Frame.voices * Vocab.classes * t.d);
  holds "the phase table" phase.q (Jsb.bar_steps * t.d);
  List.iter2_exn
    (List.mapi (Params_data.to_list t.params) ~f:(fun at tensor ->
       Printf.sprintf "tensor %d" at, tensor))
    (sizes ~d:t.d ~layers:(layers t))
    ~f:(fun (name, tensor) size -> holds name tensor.q size)
;;

let rom_bits t = Nn_quantized.rom_bits (Params_data.to_list t.params)

(* the base of each tensor inside the ROM: the exclusive prefix scan of the sizes, handed
   back through the one definition of the order *)
let rom_bases t =
  Nn_quantized.bases_of
    (Array.of_list_map (Params_data.to_list t.params) ~f:(fun tensor ->
       Array.length tensor.q))
  |> Array.to_list
  |> Params_data.of_list ~layers:(layers t)
;;

(* the tensors the file carries beside its numbered weights *)
let exponents_tensor = "exponents"
let temper_tensor = "temper"
let min_weight_tensor = "min_weight"
let heads_tensor = "heads"
let context_tensor = "context"
let slope_span_tensor = "slope_span"
let beside_the_weights = 6

let of_int8_checkpoint path =
  let file = Contract_file.open_ path in
  let values = Contract_file.values file in
  let only = Contract_file.only file in
  let count = Contract_file.tensor_count file ~beside:beside_the_weights in
  let layers, spare =
    (count - 2) / Params_data.per_layer, (count - 2) % Params_data.per_layer
  in
  if spare <> 0 || layers < 1
  then invalid_argf "%s: %d tensors is no quantized step model" path count ();
  let exponents = values exponents_tensor in
  if Array.length exponents <> count
  then
    invalid_argf "%s: %d exponents for %d tensors" path (Array.length exponents) count ();
  let temper = Contract_file.scale file temper_tensor in
  let params =
    List.init count ~f:(fun at -> { q = values (Int.to_string at); e = exponents.(at) })
    |> Params_data.of_list ~layers
  in
  let seats = params.Params_data.seats in
  let model =
    { d = Array.length seats.q / (Frame.voices * Vocab.classes)
    ; heads = only heads_tensor
    ; context = only context_tensor
    ; slope_span = only slope_span_tensor
    ; params
    ; temper
    ; min_weight = only min_weight_tensor
    }
  in
  check_shape model;
  model
;;

module For_test = struct
  type shape =
    { d : int
    ; layers : int
    ; heads : int
    ; context : int
    ; slope_span : int
    }

  let shape = { d = 32; layers = 1; heads = 2; context = 16; slope_span = 4 }

  (* the shape the ear elected on 2026-08-18, which the flash carried until era six took
     it: the cost model of a real step is read at this shape and at no other *)
  let elected = { d = 64; layers = 6; heads = 4; context = 256; slope_span = 4 }

  (* the stated exponent of the frozen eras; [Nn_quantized.For_test.drawn_tensor] carries
     the measurement behind it, and the seat and phase tables share it, which is what
     [check_shape] holds *)
  let drawn_exponent = 10
  let tensor = Nn_quantized.For_test.drawn_tensor ~e:drawn_exponent

  let drawn { d; layers; heads; context; slope_span } ~seed =
    let (_ : Prng.state), tensors =
      Prng.run
        (Prng.all
           (List.map (sizes ~d ~layers) ~f:(fun count -> Prng.normals ~count ~scale:0.02)))
        (Prng.create_folded ~seed)
    in
    (* THE ELECTED POLICY, STATED. The temper is [Constants.temper_at_one] and the floor
       is the elected min-p 0.05 as a share of the peak weight 2^15, which is
       [jax/fixed.py]'s [min_weight_of] and what [test_fixed.py] pins. The elected numbers
       themselves live above the seam now, in [ELECTED_TEMPERATURE] and [ELECTED_MIN_P] of
       [jax/fixed.py]. *)
    (* [Params_data.of_list] owns the order of the parameters; the draw states it nowhere. *)
    { d
    ; heads
    ; context
    ; slope_span
    ; temper = Constants.temper_at_one
    ; min_weight = 1638
    ; params = Params_data.of_list ~layers (List.map tensors ~f:tensor)
    }
  ;;
end
