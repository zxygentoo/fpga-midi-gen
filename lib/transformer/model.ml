open Core
module Nn_quantized = Mgen_nn.Quantized

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
  assert (Int.is_pow2 t.d);
  assert (Int.is_pow2 t.context);
  assert (t.heads > 0 && t.d % t.heads = 0);
  assert (Int.floor_log2 (t.d / t.heads) % 2 = 0);
  assert (Array.length tensors > 0);
  (* the seat rows and the phase row add row for row — the Embed op of the circuit walks
     them as one tensor — thus one exponent covers both. The four seat tables share it for
     the same reason: they stand in one tensor. *)
  assert (phase.e = seats.e);
  assert (Array.length seats.q = Frame.voices * Vocab.classes * t.d);
  assert (Array.length phase.q = Jsb.bar_steps * t.d);
  List.iter2_exn
    (Params_data.to_list t.params)
    (sizes ~d:t.d ~layers:(layers t))
    ~f:(fun tensor size -> assert (Array.length tensor.q = size))
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

(* The ring keeps the top byte of a Q12 row: the circuit stores eight bits and restores
   eight zero low bits at the read, thus the granularity is 2^-4 and the format stays Q12.
   The query does not pass here — only the stored rows coarsen. *)
(* [asr] and [lsl] associate to the right; the parentheses are the expression *)
let coarse_to_ring (row : int array) = Array.map row ~f:(fun v -> (v asr 8) lsl 8)

(* the tensors the file carries beside its numbered weights *)
let exponents_tensor = "exponents"
let temper_tensor = "temper"
let min_weight_tensor = "min_weight"
let heads_tensor = "heads"
let context_tensor = "context"
let slope_span_tensor = "slope_span"
let beside_the_weights = 6

let of_int8_checkpoint path =
  let archive = Nx_io.load_safetensors path in
  let values name =
    match Stdlib.Hashtbl.find_opt archive name with
    | None -> invalid_argf "%s has no tensor %s" path name ()
    | Some packed ->
      Array.map (Nx.to_array (Nx_io.to_typed Nx.int32 packed)) ~f:Int32.to_int_exn
  in
  let only name =
    match values name with
    | [| value |] -> value
    | row ->
      invalid_argf "%s: %s holds %d values, not one" path name (Array.length row) ()
  in
  let count = Stdlib.Hashtbl.length archive - beside_the_weights in
  let layers, spare =
    (count - 2) / Params_data.per_layer, (count - 2) % Params_data.per_layer
  in
  if spare <> 0 || layers < 1
  then invalid_argf "%s: %d tensors is no quantized step model" path count ();
  let exponents = values exponents_tensor in
  if Array.length exponents <> count
  then
    invalid_argf "%s: %d exponents for %d tensors" path (Array.length exponents) count ();
  let temper =
    match values temper_tensor with
    | [| q_value; q |] -> { Constants.q_value; q }
    | row ->
      invalid_argf "%s: the temper holds %d values, not two" path (Array.length row) ()
  in
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

  (* THE DRAW IS THE DRAW OF THE ERA and it quantizes here, where every other model of the
     era is quantized above the seam. It is a TEST model and not a checkpoint: the walk it
     makes is the one every expect test of this library and of the socket simulation has
     recorded, thus the draw may not move. The rule of the two tables is the quantizer's —
     they share one exponent because their rows add. *)
  let drawn { d; layers; heads; context; slope_span } ~seed =
    let (_ : Prng.state), tensors =
      Prng.run
        (Prng.all
           (List.map (sizes ~d ~layers) ~f:(fun count -> Prng.normals ~count ~scale:0.02)))
        (Prng.create_folded ~seed)
    in
    let { Params_data.seats; phase; layers = drawn_layers } =
      Params_data.of_list ~layers tensors
    in
    let e =
      Nn_quantized.max_exponent
        (Float.max (Nn_quantized.max_abs seats) (Nn_quantized.max_abs phase))
    in
    let temper, min_weight =
      Nn_quantized.policy
        ~temperature:Mgen_nn.Policy.elected_temperature
        ~min_p:Mgen_nn.Policy.elected_min_p
    in
    { d
    ; heads
    ; context
    ; slope_span
    ; temper
    ; min_weight
    ; params =
        { Params_data.seats = Nn_quantized.quantize ~e seats
        ; phase = Nn_quantized.quantize ~e phase
        ; layers =
            Array.map
              drawn_layers
              ~f:(fun (l : Nn_quantized.Tensor.floats Params_data.layer) ->
                { Params_data.wq = Nn_quantized.quantize l.wq
                ; wk = Nn_quantized.quantize l.wk
                ; wv = Nn_quantized.quantize l.wv
                ; wo = Nn_quantized.quantize l.wo
                ; w1 = Nn_quantized.quantize l.w1
                ; w2 = Nn_quantized.quantize l.w2
                })
        }
    }
  ;;
end
