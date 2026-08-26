(* The masked canvas as a circuit: the note source of era six — see docs/diffusion_rtl.md
   for the design and its reasons.

   The shape, in five layers. L0 is the shared units — [Prng.Rtl] in the core, and [Exp2],
   the pick rules and the clamp of [Mgen_nn]. L1 is the elaboration: one model at one
   geometry as a value — the layer table, the ROM image, the constants and the cost. L2 is
   the column array, L3 the epilogue behind its drain chain, and L4 the walk that drives
   them and answers the socket.

   THE ELABORATION IS THE ONLY PLACE THAT READS THE MODEL. Every width, every depth and
   every base comes out of it, thus the circuit states no dimension of its own and a new
   checkpoint moves nothing but the elaboration. *)

open Core
module Bits = Hardcaml.Bits
module Nn = Mgen_nn.Quantized

(* ==================================================================== *)
(* L1 — the elaboration *)
(* ==================================================================== *)

(** A layer's role in the trunk. It states the two ends, the ReLU and the residual
    together: the stem decodes the planes into X, a pair opens X into Y and closes Y back
    into X, and the head reads X and states the logits of one step. Four roles and no
    independent flags, thus no table can say that a layer closes a pair and skips the
    residual. *)
type role =
  | Stem
  | Pair_open
  | Pair_close
  | Head

type layer =
  { role : role
  ; inputs : int
  ; outputs : int
  ; groups : int
  ; weight_base : int
  ; channel_base : int
  }

type elaboration =
  { steps : int
  ; rows : int
  ; lanes : int
  ; walk : int
  ; store_channels : int
  ; layers : layer array
  ; rom : Bits.t array
  ; constants : Bits.t array
  ; alpha : Bits.t array
  ; temper : Nn.Constants.scale
  }

(* the taps of one 3 by 3 kernel *)
let taps = 9

(* the 24-bit grid of the generator: an anneal threshold is [floor (alpha * 2 ** 24)],
   thus a mask uniform compares against it exactly *)
let alpha_bits = 24

(* The fields of a constant word, low to high: the bias, the shift of the gain, and the
   value of the gain. The three stand at one address because the epilogue wants the three
   at one time, thus no two of them can fall out of step.

   THE SHIFT FIELD SIZES ON THE RULE AND NOT ON THE CHECKPOINT. The q of a gain is
   [e + weight_exponent], and the two exponent rules cap at 30 and at 14, thus 44 is the
   largest q a quantizer can state and six bits hold it. A field sized on the elected
   model's own peak would save a bit or two and would make a drawn-weight timing probe
   elaborate a DIFFERENT netlist from the trained build; the bits are worth less than
   that. [elaborate] refuses a q the field cannot hold. *)
let bias_bits = 16
let shift_bits = 6
let gain_bits = 16
let constant_bits = bias_bits + shift_bits + gain_bits

(* One uniform is three steps of the generator — [Prng.uniform] takes three bytes — and
   the machine takes one cycle for each step. *)
let uniform_cycles = 3

let role_at ~count index =
  if index = 0
  then Stem
  else if index = count - 1
  then Head
  else if index % 2 = 1
  then Pair_open
  else Pair_close
;;

(* the dwell of one (column, group): one cycle for each (tap, input channel) pair *)
let dwell layer = taps * layer.inputs

(* One layer, exactly: the dwells of every column and group, and one drain tail behind the
   last of them. The count is exact and not a bound, because [elaborate] refuses a layer
   whose dwell is shorter than its drain — no dwell ever waits for the chain to empty.
   What the layer turn itself costs is the cycle bench's to measure. *)
let layer_cycles elaboration layer =
  (elaboration.steps * layer.groups * dwell layer) + elaboration.rows
;;

let forward_cycles elaboration =
  Array.sum (module Int) elaboration.layers ~f:(layer_cycles elaboration)
;;

let canvas_cells elaboration = elaboration.steps * Frame.voices

(* one uniform for each cell in the cell order: the opening, and the mask of each pass *)
let cell_walk_cycles elaboration = canvas_cells elaboration * uniform_cycles

(* One pass, LESS THE DRAW: the mask and the forward. The draw's cycles are L4's and the
   bench's — the design estimates about 2 [rows] for each hidden cell — and a number this
   module cannot state exactly it does not state. *)
let pass_cycles elaboration = cell_walk_cycles elaboration + forward_cycles elaboration

let bases_of sizes =
  Array.of_list
    (List.folding_map (Array.to_list sizes) ~init:0 ~f:(fun base size ->
       base + size, base))
;;

let elaborate ?(rows = Diffusion.rows) (model : Quantized.Model.t) ~steps ~lanes ~walk =
  Quantized.Model.check_shape model;
  if steps < 1 then invalid_argf "a canvas of %d steps" steps ();
  if rows < 1 then invalid_argf "a column of %d rows" rows ();
  if lanes < 1 then invalid_argf "a group of %d lanes" lanes ();
  if walk < 1 then invalid_argf "a walk of %d passes" walk ();
  let twin = model.layers in
  let count = Array.length twin in
  let groups_of (l : Quantized.Model.layer) = (l.outputs + lanes - 1) / lanes in
  (* The drain rule: the chain is [rows] stages and it must empty before the next dwell
     captures the array. It is a check and not a comment, thus [layer_cycles] states an
     exact count and never a bound. *)
  Array.iteri twin ~f:(fun at (l : Quantized.Model.layer) ->
    if taps * l.inputs < rows
    then
      invalid_argf
        "layer %d dwells %d cycles and its drain takes %d: the chain cannot empty"
        at
        (taps * l.inputs)
        rows
        ());
  let layers =
    let weight_bases =
      bases_of (Array.map twin ~f:(fun l -> l.inputs * taps * groups_of l))
    in
    let channel_bases = bases_of (Array.map twin ~f:(fun l -> l.outputs)) in
    Array.mapi twin ~f:(fun at (l : Quantized.Model.layer) ->
      { role = role_at ~count at
      ; inputs = l.inputs
      ; outputs = l.outputs
      ; groups = groups_of l
      ; weight_base = weight_bases.(at)
      ; channel_base = channel_bases.(at)
      })
  in
  (* The weight image in the DWELL order. The twin gives the checkpoint order — for each
     layer the flat kernel [3; 3; inputs; outputs] — and the dwell walks the input
     channel, then the tap, then the group. The permutation stands here, thus the twin
     stays the authority on every value and the circuit reads one counter. *)
  let image = Quantized.Model.rom_bits model in
  let image_bases = Quantized.Model.rom_bases model in
  let layer_words at (l : Quantized.Model.layer) =
    let base = image_bases.(at) in
    let byte ~cin ~tap ~group lane =
      let channel = (group * lanes) + lane in
      (* A group that runs past the channels takes a zero byte: its lane multiplies by
         zero and the drain does not read it. The padding keeps every row of the image a
         whole number of words, thus the address only counts. *)
      if channel >= l.outputs
      then Bits.zero 8
      else image.(base + ((((tap * l.inputs) + cin) * l.outputs) + channel))
    in
    let word ~cin ~tap ~group =
      Bits.concat_lsb (List.init lanes ~f:(byte ~cin ~tap ~group))
    in
    List.concat_map (List.range 0 l.inputs) ~f:(fun cin ->
      List.concat_map (List.range 0 taps) ~f:(fun tap ->
        List.init (groups_of l) ~f:(fun group -> word ~cin ~tap ~group)))
  in
  let rom = Array.of_list (List.concat (List.mapi (Array.to_list twin) ~f:layer_words)) in
  let constant_word (gain : Nn.Constants.scale) bias =
    if gain.q < 0 || gain.q >= 1 lsl shift_bits
    then invalid_argf "a gain shift of %d does not fit %d bits" gain.q shift_bits ();
    Bits.concat_lsb
      [ Bits.of_signed_int ~width:bias_bits bias
      ; Bits.of_unsigned_int ~width:shift_bits gain.q
      ; Bits.of_signed_int ~width:gain_bits gain.q_value
      ]
  in
  let constants =
    Array.concat_map twin ~f:(fun (l : Quantized.Model.layer) ->
      Array.mapi l.gain ~f:(fun channel gain -> constant_word gain l.bias.(channel)))
  in
  let alpha =
    Array.init walk ~f:(fun pass ->
      Bits.of_unsigned_int ~width:alpha_bits (Diffusion.anneal_threshold ~step:pass ~walk))
  in
  (* The channels X and Y each hold: the widest layer that writes a store. The head writes
     no store, thus its [voices] channels size nothing. *)
  let store_channels =
    Array.foldi twin ~init:0 ~f:(fun at widest (l : Quantized.Model.layer) ->
      match role_at ~count at with
      | Head -> widest
      | Stem | Pair_open | Pair_close -> max widest l.outputs)
  in
  { steps
  ; rows
  ; lanes
  ; walk
  ; store_channels
  ; layers
  ; rom
  ; constants
  ; alpha
  ; temper = model.temper
  }
;;

(* The elaboration as a table. The schedule prints — the discipline of the eras — thus the
   cost model of the document and the machine cannot part without a test saying so. *)
let elaboration_to_string elaboration =
  let ends = function
    | Stem -> "planes -> X"
    | Pair_open -> "X -> Y"
    | Pair_close -> "Y + X -> X"
    | Head -> "X -> logits"
  in
  let layer_line at layer =
    sprintf
      "%3d  %-12s %4d %4d %3d %5d %7d %6d %10d"
      at
      (ends layer.role)
      layer.inputs
      layer.outputs
      layer.groups
      (dwell layer)
      layer.weight_base
      layer.channel_base
      (layer_cycles elaboration layer)
  in
  let head =
    [ sprintf
        "T %d, P %d, H %d, G %d, N %d"
        elaboration.steps
        elaboration.rows
        elaboration.store_channels
        elaboration.lanes
        elaboration.walk
    ; sprintf
        "%3s  %-12s %4s %4s %3s %5s %7s %6s %10s"
        "at"
        "ends"
        "cin"
        "cout"
        "gr"
        "dwell"
        "w base"
        "c base"
        "cycles"
    ]
  in
  let foot =
    [ sprintf
        "rom %d words of %d bits; constants %d words of %d bits; alpha %d of %d bits"
        (Array.length elaboration.rom)
        (elaboration.lanes * 8)
        (Array.length elaboration.constants)
        constant_bits
        (Array.length elaboration.alpha)
        alpha_bits
    ; sprintf
        "the array is %d by %d, thus %d lanes"
        elaboration.rows
        elaboration.lanes
        (elaboration.rows * elaboration.lanes)
    ; sprintf
        "a store is %d columns of %d bits"
        (elaboration.steps * elaboration.store_channels)
        (elaboration.rows * 16)
    ; sprintf
        "forward %d cycles, the cell walk %d, a pass %d less the draw"
        (forward_cycles elaboration)
        (cell_walk_cycles elaboration)
        (pass_cycles elaboration)
    ]
  in
  head @ Array.to_list (Array.mapi elaboration.layers ~f:layer_line) @ foot
  |> String.concat ~sep:"\n"
;;

let%expect_test "the elaboration of the elected rung" =
  (* THE SHAPE OF `l16-h16-100k`, ON DRAWN WEIGHTS. A cycle count reads the shape and
     never a value, thus a test states the elected rung's geometry without a checkpoint
     file that git ignores. The rung's real weights arrive at [gen_verilog]. *)
  let config = { Diffusion.Config.layers = 16; width = 16 } in
  let model = Quantized.Model.For_test.init config ~seed:1 in
  print_endline (elaboration_to_string (elaborate model ~steps:128 ~lanes:4 ~walk:512));
  [%expect
    {|
    T 128, P 48, H 16, G 4, N 512
     at  ends          cin cout  gr dwell  w base c base     cycles
      0  planes -> X     8   16   4    72       0      0      36912
      1  X -> Y         16   16   4   144     288     16      73776
      2  Y + X -> X     16   16   4   144     864     32      73776
      3  X -> Y         16   16   4   144    1440     48      73776
      4  Y + X -> X     16   16   4   144    2016     64      73776
      5  X -> Y         16   16   4   144    2592     80      73776
      6  Y + X -> X     16   16   4   144    3168     96      73776
      7  X -> Y         16   16   4   144    3744    112      73776
      8  Y + X -> X     16   16   4   144    4320    128      73776
      9  X -> Y         16   16   4   144    4896    144      73776
     10  Y + X -> X     16   16   4   144    5472    160      73776
     11  X -> Y         16   16   4   144    6048    176      73776
     12  Y + X -> X     16   16   4   144    6624    192      73776
     13  X -> Y         16   16   4   144    7200    208      73776
     14  Y + X -> X     16   16   4   144    7776    224      73776
     15  X -> logits    16    4   1   144    8352    240      18480
    rom 8496 words of 32 bits; constants 244 words of 38 bits; alpha 512 of 24 bits
    the array is 48 by 4, thus 192 lanes
    a store is 2048 columns of 768 bits
    forward 1088256 cycles, the cell walk 1536, a pass 1089792 less the draw
    |}]
;;

let%expect_test "the ROM image is the twin's image in the dwell order" =
  (* The packing must be a BIJECTION onto the twin's kernels: every weight stands at one
     dwell address, no two weights share one, and a lane past the channels reads zero. H 6
     at G 4 makes the ragged group the elected shapes never make, thus the padding is
     under test and not only described. *)
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let elaboration = elaborate model ~steps:8 ~lanes:4 ~walk:4 in
  let kernels = Array.of_list (Quantized.Model.rom_tensors model) in
  let seen = Array.map kernels ~f:(fun k -> Array.map k.q ~f:(fun _ -> 0)) in
  let pad = ref 0 in
  let pad_not_zero = ref 0 in
  let disagree = ref 0 in
  Array.iteri elaboration.layers ~f:(fun at layer ->
    let kernel = kernels.(at) in
    let word ~cin ~tap ~group =
      elaboration.rom.(layer.weight_base + ((((cin * taps) + tap) * layer.groups) + group))
    in
    let lane_byte word lane =
      Bits.to_unsigned_int (Bits.select word ~high:((lane * 8) + 7) ~low:(lane * 8))
    in
    for cin = 0 to layer.inputs - 1 do
      for tap = 0 to taps - 1 do
        for group = 0 to layer.groups - 1 do
          for lane = 0 to elaboration.lanes - 1 do
            let byte = lane_byte (word ~cin ~tap ~group) lane in
            let channel = (group * elaboration.lanes) + lane in
            if channel >= layer.outputs
            then (
              Int.incr pad;
              if byte <> 0 then Int.incr pad_not_zero)
            else (
              (* the twin's own layout: the flat kernel reads as [3; 3; inputs; outputs] *)
              let index = (((tap * layer.inputs) + cin) * layer.outputs) + channel in
              seen.(at).(index) <- seen.(at).(index) + 1;
              if byte <> kernel.q.(index) land 255 then Int.incr disagree)
          done
        done
      done
    done);
  let counts = Array.concat_map seen ~f:Fn.id in
  printf
    "%d words, %d weights, each seen %s\n"
    (Array.length elaboration.rom)
    (Array.length counts)
    (if Array.for_all counts ~f:(fun n -> n = 1) then "one time" else "WRONG");
  printf
    "%d padded lanes, %d of them not zero; %d bytes disagree with the twin\n"
    !pad
    !pad_not_zero
    !disagree;
  [%expect
    {|
    414 words, 1296 weights, each seen one time
    360 padded lanes, 0 of them not zero; 0 bytes disagree with the twin
    |}]
;;

let%expect_test "a constant word carries the twin's gain, shift and bias" =
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let elaboration = elaborate model ~steps:8 ~lanes:4 ~walk:4 in
  let field word ~low ~width = Bits.select word ~high:(low + width - 1) ~low in
  let disagree = ref 0 in
  Array.iteri elaboration.layers ~f:(fun at layer ->
    let twin = model.layers.(at) in
    for channel = 0 to layer.outputs - 1 do
      let word = elaboration.constants.(layer.channel_base + channel) in
      let bias = Bits.to_signed_int (field word ~low:0 ~width:bias_bits) in
      let shift = Bits.to_unsigned_int (field word ~low:bias_bits ~width:shift_bits) in
      let gain =
        Bits.to_signed_int (field word ~low:(bias_bits + shift_bits) ~width:gain_bits)
      in
      let { Nn.Constants.q_value; q } = twin.gain.(channel) in
      if bias <> twin.bias.(channel) || shift <> q || gain <> q_value
      then Int.incr disagree
    done);
  printf
    "%d constant words of %d bits, %d disagree with the twin\n"
    (Array.length elaboration.constants)
    constant_bits
    !disagree;
  [%expect {| 22 constant words of 38 bits, 0 disagree with the twin |}]
;;

let%expect_test "the elaboration refuses what the machine cannot hold" =
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let refuse name f =
    match f () with
    | (_ : elaboration) -> printf "%s: NOT REFUSED\n" name
    | exception Invalid_argument message -> printf "%s: %s\n" name message
  in
  (* the drain rule: at P 60 the H 6 layers dwell 54 cycles and the chain is 60 stages *)
  refuse "a chain that cannot empty" (fun () ->
    elaborate model ~rows:60 ~steps:8 ~lanes:4 ~walk:4);
  refuse "a walk of no passes" (fun () -> elaborate model ~steps:8 ~lanes:4 ~walk:0);
  refuse "a canvas of no steps" (fun () -> elaborate model ~steps:0 ~lanes:4 ~walk:4);
  refuse "a group of no lanes" (fun () -> elaborate model ~steps:8 ~lanes:0 ~walk:4);
  [%expect
    {|
    a chain that cannot empty: layer 1 dwells 54 cycles and its drain takes 60: the chain cannot empty
    a walk of no passes: a walk of 0 passes
    a canvas of no steps: a canvas of 0 steps
    a group of no lanes: a group of 0 lanes
    |}]
;;
