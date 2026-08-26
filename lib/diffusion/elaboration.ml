(* L1 of the diffusion source — see elaboration.mli for the contract and
   docs/diffusion_rtl.md for the design. What stands here is the WHY of each rule; the
   interface states what each one is. *)

open Core
module Bits = Hardcaml.Bits
module Nn_quantized = Mgen_nn.Quantized

type layer_role =
  | Stem
  | Pair_open
  | Pair_close
  | Head

type layer =
  { role : layer_role
  ; inputs : int
  ; outputs : int
  ; groups : int
  ; weight_base : int
  ; norm_base : int
  }

type t =
  { steps : int
  ; rows : int
  ; lanes : int
  ; walk : int
  ; store_channels : int
  ; layers : layer array
  ; weight_rom : Bits.t array
  ; norm_rom : Bits.t array
  ; alpha_rom : Bits.t array
  ; openings : Diffusion.opening array
  ; temper : Nn_quantized.Constants.scale
  }

(* the taps of one 3 by 3 kernel *)
let taps = 9

(* the 24-bit grid of the generator: an anneal threshold is [floor (alpha * 2 ** 24)],
   thus a mask uniform compares against it exactly *)
let alpha_bits = 24

(* The fields of a norm word, low to high: the bias, the shift of the gain, and the value
   of the gain. The three stand at one address because the epilogue wants the three at one
   time, thus no two of them can fall out of step.

   THE SHIFT FIELD SIZES ON THE RULE AND NOT ON THE CHECKPOINT. The q of a gain is
   [e + weight_exponent], and the two exponent rules cap at 30 and at 14, thus 44 is the
   largest q a quantizer can state and six bits hold it. A field sized on the elected
   model's own peak would save a bit or two and would make a drawn-weight timing probe
   create a DIFFERENT netlist from the trained build; the bits are worth less than that.
   [create] refuses a q the field cannot hold. *)
let bias_bits = 16
let shift_bits = 6
let gain_bits = 16
let norm_bits = bias_bits + shift_bits + gain_bits

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
   last of them. The count is exact and not a bound, because [create] refuses a layer
   whose dwell is shorter than its drain — no dwell ever waits for the chain to empty.
   What the layer turn itself costs is the cycle bench's to measure. *)
let layer_cycles t layer = (t.steps * layer.groups * dwell layer) + t.rows
let forward_cycles t = Array.sum (module Int) t.layers ~f:(layer_cycles t)
let canvas_cells t = t.steps * Frame.voices

(* one uniform for each cell in the cell order: the opening, and the mask of each pass *)
let cell_walk_cycles t = canvas_cells t * uniform_cycles

(* One pass, LESS THE DRAW: the mask and the forward. The draw's cycles are L4's and the
   bench's — the design estimates about 2 [rows] for each hidden cell — and a number this
   module cannot state exactly it does not state. *)
let pass_cycles t = cell_walk_cycles t + forward_cycles t

(* THE ONE PACKER OF A NORM WORD. The order is a fact of this function and of nothing
   else: a consumer that slices another way disagrees with it, and the gate that packs
   here and slices there is what says so. *)
let norm_word (gain : Nn_quantized.Constants.scale) ~bias =
  if gain.q < 0 || gain.q >= 1 lsl shift_bits
  then invalid_argf "a gain shift of %d does not fit %d bits" gain.q shift_bits ();
  Bits.concat_lsb
    [ Bits.of_signed_int ~width:bias_bits bias
    ; Bits.of_unsigned_int ~width:shift_bits gain.q
    ; Bits.of_signed_int ~width:gain_bits gain.q_value
    ]
;;

(* WHERE A STORE HOLDS A COLUMN IS A FACT OF THIS FUNCTION AND OF NOTHING ELSE. The map is
   t-major, thus the G writes of a group land consecutive; nothing else distinguishes the
   orders, and the circuit's ports and the stream instrument therefore slice ONE rule. *)
let column_address t ~step ~channel = (step * t.store_channels) + channel
let store_depth t = t.steps * t.store_channels

let bases_of sizes =
  Array.of_list
    (List.folding_map (Array.to_list sizes) ~init:0 ~f:(fun base size ->
       base + size, base))
;;

let create ?(rows = Diffusion.rows) (model : Quantized.Model.t) ~steps ~lanes ~walk =
  Quantized.Model.check_shape model;
  if steps < 1 then invalid_argf "a canvas of %d steps" steps ();
  if rows < 1 then invalid_argf "a column of %d rows" rows ();
  if lanes < 1 then invalid_argf "a group of %d lanes" lanes ();
  if walk < 1 then invalid_argf "a walk of %d passes" walk ();
  let twin = model.layers in
  let count = Array.length twin in
  let groups_of (l : Quantized.Model.layer) = (l.outputs + lanes - 1) / lanes in
  (* THE DWELL MUST COVER THE DRAIN AND THE BAND LOADS BEHIND IT, and the tighter of the
     two rules is the one stated. The chain is [rows] stages and must empty before the
     next dwell captures the array. Behind it, the residual columns and the norm words of
     the next group are fetched the moment the drain has read its last residual row — one
     address for each lane, two cycles of read latency behind them — because one buffer
     serves every group and nothing is doubled. A dwell shorter than the sum leaves the
     band half-loaded when the next drain reads it, and NOTHING DOWNSTREAM SAYS SO: the
     arithmetic is silently wrong and the write stream would have to catch it. It is a
     check and not a comment, thus [layer_cycles] states an exact count and never a bound. *)
  let dwell_floor = rows + lanes + 2 in
  Array.iteri twin ~f:(fun at (l : Quantized.Model.layer) ->
    if taps * l.inputs < dwell_floor
    then
      invalid_argf
        "layer %d dwells %d cycles and its drain and band loads need %d: the dwell is \
         short"
        at
        (taps * l.inputs)
        dwell_floor
        ());
  let layers =
    let weight_bases =
      bases_of (Array.map twin ~f:(fun l -> l.inputs * taps * groups_of l))
    in
    let norm_bases = bases_of (Array.map twin ~f:(fun l -> groups_of l * lanes)) in
    Array.mapi twin ~f:(fun at (l : Quantized.Model.layer) ->
      { role = role_at ~count at
      ; inputs = l.inputs
      ; outputs = l.outputs
      ; groups = groups_of l
      ; weight_base = weight_bases.(at)
      ; norm_base = norm_bases.(at)
      })
  in
  (* The weight image in the DWELL order. The twin gives the checkpoint order — for each
     layer the flat kernel [3; 3; inputs; outputs] — and the dwell walks the group, then
     the input channel, then the tap. The permutation stands here, thus the twin stays the
     authority on every value and THE CIRCUIT READS ONE COUNTER: a group holds its taps
     and channels back to back, and the groups stand in order, thus one column's dwell
     walks a layer's whole range straight through and the address reloads once for each
     column. Any other order makes the address a stride and not a count. *)
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
    List.concat_map
      (List.range 0 (groups_of l))
      ~f:(fun group ->
        List.concat_map (List.range 0 l.inputs) ~f:(fun cin ->
          List.init taps ~f:(fun tap -> word ~cin ~tap ~group)))
  in
  let weight_rom =
    Array.of_list (List.concat (List.mapi (Array.to_list twin) ~f:layer_words))
  in
  (* The norms pad to whole groups with a zero word, as the weight image pads with zero
     bytes. THE PADDING IS NOT TIDINESS: without it a ragged group's fetch runs past its
     layer's range — and off the end of the image at the last layer, which a head of four
     channels in a group of five really reaches. With it the address is
     [norm_base + group * lanes + lane] at every layer and every group. *)
  let norm_rom =
    Array.concat_map twin ~f:(fun (l : Quantized.Model.layer) ->
      Array.init
        (groups_of l * lanes)
        ~f:(fun channel ->
          if channel >= l.outputs
          then Bits.zero norm_bits
          else norm_word l.gain.(channel) ~bias:l.bias.(channel)))
  in
  let alpha_rom =
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
  ; weight_rom
  ; norm_rom
  ; alpha_rom
    (* the walk's opening, carried and never restated: the circuit reads one value for
       everything it holds, and [Diffusion] stays the authority on what a seat may sing *)
  ; openings = Diffusion.seat_openings
  ; temper = model.temper
  }
;;

(* The table. The schedule prints — the discipline of the eras — thus the cost model of
   the document and the machine cannot part without a test saying so. *)
let to_string t =
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
      layer.norm_base
      (layer_cycles t layer)
  in
  let head =
    [ sprintf
        "T %d, P %d, H %d, G %d, N %d"
        t.steps
        t.rows
        t.store_channels
        t.lanes
        t.walk
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
        "the weight ROM %d words of %d bits, the norms %d of %d, the anneal %d of %d"
        (Array.length t.weight_rom)
        (t.lanes * 8)
        (Array.length t.norm_rom)
        norm_bits
        (Array.length t.alpha_rom)
        alpha_bits
    ; sprintf "the array is %d by %d, thus %d lanes" t.rows t.lanes (t.rows * t.lanes)
    ; sprintf
        "the seats open inside the classes %s"
        (String.concat
           ~sep:", "
           (List.map (Array.to_list t.openings) ~f:(fun { Diffusion.low; width } ->
              sprintf "%d to %d" low (low + width - 1))))
    ; sprintf
        "a store is %d columns of %d bits, t-major: step * %d + channel"
        (store_depth t)
        (t.rows * 16)
        t.store_channels
    ; sprintf
        "forward %d cycles, the cell walk %d, a pass %d less the draw"
        (forward_cycles t)
        (cell_walk_cycles t)
        (pass_cycles t)
    ]
  in
  head @ Array.to_list (Array.mapi t.layers ~f:layer_line) @ foot
  |> String.concat ~sep:"\n"
;;

let%expect_test "the elaboration of the elected rung" =
  (* THE SHAPE OF `l16-h16-100k`, ON DRAWN WEIGHTS. A cycle count reads the shape and
     never a value, thus a test states the elected rung's geometry without a checkpoint
     file that git ignores. The rung's real weights arrive at [gen_verilog]. *)
  let config = { Diffusion.Config.layers = 16; width = 16 } in
  let model = Quantized.Model.For_test.init config ~seed:1 in
  print_endline (to_string (create model ~steps:128 ~lanes:4 ~walk:512));
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
    the weight ROM 8496 words of 32 bits, the norms 244 of 38, the anneal 512 of 24
    the array is 48 by 4, thus 192 lanes
    the seats open inside the classes 1 to 31, 11 to 34, 17 to 39, 25 to 46
    a store is 2048 columns of 768 bits, t-major: step * 16 + channel
    forward 1088256 cycles, the cell walk 1536, a pass 1089792 less the draw
    |}]
;;

let%expect_test "the ROM walks as one counter in the dwell order" =
  (* THE TEST WALKS THE ROM WITH A PLAIN COUNTER, as the circuit does, and demands at each
     step the weight that step of the dwell needs. A packing that is a valid permutation
     but a bad walk therefore fails here and not on the board: a bijection test alone
     passes on any order, and the order is the whole point of the permutation.

     It also holds the two other properties: every weight stands at one address and no two
     share one, and a lane past the channels reads zero. H 6 at G 4 makes the ragged group
     the elected shapes never make, thus the padding is under test and not only described. *)
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let t = create model ~steps:8 ~lanes:4 ~walk:4 in
  let kernels = Array.of_list (Quantized.Model.rom_tensors model) in
  let seen = Array.map kernels ~f:(fun k -> Array.map k.q ~f:(fun _ -> 0)) in
  let lane_byte word lane =
    Bits.to_unsigned_int (Bits.select word ~high:((lane * 8) + 7) ~low:(lane * 8))
  in
  let address = ref 0 in
  let pad = ref 0 in
  let pad_not_zero = ref 0 in
  let disagree = ref 0 in
  let out_of_step = ref 0 in
  Array.iteri t.layers ~f:(fun at layer ->
    let kernel = kernels.(at) in
    if !address <> layer.weight_base then Int.incr out_of_step;
    for group = 0 to layer.groups - 1 do
      for cin = 0 to layer.inputs - 1 do
        for tap = 0 to taps - 1 do
          let word = t.weight_rom.(!address) in
          Int.incr address;
          for lane = 0 to t.lanes - 1 do
            let byte = lane_byte word lane in
            let channel = (group * t.lanes) + lane in
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
    "%d words, the counter ended at %d; %d layer bases out of step\n"
    (Array.length t.weight_rom)
    !address
    !out_of_step;
  printf
    "%d weights, each seen %s\n"
    (Array.length counts)
    (if Array.for_all counts ~f:(fun n -> n = 1) then "one time" else "WRONG");
  printf
    "%d padded lanes, %d of them not zero; %d bytes disagree with the twin\n"
    !pad
    !pad_not_zero
    !disagree;
  [%expect
    {|
    414 words, the counter ended at 414; 0 layer bases out of step
    1296 weights, each seen one time
    360 padded lanes, 0 of them not zero; 0 bytes disagree with the twin
    |}]
;;

let%expect_test "a norm word carries the twin's gain, shift and bias" =
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let t = create model ~steps:8 ~lanes:4 ~walk:4 in
  let field word ~low ~width = Bits.select word ~high:(low + width - 1) ~low in
  let disagree = ref 0 in
  let pad = ref 0 in
  let pad_not_zero = ref 0 in
  Array.iteri t.layers ~f:(fun at layer ->
    let twin = model.layers.(at) in
    (* the whole group, thus the padded lanes of a ragged one are under test: they must
       read zero, or a fetch past a layer's range states a norm that is not its own *)
    for channel = 0 to (layer.groups * t.lanes) - 1 do
      let word = t.norm_rom.(layer.norm_base + channel) in
      if channel >= layer.outputs
      then (
        Int.incr pad;
        if Bits.to_unsigned_int word <> 0 then Int.incr pad_not_zero)
      else (
        let bias = Bits.to_signed_int (field word ~low:0 ~width:bias_bits) in
        let shift = Bits.to_unsigned_int (field word ~low:bias_bits ~width:shift_bits) in
        let gain =
          Bits.to_signed_int (field word ~low:(bias_bits + shift_bits) ~width:gain_bits)
        in
        let { Nn_quantized.Constants.q_value; q } = twin.gain.(channel) in
        if bias <> twin.bias.(channel) || shift <> q || gain <> q_value
        then Int.incr disagree)
    done);
  printf
    "%d norm words of %d bits, %d disagree with the twin\n"
    (Array.length t.norm_rom)
    norm_bits
    !disagree;
  printf "%d padded words, %d of them not zero\n" !pad !pad_not_zero;
  [%expect
    {|
    28 norm words of 38 bits, 0 disagree with the twin
    6 padded words, 0 of them not zero
    |}]
;;

let%expect_test "the elaboration refuses what the machine cannot hold" =
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let refuse name f =
    match f () with
    | (_ : t) -> printf "%s: NOT REFUSED\n" name
    | exception Invalid_argument message -> printf "%s: %s\n" name message
  in
  (* the drain rule: at P 60 the H 6 layers dwell 54 cycles and the chain is 60 stages *)
  refuse "a chain that cannot empty" (fun () ->
    create model ~rows:60 ~steps:8 ~lanes:4 ~walk:4);
  (* THE BAND RULE, AND THE GAP IT CLOSES. At P 48 and G 5 the H 6 layers dwell 54 cycles
     and the chain of 48 empties inside that: the array's rule alone admits this shape.
     The band loads behind the drain need 55, thus the engine would read a half-loaded
     residual band and no gate below it would say so. G 5 is the fused rung's geometry,
     thus this is a shape the ladder could really elaborate. *)
  refuse "a band load the dwell outruns" (fun () ->
    create model ~steps:8 ~lanes:5 ~walk:4);
  refuse "a walk of no passes" (fun () -> create model ~steps:8 ~lanes:4 ~walk:0);
  refuse "a canvas of no steps" (fun () -> create model ~steps:0 ~lanes:4 ~walk:4);
  refuse "a group of no lanes" (fun () -> create model ~steps:8 ~lanes:0 ~walk:4);
  [%expect
    {|
    a chain that cannot empty: layer 1 dwells 54 cycles and its drain and band loads need 66: the dwell is short
    a band load the dwell outruns: layer 1 dwells 54 cycles and its drain and band loads need 55: the dwell is short
    a walk of no passes: a walk of 0 passes
    a canvas of no steps: a canvas of 0 steps
    a group of no lanes: a group of 0 lanes
    |}]
;;
