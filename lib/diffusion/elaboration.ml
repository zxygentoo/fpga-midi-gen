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

(* ONE BANK OF THE WEIGHT ROM: where it starts in the flat image, and the words it
   addresses. The depth is a power of two and the base is a multiple of it, thus the
   offset inside a bank is the LOW BITS of the flat address and the bank is the bits above
   them — no subtractor stands on the address side. *)
type weight_bank =
  { base : int
  ; depth : int
  }

type t =
  { steps : int
  ; rows : int
  ; lanes : int
  ; walk : int
  ; store_channels : int
  ; layers : layer array
  ; weight_rom : Bits.t array
  ; weight_banks : weight_bank array
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
type 'a norm_fields =
  { gain : 'a
  ; shift : 'a
  ; bias : 'a
  }

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

(* THE OTHER MAP OF THE IMAGES: which output channel a lane of a group names. The weight
   image and the norm image are both walked by it, and both pad where it runs past a
   layer's channels. It is a function so that [Rtl] can state the same rule to the circuit
   rather than let the circuit restate it. *)
let channel_of t ~group ~lane = (group * t.lanes) + lane

(* The widths the circuit sizes its address ports on. They follow from the table, thus
   they stand here and a unit that derived them again would be free to derive them
   differently. A ragged group runs past its layer's own channels — a head of four seats
   in a group of three reaches channel five — thus the channel width follows the GROUPS
   and not the outputs. *)
let store_bits t = Bits.address_bits_for (store_depth t)

let widest_channel t =
  Array.fold t.layers ~init:1 ~f:(fun widest l ->
    max widest (max l.outputs (l.groups * t.lanes)))
;;

let channel_bits t = Bits.address_bits_for (widest_channel t + 1)

(* THE WEIGHT ROM IS BANKED BY POWERS OF TWO, AND THE REASON IS A TRAP IN THE TOOL: VIVADO
   PADS AN INFERRED ROM TO ITS FULL ADDRESS SPACE, AND THE PADDING IS SILENT. Rung 1 paid
   16 tiles for an image that stores 9, and the build absorbed it unseen. Rung 2 asked 64
   tiles against 49 free, and the mapper answered by demoting EVERY ROM of the design to
   fabric — the weights, the norms, the anneal table and the exp2 table — with no warning
   that names it. A bank whose depth is a power of two has no address space above its own
   depth, thus the pad becomes the elaboration's own: bounded, and printed by [to_string].

   THE PLAN IS THE TOP BIT OF THE COUNT AND ONE TAIL, and it is taken only where it costs
   less than one bank. Rung 1 banks its 8496 words as 8192 and 512 against a single ROM of
   16384; rung 2 banks its 36144 as 32768 and 4096 against 65536. A third bank would save
   half a tile and buy a second level of mux, thus the split stops at two.

   A BANK IS NEVER BELOW 512 WORDS. A 512 by 36 RAMB18 is the smallest tile a word of this
   width fills, thus a shallower bank holds less for the same tile — and the floor is what
   makes a comparison of DEPTHS a comparison of tiles. It also holds the alignment: a head
   under the floor never wins the comparison, thus every base a plan states is a multiple
   of its own bank's depth. *)
let smallest_bank = 512
let bank_depth words = max smallest_bank (Int.ceil_pow2 (max 1 words))

let weight_bank_plan words =
  let whole = [ { base = 0; depth = bank_depth words } ] in
  let head = Int.floor_pow2 (max 1 words) in
  let tail = bank_depth (words - head) in
  if head < words && head >= smallest_bank && head + tail < bank_depth words
  then [ { base = 0; depth = head }; { base = head; depth = tail } ]
  else whole
;;

(* THE BANK AS THE BITSTREAM CARRIES IT: the bank's slice of the image, then its pad. The
   flat image stays the one authority on every value and on the dwell order; a bank is a
   window on it, and the pad stands above the last word alone. *)
let weight_bank_image t { base; depth } =
  let words = Array.length t.weight_rom in
  Array.init depth ~f:(fun at ->
    if base + at < words
    then t.weight_rom.(base + at)
    else Bits.zero (Bits.width t.weight_rom.(0)))
;;

(* WHICH BANK HOLDS A FLAT WEIGHT ADDRESS: the last one whose base the address has
   reached, because the banks tile the image from address zero. This is the whole of the
   tiling rule, and [Rtl.weight_bank] states it to the circuit; the gate below holds the
   two together over every address either can name. *)
let weight_bank_at t address =
  Array.foldi t.weight_banks ~init:0 ~f:(fun at select bank ->
    if address >= bank.base then at else select)
;;

(* THE MAPS THE IMAGE AND THE CIRCUIT MUST AGREE ON, over any combinational type. Signal
   land cannot call the software functions above — it holds signals and not integers —
   thus era five's answer was to state each map twice and let a gate weld them at run
   time. The vocabulary already had the better answer: one rule over [Comb], evaluated at
   [Bits] by a test and elaborated at [Signal] by the circuit, which is what [Vocab.Rtl]
   does for the class-to-code map. The two halves then stop being two.

   THE PIN IS A LABELLED ARGUMENT AND NOT A SECOND FUNCTOR ARGUMENT. The address maps
   carry a multiply and the multiply is kept out of the DSPs; [add_attribute] is Signal's
   alone and means nothing to a [Bits] value, thus the pin cannot live inside [Comb]. As a
   functor argument it would make [Epilogue] — which wants the norm fields, and multiplies
   nothing — name the array's rule for no reason, and would put this module's own
   instantiation in debt to one two levels above it. *)
module Rtl = struct
  module Make (Comb : Hardcaml.Comb.S) = struct
    open Comb

    (* [major * stride + minor] at [width] bits. THE MULTIPLY IS REAL AND NOT A
       CONCATENATION — neither stride is a power of two at every shape, and a
       concatenation would silently stride by [2 ** minor_bits]. *)
    let flat_index ~pin ~width ~stride major minor =
      let scaled =
        pin
          (uresize major ~width
           *: of_unsigned_int ~width:(address_bits_for (stride + 1)) stride)
      in
      sel_bottom scaled ~width +: uresize minor ~width
    ;;

    let column_address ~pin t ~step ~channel =
      flat_index ~pin ~width:(store_bits t) ~stride:t.store_channels step channel
    ;;

    let channel_of ~pin t ~group ~lane =
      flat_index ~pin ~width:(channel_bits t) ~stride:t.lanes group lane
    ;;

    (* WHICH BANK OF THE WEIGHT ROM A FLAT ADDRESS FALLS IN: the last one whose base the
       address has reached, because the banks tile the image from address zero. At both
       elected plans that is one comparison against a power of two — the top address bit
       and nothing more — and the offset inside the bank is the low bits, thus no
       subtractor stands anywhere on the address side. *)
    let weight_bank t ~address =
      let bits = address_bits_for (Array.length t.weight_banks) in
      Array.foldi t.weight_banks ~init:(zero bits) ~f:(fun at select bank ->
        if at = 0
        then select
        else
          mux2
            (address >=: of_unsigned_int ~width:(width address) bank.base)
            (of_unsigned_int ~width:bits at)
            select)
    ;;

    let norm_fields word =
      let field ~low ~bits = select word ~high:(low + bits - 1) ~low in
      { bias = field ~low:0 ~bits:bias_bits
      ; shift = field ~low:bias_bits ~bits:shift_bits
      ; gain = field ~low:(bias_bits + shift_bits) ~bits:gain_bits
      }
    ;;
  end

  include Make (Hardcaml.Signal)
end

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
  let weight_banks = Array.of_list (weight_bank_plan (Array.length weight_rom)) in
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
  ; weight_banks
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
    ; sprintf
        "the weight ROM banks %s, %d words of pad"
        (String.concat
           ~sep:" + "
           (List.map (Array.to_list t.weight_banks) ~f:(fun bank ->
              Int.to_string bank.depth)))
        (Array.sum (module Int) t.weight_banks ~f:(fun b -> b.depth)
         - Array.length t.weight_rom)
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
  (* THE SHAPE OF `l64-h16-100k`, ON DRAWN WEIGHTS. A cycle count reads the shape and
     never a value, thus a test states the elected rung's geometry without a checkpoint
     file that git ignores. The rung's real weights arrive at [gen_verilog]. *)
  let config = { Diffusion.Config.layers = 64; width = 16 } in
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
     15  X -> Y         16   16   4   144    8352    240      73776
     16  Y + X -> X     16   16   4   144    8928    256      73776
     17  X -> Y         16   16   4   144    9504    272      73776
     18  Y + X -> X     16   16   4   144   10080    288      73776
     19  X -> Y         16   16   4   144   10656    304      73776
     20  Y + X -> X     16   16   4   144   11232    320      73776
     21  X -> Y         16   16   4   144   11808    336      73776
     22  Y + X -> X     16   16   4   144   12384    352      73776
     23  X -> Y         16   16   4   144   12960    368      73776
     24  Y + X -> X     16   16   4   144   13536    384      73776
     25  X -> Y         16   16   4   144   14112    400      73776
     26  Y + X -> X     16   16   4   144   14688    416      73776
     27  X -> Y         16   16   4   144   15264    432      73776
     28  Y + X -> X     16   16   4   144   15840    448      73776
     29  X -> Y         16   16   4   144   16416    464      73776
     30  Y + X -> X     16   16   4   144   16992    480      73776
     31  X -> Y         16   16   4   144   17568    496      73776
     32  Y + X -> X     16   16   4   144   18144    512      73776
     33  X -> Y         16   16   4   144   18720    528      73776
     34  Y + X -> X     16   16   4   144   19296    544      73776
     35  X -> Y         16   16   4   144   19872    560      73776
     36  Y + X -> X     16   16   4   144   20448    576      73776
     37  X -> Y         16   16   4   144   21024    592      73776
     38  Y + X -> X     16   16   4   144   21600    608      73776
     39  X -> Y         16   16   4   144   22176    624      73776
     40  Y + X -> X     16   16   4   144   22752    640      73776
     41  X -> Y         16   16   4   144   23328    656      73776
     42  Y + X -> X     16   16   4   144   23904    672      73776
     43  X -> Y         16   16   4   144   24480    688      73776
     44  Y + X -> X     16   16   4   144   25056    704      73776
     45  X -> Y         16   16   4   144   25632    720      73776
     46  Y + X -> X     16   16   4   144   26208    736      73776
     47  X -> Y         16   16   4   144   26784    752      73776
     48  Y + X -> X     16   16   4   144   27360    768      73776
     49  X -> Y         16   16   4   144   27936    784      73776
     50  Y + X -> X     16   16   4   144   28512    800      73776
     51  X -> Y         16   16   4   144   29088    816      73776
     52  Y + X -> X     16   16   4   144   29664    832      73776
     53  X -> Y         16   16   4   144   30240    848      73776
     54  Y + X -> X     16   16   4   144   30816    864      73776
     55  X -> Y         16   16   4   144   31392    880      73776
     56  Y + X -> X     16   16   4   144   31968    896      73776
     57  X -> Y         16   16   4   144   32544    912      73776
     58  Y + X -> X     16   16   4   144   33120    928      73776
     59  X -> Y         16   16   4   144   33696    944      73776
     60  Y + X -> X     16   16   4   144   34272    960      73776
     61  X -> Y         16   16   4   144   34848    976      73776
     62  Y + X -> X     16   16   4   144   35424    992      73776
     63  X -> logits    16    4   1   144   36000   1008      18480
    the weight ROM 36144 words of 32 bits, the norms 1012 of 38, the anneal 512 of 24
    the weight ROM banks 32768 + 4096, 720 words of pad
    the array is 48 by 4, thus 192 lanes
    the seats open inside the classes 1 to 31, 11 to 34, 17 to 39, 25 to 46
    a store is 2048 columns of 768 bits, t-major: step * 16 + channel
    forward 4629504 cycles, the cell walk 1536, a pass 4631040 less the draw
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

let%expect_test "the banks re-concatenate into the image, and the circuit finds them" =
  (* THE BANKING IS A PERMUTATION OF ADDRESS SPACE AND NEVER OF VALUES, and this walks it
     THE WAY THE CIRCUIT READS IT: the flat counter names the bank by the bits above its
     depth and the word inside it by the bits below, thus a base that is not a multiple of
     its own bank's depth fails here and not in a mapping report. The walk above holds the
     dwell ORDER; this holds that the memories the bitstream really carries stand in that
     order word for word, and that the pad above the last word is zero and is the size the
     print states.

     A bijection over the flat image passes any bank order, thus the walk is the counter's
     own — as the walk above is. And the decode stands beside it: [Rtl.Make (Bits)] is the
     very function [Forward] elaborates at [Signal], thus the circuit's select and the
     software's tiling are one rule at every address. THE ELECTED RUNG IS THE SHAPE THAT
     REALLY BANKS: the tiny shapes hold one bank and would pass any decode. *)
  let module Map = Rtl.Make (Bits) in
  let case ~name t =
    let words = Array.length t.weight_rom in
    let images = Array.map t.weight_banks ~f:(weight_bank_image t) in
    let depth = Array.sum (module Int) images ~f:Array.length in
    let apart = ref 0 in
    let pad_not_zero = ref 0 in
    let decoded_apart = ref 0 in
    for address = 0 to depth - 1 do
      let bank = weight_bank_at t address in
      let circuit =
        Map.weight_bank
          t
          ~address:(Bits.of_unsigned_int ~width:(Bits.address_bits_for words) address)
      in
      if Bits.to_unsigned_int circuit <> bank then Int.incr decoded_apart;
      (* the circuit's own offset: the low bits of the flat address, and nothing else *)
      let image = images.(bank) in
      let word = image.(address land (Array.length image - 1)) in
      if address < words
      then (if Bits.compare word t.weight_rom.(address) <> 0 then Int.incr apart)
      else if Bits.to_unsigned_int word <> 0
      then Int.incr pad_not_zero
    done;
    printf
      "%s: %d words banked %s, %d of pad\n"
      name
      words
      (String.concat
         ~sep:" + "
         (List.map (Array.to_list t.weight_banks) ~f:(fun bank ->
            Int.to_string bank.depth)))
      (depth - words);
    printf
      "  %d addresses walked: %d apart from the image, %d pad words not zero, %d banks \
       decoded apart\n"
      depth
      !apart
      !pad_not_zero
      !decoded_apart
  in
  let rung = { Diffusion.Config.layers = 64; width = 16 } in
  case
    ~name:"the elected rung"
    (create (Quantized.Model.For_test.init rung ~seed:1) ~steps:128 ~lanes:4 ~walk:512);
  case
    ~name:"a shape of one bank"
    (create
       Quantized.Model.(For_test.init For_test.config ~seed:7)
       ~steps:8
       ~lanes:4
       ~walk:4);
  [%expect
    {|
    the elected rung: 36144 words banked 32768 + 4096, 720 of pad
      36864 addresses walked: 0 apart from the image, 0 pad words not zero, 0 banks decoded apart
    a shape of one bank: 414 words banked 512, 98 of pad
      512 addresses walked: 0 apart from the image, 0 pad words not zero, 0 banks decoded apart
    |}]
;;

let%expect_test "a norm word carries the twin's gain, shift and bias" =
  let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
  let t = create model ~steps:8 ~lanes:4 ~walk:4 in
  (* THE TEST SLICES WITH THE CIRCUIT'S OWN UNPACKER and never with a third reading of the
     field order: [Rtl.Make (Bits)] is the very function the epilogue elaborates. *)
  let module Fields = Rtl.Make (Bits) in
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
        let { gain; shift; bias } = Fields.norm_fields word in
        let { Nn_quantized.Constants.q_value; q } = twin.gain.(channel) in
        if Bits.to_signed_int bias <> twin.bias.(channel)
           || Bits.to_unsigned_int shift <> q
           || Bits.to_signed_int gain <> q_value
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

let%expect_test "the circuit states the maps the software states" =
  (* ONE RULE, TWO HALVES, HELD TOGETHER. [Rtl.Make (Bits)] evaluates exactly the
     functions [Forward] elaborates at [Signal], thus this holds the circuit's arithmetic
     against the software's over every address either can name — the gate [Vocab.Rtl]
     already stands for the vocabulary's own map. Before the functor these two were
     separate statements and only the store-write stream said whether they agreed.

     The pin is the identity here: an attribute means nothing to a value, and it is the
     multiply's placement and not its arithmetic. The shapes below make a RAGGED group — H
     6 at G 4 — thus the channel map is under test past a layer's own channels, which is
     where the padding lives. *)
  let module Map = Rtl.Make (Bits) in
  let case ~steps ~lanes =
    let model = Quantized.Model.(For_test.init For_test.config ~seed:7) in
    let t = create model ~steps ~lanes ~walk:4 in
    let addresses = ref 0
    and channels = ref 0
    and apart = ref 0 in
    let widest = Array.fold t.layers ~init:0 ~f:(fun n l -> max n l.groups) in
    for step = 0 to t.steps - 1 do
      for channel = 0 to t.store_channels - 1 do
        Int.incr addresses;
        let circuit =
          Map.column_address
            ~pin:Fn.id
            t
            ~step:(Bits.of_unsigned_int ~width:(Bits.address_bits_for t.steps) step)
            ~channel:(Bits.of_unsigned_int ~width:(channel_bits t) channel)
        in
        if Bits.to_unsigned_int circuit <> column_address t ~step ~channel
        then Int.incr apart
      done
    done;
    for group = 0 to widest - 1 do
      for lane = 0 to t.lanes - 1 do
        Int.incr channels;
        let circuit =
          Map.channel_of
            ~pin:Fn.id
            t
            ~group:(Bits.of_unsigned_int ~width:(channel_bits t) group)
            ~lane:(Bits.of_unsigned_int ~width:(Bits.address_bits_for t.lanes) lane)
        in
        if Bits.to_unsigned_int circuit <> channel_of t ~group ~lane then Int.incr apart
      done
    done;
    printf
      "T %d, G %d: %d store addresses and %d channels, %d apart\n"
      steps
      lanes
      !addresses
      !channels
      !apart
  in
  case ~steps:8 ~lanes:4;
  (* a stride that is not a power of two, where a concatenation would pass the one above *)
  case ~steps:5 ~lanes:3;
  case ~steps:12 ~lanes:1;
  (* AND THE NORM WORD, PACKED HERE AND SLICED THERE, over the whole range a quantizer can
     state: every shift the field holds, against the two rails of the gain and the bias. *)
  let words = ref 0
  and apart = ref 0 in
  List.iter [ 0; 1; 30; 44; 63 ] ~f:(fun q ->
    List.iter [ 0; 1; -1; 32767; -32768 ] ~f:(fun q_value ->
      List.iter [ 0; 1; -1; 32767; -32768 ] ~f:(fun bias ->
        Int.incr words;
        let { gain; shift; bias = read } =
          Map.norm_fields (norm_word { Nn_quantized.Constants.q_value; q } ~bias)
        in
        if Bits.to_signed_int gain <> q_value
           || Bits.to_unsigned_int shift <> q
           || Bits.to_signed_int read <> bias
        then Int.incr apart)));
  printf "%d norm words packed and sliced, %d apart\n" !words !apart;
  [%expect
    {|
    T 8, G 4: 48 store addresses and 8 channels, 0 apart
    T 5, G 3: 30 store addresses and 6 channels, 0 apart
    T 12, G 1: 72 store addresses and 6 channels, 0 apart
    125 norm words packed and sliced, 0 apart
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
