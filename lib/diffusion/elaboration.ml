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

(* ONE BANK OF A MEMORY: where it starts in the flat address space, and the words it
   addresses. The depth is a power of two and the base is a multiple of it, thus the
   offset inside a bank is the LOW BITS of the flat address and the bank is the bits above
   them — no subtractor stands on the address side. The weight ROM and the two activation
   stores bank by the one rule; only the ROM carries an image. *)
type bank =
  { base : int
  ; depth : int
  }

(* ONE TURN OF THE WALK: the layer of phase A, and the layer of phase B where the turn is
   a pair. The stem and the head are turns of one phase. A turn is what the engine primes
   once and drains once; inside a pair the two layers interleave, thus neither of them is
   a unit the walk can name. *)
type turn =
  { first : int
  ; second : int option
  }

type t =
  { steps : int
  ; rows : int
  ; lanes : int
  ; walk : int
  ; store_channels : int
  ; layers : layer array
  ; turns : turn array
  ; weight_rom : Bits.t array
  ; weight_banks : bank array
  ; store_banks : bank array
  ; ring_banks : bank array
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

(* THE TURNS OF A MODEL: the stem alone, then one turn for each pair, then the head alone.
   The roles already state the shape — [create] builds them with [role_at] — thus this
   reads them back rather than counting layers a second way, and a model whose roles do
   not walk stem, pairs, head raises here and not in a waveform. *)
let turns_of roles =
  let count = Array.length roles in
  let rec walk at =
    if at >= count
    then []
    else (
      match roles.(at) with
      | Stem | Head -> { first = at; second = None } :: walk (at + 1)
      | Pair_open ->
        (match if at + 1 < count then Some roles.(at + 1) else None with
         | Some Pair_close -> { first = at; second = Some (at + 1) } :: walk (at + 2)
         | _ ->
           invalid_argf
             "layer %d opens a pair that layer %d does not close"
             at
             (at + 1)
             ())
      | Pair_close -> invalid_argf "layer %d closes a pair that nothing opened" at ())
  in
  Array.of_list (walk 0)
;;

(* the dwell of one (column, group): one cycle for each (tap, input channel) pair *)
let dwell layer = taps * layer.inputs

(* THE PHASES OF A TURN. A pair runs its opening layer in phase A and its closing layer in
   phase B; the stem and the head have phase A alone. The phase is one bit and it travels
   in the frames, because inside a pair the lead frame can be in B while the now frame is
   still in A. *)
let phase_a = 0
let phase_b = 1

(* A TURN OF ONE PHASE ANSWERS ITS OWN LAYER AT EITHER PHASE, as [Rtl.layer_of] does: the
   two halves of one rule cannot differ on the edge. *)
let layer_of_phase turn phase =
  if phase = phase_b then Option.value turn.second ~default:turn.first else turn.first
;;

let is_pair turn = Option.is_some turn.second

(* ONE BLOCK OF A TURN: the layer it runs, the column of the canvas it works on, and the
   group of output channels. A block dwells [dwell layer] cycles. *)
type block =
  { layer : int
  ; column : int
  ; group : int
  }

(* THE NEST OF A TURN, as the lead frame carries it: the input channel inside a block, the
   group inside a phase, the pair's step counter, and the phase. [Rtl.next_block] advances
   it and [blocks_of_turn] lists what it visits. *)
type 'a nest =
  { cin : 'a
  ; group : 'a
  ; step : 'a
  ; phase : 'a
  }

(* what one advance of the nest gives: the state after it, and whether the turn closed *)
type 'a block_walk =
  { next : 'a nest
  ; ends : 'a
  }

(* THE ORDER OF A PAIR'S BLOCKS, AND WHY B TRAILS A BY TWO. With [s] the pair's step
   counter from 0 to T + 1, the turn runs A at column [s] while [s < T] and B at column
   [s - 2] while [s >= 2]:

   A0, A1, A2 B0, A3 B1, ..., A(T-1) B(T-3), B(T-2), B(T-1).

   B at c reads Y at c + 1, thus A at c + 2 must have written it — and one WHOLE block
   must stand between that write and this read, because a flush lands one epilogue behind
   its drain. A lag of one would make every column wait for a flush and "the same cycle
   count" would be false. The lag also frees X: A at c + 1 was the last reader of X at c,
   thus B at c may overwrite it in place. And it bounds the ring at four columns.

   THE PAIR NEEDS TWO COLUMNS. At T 1 the counter's [s = 1] holds no block at all — A is
   past its last column and B has not reached its first — and a nest that advances one
   step for each block would stall there. [create] refuses it. *)
let phases_at turn ~steps ~s =
  let a = if s < steps then [ phase_a ] else [] in
  let b = if is_pair turn && s >= 2 then [ phase_b ] else [] in
  a @ b
;;

let column_of_phase ~s phase = if phase = phase_b then s - 2 else s

let blocks_of_turn t turn =
  let last_s = if is_pair turn then t.steps + 1 else t.steps - 1 in
  List.range 0 (last_s + 1)
  |> List.concat_map ~f:(fun s ->
    phases_at turn ~steps:t.steps ~s
    |> List.concat_map ~f:(fun phase ->
      let at = layer_of_phase turn phase in
      List.init t.layers.(at).groups ~f:(fun group ->
        { layer = at; column = column_of_phase ~s phase; group })))
;;

(* One turn, exactly: the dwells of every column and group of its one or two layers, and
   ONE drain tail behind the last of them. The count is exact and not a bound, because
   [create] refuses a layer whose dwell is shorter than the drain and the loads behind it.
   What the turn itself costs — the preamble, the tail — is the cycle bench's to measure. *)
let layer_dwell_cycles t layer = t.steps * layer.groups * dwell layer

let turn_cycles t turn =
  let of_layer at = layer_dwell_cycles t t.layers.(at) in
  of_layer turn.first
  + (match turn.second with
    | Some at -> of_layer at
    | None -> 0)
  + t.rows
;;

let forward_cycles t = Array.sum (module Int) t.turns ~f:(turn_cycles t)
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

(* THE Y RING. The fused pair never lets Y exist as a tensor: B at column c reads Y at c -
   1, c and c + 1, and A runs two columns ahead of B, thus FOUR columns of Y are live at
   any moment — c - 1, c, c + 1 and the c + 2 that A has just written. Y at c - 2 died
   with B at c - 1.

   FOUR IS A POWER OF TWO AND THAT IS THE WHOLE OF THE ADDRESS. The ring's step is the low
   two bits of the semantic step, thus no modulo and no compare stands anywhere: the
   engine drives the semantic column and the map takes it. *)
let ring_steps = 4
let ring_depth t = ring_steps * t.store_channels

let ring_address t ~step ~channel =
  (step land (ring_steps - 1) * t.store_channels) + channel
;;

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
let ring_bits t = Bits.address_bits_for (ring_depth t)

let widest_channel t =
  Array.fold t.layers ~init:1 ~f:(fun widest l ->
    max widest (max l.outputs (l.groups * t.lanes)))
;;

let channel_bits t = Bits.address_bits_for (widest_channel t + 1)

(* THE MEMORIES ARE BANKED BY POWERS OF TWO, AND THE REASON IS A TRAP IN THE TOOL: VIVADO
   PADS AN INFERRED MEMORY TO ITS FULL ADDRESS SPACE, AND THE PADDING IS SILENT. Rung 1
   paid 16 tiles for an image that stores 9, and the build absorbed it unseen. Rung 2
   asked 64 tiles against 49 free, and the mapper answered by demoting EVERY ROM of the
   design to fabric — the weights, the norms, the anneal table and the exp2 table — with
   no warning that names it. A bank whose depth is a power of two has no address space
   above its own depth, thus the pad becomes the elaboration's own: bounded, and printed
   by [to_string].

   THE RULE HOLDS FOR A RAM AS IT HOLDS FOR A ROM, and the rung-3 measurement build of
   2026-08-27 is what says so: a store of 1280 columns mapped as [2048x768], the same 43
   tiles rung 2 pays for 2048 columns. At T 128 and H 20 a store is 2560 columns; unbanked
   the mapper rounds it to 4096 and the store costs 86 tiles alone, banked as 2048 + 512
   it costs 43 + 11 = 54.

   THE PLAN IS THE TOP BIT OF THE COUNT AND ONE TAIL, and it is taken only where it costs
   less than one bank. Rung 1 banks its 8496 weight words as 8192 and 512 against a single
   ROM of 16384; rung 2 banks its 36144 as 32768 and 4096 against 65536. A third bank
   would save half a tile and buy a second level of mux, thus the split stops at two.

   A BANK IS NEVER BELOW 512 WORDS. A 512 by 36 RAMB18 is the smallest tile a word of this
   width fills, thus a shallower bank holds less for the same tile — and the floor is what
   makes a comparison of DEPTHS a comparison of tiles. It holds for a 768-bit word too: a
   512 by 768 store is eleven such tiles with no waste. It also holds the alignment: a
   head under the floor never wins the comparison, thus every base a plan states is a
   multiple of its own bank's depth. *)
let smallest_bank = 512
let bank_depth words = max smallest_bank (Int.ceil_pow2 (max 1 words))

let bank_plan words =
  let whole = [ { base = 0; depth = bank_depth words } ] in
  let head = Int.floor_pow2 (max 1 words) in
  let tail = bank_depth (words - head) in
  if head < words && head >= smallest_bank && head + tail < bank_depth words
  then [ { base = 0; depth = head }; { base = head; depth = tail } ]
  else whole
;;

(* THE BANK AS THE BITSTREAM CARRIES IT: the bank's slice of the image, then its pad. The
   flat image stays the one authority on every value and on the dwell order; a bank is a
   window on it, and the pad stands above the last word alone. A store holds no image —
   the circuit writes it — thus this belongs to the ROM alone. *)
let weight_bank_image t { base; depth } =
  let words = Array.length t.weight_rom in
  Array.init depth ~f:(fun at ->
    if base + at < words
    then t.weight_rom.(base + at)
    else Bits.zero (Bits.width t.weight_rom.(0)))
;;

(* WHICH BANK HOLDS A FLAT ADDRESS: the last one whose base the address has reached,
   because the banks tile the address space from zero. This is the whole of the tiling
   rule, and [Rtl.bank_at] states it to the circuit; the gate below holds the two together
   over every address either can name, in the weight plan and in the store plan. *)
let bank_at banks address =
  Array.foldi banks ~init:0 ~f:(fun at select bank ->
    if address >= bank.base then at else select)
;;

(* how [to_string] says a plan: one bank reads differently from two *)
let banks_phrase banks =
  if Array.length banks = 1
  then sprintf "in one bank of %d" banks.(0).depth
  else
    sprintf
      "in banks of %s"
      (String.concat
         ~sep:" + "
         (List.map (Array.to_list banks) ~f:(fun b -> Int.to_string b.depth)))
;;

(* the depths of a plan as [to_string] and the gates print them *)
let bank_depths banks =
  String.concat
    ~sep:" + "
    (List.map (Array.to_list banks) ~f:(fun b -> Int.to_string b.depth))
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

    (* THE RING'S MAP, AND THE ONLY PLACE ITS GEOMETRY IS STATED. The ring holds
       [ring_steps] columns and that is a power of two, thus the low bits of the SEMANTIC
       column are the ring's own step: the engine drives the column it means and this
       takes the bits it keeps. No modulo, no compare, no second counter. *)
    let ring_address ~pin t ~step ~channel =
      flat_index
        ~pin
        ~width:(ring_bits t)
        ~stride:t.store_channels
        (sel_bottom step ~width:(address_bits_for ring_steps))
        channel
    ;;

    let channel_of ~pin t ~group ~lane =
      flat_index ~pin ~width:(channel_bits t) ~stride:t.lanes group lane
    ;;

    (* WHICH BANK OF A PLAN A FLAT ADDRESS FALLS IN: the last one whose base the address
       has reached, because the banks tile the address space from zero. At every elected
       plan that is one comparison against a power of two — the top address bit and
       nothing more — and the offset inside the bank is the low bits, thus no subtractor
       stands anywhere on the address side. *)
    let bank_at banks ~address =
      let bits = address_bits_for (Array.length banks) in
      Array.foldi banks ~init:(zero bits) ~f:(fun at select bank ->
        if at = 0
        then select
        else
          mux2
            (address >=: of_unsigned_int ~width:(width address) bank.base)
            (of_unsigned_int ~width:bits at)
            select)
    ;;

    (* WHICH LAYER A FRAME IS IN: the turn states one or two, and the phase picks. Every
       fact of the table is muxed by this and not by the turn, thus the table's mux is the
       one it always was and only its index has learned to travel. *)
    let layer_of t ~turn ~phase =
      let width = address_bits_for (Array.length t.layers) in
      mux
        turn
        (List.map (Array.to_list t.turns) ~f:(fun tn ->
           let at = of_unsigned_int ~width tn.first in
           match tn.second with
           | None -> at
           | Some second -> mux2 phase (of_unsigned_int ~width second) at))
    ;;

    (* THE NEST OF A TURN, AS THE CIRCUIT WALKS IT. The state is the input channel, the
       group, the pair's step counter and the phase; the counts come from the LEAD layer,
       because it is the lead frame that walks and the lead frame's layer states them.

       The order is [blocks_of_turn]'s and the gate beside it holds the two together over
       every block of every shape. A block closes when the last channel of its last group
       retires. Then the phase turns to B where B is live at this step; otherwise the step
       advances, and the phase opens at A while A is still inside the canvas. A turn ends
       on the close of its last step's last phase. *)
    let next_block t ~is_pair ~cin_count ~group_count { cin; group; step; phase } =
      let at n = of_unsigned_int ~width:(width step) n in
      let last_cin = cin ==: cin_count -:. 1 in
      let last_group = group ==: group_count -:. 1 in
      let closes = last_cin &: last_group in
      let in_a = ~:phase in
      (* B is live once the step has reached column two; A is live while the step is still
         inside the canvas *)
      let b_live = is_pair &: (step >=: at 2) in
      (* A TURN OF ONE PHASE NEVER LEAVES A. Only a pair runs past its last column, and
         only there does the step open at B. *)
      let a_live_next = ~:is_pair |: (step +:. 1 <: at t.steps) in
      let turns_to_b = closes &: in_a &: b_live in
      let turns_step = closes &: ~:turns_to_b in
      let last_step = mux2 is_pair (at (t.steps + 1)) (at (t.steps - 1)) in
      { next =
          { cin = mux2 last_cin (zero (width cin)) (cin +:. 1)
          ; group = mux2 closes (zero (width group)) (mux2 last_cin (group +:. 1) group)
          ; step = mux2 turns_step (step +:. 1) step
          ; phase = mux2 turns_to_b vdd (mux2 turns_step (mux2 a_live_next gnd vdd) phase)
          }
      ; ends = turns_step &: (step ==: last_step)
      }
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
  (* THE DWELL MUST COVER THE DRAIN, THE BAND LOADS BEHIND IT, AND THE NEXT BLOCK'S FETCH
     AHEAD OF IT. The chain is [rows] stages and must empty before the next dwell captures
     the array. Behind it, the residual columns and the norm words of the group are
     fetched the moment that drain has read its last residual row — one address for each
     lane, two cycles of read latency behind them — because one buffer serves every group
     and nothing is doubled. That is [rows + lanes + 2], and it was the whole rule while a
     layer was the unit of the walk.

     THE FUSED PAIR ADDS THE LAST INPUT CHANNEL, AND IT COSTS NINE. Inside a pair no
     preamble stands between the blocks: the next block's three columns are fetched under
     the LAST input channel of the running one, which is [taps] cycles wide. In a B block
     the X port carries the residual load, and the block after it is an A whose fetch
     needs X. The two windows must not overlap, thus the load — which ends by
     [rows + lanes + 2] — must close before the last channel opens at [dwell - taps].

     NOTHING DOWNSTREAM SAYS SO IF IT DOES NOT. A half-loaded band and a column fetched
     from the wrong memory are both silently wrong arithmetic, and only the write stream
     would catch them. It is a check and not a comment, thus [turn_cycles] states an exact
     count and never a bound. H 6 at G 4 dwells 54 against a floor of 54 under the old
     rule and 63 under this one: it is the shape the fused engine would read wrong. *)
  let dwell_floor = rows + lanes + 2 + taps in
  Array.iteri twin ~f:(fun at (l : Quantized.Model.layer) ->
    if taps * l.inputs < dwell_floor
    then
      invalid_argf
        "layer %d dwells %d cycles; its drain, band loads and the next block's fetch \
         need %d: the dwell is short"
        at
        (taps * l.inputs)
        dwell_floor
        ());
  (* A PAIR NEEDS TWO COLUMNS. B trails A by two, thus the pair's step counter walks 0 to
     T + 1; at T 1 the step 1 holds no block at all — A is past its last column and B has
     not reached its first — and a nest that advances one step for each block would stall
     there. Two columns is not a canvas either, but it is a shape the nest can walk. *)
  if steps < 2 && count > 2
  then invalid_argf "a canvas of %d steps cannot carry a fused pair" steps ();
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
  let weight_banks = Array.of_list (bank_plan (Array.length weight_rom)) in
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
  ; turns = turns_of (Array.map layers ~f:(fun l -> l.role))
  ; ring_banks = Array.of_list (bank_plan (ring_steps * store_channels))
  ; weight_rom
  ; weight_banks
  ; store_banks = Array.of_list (bank_plan (steps * store_channels))
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
      (layer_dwell_cycles t layer)
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
        (bank_depths t.weight_banks)
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
        "a store is %d columns of %d bits %s, t-major: step * %d + channel"
        (store_depth t)
        (t.rows * 16)
        (banks_phrase t.store_banks)
        t.store_channels
    ; sprintf
        "the Y ring is %d columns of %d bits %s, and B trails A by two"
        (ring_depth t)
        (t.rows * 16)
        (banks_phrase t.ring_banks)
    ; sprintf
        "the turns are %s"
        (String.concat
           ~sep:", "
           (List.map (Array.to_list t.turns) ~f:(fun turn ->
              match turn.second with
              | None -> sprintf "%d" turn.first
              | Some second -> sprintf "%d+%d" turn.first second)))
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

(* THE TINY SHAPE THE GATES BELOW ELABORATE, and it is NOT the twin's own
   [Quantized.Model.For_test.config]. At H 6 a layer dwells 54 cycles and the fused floor
   asks 63: the model the twin tests with is a model this elaboration refuses, and the
   refusal gate at the foot of the file is where that shape belongs. One channel wider
   walks every map the same way. *)
let tiny_shape = { Diffusion.Config.layers = 4; width = 8 }

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
      0  planes -> X     8   16   4    72       0      0      36864
      1  X -> Y         16   16   4   144     288     16      73728
      2  Y + X -> X     16   16   4   144     864     32      73728
      3  X -> Y         16   16   4   144    1440     48      73728
      4  Y + X -> X     16   16   4   144    2016     64      73728
      5  X -> Y         16   16   4   144    2592     80      73728
      6  Y + X -> X     16   16   4   144    3168     96      73728
      7  X -> Y         16   16   4   144    3744    112      73728
      8  Y + X -> X     16   16   4   144    4320    128      73728
      9  X -> Y         16   16   4   144    4896    144      73728
     10  Y + X -> X     16   16   4   144    5472    160      73728
     11  X -> Y         16   16   4   144    6048    176      73728
     12  Y + X -> X     16   16   4   144    6624    192      73728
     13  X -> Y         16   16   4   144    7200    208      73728
     14  Y + X -> X     16   16   4   144    7776    224      73728
     15  X -> Y         16   16   4   144    8352    240      73728
     16  Y + X -> X     16   16   4   144    8928    256      73728
     17  X -> Y         16   16   4   144    9504    272      73728
     18  Y + X -> X     16   16   4   144   10080    288      73728
     19  X -> Y         16   16   4   144   10656    304      73728
     20  Y + X -> X     16   16   4   144   11232    320      73728
     21  X -> Y         16   16   4   144   11808    336      73728
     22  Y + X -> X     16   16   4   144   12384    352      73728
     23  X -> Y         16   16   4   144   12960    368      73728
     24  Y + X -> X     16   16   4   144   13536    384      73728
     25  X -> Y         16   16   4   144   14112    400      73728
     26  Y + X -> X     16   16   4   144   14688    416      73728
     27  X -> Y         16   16   4   144   15264    432      73728
     28  Y + X -> X     16   16   4   144   15840    448      73728
     29  X -> Y         16   16   4   144   16416    464      73728
     30  Y + X -> X     16   16   4   144   16992    480      73728
     31  X -> Y         16   16   4   144   17568    496      73728
     32  Y + X -> X     16   16   4   144   18144    512      73728
     33  X -> Y         16   16   4   144   18720    528      73728
     34  Y + X -> X     16   16   4   144   19296    544      73728
     35  X -> Y         16   16   4   144   19872    560      73728
     36  Y + X -> X     16   16   4   144   20448    576      73728
     37  X -> Y         16   16   4   144   21024    592      73728
     38  Y + X -> X     16   16   4   144   21600    608      73728
     39  X -> Y         16   16   4   144   22176    624      73728
     40  Y + X -> X     16   16   4   144   22752    640      73728
     41  X -> Y         16   16   4   144   23328    656      73728
     42  Y + X -> X     16   16   4   144   23904    672      73728
     43  X -> Y         16   16   4   144   24480    688      73728
     44  Y + X -> X     16   16   4   144   25056    704      73728
     45  X -> Y         16   16   4   144   25632    720      73728
     46  Y + X -> X     16   16   4   144   26208    736      73728
     47  X -> Y         16   16   4   144   26784    752      73728
     48  Y + X -> X     16   16   4   144   27360    768      73728
     49  X -> Y         16   16   4   144   27936    784      73728
     50  Y + X -> X     16   16   4   144   28512    800      73728
     51  X -> Y         16   16   4   144   29088    816      73728
     52  Y + X -> X     16   16   4   144   29664    832      73728
     53  X -> Y         16   16   4   144   30240    848      73728
     54  Y + X -> X     16   16   4   144   30816    864      73728
     55  X -> Y         16   16   4   144   31392    880      73728
     56  Y + X -> X     16   16   4   144   31968    896      73728
     57  X -> Y         16   16   4   144   32544    912      73728
     58  Y + X -> X     16   16   4   144   33120    928      73728
     59  X -> Y         16   16   4   144   33696    944      73728
     60  Y + X -> X     16   16   4   144   34272    960      73728
     61  X -> Y         16   16   4   144   34848    976      73728
     62  Y + X -> X     16   16   4   144   35424    992      73728
     63  X -> logits    16    4   1   144   36000   1008      18432
    the weight ROM 36144 words of 32 bits, the norms 1012 of 38, the anneal 512 of 24
    the weight ROM banks 32768 + 4096, 720 words of pad
    the array is 48 by 4, thus 192 lanes
    the seats open inside the classes 1 to 31, 11 to 34, 17 to 39, 25 to 46
    a store is 2048 columns of 768 bits in one bank of 2048, t-major: step * 16 + channel
    the Y ring is 64 columns of 768 bits in one bank of 512, and B trails A by two
    the turns are 0, 1+2, 3+4, 5+6, 7+8, 9+10, 11+12, 13+14, 15+16, 17+18, 19+20, 21+22, 23+24, 25+26, 27+28, 29+30, 31+32, 33+34, 35+36, 37+38, 39+40, 41+42, 43+44, 45+46, 47+48, 49+50, 51+52, 53+54, 55+56, 57+58, 59+60, 61+62, 63
    forward 4628016 cycles, the cell walk 1536, a pass 4629552 less the draw
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
  let model = Quantized.Model.For_test.init tiny_shape ~seed:7 in
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
    504 words, the counter ended at 504; 0 layer bases out of step
    2016 weights, each seen one time
    0 padded lanes, 0 of them not zero; 0 bytes disagree with the twin
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
     REALLY BANKS: the tiny shapes hold one bank and would pass any decode.

     THE STORES BANK BY THE SAME PLAN AND THE DECODE RUNS OVER THEM TOO. A store carries
     no image — the circuit writes it — thus what there is to hold is the tiling, and its
     addresses are its columns and no pad above them. *)
  let module Map = Rtl.Make (Bits) in
  let decoded_apart banks ~address_bits ~addresses =
    let apart = ref 0 in
    for address = 0 to addresses - 1 do
      let circuit =
        Map.bank_at banks ~address:(Bits.of_unsigned_int ~width:address_bits address)
      in
      if Bits.to_unsigned_int circuit <> bank_at banks address then Int.incr apart
    done;
    !apart
  in
  let case ~name t =
    let words = Array.length t.weight_rom in
    let images = Array.map t.weight_banks ~f:(weight_bank_image t) in
    let depth = Array.sum (module Int) images ~f:Array.length in
    let apart = ref 0 in
    let pad_not_zero = ref 0 in
    for address = 0 to depth - 1 do
      (* the circuit's own offset: the low bits of the flat address, and nothing else *)
      let image = images.(bank_at t.weight_banks address) in
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
      (bank_depths t.weight_banks)
      (depth - words);
    printf
      "  %d addresses walked: %d apart from the image, %d pad words not zero, %d banks \
       decoded apart\n"
      depth
      !apart
      !pad_not_zero
      (decoded_apart
         t.weight_banks
         ~address_bits:(Bits.address_bits_for words)
         ~addresses:depth);
    printf
      "  a store of %d columns banks %s: %d banks decoded apart\n"
      (store_depth t)
      (bank_depths t.store_banks)
      (decoded_apart
         t.store_banks
         ~address_bits:(store_bits t)
         ~addresses:(store_depth t))
  in
  let rung = { Diffusion.Config.layers = 64; width = 16 } in
  case
    ~name:"the elected rung"
    (create (Quantized.Model.For_test.init rung ~seed:1) ~steps:128 ~lanes:4 ~walk:512);
  case
    ~name:"a shape of one bank"
    (create (Quantized.Model.For_test.init tiny_shape ~seed:7) ~steps:8 ~lanes:4 ~walk:4);
  (* A STORE THAT REALLY BANKS, and the elected rung's does not: 129 steps of 8 channels
     are 1032 columns, thus the plan splits where the two rungs above hold one bank. *)
  case
    ~name:"a store of two banks"
    (create
       (Quantized.Model.For_test.init { Diffusion.Config.layers = 4; width = 8 } ~seed:3)
       ~steps:129
       ~lanes:2
       ~walk:4);
  [%expect
    {|
    the elected rung: 36144 words banked 32768 + 4096, 720 of pad
      36864 addresses walked: 0 apart from the image, 0 pad words not zero, 0 banks decoded apart
      a store of 2048 columns banks 2048: 0 banks decoded apart
    a shape of one bank: 504 words banked 512, 8 of pad
      512 addresses walked: 0 apart from the image, 0 pad words not zero, 0 banks decoded apart
      a store of 64 columns banks 512: 0 banks decoded apart
    a store of two banks: 1008 words banked 1024, 16 of pad
      1024 addresses walked: 0 apart from the image, 0 pad words not zero, 0 banks decoded apart
      a store of 1032 columns banks 1024 + 512: 0 banks decoded apart
    |}]
;;

let%expect_test "the circuit walks the turn the block order states" =
  (* THE WALK CANNOT DRIFT FROM THE ORDER THE COST MODEL COUNTS. [blocks_of_turn] is what
     [turn_cycles] sums and what the stream gate reads the writes in; [Rtl.next_block] is
     what the engine's lead frame really does. They are one rule stated twice, thus a gate
     welds them — the era's answer to two statements of one map, as [Rtl.bank_at] is.

     THE FUSED PAIR IS THE SHAPE THAT REALLY INTERLEAVES: a turn of one phase would pass
     any phase logic. The lag of two is visible in the print — A0, A1, A2 B0, ... — and a
     schedule that drifted by a column would move that line. *)
  let module Map = Rtl.Make (Bits) in
  let case ~name t =
    let step_bits = Bits.address_bits_for (t.steps + 2) in
    let cin_bits =
      Bits.address_bits_for
        (1 + Array.fold t.layers ~init:1 ~f:(fun w l -> max w l.inputs))
    in
    let group_bits =
      Bits.address_bits_for
        (1 + Array.fold t.layers ~init:1 ~f:(fun w l -> max w l.groups))
    in
    let apart = ref 0 in
    let walked = ref 0 in
    let first = ref "" in
    Array.iteri t.turns ~f:(fun at turn ->
      let want = blocks_of_turn t turn in
      let got = ref [] in
      let nest =
        ref
          { cin = Bits.zero cin_bits
          ; group = Bits.zero group_bits
          ; step = Bits.zero step_bits
          ; phase = Bits.gnd
          }
      in
      let ended = ref false in
      let guard =
        ref
          (10
           * List.length want
           * (1 + Array.fold t.layers ~init:1 ~f:(fun w l -> max w l.inputs)))
      in
      while (not !ended) && !guard > 0 do
        Int.decr guard;
        let n = !nest in
        let phase = Bits.to_unsigned_int n.phase in
        let layer = layer_of_phase turn phase in
        let l = t.layers.(layer) in
        if Bits.to_unsigned_int n.cin = 0
        then
          got
          := { layer
             ; column = column_of_phase ~s:(Bits.to_unsigned_int n.step) phase
             ; group = Bits.to_unsigned_int n.group
             }
             :: !got;
        let walk =
          Map.next_block
            t
            ~is_pair:(if is_pair turn then Bits.vdd else Bits.gnd)
            ~cin_count:(Bits.of_unsigned_int ~width:cin_bits l.inputs)
            ~group_count:(Bits.of_unsigned_int ~width:group_bits l.groups)
            n
        in
        Int.incr walked;
        if Bits.to_bool walk.ends then ended := true else nest := walk.next
      done;
      let got = List.rev !got in
      (* the picture is one name for each BLOCK and not for each group: the lag of two is
         what the line is for, and the groups inside a block would hide it *)
      if at = 1
      then
        first
        := String.concat
             ~sep:" "
             (List.filter_map
                (List.take got (8 * t.layers.(turn.first).groups))
                ~f:(fun b ->
                  if b.group = 0
                  then
                    Some
                      (sprintf
                         "%s%d"
                         (if b.layer = turn.first then "A" else "B")
                         b.column)
                  else None));
      if not
           (List.equal
              (fun a b -> a.layer = b.layer && a.column = b.column && a.group = b.group)
              want
              got)
      then Int.incr apart);
    printf
      "%s: %d turns, %d blocks, %d cycles walked, %d turns apart\n"
      name
      (Array.length t.turns)
      (Array.sum (module Int) t.turns ~f:(fun turn -> List.length (blocks_of_turn t turn)))
      !walked
      !apart;
    if not (String.is_empty !first) then printf "  the first pair opens %s\n" !first
  in
  case
    ~name:"a shape of two pairs"
    (create
       (Quantized.Model.For_test.init { Diffusion.Config.layers = 6; width = 8 } ~seed:1)
       ~steps:5
       ~lanes:2
       ~walk:4);
  case
    ~name:"the elected rung"
    (create
       (Quantized.Model.For_test.init
          { Diffusion.Config.layers = 64; width = 16 }
          ~seed:1)
       ~steps:128
       ~lanes:4
       ~walk:512);
  [%expect
    {|
    a shape of two pairs: 4 turns, 110 blocks, 880 cycles walked, 0 turns apart
      the first pair opens A0 A1 A2 B0 A3 B1 A4 B2
    the elected rung: 33 turns, 32384 blocks, 514048 cycles walked, 0 turns apart
      the first pair opens A0 A1 A2 B0 A3 B1 A4 B2
    |}]
;;

let%expect_test "a norm word carries the twin's gain, shift and bias" =
  let model = Quantized.Model.For_test.init tiny_shape ~seed:7 in
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
    0 padded words, 0 of them not zero
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
    let model = Quantized.Model.For_test.init tiny_shape ~seed:7 in
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
    T 8, G 4: 64 store addresses and 8 channels, 0 apart
    T 5, G 3: 40 store addresses and 9 channels, 0 apart
    T 12, G 1: 96 store addresses and 8 channels, 0 apart
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
  (* THE FUSED FLOOR, AND THE SHAPE IT CLOSES. At P 48 and G 4 the H 6 layers dwell 54
     cycles: the drain and the band loads need 54, thus the unfused machine accepted this
     shape exactly. Fused, the next block's three columns are fetched under the last input
     channel — nine cycles — and in a B block the X port is still carrying the residual
     load, thus the fetch would read the wrong memory and no gate below it would say so.
     One channel wider clears it: H 7 dwells 63 against 63. *)
  refuse "a fetch the load outruns" (fun () -> create model ~steps:8 ~lanes:4 ~walk:4);
  refuse "the channel that clears it" (fun () ->
    create
      (Quantized.Model.For_test.init { Diffusion.Config.layers = 4; width = 7 } ~seed:7)
      ~steps:8
      ~lanes:4
      ~walk:4);
  refuse "a walk of no passes" (fun () -> create model ~steps:8 ~lanes:4 ~walk:0);
  refuse "a canvas of no steps" (fun () -> create model ~steps:0 ~lanes:4 ~walk:4);
  refuse "a group of no lanes" (fun () -> create model ~steps:8 ~lanes:0 ~walk:4);
  [%expect
    {|
    a chain that cannot empty: layer 0 dwells 72 cycles; its drain, band loads and the next block's fetch need 75: the dwell is short
    a band load the dwell outruns: layer 1 dwells 54 cycles; its drain, band loads and the next block's fetch need 64: the dwell is short
    a fetch the load outruns: layer 1 dwells 54 cycles; its drain, band loads and the next block's fetch need 63: the dwell is short
    the channel that clears it: NOT REFUSED
    a walk of no passes: a walk of 0 passes
    a canvas of no steps: a canvas of 0 steps
    a group of no lanes: a group of 0 lanes
    |}]
;;
