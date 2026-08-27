(* The column engine — see forward.mli for the contract and docs/diffusion_rtl.md, "The
   dwell", "The drain" and "The memories and their ports", for the design. What stands
   here is the WHY of each rule.

   TWO FRAMES, TWO CYCLES APART, AND EVERY RULE OF THIS UNIT SITS IN ONE OF THEM. The LEAD
   frame addresses the memories; the NOW frame — the lead delayed by two — is where the
   term really happens, because a read is an address register, the memory and a data
   register. One counter therefore serves both: [now] is [lead] through two registers, and
   a value the memory answers at [lead] stands on the wire exactly when [now] reaches it.

   The freeze is the same discipline. [run] holds every register of both frames and every
   memory register between them, thus a wait costs cycles and moves nothing. The array,
   the epilogue, the drain counter and the bands DO NOT freeze: they are a pipeline the
   terms already left, and the array's chain runs on a counter of its own. *)

open Core
open Hardcaml
open Signal

(* the activation format of the twin: what a column carries in each row *)
let activation_bits = Quantized.activation_bits

module State = struct
  type t =
    | Idle (* the rest, and the one state that reads [start] *)
    | Prime (* the layer's preamble: the window fills, and no term retires behind it *)
    | Dwell (* the work: one term a cycle, over the taps and channels of a group *)
    | Offer
      (* the head alone: the logit file stands whole, offered as a level until the draw
         takes it *)
    | Turn (* the chain empties, then the next layer primes — or Idle, behind the head *)
  [@@deriving compare ~localize, enumerate, sexp_of]
end

module Make (M : sig
    val e : Elaboration.t
  end) =
struct
  let e : Elaboration.t = M.e
  let steps = e.steps
  let rows = e.rows
  let lanes = e.lanes
  let layers = e.layers
  let taps = Elaboration.taps
  let voices = Frame.voices
  let planes = 2 * voices
  let layer_count = Array.length layers
  let widest f = Array.fold layers ~init:1 ~f:(fun widest l -> max widest (f l))
  let max_inputs = widest (fun l -> l.Elaboration.inputs)
  let max_groups = widest (fun l -> l.Elaboration.groups)
  let step_bits = address_bits_for steps

  (* A COUNTER HOLDS ITS COUNT AND NOT ONLY ITS LAST INDEX. The table states [inputs] and
     [groups] as values, and a width that only addressed them would refuse the count
     itself. *)
  let cin_bits = address_bits_for (max_inputs + 1)
  let group_bits = address_bits_for (max_groups + 1)
  let tap_bits = address_bits_for taps
  let lane_bits = address_bits_for lanes
  let layer_bits = address_bits_for layer_count
  let turn_count = Array.length e.turns
  let turn_bits = address_bits_for turn_count

  (* THE PAIR'S STEP COUNTER RUNS PAST THE CANVAS. B trails A by two, thus [s] walks 0 to
     T + 1 and its width is not the column's. The semantic column is [s] in phase A and
     [s - 2] in phase B, and that one is [step_bits] wide as it always was. *)
  let turn_step_bits = address_bits_for (steps + 2)
  let seat_bits = address_bits_for voices
  let plane_bits = address_bits_for planes
  let column_bits = rows * activation_bits
  let weight_bits = address_bits_for (Array.length e.weight_rom)
  let norm_bits = address_bits_for (Array.length e.norm_rom)

  (* the address widths are the elaboration's, read and not derived again *)
  let store_bits = Elaboration.store_bits e
  let channel_bits = Elaboration.channel_bits e

  module Lanes = Column_array.Make (struct
      let rows = rows
      let lanes = lanes
    end)

  module Tail = Epilogue.Make (struct
      let rows = rows
      let lanes = lanes
    end)

  module I = struct
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; start : 'a
      ; plane_column : 'a [@bits column_bits]
      ; logit_seat : 'a [@bits seat_bits]
      ; step_taken : 'a
      }
    [@@deriving hardcaml]
  end

  module O = struct
    type 'a t =
      { busy : 'a
      ; plane_step : 'a [@bits step_bits]
      ; plane : 'a [@bits plane_bits]
      ; step_ready : 'a
      ; logits : 'a [@bits column_bits]
      }
    [@@deriving hardcaml]
  end

  (* ---------------------------------------------------------------- *)
  (* the maps and the slicing, which follow the shape alone *)
  (* ---------------------------------------------------------------- *)

  (* THE TWO MAPS ARE THE ELABORATION'S, ELABORATED — not a second statement of them. The
     array owns the DSPs, thus every address product is pinned, and the rule is fixed here
     one time rather than at each of the five addresses below. *)
  let column_address = Elaboration.Rtl.column_address ~pin:Column_array.no_dsp e
  let ring_address = Elaboration.Rtl.ring_address ~pin:Column_array.no_dsp e
  let channel_of = Elaboration.Rtl.channel_of ~pin:Column_array.no_dsp e

  (* A COLUMN BANK IS SLICED, AND ITS TAKE STANDS IN REPLICAS — ring 3's broadcast
     families: one decode or one strobe drove the 768 register pins of a whole column, at
     up to 12 ns of route. A bank is [column_slices] register slices, and each slice's
     take is its own copy of the same condition, [dont_touch] so the tools keep the copies
     apart — no driver reaches more than [Column_array.slice_rows] rows, and the placer
     lays each beside its slice. The values and the writes stay whole columns; the slicing
     is invisible outside these helpers. *)
  let column_slices = Column_array.slices_for ~rows

  (* the replica slices of a word of [bits]: [Column_array.slice_rows] rows of activations
     each, and the last one short. A column bank takes these of a column; the bank mux of
     a store takes the same slices of the same word. *)
  let word_slices bits =
    let span = Column_array.slice_rows * activation_bits in
    List.init
      ((bits + span - 1) / span)
      ~f:(fun s ->
        let low = s * span in
        min bits (low + span) - 1, low)
  ;;

  let column_slice_ranges = word_slices column_bits
  let slice_range s = List.nth_exn column_slice_ranges s
  let bank_value bank = concat_lsb (List.map bank ~f:(fun v -> v.Always.Variable.value))

  (* [replicas ~count make] is [count] statements of the same signal, all but the first
     [dont_touch] so the tools keep the copies apart *)
  let replicas ~count make =
    List.init count ~f:(fun s ->
      if s = 0
      then make ()
      else add_attribute (make ()) (Rtl_attribute.Vivado.dont_touch true))
  ;;

  let replicated_takes make = replicas ~count:column_slices make

  let create (i : _ I.t) : _ O.t =
    let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
    (* the datapath and the memories hold no clear: what is real is what the strobes mark *)
    let dspec = Reg_spec.create ~clock:i.clock () in
    let open Always in
    let sm = State_machine.create (module State) spec in
    (* the state register carries its own name: the cycle bench counts what a pass spends
       outside its dwells by reading it, and a level named beside it would be a signal
       nothing drives and the tools would prune it away *)
    let _ = sm.current -- "state" in
    (* THE FREEZE. Both frames and every memory register between them advance together, or
       not at all. Nothing downstream of a term freezes: the array's chain and the
       epilogue are a pipeline the terms have already left. *)
    let run = sm.is Prime |: sm.is Dwell in
    let hold x = reg dspec ~enable:run x in
    (* ---------------------------------------------------------------- *)
    (* the layer, and the facts the table states about it *)
    (* ---------------------------------------------------------------- *)
    (* THE TURN IS THE REGISTER; THE LAYER TRAVELS IN THE FRAMES. A layer used to end
       before the next began, thus one register named it and every fact muxed by it.
       Inside a pair the lead frame can be in B while the now frame is still in A and the
       flush trails both, thus what the machine holds is the TURN and each frame carries
       its own phase. [Elaboration.Rtl.layer_of] turns the two into the table's index, and
       the table's mux is the one it always was. *)
    let turn = Variable.reg spec ~width:turn_bits in
    let layer_of ~phase = Elaboration.Rtl.layer_of e ~turn:turn.value ~phase in
    let fact at ~width f =
      mux at (List.map (Array.to_list layers) ~f:(fun l -> of_unsigned_int ~width (f l)))
    in
    let flag at f =
      mux at (List.map (Array.to_list layers) ~f:(fun l -> if f l then vdd else gnd))
    in
    (* A LAYER'S ROLE STATES ITS TWO ENDS, ITS RELU AND ITS RESIDUAL TOGETHER, thus every
       flag below reads the one field and no two of them can disagree. *)
    let by_role at f = flag at (fun l -> f l.Elaboration.role) in
    let is_head_of at =
      by_role at (function
        | Head -> true
        | _ -> false)
    in
    let is_join_of at =
      by_role at (function
        | Pair_close -> true
        | _ -> false)
    in
    (* THE STEM IS ITS OWN TURN, thus its plane select is a fact of the turn alone and
       needs no phase: a preamble stands between it and everything after it. *)
    let is_stem =
      mux
        turn.value
        (List.map (Array.to_list e.turns) ~f:(fun tn ->
           match layers.(tn.Elaboration.first).role with
           | Stem -> vdd
           | _ -> gnd))
    in
    let is_pair =
      mux
        turn.value
        (List.map (Array.to_list e.turns) ~f:(fun tn ->
           if Elaboration.is_pair tn then vdd else gnd))
    in
    (* ---------------------------------------------------------------- *)
    (* THE LEAD FRAME: the block being addressed *)
    (* ---------------------------------------------------------------- *)
    let lead_s = Variable.reg spec ~width:turn_step_bits in
    let lead_phase = Variable.reg spec ~width:1 in
    let lead_group = Variable.reg spec ~width:group_bits in
    let lead_cin = Variable.reg spec ~width:cin_bits in
    let lead_cycle = Variable.reg spec ~width:tap_bits in
    (* the preamble fetches the FIRST block instead of the next one, and holds the nest
       still while it does: one short preamble at each TURN — not at each layer — and the
       rotation then hides every fetch of the turn under a running dwell, at a phase
       change as at a column change *)
    let priming = Variable.reg spec ~width:1 in
    let lead_layer = layer_of ~phase:lead_phase.value in
    let cin_count = fact lead_layer ~width:cin_bits (fun l -> l.inputs) in
    let group_count = fact lead_layer ~width:group_bits (fun l -> l.groups) in
    let last_cycle = lead_cycle.value ==:. taps - 1 in
    let last_cin = lead_cin.value ==: cin_count -:. 1 in
    let last_group = lead_group.value ==: group_count -:. 1 in
    (* THE BLOCK AFTER THE LEAD'S, BY THE ELABORATION'S OWN RULE. [next_block] is the
       circuit half of [blocks_of_turn]: the fetches take it, and the nest takes it at the
       close of the block. A phase change is a block change like any other. *)
    let block_closes = last_cin &: last_group in
    let walk =
      Elaboration.Rtl.next_block
        e
        ~is_pair
        ~cin_count
        ~group_count
        { Elaboration.cin = lead_cin.value
        ; group = lead_group.value
        ; step = lead_s.value
        ; phase = lead_phase.value
        }
    in
    let next = walk.next in
    (* THE FETCH FRAME IS THE LEAD ADVANCED BY ONE CHANNEL, and at the close of a block
       that is the NEXT BLOCK'S — its channel, its column and its PHASE. The phase is what
       is new: the columns fetched under B's last channel are A's, and they come from a
       different memory and a different column. *)
    (* THE FETCH FRAME STANDS BEHIND A REGISTER, AND IT COSTS NO CYCLE. The fetch of slot
       [s] goes out at lead cycle [3 s + 2] — 2, 5 and 8 — while the lead nest advances at
       cycle 0, thus these coordinates have two cycles of slack that a combinational
       [next_block] threw away. Registered, they stand from cycle 1 and every fetch reads
       them at 2 or later; inside a channel they do not move, thus the delay changes no
       value anywhere. The cone behind them — [next_block], the [s - 2] of [column_of],
       [column_address] and a bank's [uresize] — was the fused round's own timing family,
       eleven levels into a store's address pins at −0.118. *)
    let fetch_cin = hold (mux2 priming.value lead_cin.value next.cin) in
    let fetch_s = hold (mux2 priming.value lead_s.value next.step) in
    let fetch_phase = hold (mux2 priming.value lead_phase.value next.phase) in
    let fetch_layer = layer_of ~phase:fetch_phase in
    let fetch_reads_y = is_join_of fetch_layer in
    (* the semantic column of a block: [s] in phase A, [s - 2] in phase B, and the stem
       and the head are turns of phase A alone *)
    let column_of ~s ~phase = sel_bottom (mux2 phase (s -:. 2) s) ~width:step_bits in
    let fetch_step = column_of ~s:fetch_s ~phase:fetch_phase in
    (* THE TAP ORDER IS DY-MAJOR, AND THE ROTATION LEANS ON IT: tap [k] takes the time
       slot [k / 3] and the pitch shift [k mod 3]. The fetch of slot [s] goes out at the
       lead cycle [3 s + 2] and lands when [now] reaches that same cycle, which is the
       slot's own last read. *)
    let by_cycle cycle table =
      mux cycle (List.map table ~f:(fun v -> of_unsigned_int ~width:2 v))
    in
    let slot_of cycle = by_cycle cycle [ 0; 0; 0; 1; 1; 1; 2; 2; 2 ] in
    let shift_of cycle = by_cycle cycle [ 0; 1; 2; 0; 1; 2; 0; 1; 2 ] in
    (* the step the fetched slot names: the slot below the block's column, the column, or
       the slot above it. THE CALLER OWNS THE EDGE — a step outside the roll takes the
       zero column, and this unit drives no address for it. *)
    let roll_step =
      let wide = uresize fetch_step ~width:(step_bits + 2) in
      wide +: uresize (slot_of lead_cycle.value) ~width:(step_bits + 2) -:. 1
    in
    let out_of_roll = msb roll_step |: (roll_step >=:. steps) in
    let tap_step =
      mux2 out_of_roll (zero step_bits) (sel_bottom roll_step ~width:step_bits)
    in
    let tap_address = column_address ~step:tap_step ~channel:fetch_cin in
    (* the ring takes the SEMANTIC column and keeps the bits it holds; nothing else states
       the ring's geometry *)
    let tap_ring_address = ring_address ~step:tap_step ~channel:fetch_cin in
    (* the fetch travels to the NOW frame beside its data: the zero flag says the slot
       takes zero, and the step and the plane are what the canvas answers *)
    (* THE SOURCE OF A FETCHED COLUMN TRAVELS WITH THE FETCH. Which memory a column comes
       from is a fact of the FETCH's layer and not of the slot's: the columns that land
       under B's last input channel are A's and come from X, and a slot that read the
       layer register would take Y. The bit rides beside the zero flag, and the slot takes
       what its own fetch named. *)
    let fetch_word =
      concat_lsb
        [ out_of_roll; tap_step; sel_bottom fetch_cin ~width:plane_bits; fetch_reads_y ]
    in
    let landed = hold (hold fetch_word) in
    let landed_zero = bit landed ~pos:0 in
    let landed_step = select landed ~high:step_bits ~low:1 in
    let landed_plane =
      select landed ~high:(step_bits + plane_bits) ~low:(step_bits + 1)
    in
    let landed_reads_y = bit landed ~pos:(1 + step_bits + plane_bits) in
    (* ---------------------------------------------------------------- *)
    (* THE NOW FRAME: the term that really happens *)
    (* ---------------------------------------------------------------- *)
    (* THE NOW FRAME CARRIES THE PHASE AND THE SEMANTIC COLUMN. The phase names the layer
       every rule of this frame reads; the column is what the band load and the state
       machine mean by a step, and [s] never leaves the lead frame. [ends] rides too: the
       turn closes on the now frame's own last block and not on a column count. *)
    let lead_column = column_of ~s:lead_s.value ~phase:lead_phase.value in
    let lead_word =
      concat_lsb
        [ sm.is Dwell
        ; lead_cycle.value
        ; lead_cin.value
        ; lead_group.value
        ; lead_column
        ; lead_phase.value
        ; walk.ends
        ]
    in
    let now_word = hold (hold lead_word) in
    let at low width = select now_word ~high:(low + width - 1) ~low in
    let now_valid = bit now_word ~pos:0 &: run in
    let now_cycle = at 1 tap_bits in
    let now_cin = at (1 + tap_bits) cin_bits in
    let now_group = at (1 + tap_bits + cin_bits) group_bits in
    let now_step = at (1 + tap_bits + cin_bits + group_bits) step_bits in
    let now_phase = at (1 + tap_bits + cin_bits + group_bits + step_bits) 1 in
    let now_ends = at (2 + tap_bits + cin_bits + group_bits + step_bits) 1 in
    let now_layer = layer_of ~phase:now_phase in
    let now_last_cin =
      now_cin ==: fact now_layer ~width:cin_bits (fun l -> l.inputs) -:. 1
    in
    let now_last_cycle = now_cycle ==:. taps - 1 in
    let now_last_group =
      now_group ==: fact now_layer ~width:group_bits (fun l -> l.groups) -:. 1
    in
    (* THE DRAIN IS A FRAME OF ITS OWN, AND IT CARRIES THE PHASE TOO. The chain of a group
       empties over [rows] cycles behind the term that captured it, and the epilogue
       answers three behind that — thus THE LAST GROUP OF A BLOCK DRAINS UNDER THE NEXT
       BLOCK, which inside a pair is the other layer. A join flag read from the now frame
       would then add A's residual to nothing and drop B's, and the ReLU would follow the
       wrong layer; the twin sees a column of zeros where it wants the residual.

       It was invisible while a layer was the unit of the walk: a layer's last group
       drained under [Turn], where the layer register still stood at its own value. The
       phase is captured where the array captures the sums, and it holds until the
       epilogue of that group is done — a whole dwell away, which the dwell floor
       guarantees. *)
    let drain_phase = Variable.reg spec ~width:1 in
    let drain_layer = layer_of ~phase:drain_phase.value in
    let takes_relu =
      by_role drain_layer (function
        | Stem | Pair_open -> true
        | _ -> false)
    in
    let is_join = is_join_of drain_layer in
    (* ---------------------------------------------------------------- *)
    (* the memories: the address registers before them and the data registers behind them,
       which is era four's rule and the reason every read of this unit is two cycles *)
    (* ---------------------------------------------------------------- *)
    (* ONE BANKED MEMORY PORT, AND BOTH MEMORY CLASSES READ THROUGH IT. The weight ROM and
       the two activation stores differ in one thing — the ROM's banks carry an image and
       the stores' banks are written — thus one port states the rule for both and neither
       can drift from it.

       READ ERA FOUR'S WAY: [hold] stands on the address and again on the data, thus the
       tools cannot retime the data register onto the address pins and rebuild the address
       cone inside each primitive. THE DATA HOLD STANDS INSIDE THE BANK, because it is the
       block RAM's own latch: evict it into fabric and Vivado absorbs the ADDRESS hold
       into the latch instead, and the whole address cone lands on the pins. The first
       banked build measured exactly that — 1424 registers into fabric, an eight-level
       cone on a store's address pins, and setup lost by 0.354.

       AND EACH MEMORY STANDS IN THE BANKS ITS PLAN STATES. Vivado rounds the depth of a
       RAM up to a power of two as it rounds a ROM's, and warns of nothing: the rung-3
       measurement build of 2026-08-27 mapped a store of 1280 columns as [2048x768], the
       same 43 tiles rung 2 pays for 2048. One memory for each bank; the one address feeds
       all of them as it stands, because a base is a multiple of its own bank's depth and
       the offset is therefore the low bits; and a write reaches the bank its own address
       selects.

       THE MUX STANDS BEHIND THE DATA HOLDS, thus the select rides two [hold]s — one in
       step with the address hold and one with the data hold. No cycle is added: the read
       is the two it always was, and nothing stands between a bank and its own latch. The
       select rides one replica for each slice of the word: a weight word of [lanes] bytes
       is one slice, a column of 768 bits is six, as every array-scale take.

       WITH ONE BANK NONE OF THIS ELABORATES — no mux, no select, and the write enable
       passes as it stands — thus a shape that does not bank builds the memory it always
       did. *)
    let block_memory ?image ~banks ~address ~write_enable ~write_address ~write_data () =
      let bank_write_enable =
        if Array.length banks = 1
        then fun (_ : int) -> write_enable
        else (
          let select = Elaboration.Rtl.bank_at banks ~address:write_address in
          fun at -> write_enable &: (select ==:. at))
      in
      let read_bank at (bank : Elaboration.bank) =
        let bits = address_bits_for bank.depth in
        (multiport_memory
           ~attributes:[ Rtl_attribute.Vivado.Ram_style.block ]
           ?initialize_to:(Option.map image ~f:(fun words -> words bank))
           bank.depth
           ~write_ports:
             [| { Write_port.write_clock = i.clock
                ; write_address = uresize write_address ~width:bits
                ; write_enable = bank_write_enable at
                ; write_data
                }
             |]
           ~read_addresses:[| hold (uresize address ~width:bits) |]).(0)
        |> hold
      in
      match List.mapi (Array.to_list banks) ~f:read_bank with
      | [ read ] -> read
      | reads ->
        let slices = word_slices (width (List.hd_exn reads)) in
        replicas ~count:(List.length slices) (fun () ->
          hold (hold (Elaboration.Rtl.bank_at banks ~address)))
        |> List.map2_exn slices ~f:(fun (high, low) which ->
          mux which (List.map reads ~f:(fun read -> select read ~high ~low)))
        |> concat_lsb
    in
    (* a ROM of one bank is the same port with an image behind it and its write side wired
       off: the norms want it, and nothing about them asks for a plan *)
    let rom image address =
      let size = Array.length image in
      block_memory
        ~image:(fun (_ : Elaboration.bank) -> image)
        ~banks:[| { Elaboration.base = 0; depth = size } |]
        ~address
        ~write_enable:gnd
        ~write_address:(zero (address_bits_for size))
        ~write_data:(zero (Bits.width image.(0)))
        ()
    in
    (* THE WEIGHT ADDRESS ONLY COUNTS. The image is packed in the dwell order, thus one
       column's dwell walks a layer's whole range straight through and the address reloads
       one time for each column. *)
    let weight_address = Variable.reg spec ~width:weight_bits in
    (* THE WEIGHT ROM STANDS IN THE BANKS ITS PLAN STATES, and the port above is the whole
       of what the circuit owes that plan. Vivado pads an inferred ROM to its full address
       space and warns of nothing: rung 2 asked 64 tiles against 49 free and the mapper
       demoted every ROM of the design to fabric. The one counter feeds every bank as it
       stands, and the mux stands behind the data holds, thus nothing stands between the
       counter and the memories nor between a bank and its own latch. The mux is also
       BEFORE the operand replicas of the array, thus the replica bank still breaks the
       broadcast and the mux's own fanout is the replica count and no more. *)
    let weights =
      block_memory
        ~image:(Elaboration.weight_bank_image e)
        ~banks:e.weight_banks
        ~address:weight_address.value
        ~write_enable:gnd
        ~write_address:(zero weight_bits)
        ~write_data:(zero (Bits.width e.weight_rom.(0)))
        ()
    in
    (* ---------------------------------------------------------------- *)
    (* the bands: the window, the residual columns, the output columns and the norm bank *)
    (* ---------------------------------------------------------------- *)
    (* the registers of a sliced bank, and the write that fills one: the slicing rule is
       [slice_range]'s, above, and it is invisible past these two *)
    let column_bank () =
      List.init column_slices ~f:(fun s ->
        let high, low = slice_range s in
        Variable.reg dspec ~width:(high - low + 1))
    in
    let write_bank bank ~takes ~column =
      proc
        (List.mapi bank ~f:(fun s v ->
           let high, low = slice_range s in
           when_ (List.nth_exn takes s) [ v <-- select column ~high ~low ]))
    in
    let slots = List.init 3 ~f:(fun (_ : int) -> column_bank ()) in
    let residual_band =
      List.init lanes ~f:(fun (_ : int) -> Variable.reg dspec ~width:column_bits)
    in
    let output_band = List.init lanes ~f:(fun (_ : int) -> column_bank ()) in
    let norm_bank =
      List.init lanes ~f:(fun (_ : int) ->
        Variable.reg dspec ~width:Elaboration.norm_bits)
    in
    let logit_file = List.init voices ~f:(fun (_ : int) -> column_bank ()) in
    (* ---------------------------------------------------------------- *)
    (* the array and the epilogue *)
    (* ---------------------------------------------------------------- *)
    (* the names the waveform test and the stream gate read; the parentheses are
       load-bearing, as they are at the store probes below *)
    let _ = lead_cycle.value -- "lead_cycle" in
    let _ = lead_phase.value -- "lead_phase" in
    let _ = lead_s.value -- "lead_s" in
    let _ = lead_column -- "lead_column" in
    let _ = now_cycle -- "now_cycle" in
    let term = now_valid -- "term" in
    let term_first =
      (now_valid &: (now_cin ==:. 0) &: (now_cycle ==:. 0)) -- "term_first"
    in
    let term_last = (now_valid &: now_last_cin &: now_last_cycle) -- "term_last" in
    let capture_drain_phase = when_ term_last [ drain_phase <-- now_phase ] in
    let column_now = mux (slot_of now_cycle) (List.map slots ~f:bank_value) in
    let drained =
      Lanes.create
        { Lanes.I.clock = i.clock
        ; clear = i.clear
        ; term
        ; term_first
        ; term_last
        ; column = column_now
        ; row_shift = shift_of now_cycle
        ; weights
        }
    in
    (* the residual row the join adds: the head of each band column, which the drain
       shifts out one row a cycle in the row order the chain gives *)
    let residual_row =
      concat_lsb
        (List.map residual_band ~f:(fun column ->
           sel_bottom column.value ~width:activation_bits))
    in
    let tail =
      Tail.create
        { Tail.I.clock = i.clock
        ; clear = i.clear
        ; drained = drained.drained
        ; row = drained.row
        ; sums = drained.sums
        ; residual = residual_row
        ; norms = concat_lsb (List.map norm_bank ~f:(fun w -> w.value))
        ; relu = takes_relu
        ; join = is_join
        }
    in
    let _ = drained.drained -- "drained" in
    let _ = drained.row -- "drain_row" in
    let _ = tail.valid -- "band_row" in
    let last_drained = drained.drained &: (drained.row ==:. rows - 1) in
    let band_whole = tail.valid &: (tail.activation_row ==:. rows - 1) in
    (* the output band's takes: the same delay of the same strobe [tail.valid] states —
       ring 3's fourth family, one flop at 3 073 pins — one registered copy for each slice
       of each band column. The flush and the seam keep the epilogue's own. *)
    let band_takes =
      let pre = pipeline spec ~n:(Epilogue.latency - 1) drained.drained in
      List.init lanes ~f:(fun (_ : int) ->
        List.init column_slices ~f:(fun (_ : int) ->
          add_attribute (reg spec pre) (Rtl_attribute.Vivado.dont_touch true)))
    in
    (* ---------------------------------------------------------------- *)
    (* THE BAND LOADS. The residual columns and the norm words of the group whose terms
       are running now are fetched the moment the drain BEFORE it has read its last row,
       thus one buffer serves both and a dwell that covers its drain covers this too. *)
    (* ---------------------------------------------------------------- *)
    (* THE LOAD CARRIES ITS OWN PHASE, as it carries its own column and group. It fires
       inside a dwell and holds for [lanes] cycles; the now frame is stable across those,
       but the load's decisions — which norms, and whether X owes it a residual — belong
       to the block it was fired for and are captured with it. *)
    let loading = Variable.reg spec ~width:1 in
    let load_index = Variable.reg spec ~width:lane_bits in
    let load_step = Variable.reg spec ~width:step_bits in
    let load_group = Variable.reg spec ~width:group_bits in
    let load_phase = Variable.reg spec ~width:1 in
    let load_layer = layer_of ~phase:load_phase.value in
    let load_join = is_join_of load_layer in
    let load_valid = hold (hold loading.value) in
    let load_landed = hold (hold load_index.value) in
    let load_channel = channel_of ~group:load_group.value ~lane:load_index.value in
    let residual_address = column_address ~step:load_step.value ~channel:load_channel in
    let norm_address =
      fact load_layer ~width:norm_bits (fun l -> l.norm_base)
      +: uresize load_channel ~width:norm_bits
    in
    let norms = rom e.norm_rom norm_address in
    (* ---------------------------------------------------------------- *)
    (* THE FLUSH. The output band stands whole one epilogue behind the drain's last row;
       its [lanes] columns then go out one a cycle, and a lane past the layer's channels
       writes nothing — the padding of the elaboration reaches the band, not the store. *)
    (* ---------------------------------------------------------------- *)
    let flushing = Variable.reg spec ~width:1 in
    let flush_index = Variable.reg spec ~width:lane_bits in
    let flush_s = Variable.reg spec ~width:turn_step_bits in
    let flush_phase = Variable.reg spec ~width:1 in
    let flush_group = Variable.reg spec ~width:group_bits in
    let flush_layer = layer_of ~phase:flush_phase.value in
    let flush_step = column_of ~s:flush_s.value ~phase:flush_phase.value in
    let out_count = fact flush_layer ~width:channel_bits (fun l -> l.outputs) in
    let is_head = is_head_of flush_layer in
    let writes_y =
      by_role flush_layer (function
        | Pair_open -> true
        | _ -> false)
    in
    let flush_channel = channel_of ~group:flush_group.value ~lane:flush_index.value in
    let flush_last = flush_index.value ==:. lanes - 1 in
    let flush_done = flushing.value &: flush_last in
    let flush_real = flushing.value &: (flush_channel <: out_count) in
    let flush_column = mux flush_index.value (List.map output_band ~f:bank_value) in
    let store_write = flush_real &: ~:is_head in
    (* THE PROBE NAMES ARE A CONTRACT: the stream gate reads these five by name. The
       parentheses are load-bearing — [--] binds tighter than [&:], thus a name written
       without them lands on the last operand and the gate reads a signal nobody meant.

       ONE FLUSH NEST STATES BOTH DESTINATIONS, thus ONE address probe does. Y no longer
       holds a tensor and its port takes a ring address, but the COLUMN a write means is
       the flush nest's own and it is the same for X and for Y; a second name on it would
       be a signal nothing drives, and Hardcaml prunes those. *)
    let flush_address =
      column_address ~step:flush_step ~channel:flush_channel -- "flush_address"
    in
    let x_write = (store_write &: ~:writes_y) -- "x_write" in
    let y_write = (store_write &: writes_y) -- "y_write" in
    let x_data = flush_column -- "x_data" in
    let y_data = flush_column -- "y_data" in
    (* THE X PORT IS ARBITRATED BY THE CYCLE AND NOT BY THE LAYER. It used to be the
       layer's: a join layer pointed X at the residual for its whole run, because its taps
       read Y and nothing else wanted the port. Fused, the fetch of the next A block goes
       out under B's LAST INPUT CHANNEL and needs X while the now frame is still in B.
       Thus the residual takes the port only in the cycles the load really addresses it,
       and the dwell floor of the elaboration is what keeps the two windows apart. *)
    let x_read =
      block_memory
        ~banks:e.store_banks
        ~address:(mux2 (loading.value &: load_join) residual_address tap_address)
        ~write_enable:x_write
        ~write_address:flush_address
        ~write_data:x_data
        ()
    in
    (* Y IS A RING OF FOUR COLUMNS AND NOT A TENSOR. Both its ports take the ring's map of
       the semantic column; no load ever touches this port, thus its read is the taps'
       alone. *)
    let y_read =
      block_memory
        ~banks:e.ring_banks
        ~address:tap_ring_address
        ~write_enable:y_write
        ~write_address:(ring_address ~step:flush_step ~channel:flush_channel)
        ~write_data:y_data
        ()
    in
    (* ---------------------------------------------------------------- *)
    (* the wiring of the bands *)
    (* ---------------------------------------------------------------- *)
    let tap_column =
      mux2
        landed_zero
        (zero column_bits)
        (mux2 is_stem i.plane_column (mux2 landed_reads_y y_read x_read))
    in
    let step_ready = Variable.reg spec ~width:1 in
    let turn_drained = Variable.reg spec ~width:1 in
    (* the window rotation: each slot takes its new column on the edge of its own last
       read, thus the operand register takes the old value on that very edge *)
    let rotate_window =
      proc
        (List.mapi slots ~f:(fun at slot ->
           let takes =
             replicated_takes (fun () -> run &: (now_cycle ==:. (3 * at) + 2))
           in
           let (_ : Signal.t) = List.hd_exn takes -- sprintf "slot_%d" at in
           write_bank slot ~takes ~column:tap_column))
    in
    (* the residual band shifts a row out at every drained row, and takes a whole column
       at a load *)
    let shift_residual_band =
      proc
        (List.mapi residual_band ~f:(fun at column ->
           proc
             [ when_ drained.drained [ column <-- srl column.value ~by:activation_bits ]
             ; when_ (load_valid &: (load_landed ==:. at)) [ column <-- x_read ]
             ]))
    in
    (* the output band shifts a row in at every epilogue row, thus after [rows] of them it
       holds the column in the store's own order *)
    let shift_output_band =
      proc
        (List.mapi output_band ~f:(fun at column ->
           let shifted =
             select
               (select
                  tail.activations
                  ~high:((at * activation_bits) + activation_bits - 1)
                  ~low:(at * activation_bits)
                @: bank_value column)
               ~high:(column_bits + activation_bits - 1)
               ~low:activation_bits
           in
           write_bank column ~takes:(List.nth_exn band_takes at) ~column:shifted))
    in
    let take_norms =
      proc
        (List.mapi norm_bank ~f:(fun at word ->
           when_ (load_valid &: (load_landed ==:. at)) [ word <-- norms ]))
    in
    (* the head writes one step and not a tensor: the band's columns are the seat files of
       the step, and a lane past the four seats writes nothing *)
    let take_logits =
      proc
        (List.mapi logit_file ~f:(fun seat column ->
           let takes =
             replicated_takes (fun () ->
               flushing.value &: is_head &: (flush_channel ==:. seat))
           in
           write_bank column ~takes ~column:flush_column))
    in
    compile
      [ rotate_window; shift_residual_band; shift_output_band; take_norms; take_logits ];
    (* ---------------------------------------------------------------- *)
    (* the walk of the counters *)
    (* ---------------------------------------------------------------- *)
    let start_load = Variable.wire ~default:gnd () in
    let lead_nest =
      (* THE LEAD NEST. It walks the column, the group, the input channel and the tap, and
         the preamble holds it still for one block. *)
      when_
        run
        [ if_
            last_cycle
            [ lead_cycle <--. 0
            ; when_
                ~:(priming.value)
                [ lead_cin <-- next.cin
                ; lead_group <-- next.group
                ; lead_s <-- next.step
                ; lead_phase <-- next.phase
                ]
            ]
            [ lead_cycle <-- lead_cycle.value +:. 1 ]
        ; (* THE WEIGHT ADDRESS ONLY COUNTS, AND RELOADS AT EACH NEW BLOCK — to the base
             of the block it is about to walk, which at a phase change is the OTHER
             layer's. The fetch frame already names that layer. *)
          if_
            priming.value
            [ weight_address
              <-- fact lead_layer ~width:weight_bits (fun l -> l.weight_base)
            ]
            [ if_
                (last_cycle &: block_closes)
                [ weight_address
                  <-- fact fetch_layer ~width:weight_bits (fun l -> l.weight_base)
                ]
                [ weight_address <-- weight_address.value +:. 1 ]
            ]
        ]
    in
    let band_load =
      (* THE BAND LOAD. The drain before this dwell has just read its last residual row,
         thus the buffer is free and the columns of this dwell's group may land in it. It
         walks under [run] like the frames it is timed against, and the resume of a wait
         states it again rather than carry a half-walked one across. *)
      if_
        (start_load.value |: (run &: last_drained))
        [ (* THE START STANDS OUTSIDE [run]. A layer's first group is asked for while the
             machine is still idle or still turning, thus a trigger that waited for [run]
             would leave the first group of every layer with the norms of the last one. *)
          loading <-- vdd
        ; load_index <--. 0
        ; if_
            start_load.value
            [ load_step <--. 0; load_group <--. 0; load_phase <--. 0 ]
            [ load_step <-- now_step; load_group <-- now_group; load_phase <-- now_phase ]
        ]
        [ when_
            (run &: loading.value)
            [ load_index <-- load_index.value +:. 1
            ; when_ (load_index.value ==:. lanes - 1) [ loading <-- gnd ]
            ]
        ]
    in
    let open_flush =
      (* THE FLUSH, and the nest that walks it. The flushes stand in the order the dwells
         do, thus a counter of its own states the column and the group of each one and no
         tag has to travel the length of the drain. *)
      when_ band_whole [ flushing <-- vdd; flush_index <--. 0 ]
    in
    (* THE FLUSH WALKS THE BLOCK ORDER TOO, AND BY THE SAME RULE. The flushes stand in the
       order the dwells do, thus this nest must turn a phase where the lead nest turned
       one. It has no input channels of its own — a flush walks lanes — thus it enters
       [next_block] with its channel already at the last, and every advance is a whole
       block. *)
    let flush_cin_count = fact flush_layer ~width:cin_bits (fun l -> l.inputs) in
    let flush_group_count = fact flush_layer ~width:group_bits (fun l -> l.groups) in
    let flush_walk =
      Elaboration.Rtl.next_block
        e
        ~is_pair
        ~cin_count:flush_cin_count
        ~group_count:flush_group_count
        { Elaboration.cin = flush_cin_count -:. 1
        ; group = flush_group.value
        ; step = flush_s.value
        ; phase = flush_phase.value
        }
    in
    let walk_flush =
      when_
        (flushing.value &: ~:band_whole)
        [ flush_index <-- flush_index.value +:. 1
        ; when_
            flush_last
            [ flushing <-- gnd
            ; flush_group <-- flush_walk.next.group
            ; flush_s <-- flush_walk.next.step
            ; flush_phase <-- flush_walk.next.phase
            ; when_ flush_walk.ends [ turn_drained <-- vdd ]
            ]
        ]
    in
    let head_seam =
      (* the head's seam: the level rises when the file stands whole and falls on the edge
         behind the acknowledgement *)
      when_
        (flush_done &: is_head &: (flush_group.value ==: flush_group_count -:. 1))
        [ step_ready <-- vdd ]
    in
    let machine =
      sm.switch
        [ ( State.Idle
          , [ when_
                i.start
                [ turn <--. 0
                ; lead_s <--. 0
                ; lead_phase <--. 0
                ; lead_group <--. 0
                ; lead_cin <--. 0
                ; lead_cycle <--. 0
                ; flush_s <--. 0
                ; flush_phase <--. 0
                ; flush_group <--. 0
                ; flushing <-- gnd
                ; turn_drained <-- gnd
                ; step_ready <-- gnd
                ; priming <-- vdd
                ; start_load <-- vdd
                ; sm.set_next Prime
                ]
            ] )
        ; Prime, [ when_ last_cycle [ priming <-- gnd; sm.set_next Dwell ] ]
        ; ( Dwell
          , [ (* THE TURN CLOSES ON ITS OWN LAST BLOCK, which the nest states and no
                 column count can: a pair's last block is B at the last column and its
                 step counter stands two past it. *)
              when_
                (term_last &: now_last_group)
                [ if_
                    (is_head_of now_layer)
                    [ sm.set_next Offer ]
                    [ when_ now_ends [ sm.set_next Turn ] ]
                ]
            ] )
        ; ( Offer
          , [ when_
                (step_ready.value &: i.step_taken)
                [ step_ready <-- gnd
                ; if_
                    turn_drained.value
                    [ sm.set_next Turn ]
                    [ (* the step behind the wait opens at group 0 and the bank still
                         holds the last group's norms, thus the load is stated again *)
                      start_load <-- vdd
                    ; sm.set_next Dwell
                    ]
                ]
            ] )
        ; ( Turn
          , [ when_
                turn_drained.value
                [ turn_drained <-- gnd
                ; if_
                    (turn.value ==:. turn_count - 1)
                    [ sm.set_next Idle ]
                    [ turn <-- turn.value +:. 1
                    ; lead_s <--. 0
                    ; lead_phase <--. 0
                    ; lead_group <--. 0
                    ; lead_cin <--. 0
                    ; lead_cycle <--. 0
                    ; flush_s <--. 0
                    ; flush_phase <--. 0
                    ; flush_group <--. 0
                    ; priming <-- vdd
                    ; start_load <-- vdd
                    ; sm.set_next Prime
                    ]
                ]
            ] )
        ]
    in
    compile
      [ lead_nest
      ; capture_drain_phase
      ; band_load
      ; open_flush
      ; walk_flush
      ; head_seam
      ; machine
      ];
    { O.busy = ~:(sm.is Idle)
    ; plane_step = landed_step
    ; plane = landed_plane
    ; step_ready = step_ready.value
    ; logits = mux i.logit_seat (List.map logit_file ~f:bank_value)
    }
  ;;
end

(* ==================================================================== *)
(* The bench *)
(* ==================================================================== *)

(* INSTRUMENT 3, AND ITS NAMES ARE A CONTRACT. The gate probes the store write signals by
   name — [x_write], [x_address], [x_data] and the same for Y — thus a rename cannot
   silently blind it. Era five's four datapath faults all lived in exactly this layer and
   none of them moved a frame; that is why the instrument exists.

   The canvas is MODELLED and not instantiated: [Canvas] carries its own gate, thus the
   bench answers the stem's plane column with the twin's own decode and keeps this gate
   about the engine.

   THE MODEL IS ONE CYCLE LATE AND THAT IS SOUND HERE, NOT A SEAM. [Canvas] answers
   combinationally in the cycle its address stands; this bench reads the address after a
   cycle and answers in the next one. The two agree because the landed address is
   BLOCK-STABLE: it holds through the load cycle and the cycle before it, thus a column
   read one cycle early is the same column. What the shortcut does not exercise is the
   combinational path itself, and that waits for the canvas agreement of S4. *)
module Bench (M : sig
    val e : Elaboration.t
  end) =
struct
  module Engine = Make (M)
  module Sim = Cyclesim.With_interface (Engine.I) (Engine.O)

  let e : Elaboration.t = M.e
  let steps = e.steps
  let rows = e.rows
  let voices = Frame.voices

  (* one column write of a store, as the probe sees it *)
  type write =
    { to_y : bool
    ; address : int
    ; column : int array
    }

  (* what one pass did: the columns the stores took in the order they went out, the logit
     columns of every step the head offered, and where its cycles went *)
  type pass =
    { written : write list
    ; offered : int array array list
    ; cycles : int
    ; priming : int (** the preambles: one short block at each layer *)
    ; turning : int (** the turns: the tail of the last drain of each layer *)
    ; waiting : int (** the head's waits, which Phase I spends and Phase II would not *)
    }

  (* one column as the twin holds it: [rows] signed activations, row 0 first *)
  let column_of bits =
    Bits.split_lsb ~part_width:activation_bits bits
    |> List.map ~f:Bits.to_signed_int
    |> Array.of_list
  ;;

  (* [run ~planes] is one forward pass, and [planes] is the canvas: the [rows] activations
     of one step and one plane. The stream gate hands the twin's own decode over; a
     picture at a shape the twin does not hold hands over one of its own. *)
  let run ?(trace = false) ?(read_logits = true) ~planes () =
    (* THE TRACE IS THE NAMED SIGNALS AND NOT EVERY SIGNAL, the rule [Source]'s harness
       states: this is the largest circuit of the library and the gates run tens of
       thousands of cycles through it. Every signal the probes look up and every signal
       the waveforms display carries a [--] name, thus [`All_named] reaches all of them. *)
    let sim = Sim.create ~config:(Cyclesim.Config.trace `All_named) Engine.create in
    let waves, sim =
      if trace
      then (
        let waves, sim = Cyclesim.Waveform.create sim in
        Some waves, sim)
      else None, sim
    in
    let inp = Cyclesim.inputs sim in
    let out = Cyclesim.outputs sim in
    let node name =
      Option.value_exn (Cyclesim.lookup_node_by_name sim name) ~message:name
    in
    (* ONE FLUSH NEST, ONE ADDRESS. The column a write means is the flush nest's own and
       it is the same whichever store takes it; Y's port carries a ring address below it,
       and the gate holds the SEMANTIC write. *)
    let address = node "flush_address" in
    let probe write data = node write, address, node data in
    let x = probe "x_write" "x_data" in
    let y = probe "y_write" "y_data" in
    (* What a pass spends outside its dwells. The state's encoding is its index in
       [State.all] — Hardcaml's binary encoding — and a change of that would not read as a
       small error here: [List.nth_exn] refuses an index the enumeration does not hold. *)
    (* the state is a register and not a node, thus it is looked up as either *)
    let state =
      Option.value_exn (Cyclesim.lookup_node_or_reg_by_name sim "state") ~message:"state"
    in
    let spent = Array.create ~len:(List.length State.all) 0 in
    let tick () =
      let at = Cyclesim.Node.to_int state in
      spent.(at) <- spent.(at) + 1
    in
    let in_state which =
      spent.(fst
               (List.findi_exn State.all ~f:(fun (_ : int) s -> State.compare s which = 0)))
    in
    let written = ref [] in
    let offered = ref [] in
    let cycles = ref 0 in
    let take ~to_y (enable, address, data) =
      if Cyclesim.Node.to_int enable = 1
      then
        written
        := { to_y
           ; address = Cyclesim.Node.to_int address
           ; column = column_of (Cyclesim.Node.to_bits data)
           }
           :: !written
    in
    let plain () =
      Cyclesim.cycle sim;
      Int.incr cycles;
      take ~to_y:false x;
      take ~to_y:true y;
      tick ();
      (* the canvas answers the port of the next cycle, combinational from its registers *)
      inp.plane_column
      := Bits.concat_lsb
           (List.map
              (Array.to_list
                 (planes
                    ~step:(Bits.to_unsigned_int !(out.plane_step))
                    ~plane:(Bits.to_unsigned_int !(out.plane))))
              ~f:(Bits.of_signed_int ~width:activation_bits))
    in
    (* THE SEAT SWEEP COSTS CYCLES AND THAT IS THE POINT. The machine is frozen while the
       level stands, thus the file stands still and one cycle for each seat reads it
       whole. A sweep inside one cycle does not: the simulator answers a port with the
       input the last cycle carried, and every seat then reads the seat the sweep read
       before it — which this bench first read as three seats of four agreeing. *)
    let cycle () =
      plain ();
      if not (Bits.to_bool !(out.step_ready))
      then inp.step_taken := Bits.gnd
      else if not read_logits
      then
        (* the tie of the cycle bench: the level answers itself, thus no wait is counted *)
        inp.step_taken := Bits.vdd
      else (
        let columns =
          Array.init voices ~f:(fun seat ->
            Prng.For_test.set inp.logit_seat seat;
            plain ();
            column_of !(out.logits))
        in
        offered := columns :: !offered;
        inp.step_taken := Bits.vdd;
        plain ();
        inp.step_taken := Bits.gnd)
    in
    inp.start := Bits.vdd;
    cycle ();
    inp.start := Bits.gnd;
    while Bits.to_bool !(out.busy) do
      cycle ()
    done;
    (* the last write leaves the probe one cycle behind the fall of [busy] *)
    cycle ();
    ( waves
    , { written = List.rev !written
      ; offered = List.rev !offered
      ; cycles = !cycles
      ; priming = in_state Prime
      ; waiting = in_state Offer
      ; turning = in_state Turn
      } )
  ;;
end

let%expect_test "the schedule of one column: the preamble, the nine terms, the drain" =
  (* THE SCHEDULE'S VISUAL GATE, at a shape a picture can hold: P 6, G 2, T 3 and a trunk
     of three channels — the narrowest the fused floor admits at P 6. The stem is the
     widest dwell of any layer — its input channels are the eight planes, and no shape
     makes them fewer — thus its first column is the whole of the rotation in one place.

     WINDOW ONE is the preamble and the first block. The lead cycle counts 0 to 8 with no
     term under it, and the three fetches it sends at 0, 3 and 6 land at [slot_0],
     [slot_1] and [slot_2] two cycles later — the last of them one cycle before the first
     term. The terms then run a block a channel with no gap, and every slot reloads nine
     cycles behind the load before it, on the edge of its own last read: that is the
     rotation.

     WINDOW TWO is the close of that column. [term_last] captures the array; [drained]
     stands [rows] cycles later and the epilogue answers three behind it; the column
     writes go out one a cycle when the band stands whole. THE NEXT DWELL IS ALREADY
     RUNNING UNDER ALL OF IT — [term] never falls — which is what "the dwells stand back
     to back" means and what the exact cycle counts lean on. *)
  let config = { Diffusion.Config.layers = 4; width = 3 } in
  let model = Quantized.Model.For_test.init config ~seed:1 in
  let elaboration = Elaboration.create ~rows:6 model ~steps:3 ~lanes:2 ~walk:4 in
  let module B =
    Bench (struct
      let e = elaboration
    end)
  in
  (* the twin's decode reads its own [rows], thus this picture answers the stem with a
     column of its own: what stands here is the SCHEDULE, and the arithmetic has its own
     gate below *)
  let planes ~step ~plane =
    Array.init elaboration.rows ~f:(fun row ->
      if row = (step + plane) % elaboration.rows then 64 else 0)
  in
  let waves, (_ : B.pass) = B.run ~trace:true ~planes () in
  let waves = Option.value_exn waves ~message:"a traced run gives a waveform" in
  let window ~start_cycle =
    Hardcaml_waveterm.Waveform.expect
      ~display_rules:
        [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Bit
            [ "term"
            ; "term_first"
            ; "term_last"
            ; "slot_0"
            ; "slot_1"
            ; "slot_2"
            ; "drained"
            ; "band_row"
            ; "x_write"
            ]
        ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
            ~wave_format:Wave_format.Unsigned_int
            [ "lead_cycle"; "drain_row" ]
        ]
      ~show_digest:false
      ~wave_width:0
      ~display_width:80
      ~start_cycle
      waves
  in
  window ~start_cycle:0;
  window ~start_cycle:78;
  [%expect
    {|
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │term              ││                        ┌─────────────────────────────────│
    │                  ││────────────────────────┘                                 │
    │term_first        ││                        ┌─┐                               │
    │                  ││────────────────────────┘ └───────────────────────────────│
    │term_last         ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │slot_0            ││          ┌─┐               ┌─┐               ┌─┐         │
    │                  ││──────────┘ └───────────────┘ └───────────────┘ └─────────│
    │slot_1            ││                ┌─┐               ┌─┐               ┌─┐   │
    │                  ││────────────────┘ └───────────────┘ └───────────────┘ └───│
    │slot_2            ││                      ┌─┐               ┌─┐               │
    │                  ││──────────────────────┘ └───────────────┘ └───────────────│
    │drained           ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │band_row          ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │x_write           ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │                  ││────┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─│
    │lead_cycle        ││ 0  │1│2│3│4│5│6│7│8│0│1│2│3│4│5│6│7│8│0│1│2│3│4│5│6│7│8│0│
    │                  ││────┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─│
    │                  ││──────────────────────────────────────────────────────────│
    │drain_row         ││ 0                                                        │
    │                  ││──────────────────────────────────────────────────────────│
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │term              ││──────────────────────────────────────────────────────────│
    │                  ││                                                          │
    │term_first        ││            ┌─┐                                           │
    │                  ││────────────┘ └───────────────────────────────────────────│
    │term_last         ││          ┌─┐                                             │
    │                  ││──────────┘ └─────────────────────────────────────────────│
    │slot_0            ││                ┌─┐               ┌─┐               ┌─┐   │
    │                  ││────────────────┘ └───────────────┘ └───────────────┘ └───│
    │slot_1            ││    ┌─┐               ┌─┐               ┌─┐               │
    │                  ││────┘ └───────────────┘ └───────────────┘ └───────────────│
    │slot_2            ││          ┌─┐               ┌─┐               ┌─┐         │
    │                  ││──────────┘ └───────────────┘ └───────────────┘ └─────────│
    │drained           ││                  ┌───────────┐                           │
    │                  ││──────────────────┘           └───────────────────────────│
    │band_row          ││                            ┌───────────┐                 │
    │                  ││────────────────────────────┘           └─────────────────│
    │x_write           ││                                        ┌───┐             │
    │                  ││────────────────────────────────────────┘   └─────────────│
    │                  ││──┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─│
    │lead_cycle        ││ 5│6│7│8│0│1│2│3│4│5│6│7│8│0│1│2│3│4│5│6│7│8│0│1│2│3│4│5│6│
    │                  ││──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─│
    │                  ││────────────────────┬─┬─┬─┬─┬─────────────────────────────│
    │drain_row         ││ 0                  │1│2│3│4│5                            │
    │                  ││────────────────────┴─┴─┴─┴─┴─────────────────────────────│
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    |}]
;;

let%expect_test "the pair interleaves, and the picture is the schedule" =
  (* THE SCHEDULE'S OWN PICTURE. One pair at a shape a window can hold: P 6, G 2, T 4 and
     a trunk of three channels. What the waves state is the ORDER — [lead_phase] falls for
     A and rises for B, and [lead_column] is the canvas column each block works on — thus
     A0, A1, A2 B0, A3 B1, B2, B3 stands in one place, with the lag of two visible as the
     distance between a rise of [lead_phase] and the column it names.

     [y_write] AND [x_write] SAY WHERE EACH BLOCK'S COLUMNS WENT: an A block writes the
     ring and a B block writes X, and they alternate from the third block on. The flush
     trails the dwell by a drain and an epilogue, thus a write stands under the block
     AFTER the one that made it — which is the whole reason the lag is two and not one. *)
  let config = { Diffusion.Config.layers = 4; width = 3 } in
  let model = Quantized.Model.For_test.init config ~seed:1 in
  let elaboration = Elaboration.create ~rows:6 model ~steps:4 ~lanes:2 ~walk:4 in
  let module B =
    Bench (struct
      let e = elaboration
    end)
  in
  let planes ~step ~plane =
    Array.init elaboration.rows ~f:(fun row ->
      if row = (step + plane) % elaboration.rows then 64 else 0)
  in
  let waves, (_ : B.pass) = B.run ~trace:true ~planes () in
  let waves = Option.value_exn waves ~message:"a traced run gives a waveform" in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:
      [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:Wave_format.Bit
          [ "lead_phase"; "y_write"; "x_write" ]
      ; Hardcaml_waveterm.Display_rule.port_name_is_one_of
          ~wave_format:(Wave_format.Unsigned_int : Wave_format.t)
          [ "lead_column"; "lead_s" ]
      ; Hardcaml_waveterm.Display_rule.default
      ]
    ~wave_width:(-5)
    ~display_width:80
    ~display_height:18
    ~start_cycle:(Elaboration.turn_cycles elaboration elaboration.turns.(0) + 12)
    waves;
  [%expect
    {|
    ┌Signals───────────┐┌Waves─────────────────────────────────────────────────────┐
    │lead_phase        ││                                    ┌──────────┐          │
    │                  ││────────────────────────────────────┘          └──────────│
    │y_write           ││            ┌┐    ╥    ╥    ┌┐    ╥    ╥               ┌┐ │
    │                  ││────────────┘└────╨────╨────┘└────╨────╨───────────────┘└─│
    │x_write           ││ ╥                                           ╥    ╥       │
    │                  ││─╨───────────────────────────────────────────╨────╨───────│
    │                  ││──────────────┬──────────┬──────────┬──────────┬──────────│
    │lead_column       ││ 0            │1         │2         │0         │3         │
    │                  ││──────────────┴──────────┴──────────┴──────────┴──────────│
    │                  ││──┬───────────┬──────────┬─────────────────────┬──────────│
    │lead_s            ││ 4│0          │1         │2                    │3         │
    │                  ││──┴───────────┴──────────┴─────────────────────┴──────────│
    │clear             ││                                                          │
    │                  ││──────────────────────────────────────────────────────────│
    │clock             ││╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥│
    │                  ││╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨│
    └──────────────────┘└──────────────────────────────────────────────────────────┘
    6bc98568b57b46d22731f0145f1c1770
    |}]
;;

let%expect_test "the store writes are the twin's, write for write" =
  (* INSTRUMENT 3. Era five's four faults — a weight address whose stride was not the
     tensor's, a channel block read at the gate's offset, an operand taken on the address
     side of a two-cycle read, and a ring run off its end — were all faults of this
     composition layer, and none of them moved a frame. This holds EVERY column the engine
     writes against [Quantized.For_test.layer_writes]: the address stands in the
     elaboration's own map, the datum equals the twin's, and each destination column is
     written exactly one time for each layer. The head writes no store, thus its gate is
     the logit face, read through the ports at every step the level offers. *)
  let case ~name ~width ~lanes ~pairs ~steps ~seed =
    let config = { Diffusion.Config.layers = 2 + (2 * pairs); width } in
    let model = Quantized.Model.For_test.init config ~seed in
    let elaboration = Elaboration.create model ~steps ~lanes ~walk:8 in
    let module B =
      Bench (struct
        let e = elaboration
      end)
    in
    let rows = elaboration.rows in
    let voices = Frame.voices in
    let state, canvas = Diffusion.opening_canvas (Prng.create_folded ~seed) ~steps in
    let threshold = Diffusion.anneal_threshold ~step:0 ~walk:8 in
    let (_ : Prng.state), hidden = Diffusion.hidden_cells state ~steps ~threshold in
    let stem = Quantized.For_test.plane_activations canvas hidden ~steps in
    let planes ~step ~plane = Quantized.For_test.plane_column stem ~step ~plane in
    let (_ : Hardcaml_waveterm.Waveform.t option), pass = B.run ~planes () in
    let want = Quantized.For_test.layer_writes model canvas hidden ~steps in
    let store () =
      Array.init (Elaboration.store_depth elaboration) ~f:(fun (_ : int) ->
        Array.create ~len:rows 0)
    in
    let x = store () in
    let y = store () in
    let cursor = ref pass.written in
    let checked = ref 0 in
    let part = ref 0 in
    let misplaced = ref 0 in
    let head_columns (expected : int array) =
      List.iteri pass.offered ~f:(fun step columns ->
        Array.iteri columns ~f:(fun seat got ->
          Int.incr checked;
          if not
               (Array.equal
                  Int.equal
                  got
                  (Quantized.For_test.tensor_column
                     expected
                     ~step
                     ~channel:seat
                     ~channels:voices))
          then Int.incr part))
    in
    (* THE WRITES OF A TURN COME OUT INTERLEAVED, thus a turn is what the cursor takes.
       Inside a pair the blocks run A0, A1, A2 B0, A3 B1, ...: A's columns go to Y and B's
       to X in one stream, and each write says which store took it. The gate holds the
       stream by TURN and then reads each layer's tensor whole. *)
    let check_layer at (layer : Elaboration.layer) destination =
      let expected = List.nth_exn want at in
      List.iter (List.range 0 steps) ~f:(fun step ->
        List.iter (List.range 0 layer.outputs) ~f:(fun channel ->
          Int.incr checked;
          let address = Elaboration.column_address elaboration ~step ~channel in
          let want =
            Quantized.For_test.tensor_column
              expected
              ~step
              ~channel
              ~channels:layer.outputs
          in
          if not (Array.equal Int.equal destination.(address) want)
          then (
            Int.incr part;
            let rows_part =
              Array.counti want ~f:(fun r v -> v <> destination.(address).(r))
            in
            if !part <= 2
            then
              printf
                "  L%d step %d channel %d: %d of %d rows part\n    want %s\n    got  %s\n"
                at
                step
                channel
                rows_part
                rows
                (String.concat
                   ~sep:" "
                   (List.map (List.take (Array.to_list want) 12) ~f:Int.to_string))
                (String.concat
                   ~sep:" "
                   (List.map
                      (List.take (Array.to_list destination.(address)) 12)
                      ~f:Int.to_string)))))
    in
    let turn_columns (turn : Elaboration.turn) =
      let ats = turn.first :: Option.to_list turn.second in
      let count =
        List.sum (module Int) ats ~f:(fun at -> steps * elaboration.layers.(at).outputs)
      in
      let mine = List.take !cursor count in
      cursor := List.drop !cursor count;
      List.iter mine ~f:(fun (w : B.write) ->
        (if w.to_y then y else x).(w.address) <- w.column);
      (* every destination column written exactly one time, and each store taking exactly
         the layer's own count *)
      (* the destination and the address as one key: X at [address], Y above them *)
      let depth = Elaboration.store_depth elaboration in
      let placed =
        List.fold
          mine
          ~init:(Set.empty (module Int))
          ~f:(fun seen w ->
            Set.add seen (if w.to_y then depth + w.address else w.address))
      in
      if List.length mine <> count || Set.length placed <> count then Int.incr misplaced;
      List.iter ats ~f:(fun at ->
        let layer = elaboration.layers.(at) in
        let to_y =
          match layer.role with
          | Pair_open -> true
          | Stem | Pair_close | Head -> false
        in
        if List.count mine ~f:(fun w -> Bool.equal w.to_y to_y) <> steps * layer.outputs
        then Int.incr misplaced;
        check_layer at layer (if to_y then y else x))
    in
    Array.iter elaboration.turns ~f:(fun (turn : Elaboration.turn) ->
      match elaboration.layers.(turn.first).role with
      | Head -> head_columns (List.nth_exn want turn.first)
      | Stem | Pair_open | Pair_close -> turn_columns turn);
    (* THE TWO PLANS STAND IN THE LINE, thus a shape that stops crossing a bank says so
       here and does not leave a select and an offset untested in silence. *)
    let plan banks =
      String.concat
        ~sep:" + "
        (List.map (Array.to_list banks) ~f:(fun (bank : Elaboration.bank) ->
           Int.to_string bank.depth))
    in
    printf
      "%s: the weights bank %s, the stores bank %s; %d columns written, %d steps \
       offered, %d columns checked — %d part, %d misplaced\n"
      name
      (plan elaboration.weight_banks)
      (plan elaboration.store_banks)
      (List.length pass.written)
      (List.length pass.offered)
      !checked
      !part
      !misplaced
  in
  case ~name:"H 8, G 2, two pairs, T 6" ~width:8 ~lanes:2 ~pairs:2 ~steps:6 ~seed:1;
  case ~name:"H 7, G 3, one pair,  T 5" ~width:7 ~lanes:3 ~pairs:1 ~steps:5 ~seed:2;
  (* AN IMAGE THAT REALLY BANKS: 1 080 words plan as 1 024 and 512, thus this case reads
     through the bank mux and the two above read through one bank alone. The elected
     rung's own image banks, thus a shape that never crosses a bank would leave the select
     and the offset untested until a board. *)
  case ~name:"H 8, G 4, three pairs, T 6" ~width:8 ~lanes:4 ~pairs:3 ~steps:6 ~seed:3;
  (* A STORE THAT REALLY BANKS: 129 steps of 8 channels make a store of 1 032 columns,
     which plans as 1 024 and 512, thus this case reads and writes THROUGH the store's
     bank mux and its write select where the three above stand in one bank each. The
     elected geometry banks its stores at T 128 and H 20, thus a shape that never crosses
     would leave the split untested until a board. *)
  case ~name:"H 8, G 2, one pair,   T 129" ~width:8 ~lanes:2 ~pairs:1 ~steps:129 ~seed:4;
  (* THE RING WRAPS TWICE. Five columns over four ring slots is one wrap; two pairs and
     the head behind them read every wrapped column, thus a ring one column short — or a
     lag of one instead of two — writes over a column that is still live and this case is
     where it would say so. *)
  case ~name:"H 8, G 2, two pairs, T 5" ~width:8 ~lanes:2 ~pairs:2 ~steps:5 ~seed:5;
  [%expect
    {|
    H 8, G 2, two pairs, T 6: the weights bank 2048, the stores bank 512; 240 columns written, 6 steps offered, 264 columns checked — 0 part, 0 misplaced
    H 7, G 3, one pair,  T 5: the weights bank 1024, the stores bank 512; 105 columns written, 5 steps offered, 125 columns checked — 0 part, 0 misplaced
    H 8, G 4, three pairs, T 6: the weights bank 1024 + 512, the stores bank 512; 336 columns written, 6 steps offered, 360 columns checked — 0 part, 0 misplaced
    H 8, G 2, one pair,   T 129: the weights bank 1024, the stores bank 1024 + 512; 3096 columns written, 129 steps offered, 3612 columns checked — 0 part, 0 misplaced
    H 8, G 2, two pairs, T 5: the weights bank 2048, the stores bank 512; 200 columns written, 5 steps offered, 220 columns checked — 0 part, 0 misplaced
    |}]
;;

let%expect_test "the cycles of one forward, against the cost model" =
  (* INSTRUMENT 4'S DYNAMIC HALF. [Elaboration.forward_cycles] counts the dwells of every
     layer and one drain tail behind each of them, and it is a COUNT and not a bound: it
     assumes the dwells stand back to back, thus every fetch of the rotation hides under a
     running dwell. What it does not count is the preamble of each layer and the turn
     behind it, and this bench is the first measured number for them.

     The harness ties [step_taken] to [step_ready], thus no draw is waited on and what
     stands is the machine's own.

     WHAT THE NUMBERS SAY. The preamble is nine cycles at each layer and NOTHING ELSE
     GROWS: it does not follow the columns, the groups or the channels, thus every fetch
     of the rotation but the first of a layer hides under a running dwell — which is the
     claim of "The dwell" in the design, measured. The turn stands at or under the drain
     tails the model already counts. What is left is the head's wait, and it is PHASE I'S
     SERIALIZATION PRICED: about [rows] + 9 cycles at every step, growing with T and with
     nothing else, which is the number the overlap of Phase II would buy back.

     IF THE ROTATION EVER FAILS TO HIDE A FETCH the preamble line grows with the dwells
     and not with the layers, and then the design moves and not this bench. *)
  let case ~name ~width ~lanes ~pairs ~steps ~seed =
    let config = { Diffusion.Config.layers = 2 + (2 * pairs); width } in
    let model = Quantized.Model.For_test.init config ~seed in
    let elaboration = Elaboration.create model ~steps ~lanes ~walk:8 in
    let module B =
      Bench (struct
        let e = elaboration
      end)
    in
    let state, canvas = Diffusion.opening_canvas (Prng.create_folded ~seed) ~steps in
    let threshold = Diffusion.anneal_threshold ~step:0 ~walk:8 in
    let (_ : Prng.state), hidden = Diffusion.hidden_cells state ~steps ~threshold in
    let stem = Quantized.For_test.plane_activations canvas hidden ~steps in
    let planes ~step ~plane = Quantized.For_test.plane_column stem ~step ~plane in
    let (_ : Hardcaml_waveterm.Waveform.t option), pass =
      B.run ~read_logits:false ~planes ()
    in
    let model_cycles = Elaboration.forward_cycles elaboration in
    (* THE TURN IS THE UNIT NOW, not the layer: one preamble and one drain tail for each
       of them, and a pair holds two layers inside one turn. *)
    let count = Array.length elaboration.turns in
    let tails = count * elaboration.rows in
    printf
      "%s: %d cycles, the model %d (%+d)\n"
      name
      pass.cycles
      model_cycles
      (pass.cycles - model_cycles);
    printf
      "  %d preamble (%d a turn), %d head wait (%d a step), %d turn against the %d the \
       model counts, %d elsewhere\n"
      pass.priming
      (pass.priming / count)
      pass.waiting
      (pass.waiting / steps)
      pass.turning
      tails
      (pass.cycles - model_cycles - pass.priming - pass.waiting - (pass.turning - tails))
  in
  case ~name:"H 8, G 2, two pairs, T 6" ~width:8 ~lanes:2 ~pairs:2 ~steps:6 ~seed:1;
  case ~name:"H 7, G 3, one pair,  T 5" ~width:7 ~lanes:3 ~pairs:1 ~steps:5 ~seed:2;
  case ~name:"H 8, G 2, two pairs, T 12" ~width:8 ~lanes:2 ~pairs:2 ~steps:12 ~seed:1;
  [%expect
    {|
    H 8, G 2, two pairs, T 6: 10082 cycles, the model 9696 (+386)
      36 preamble (9 a turn), 354 head wait (59 a step), 178 turn against the 192 the model counts, 10 elsewhere
    H 7, G 3, one pair,  T 5: 4056 cycles, the model 3744 (+312)
      27 preamble (9 a turn), 300 head wait (60 a step), 121 turn against the 144 the model counts, 8 elsewhere
    H 8, G 2, two pairs, T 12: 19940 cycles, the model 19200 (+740)
      36 preamble (9 a turn), 708 head wait (59 a step), 178 turn against the 192 the model counts, 10 elsewhere
    |}]
;;
