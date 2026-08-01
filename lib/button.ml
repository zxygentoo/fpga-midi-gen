open Base
open Hardcaml
open Signal

module I = struct
  type 'a t =
    { clock : 'a
    ; clear : 'a
    ; button : 'a
    }
  [@@deriving hardcaml]
end

module O = struct
  type 'a t = { toggle : 'a } [@@deriving hardcaml]
end

let create ~debounce_clocks (i : _ I.t) : _ O.t =
  assert (debounce_clocks >= 2);
  let spec = Reg_spec.create ~clock:i.clock ~clear:i.clear () in
  let open Always in
  let synced = reg spec (reg spec i.button) in
  let level = Variable.reg spec ~width:1 in
  let count = Variable.reg spec ~width:(Int.ceil_log2 debounce_clocks) in
  let toggle = Variable.wire ~default:gnd () in
  let _ = level.value -- "level" in
  let full = count.value ==:. debounce_clocks - 1 in
  compile
    [ if_
        (synced ==: level.value)
        [ count <--. 0 ]
        [ count <-- count.value +:. 1
        ; when_ full [ level <-- synced; count <--. 0; when_ synced [ toggle <-- vdd ] ]
        ]
    ];
  { O.toggle = toggle.value }
;;

let%expect_test "the waveform of the debounce" =
  (* a 3-cycle window. The first burst bounces and then holds 1: one strobe, at the third
     stable cycle. The release bounces and holds 0: no strobe. A spike of one cycle moves
     nothing. *)
  let debounce_clocks = 3 in
  let module Sim = Cyclesim.With_interface (I) (O) in
  let sim = Sim.create ~config:Cyclesim.Config.trace_all (create ~debounce_clocks) in
  let waves, sim = Cyclesim.Waveform.create sim in
  let inp = Cyclesim.inputs sim in
  let drive bits =
    String.iter bits ~f:(fun b ->
      inp.button := if Char.equal b '1' then Bits.vdd else Bits.gnd;
      Cyclesim.cycle sim)
  in
  (* bounce press bounce release spike *)
  drive "0101100111111111110010100000000000010000000";
  let rules =
    [ Hardcaml_waveterm.Display_rule.port_name_is_one_of
        ~wave_format:Wave_format.(Bit_or Hex)
        [ "clock"; "button"; "level"; "toggle" ]
    ]
  in
  Hardcaml_waveterm.Waveform.expect
    ~display_rules:rules
    ~show_digest:false
    ~wave_width:(-1)
    waves;
  [%expect
    {|
    ┌Signals────────┐┌Waves──────────────────────────────────────────────┐
    │clock          ││╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥╥│
    │               ││╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨╨│
    │button         ││ ┌┐┌─┐ ┌──────────┐ ┌┐┌┐           ┌┐              │
    │               ││─┘└┘ └─┘          └─┘└┘└───────────┘└──────        │
    │level          ││            ┌───────────────┐                      │
    │               ││────────────┘               └──────────────        │
    │toggle         ││           ┌┐                                      │
    │               ││───────────┘└──────────────────────────────        │
    └───────────────┘└───────────────────────────────────────────────────┘
    |}]
;;
