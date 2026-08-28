(** The top level of the Nexys 4 board.

    The control port serves the wire protocol on the host UART and [Control_regs] holds
    the cells; the structure is the one of [docs/host_control_rtl.md]. The MIDI path is
    the model seat and nothing else: [Socket] takes any source of [Source_intf] and gives
    the line, and it holds the masked sheet of [docs/diffusion_rtl.md] — a run start draws
    one sheet whole, a step reads one of its frames, and the sequencer decodes the frame
    into the messages of the wire. The model arrives as [e] at elaboration — the bitstream
    carries the weights, thus [create] takes the elaboration of one checkpoint at one
    geometry and [gen_verilog] names both.

    NOTHING HERE NAMES AN ERA. [Source] and [Elaboration] are whatever the library this
    module opens provides, thus the seat moves with one line of the dune file and the
    source of the era before it stays in the tree, buildable and gated.

    The board shows two things: the run state on [led 0] and MIDI activity on [led 1]. The
    two are not the same lamp, because the model states nothing until it has something to
    state — era six draws the whole sheet at the push, about 5.6 s at the elected rung —
    thus [led 0] answers "did the push take" while [led 1] is still dark. The MIDI line
    idles at 1, the no-current level of the current loop; the other JD pins stay at 1.

    The model plays while RUN is 1: a host write sets the run state, and a push of the
    center button [btnC] toggles it. The seed comes from the slide switches in the same
    way — [Seed_switches] writes SEED and shows the cell on the eight digits, thus the
    board alone states which piece it plays. The decimal point is a pin of that group and
    this design does not use it, thus the top level holds it at 1 as it holds [JD[7:1]]. *)

open Hardcaml
open Signal

(* 100 MHz / 115200 baud = 868.06 *)
let host_clocks_per_bit = 868

(* 100 MHz / 31250 baud = 3200, exact *)
let midi_clocks_per_bit = 3200

(* the 100 MHz clock *)
let clocks_per_ms = 100_000
let button_debounce_ms = 10

let create ~e () =
  let clk = input "clk" 1 in
  let rstn = input "btnCpuReset" 1 in
  let rx_pin = input "RsRx" 1 in
  let run_pin = input "btnC" 1 in
  let switch_pins = input "sw" 16 in
  let clear = ~:rstn in
  let uart_rx =
    Uart_rx.create
      ~clocks_per_bit:host_clocks_per_bit
      { Uart_rx.I.clock = clk; clear; serial = rx_pin }
  in
  (* each wire breaks the order of one connection that goes in both directions; no path is
     a loop, because the far end of each one comes from a register *)
  let tx_busy = wire 1 in
  let read_data = wire 8 in
  let seed = wire 32 in
  let panel =
    Seed_switches.create
      { Seed_switches.I.clock = clk; clear; switches = switch_pins; seed }
  in
  let button =
    Button.create
      ~debounce_clocks:(button_debounce_ms * clocks_per_ms)
      { Button.I.clock = clk; clear; button = run_pin }
  in
  let control_port =
    Control_port.create
      { Control_port.I.clock = clk
      ; clear
      ; in_data = uart_rx.data
      ; in_valid = uart_rx.valid
      ; hold = tx_busy
      ; read_data
      }
  in
  let control_regs =
    Control_regs.create
      { Control_regs.I.clock = clk
      ; clear
      ; write_enable = control_port.write_enable
      ; write_address = control_port.write_address
      ; write_data = control_port.write_data
      ; commit = control_port.commit
      ; read_address = control_port.read_address
      ; run_toggle = button.toggle
      ; seed_write = panel.seed_write
      ; seed_value = panel.seed_value
      }
  in
  assign read_data control_regs.read_data;
  assign seed control_regs.params.seed;
  let model =
    (* the one line that names the model of the era *)
    Socket.create
      ~clocks_per_ms
      ~clocks_per_bit:midi_clocks_per_bit
      ~source:(Source.create ~e ~seed:control_regs.params.seed)
      { Socket.I.clock = clk; clear; params = control_regs.params }
  in
  let uart_tx =
    Uart_tx.create
      ~clocks_per_bit:host_clocks_per_bit
      { Uart_tx.I.clock = clk
      ; clear
      ; data = control_port.out_data
      ; valid = control_port.out_valid
      }
  in
  assign tx_busy uart_tx.busy;
  (* the line idles at 1, thus the LED shows the bytes and not the rest *)
  let led = concat_msb [ zero 14; ~:(model.serial); control_regs.params.run ] in
  Circuit.create_exn
    ~name:"top"
    [ output "led" led
    ; output "RsTx" uart_tx.serial
    ; output "JD" (concat_msb [ ones 7; model.serial ])
    ; output "an" panel.digit
    ; output "seg" panel.segment
    ; output "dp" vdd
    ]
;;
