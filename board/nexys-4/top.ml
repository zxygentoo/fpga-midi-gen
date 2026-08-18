(** The top level of the Nexys 4 board.

    The control port serves the wire protocol on the host UART, [Control_regs] holds the
    cells, and the MIDI path is [Midi_merge] and [Midi_out]. The structure is the one of
    [docs/host_control_rtl.md]. The model seat takes any source of [Source_intf], and it
    holds the transformer of [docs/transformer_rtl.md]: one step of music is one pass of
    the network and one frame, and the sequencer decodes the frame into the messages of
    the wire. The model arrives as [model] at elaboration — the bitstream carries the
    weights, thus [create] takes the quantized model whole and [gen_verilog] names the
    checkpoint.

    The board shows: heartbeat on [led 0], RsRx activity on [led 1], RsTx activity on
    [led 2], MIDI activity on [led 3], the busy state of the port on [led 4], and the run
    state on [led 5]. The MIDI line idles at 1, the no-current level of the current loop;
    the other JD pins stay at 1.

    The model plays while RUN is 1: a host write sets the run state, and a push of the
    center button [btnC] toggles it. [led 5] shows it. *)

open Hardcaml
open Signal

(* 100 MHz / 115200 baud = 868.06 *)
let host_clocks_per_bit = 868

(* 100 MHz / 31250 baud = 3200, exact *)
let midi_clocks_per_bit = 3200

(* the 100 MHz clock *)
let clocks_per_ms = 100_000
let button_debounce_ms = 10

let create ~model () =
  let clk = input "clk" 1 in
  let rstn = input "btnCpuReset" 1 in
  let rx_pin = input "RsRx" 1 in
  let run_pin = input "btnC" 1 in
  let clear = ~:rstn in
  let spec = Reg_spec.create ~clock:clk ~clear () in
  (* bit 26 of a 100 MHz counter toggles with a period of 1.34 s *)
  let counter = reg_fb spec ~width:27 ~f:(fun d -> d +:. 1) in
  let heartbeat = select counter ~high:26 ~low:26 in
  let uart_rx =
    Uart_rx.create
      ~clocks_per_bit:host_clocks_per_bit
      { Uart_rx.I.clock = clk; clear; serial = rx_pin }
  in
  (* each wire breaks the order of one connection that goes in both directions; no path is
     a loop, because the far end of each one comes from a register *)
  let tx_busy = wire 1 in
  let read_data = wire 8 in
  let doorbell_ready = wire 1 in
  let model_ready = wire 1 in
  let out_ready = wire 1 in
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
      ; doorbell_ready
      }
  in
  assign read_data control_regs.read_data;
  let model =
    (* the one line that names the model of the era *)
    Socket.create
      ~clocks_per_ms
      ~source:(Source.create ~model ~seed:control_regs.params.seed)
      { Socket.I.clock = clk
      ; clear
      ; params = control_regs.params
      ; midi_ready = model_ready
      }
  in
  let midi_merge =
    Midi_merge.create
      { Midi_merge.I.doorbell = control_regs.doorbell; model = model.midi; out_ready }
  in
  assign doorbell_ready midi_merge.doorbell_ready;
  assign model_ready midi_merge.model_ready;
  let midi_out =
    Midi_out.create
      ~clocks_per_bit:midi_clocks_per_bit
      { Midi_out.I.clock = clk; clear; message = midi_merge.out }
  in
  assign out_ready midi_out.ready;
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
  let led =
    concat_msb
      [ zero 10
      ; control_regs.params.run
      ; control_port.busy
      ; ~:(midi_out.serial)
      ; ~:(uart_tx.serial)
      ; ~:rx_pin
      ; heartbeat
      ]
  in
  Circuit.create_exn
    ~name:"top"
    [ output "led" led
    ; output "RsTx" uart_tx.serial
    ; output "JD" (concat_msb [ ones 7; midi_out.serial ])
    ]
;;
