(** The top level of the Nexys 4 board.

    Step 6 of the bring-up: the control port serves the wire protocol on the host UART,
    and the doorbell sends the test message on the MIDI output, [JD 0].

    The board shows: heartbeat on [led 0], RsRx activity on [led 1], RsTx activity on
    [led 2], MIDI activity on [led 3]. The MIDI line idles at 1, the no-current level of
    the current loop; the other JD pins stay at 1. *)

open Hardcaml
open Signal

(* 100 MHz / 115200 baud = 868.06 *)
let host_clocks_per_bit = 868

(* 100 MHz / 31250 baud = 3200, exact *)
let midi_clocks_per_bit = 3200

let create () =
  let clk = input "clk" 1 in
  let rstn = input "btnCpuReset" 1 in
  let rx_pin = input "RsRx" 1 in
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
  let tx_busy = wire 1 in
  let midi_busy = wire 1 in
  let control_port =
    Control_port.create
      { Control_port.I.clock = clk
      ; clear
      ; in_data = uart_rx.data
      ; in_valid = uart_rx.valid
      ; hold = tx_busy
      ; midi_hold = midi_busy
      }
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
  let midi_tx =
    Uart_tx.create
      ~clocks_per_bit:midi_clocks_per_bit
      { Uart_tx.I.clock = clk
      ; clear
      ; data = control_port.midi_data
      ; valid = control_port.midi_valid
      }
  in
  assign midi_busy midi_tx.busy;
  let led =
    concat_msb [ zero 12; ~:(midi_tx.serial); ~:(uart_tx.serial); ~:rx_pin; heartbeat ]
  in
  Circuit.create_exn
    ~name:"top"
    [ output "led" led
    ; output "RsTx" uart_tx.serial
    ; output "JD" (concat_msb [ ones 7; midi_tx.serial ])
    ]
;;
