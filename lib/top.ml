(** The top level of the Nexys 4 board.

    Step 4 of the bring-up: the control port serves the wire protocol on the host UART.
    The MSG cells are plain memory in this step; the doorbell behavior comes with the MIDI
    output.

    The board shows: heartbeat on [led 0], RsRx activity on [led 1], RsTx activity on
    [led 2]. The JD pins stay at 1, the idle level of the MIDI current loop; no current
    flows. *)

open Hardcaml
open Signal

(* 100 MHz / 115200 baud = 868.06 *)
let host_clocks_per_bit = 868

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
  let control_port =
    Control_port.create
      { Control_port.I.clock = clk
      ; clear
      ; in_data = uart_rx.data
      ; in_valid = uart_rx.valid
      ; hold = tx_busy
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
  let led = concat_msb [ zero 13; ~:(uart_tx.serial); ~:rx_pin; heartbeat ] in
  Circuit.create_exn
    ~name:"top"
    [ output "led" led; output "RsTx" uart_tx.serial; output "JD" (ones 8) ]
;;
