(** The top level of the Nexys 4 board.

    Step 3 of the bring-up: the host UART echoes at 115200 baud. A one-byte buffer holds
    the newest byte while the transmitter is busy: at one baud rate the transmitter
    becomes free a moment after the subsequent byte is complete, thus a direct connection
    loses bytes in a continuous stream.

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
    Uart_rx.create ~clocks_per_bit:host_clocks_per_bit ~clock:clk ~clear ~rxd:rx_pin
  in
  let tx_busy = wire 1 in
  let open Always in
  let pending = Variable.reg spec ~width:8 in
  let pending_valid = Variable.reg spec ~width:1 in
  let tx_valid = pending_valid.value &: ~:tx_busy in
  compile
    [ when_ tx_valid [ pending_valid <-- gnd ]
    ; when_ uart_rx.valid [ pending <-- uart_rx.data; pending_valid <-- vdd ]
    ];
  let uart_tx =
    Uart_tx.create
      ~clocks_per_bit:host_clocks_per_bit
      ~clock:clk
      ~clear
      ~data:pending.value
      ~valid:tx_valid
  in
  assign tx_busy uart_tx.busy;
  let led = concat_msb [ zero 13; ~:(uart_tx.txd); ~:rx_pin; heartbeat ] in
  Circuit.create_exn
    ~name:"top"
    [ output "led" led; output "RsTx" uart_tx.txd; output "JD" (ones 8) ]
;;
