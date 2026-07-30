(** The top level of the Nexys 4 board.

    Step 2 of the bring-up: a heartbeat on [led 0] from a free-running counter, and the
    inverse of RsRx on [led 1] as an activity light. RsTx and all JD pins stay at 1: this
    is the idle level of a UART and of the MIDI current loop, and no current flows. *)

open Hardcaml
open Signal

let create () =
  let clk = input "clk" 1 in
  let rstn = input "btnCpuReset" 1 in
  let rx = input "RsRx" 1 in
  let spec = Reg_spec.create ~clock:clk ~clear:~:rstn () in
  (* bit 26 of a 100 MHz counter toggles with a period of 1.34 s *)
  let counter = reg_fb spec ~width:27 ~f:(fun d -> d +:. 1) in
  let heartbeat = select counter ~high:26 ~low:26 in
  let rx_activity = ~:rx in
  let led = concat_msb [ zero 14; rx_activity; heartbeat ] in
  Circuit.create_exn
    ~name:"top"
    [ output "led" led; output "RsTx" vdd; output "JD" (ones 8) ]
;;
