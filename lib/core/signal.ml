open Core

(* the exit code the signal named, and 0 while none has arrived. An atomic, because a
   handler writes it and the loop reads it. *)
type stop_play = int Atomic.t

let watch_stop_play () =
  let stopped = Atomic.make 0 in
  List.iter
    [ Stdlib.Sys.sigint, 130; Stdlib.Sys.sigterm, 143 ]
    ~f:(fun (signal, code) ->
      ignore
        (Stdlib.Sys.Safe.signal
           signal
           (Stdlib.Sys.Signal_handle (fun (_ : int) -> Atomic.set stopped code))
         : Stdlib.Sys.signal_behavior));
  stopped
;;

let stop_code t = Atomic.get t
let stop_requested t = stop_code t <> 0
let exit_if_stopped t = if stop_requested t then exit (stop_code t)
