(** The host control: the contract between the host drivers and the FPGA.

    This module is the single definition of the host-control constants — the sizes, the
    operations, the status codes and the control registers. The RTL reads them at
    elaboration time and the drivers read the same ones, thus a value has one home.
    [Control_frame] is the wire codec that carries them, and the normative description is
    [docs/host_control.md].

    The module holds only the definitions of the contract, thus it has no [.mli]: the
    definitions are their own signature. *)

open Base
open Bytes_util

(** The sizes of the host control, in bytes. A wire payload is a header and then DATA. *)
module Constants = struct
  let request_header_bytes = 3 (* OP, ADDR, LEN *)
  let response_header_bytes = 2 (* OP, STATUS *)
  let max_data_len = 32

  (* the request header and the largest DATA; the FPGA discards a longer payload *)
  let max_payload_bytes = request_header_bytes + max_data_len
end

module Op = struct
  let read = 0x01
  let write = 0x02

  (* the response OP is the request OP plus this flag *)
  let response_flag = 0x80
end

module Status = struct
  type t =
    | Ok
    | Bad_op
    | Bad_address
    | Bad_length

  let to_code = function
    | Ok -> 0x00
    | Bad_op -> 0x01
    | Bad_address -> 0x02
    | Bad_length -> 0x03
  ;;

  let of_code = function
    | 0x00 -> Some Ok
    | 0x01 -> Some Bad_op
    | 0x02 -> Some Bad_address
    | 0x03 -> Some Bad_length
    | _ -> None
  ;;

  (* the name of the status in the diagnostic output *)
  let to_string = function
    | Ok -> "ok"
    | Bad_op -> "bad-op"
    | Bad_address -> "bad-address"
    | Bad_length -> "bad-length"
  ;;
end

(** The power-on values of the control cells. [Reg.fields] gives the value of each cell,
    and these are the names that a driver reads. *)
module Default = struct
  (* MIDI channel 3, the S-1 factory default. *)
  let channel = 2

  (* One step is a sixteenth, thus 200 ms puts the quarter at exactly 75 — the common
     chorale tempo. The pink era booted at 250; the transformer era's ear asked for this. *)
  let step_ms = 200
  let gate_ms = 125
  let velocity = 100
  let seed = 42 (* must not be 0 *)
end

(** The control registers. They are the local storage of the control unit, thus one byte
    of address names each one. *)
module Reg = struct
  let base = 0x00
  let size = 16
  let run = 0x0F (* bit 0; the board button also toggles it *)
  let channel = 0x0E
  let step_ms = 0x0C
  let gate_ms = 0x0A
  let velocity = 0x09
  let seed = 0x05 (* the PRNG loads it at the run start *)
  let midi_go = 0x04 (* write: send; read: 1 while a message waits *)
  let midi_len = 0x03
  let midi_msg = 0x00

  (** The range of the values that a register accepts. *)
  type bounds =
    { lower : int
    ; upper : int
    }

  (** One register: its name, the address of its first byte, the number of bytes, its
      power-on value, and its range when it has one. A value of more than one byte is
      little-endian.

      [bounds] is [None] for a cell that holds no scalar: RUN and MIDI_GO are bit fields
      and the circuit reads bit 0 alone; MIDI_MSG takes any bytes; GATE_MS takes any
      value, and one that is not less than STEP_MS only means that the gate never comes;
      STEP_MS takes any value, and 0 counts as 1; MIDI_LEN rings for 1 to 3 but rests at
      0, thus a range would not agree with its own power-on value. *)
  type field =
    { name : string
    ; address : int
    ; width : int
    ; default : int
    ; bounds : bounds option
    }

  (* Each register, from the first address upward. The RTL cells, their power-on values
     and the driver dump all read this one table. *)
  let fields =
    [ { name = "midi_msg"
      ; address = midi_msg
      ; width = Midi.max_message_bytes
      ; default = 0
      ; bounds = None
      }
    ; { name = "midi_len"; address = midi_len; width = 1; default = 0; bounds = None }
    ; { name = "midi_go"; address = midi_go; width = 1; default = 0; bounds = None }
    ; { name = "seed"
      ; address = seed
      ; width = 4
      ; default = Default.seed
      ; bounds = Some { lower = 1; upper = 0xFFFF_FFFF }
      }
    ; { name = "velocity"
      ; address = velocity
      ; width = 1
      ; default = Default.velocity
      ; bounds = Some { lower = 1; upper = 127 }
      }
    ; { name = "gate_ms"
      ; address = gate_ms
      ; width = 2
      ; default = Default.gate_ms
      ; bounds = None
      }
    ; { name = "step_ms"
      ; address = step_ms
      ; width = 2
      ; default = Default.step_ms
      ; bounds = None
      }
    ; { name = "channel"
      ; address = channel
      ; width = 1
      ; default = Default.channel
      ; bounds = Some { lower = 0; upper = 15 }
      }
    ; { name = "run"; address = run; width = 1; default = 0; bounds = None }
    ]
  ;;

  (* each default must agree with its own range: the table cannot state a power-on value
     that the same row forbids *)
  let () =
    List.iter fields ~f:(fun f ->
      Option.iter f.bounds ~f:(fun b ->
        assert (f.default >= b.lower && f.default <= b.upper)))
  ;;

  (* the table must cover each address of the section one time: this catches a width that
     does not agree with the addresses *)
  let () =
    let covered =
      List.concat_map fields ~f:(fun f -> List.init f.width ~f:(fun k -> f.address + k))
    in
    assert (List.length covered = size);
    assert (List.length (List.dedup_and_sort covered ~compare:Int.compare) = size);
    assert (List.for_all covered ~f:(fun a -> a >= base && a < base + size))
  ;;

  let field_at address = List.find_exn fields ~f:(fun f -> f.address = address)

  (* the number of bytes of the register at [address]; it raises when none starts there *)
  let width_of address = (field_at address).width

  (* the range of the register at [address], when it has one *)
  let bounds_of address = (field_at address).bounds
end

(** [build_doorbell message] is the address and the bytes of the one ascending write that
    sends [message] to the MIDI output: MIDI_MSG, then MIDI_LEN, then MIDI_GO with bit 0
    at 1. A write applies at one time, thus this one burst does the whole operation, and
    the layout of the burst has one definition — here, beside the addresses that make it.
    It raises [Invalid_argument] when [message] is not 1 to [Midi.max_message_bytes]
    bytes. *)
let build_doorbell message =
  let n = List.length message in
  if n < 1 || n > Midi.max_message_bytes
  then
    invalid_arg
      (Printf.sprintf "a test message has 1 to %d bytes, not %d" Midi.max_message_bytes n);
  let data = Bytes.make (Reg.midi_go - Reg.midi_msg + 1) '\x00' in
  List.iteri message ~f:(fun k b -> set_byte data k b);
  set_byte data (Reg.midi_len - Reg.midi_msg) n;
  set_byte data (Reg.midi_go - Reg.midi_msg) 1;
  Reg.midi_msg, data
;;
