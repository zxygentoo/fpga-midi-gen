(** The host control: the contract between the host drivers and the FPGA.

    This module is the single definition of the host-control constants — the sizes, the
    operations, the status codes and the control registers. The RTL reads them at
    elaboration time and the drivers read the same ones, thus a value has one home.
    [Control_frame] is the wire codec that carries them, and the normative description is
    [docs/host_control.md].

    The module holds only the definitions of the contract, thus it has no [.mli]: the
    definitions are their own signature. *)

open Base

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

(** The values a driver takes when a person names none. [Reg.fields] gives the power-on
    value of each cell, and the two agree for every cell but SEED: the slide switches
    write SEED at the power-on, thus the board has no seed of its own and [seed] below
    serves the host tools alone. *)
module Default = struct
  (* MIDI channel 3, the S-1 factory default. *)
  let channel = 2

  (* One step is a sixteenth, thus 200 ms puts the quarter at exactly 75 — the common
     chorale tempo. The pink era booted at 250; the transformer era's ear asked for this. *)
  let step_ms = 200
  let velocity = 100

  (* not 0: that seed holds the walk still and the piece is one chord *)
  let seed = 42
end

(** The control registers. They are the local storage of the control unit, thus one byte
    of address names each one. *)
module Reg = struct
  let base = 0x00
  let size = 9
  let run = 0x08 (* bit 0; the board button also toggles it *)
  let channel = 0x07
  let step_ms = 0x05
  let velocity = 0x04
  let seed = 0x00 (* the PRNG loads it at the run start *)

  (** The range of the values that a register accepts. *)
  type bounds =
    { lower : int
    ; upper : int
    }

  (** One register: its name, the address of its first byte, the number of bytes, its
      power-on value, and its range when it has one. A value of more than one byte is
      little-endian.

      [bounds] is [None] for a cell that holds no scalar and for a cell that takes its
      whole range: RUN is a bit field and the circuit reads bit 0 alone, STEP_MS takes any
      value because 0 counts as 1, and SEED takes any value because the slide switches can
      set 0 and the board accepts it. *)
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
    [ (* The slide switches write this cell a few cycles after the power-on, thus its
         stored default is never the seed of a run and it is 0. [Default.seed] is the seed
         a host tool takes when a person names none, which is another question. *)
      { name = "seed"; address = seed; width = 4; default = 0; bounds = None }
    ; { name = "velocity"
      ; address = velocity
      ; width = 1
      ; default = Default.velocity
      ; bounds = Some { lower = 1; upper = 127 }
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
