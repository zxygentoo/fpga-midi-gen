(** A byte memory of [Config.size] cells with one shared address port. The block knows
    nothing of its content: the defaults and the cell semantics live in the blocks that
    use it. The cells are 0 at power-on and at clear.

    The cells are flip-flops and the read is combinational: the right shape for tens of
    cells. A large memory (for example the model window) needs a block RAM and a
    registered read — a different contract. *)

open Hardcaml

module Make (Config : sig
    val size : int
  end) : sig
  module I : sig
    type 'a t =
      { clock : 'a
      ; clear : 'a
      ; write_enable : 'a (** writes [write_data] to the cell [address] *)
      ; address : 'a (** the cell index; the width follows [Config.size] *)
      ; write_data : 'a (** the byte to write *)
      }
    [@@deriving hardcaml]
  end

  module O : sig
    type 'a t =
      { read_data : 'a
      (** the value of the cell [address]; the read is combinational, and during a write
          it is the old value *)
      }
    [@@deriving hardcaml]
  end

  val create : Signal.t I.t -> Signal.t O.t
end
