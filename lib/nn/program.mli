(** L3: a step-frame program, and the compiler that folds it into cases of a program
    counter.

    THE MATHEMATICS IS A PROGRAM, NOT A STATE MACHINE. A step of a step-frame era is a
    list of operations — the forward pass, then the chain that draws one seat — and this
    module turns that list into the cases the counter runs. The two frozen eras wrote this
    text twice; it stands here once, and their five-layer records say so.

    WHAT IS AN ERA'S AND STAYS THERE: its op vocabulary, its [schedule] that builds the
    program, its [build] that gives each op an entry and a body, its [Cost] model, and L4
    — the rewind and the step strobe. L4 is where the two eras genuinely differ (era
    four's rewind clears the ring's slot and its filled flag), and ten lines of difference
    do not want a shared text with a special case in it.

    NOTHING HERE DECLARES A REGISTER OR A WIRE. Hardcaml names an unnamed signal
    [signal_<op>_N] by the ORDER OF ITS CREATION, thus a declaration moved into this
    module would create its signal at another point of the elaboration and rename every
    signal after it. The era declares [pc], [seat], its state machine and its registers,
    and passes them in. [Sampler] holds the same rule and for the same reason. *)

open Hardcaml

(** A step as a program: the forward pass, then the chain. The op is the era's own type —
    the two vocabularies only rhyme, and the rule of this library keeps rhyme in the eras
    — thus this record is polymorphic in it and reads none of it. *)
type 'op t =
  { chain : 'op list (** the ops of ONE seat; the compiler runs them once for each *)
  ; forward : 'op list (** the ops of the forward pass, once for a step *)
  }

(** The outer machine's two states. It stands here because [State_machine.create] reads
    the type and both eras stated the same two. *)
module State : sig
  type t =
    | Idle
    | Run
  [@@deriving compare ~localize, enumerate, sexp_of]
end

(** what [compile] gives back: the actions that OPEN a step, and the body the machine runs
    while the step is in flight *)
type compiled =
  { forward_entry : Always.t list
  (** the entry of the forward's first op — L4 runs this on the step strobe *)
  ; run_body : Always.t list (** the parallel case over the program counter *)
  }

(** [chain_over counter bodies] is a BESPOKE CHAIN: body [k] runs when [counter] reads
    [k], and the counter steps itself. A body states only its work and the position in the
    list states its time. The last body owns what follows it, because a chain ends either
    on a finish or on a return to 0. The counter's own width is the width of the cases,
    thus this states no number and two eras with two tick widths share it. *)
val chain_over : Always.Variable.t -> Always.t list list -> Always.t

(** [case_over counter bodies] is a PARALLEL CASE: body [k] runs when [counter] reads [k],
    and the counter does not move. The stages of a multi-stage op take this form, because
    they wait on different things — a walk, a unit, a RAM — and each moves the counter
    when its own wait is over. *)
val case_over : Always.Variable.t -> Always.t list list -> Always.t

(** [compile ~build ~pc ~seat ~idle ~forward_done prog] folds [prog] into one parallel
    case of [pc].

    AN OP'S FINISH RUNS THE NEXT OP'S ENTRY IN THE SAME CYCLE, and the pc moves with it,
    thus an op initializes its own counters and its predecessor is what runs that entry.
    [build op ~finish] gives that pair: the entry actions, and the body that runs while
    the op holds the counter.

    THE CHAIN IS ONE SEAT AND IT RUNS ONCE FOR EACH, counting down from the soprano. Its
    last op returns to its first until the bass has drawn, thus the loop closes on the op
    boundary and costs no cycle of its own. The head's entry is taken from a build of its
    own: an op's entry does not depend on its finish, thus building the head twice states
    the loop back with no circular definition, and the bodies of that second build reach
    no output and leave the circuit.

    [idle] is the era's "go idle" — a THUNK, because the era's [sm.set_next Idle] makes a
    constant when it is called and the era called it inside the chain's last op, not
    before the head was built. [forward_done ~enter_chain] is what the era runs when its
    forward pass ends: it steps the era's own step counter, moves whatever the era's
    datapath moves at a step boundary (era four's ring slot and filled flag), and either
    enters the chain or goes idle — the lead-in is that choice. *)
val compile
  :  build:('op -> finish:Always.t list -> Always.t list * Always.t list)
  -> pc:Always.Variable.t
  -> seat:Always.Variable.t
  -> idle:(unit -> Always.t list)
  -> forward_done:(enter_chain:Always.t list -> Always.t list)
  -> 'op t
  -> compiled
