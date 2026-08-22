(** The float rules both era references share: the frame's way into the network, the norm,
    the softmax, the ALiBi slope, the feed-forward, and the whole chained head — its
    logits, its loss and its draw.

    Both eras' float models are references for their integer twins, and the parts here are
    the parts where the two models were one function written twice. What stays in the era
    files is the trunk — era four's windowed attention, era five's recurrence and ring —
    and the walks over it. A function here must keep its exact arithmetic: the drift
    reports and the seed-walk gates pin float values, thus a reorder of one product
    re-records every one of them.

    The head in one paragraph: four voice classes enter through four tables that sum
    ([embedding]), and they leave through the same four tables in a chain from the soprano
    down ([seat_logits], [loss], [draw_frame]) — each seat reading the stream the seats
    above it have written, because a chord is a joint choice and not four independent
    ones. *)

(** every tensor of the float models is float32 *)
type tensor = Checkpoint.tensor

(** [seat_classes frames] is the classes of a batch of frames, apart by seat: the result
    at [seat] holds one row of classes for each walk. Four tables read four rows, thus a
    pass wants the seats apart and never the frame whole. *)
val seat_classes : int array array -> int array array array

(** RMSNorm with no scale, over the last axis: the trainers of both eras fold the scale
    away. *)
val rms_norm : tensor -> tensor

(** [softmax x ~axis] is the softmax along [axis], shifted by the peak. *)
val softmax : tensor -> axis:int -> tensor

(** [alibi_slope ~span ~heads ~head] is the (negative) ALiBi slope of one head,
    [-(2 ** -(span (head+1) / heads))]: the recency prior both eras' attention layers
    subtract, times the distance. The integer twins state the same rule as a shift, in
    [Quantized.Constants.slope_exponent]. *)
val alibi_slope : span:int -> heads:int -> head:int -> float

(** [feed_forward ~w1 ~w2 y] is era four's position-wise feed-forward, [relu (y w1) w2].
    The norm before it is the caller's, because the eras place it differently around their
    residual joins. *)
val feed_forward : w1:tensor -> w2:tensor -> tensor -> tensor

(** [embedding ~seats ~phase ~classes ~phases] is the input of one step: the four seat
    rows sum, and the bar phase adds to them. Four tables and not one shared table with a
    voice tag — the sum of four identical tags is a bias that carries nothing, and the
    voices would be thrown away on the way in; the implementation states the whole
    argument. *)
val embedding
  :  seats:tensor
  -> phase:tensor
  -> classes:int array array array
  -> phases:int array array
  -> tensor

(** [seat_logits ~seats h ~drawn] is the chained head: the logits of each seat over the
    classes, paired with its seat, the soprano first. [drawn] holds the classes the chain
    conditions on — the true frame in training, the drawn seats at the draw; only seats 3,
    2 and 1 are read. *)
val seat_logits
  :  seats:tensor
  -> tensor
  -> drawn:int array array array
  -> (int * tensor) list

(** [loss ~seats ~hidden ~windows] is the cross entropy of the frames of [windows] in nats
    for each step — the sum over the four seats, and the mean over the steps. [hidden] is
    the era's trunk: the residual stream of every position of a batch of windows. It
    raises [Invalid_argument] when [windows] is empty. *)
val loss
  :  seats:tensor
  -> hidden:(classes:int array array array -> phases:int array array -> tensor)
  -> windows:Jsb.stream list
  -> float

(** [draw_frame ~seats ~d ~temperature ~min_p ~rng stream] is one step of the chained
    draw, on the host: the soprano first, each seat under it reading the stream the seats
    above have written, every uniform from [Prng] through [Policy.draw_class]. It gives
    the classes of the frame, seat 0 first. [stream] is one row of [d]. *)
val draw_frame
  :  seats:tensor
  -> d:int
  -> temperature:float
  -> min_p:float
  -> rng:Prng.state
  -> tensor
  -> Prng.state * int list
