open Base

module Event = struct
  type t =
    | On of int
    | Off of int
  [@@deriving sexp_of]
end

(* one voice of the performance: its re-strike policy, and the note that it holds *)
module Voice = struct
  type t =
    { restrike : bool
    ; opened : int option
    }
end

type t =
  { walk : Pink.walk
  ; voices : Voice.t list (* the highest voice first, as the states come *)
  }

let create ~(model : Pink.t) ~seed =
  { walk = Pink.create ~model ~seed
  ; voices =
      List.map model.voices ~f:(fun v ->
        { Voice.restrike = v.Pink.Voice.restrike; opened = None })
  }
;;

(* a voice speaks when it is due and it holds no note, or its pitch moved, or its policy
   re-strikes a held pitch *)
let speaks (state : Pink.state) ({ restrike; opened } : Voice.t) =
  state.due
  &&
  match opened with
  | None -> true
  | Some note -> note <> state.note || restrike
;;

(* a voice closes the note that it holds, and it holds none after that *)
let close (voice : Voice.t) =
  ( { voice with Voice.opened = None }
  , Option.to_list (Option.map voice.opened ~f:(fun note -> Event.Off note)) )
;;

(* each rule gives a new voice and the events of that voice; the events come out from the
   highest voice downward, the order of the wire *)
let collect pairs = List.map pairs ~f:fst, List.concat_map pairs ~f:snd

let step t =
  let walk, states = Pink.next_step t.walk in
  let voices, events =
    collect
      (List.map2_exn states t.voices ~f:(fun state (voice : Voice.t) ->
         if speaks state voice
         then (
           let closed, off = close voice in
           ( { closed with Voice.opened = Some state.Pink.note }
           , off @ [ Event.On state.Pink.note ] ))
         else voice, []))
  in
  { walk; voices }, events
;;

let gate t =
  (* the highest voice is the head of the list, and the gate closes no other *)
  match t.voices with
  | [] -> t, []
  | top :: rest ->
    let top, events = close top in
    { t with voices = top :: rest }, events
;;

let stop t =
  let voices, events = collect (List.map t.voices ~f:close) in
  { t with voices }, events
;;

(* the events of the player are the piece: at step 1 the four voices enter from the
   soprano, and after that the low voices are silent until they move *)
let%expect_test "the events of the first steps, with the gate" =
  let player = ref (create ~model:Pink.default ~seed:Control_intf.Default.seed) in
  let show label events =
    if not (List.is_empty events)
    then
      Stdio.printf
        "%s %s\n"
        label
        (String.concat
           ~sep:" "
           (List.map events ~f:(function
             | Event.On note -> Printf.sprintf "on %d" note
             | Event.Off note -> Printf.sprintf "off %d" note)))
  in
  for number = 1 to 9 do
    let t, struck = step !player in
    let t, closed = gate t in
    player := t;
    show (Printf.sprintf "%2d  " number) struck;
    show "    gate" closed
  done;
  let _, stopped = stop !player in
  show "    stop" stopped;
  [%expect
    {|
    1   on 88 on 62 on 57 on 45
       gate off 88
    2   on 84
       gate off 84
    3   on 84
       gate off 84
    4   on 88 off 62 on 67
       gate off 88
    5   on 93
       gate off 93
    6   on 91
       gate off 91
    7   on 93
       gate off 93
    8   on 74 off 67 on 62
       gate off 74
    9   on 76
       gate off 76
       stop off 62 off 57 off 45
    |}]
;;
