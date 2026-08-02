open Base

module Event = struct
  type t =
    | On of int
    | Off of int
  [@@deriving sexp_of]
end

type t =
  { walk : Pink.walk
  ; restrikes : bool list (* the lowest voice first, as the states come *)
  ; open_notes : int option list (* the note that each voice holds, or none *)
  }

let create ~(model : Pink.t) ~seed =
  { walk = Pink.create ~model ~seed
  ; restrikes = List.rev_map model.voices ~f:(fun v -> v.Pink.Voice.restrike)
  ; open_notes = List.map model.voices ~f:(fun _ -> None)
  }
;;

(* a voice speaks when it is due and it holds no note, or its pitch moved, or its policy
   re-strikes a held pitch *)
let speaks (state : Pink.state) ~restrike ~opened =
  state.due
  &&
  match opened with
  | None -> true
  | Some note -> note <> state.note || restrike
;;

let step t =
  let walk, states = Pink.next_step t.walk in
  let voices =
    List.map3_exn states t.restrikes t.open_notes ~f:(fun state restrike opened ->
      if speaks state ~restrike ~opened
      then (
        let off = Option.to_list (Option.map opened ~f:(fun note -> Event.Off note)) in
        Some state.Pink.note, off @ [ Event.On state.Pink.note ])
      else opened, [])
  in
  { t with walk; open_notes = List.map voices ~f:fst }, List.concat_map voices ~f:snd
;;

let highest t = List.length t.open_notes - 1

let gate t =
  let top = highest t in
  ( { t with
      open_notes =
        List.mapi t.open_notes ~f:(fun k note -> if k = top then None else note)
    }
  , match List.nth t.open_notes top with
    | Some (Some note) -> [ Event.Off note ]
    | Some None | None -> [] )
;;

let stop t =
  ( { t with open_notes = List.map t.open_notes ~f:(fun _ -> None) }
  , List.filter_map t.open_notes ~f:(Option.map ~f:(fun note -> Event.Off note)) )
;;

(* the events of the player are the piece: at step 1 the four voices enter from the bass,
   and after that the low voices are silent until they move *)
let%expect_test "the events of the first steps, with the gate" =
  let player = ref (create ~model:Pink.default ~seed:Control.Default.seed) in
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
     1   on 45 on 57 on 62 on 88
        gate off 88
     2   on 84
        gate off 84
     3   on 84
        gate off 84
     4   off 62 on 67 on 88
        gate off 88
     5   on 93
        gate off 93
     6   on 91
        gate off 91
     7   on 93
        gate off 93
     8   off 67 on 62 on 74
        gate off 74
     9   on 76
        gate off 76
        stop off 45 off 57 off 62
    |}]
;;
