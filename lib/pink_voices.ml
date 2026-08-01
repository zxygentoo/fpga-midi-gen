open Base

type state =
  { note : int
  ; due : bool
  }

(* The groups from row 0 upward: the treble takes the fast rows, the bass the slow. Each
   spec is a [Pink.Params] over its group's rows. The registers are disjoint — no two
   voices can ever hold one pitch: a Note Off on the one channel releases a voice by
   pitch, thus a shared pitch lets one voice's off silence another. *)
let groups : Pink.Params.t list =
  [ { Pink.Params.rows = 3
    ; root = 60
    ; degrees = 11
    ; scale = Pink.Params.pentatonic
    ; stretch = 2
    }
  ; { Pink.Params.rows = 3
    ; root = 48
    ; degrees = 5
    ; scale = Pink.Params.pentatonic
    ; stretch = 2
    }
  ; { Pink.Params.rows = 2
    ; root = 36
    ; degrees = 5
    ; scale = Pink.Params.pentatonic
    ; stretch = 2
    }
  ]
;;

let total_rows = List.sum (module Int) groups ~f:(fun g -> g.rows)
let voices = List.length groups

type t =
  { prng : Pink.Prng.t
  ; rows : int list (* the row values; the head is row 0 *)
  ; step : int
  }

let create ~seed =
  let prng = Pink.Prng.create ~seed in
  let prng, rows =
    Pink.reroll prng (List.init total_rows ~f:(Fn.const 0)) ~count:total_rows
  in
  { prng; rows; step = 0 }
;;

let next_step t =
  let step = t.step + 1 in
  let count = Int.min total_rows (Int.ctz step + 1) in
  let prng, rows = Pink.reroll t.prng t.rows ~count in
  (* walk the groups from row 0: a group is due when the re-roll count reaches into it *)
  let states =
    List.folding_map groups ~init:(0, rows) ~f:(fun (start, remaining) params ->
      let mine = List.take remaining params.rows in
      let rest = List.drop remaining params.rows in
      let sum = List.sum (module Int) mine ~f:Fn.id in
      ( (start + params.rows, rest)
      , { note = Pink.mapper params sum; due = step = 1 || count > start } ))
  in
  { prng; rows; step }, List.rev states
;;

let%expect_test "the first steps of the power-on seed" =
  let t = ref (create ~seed:Control.Default.seed) in
  Stdio.printf "step   bass    mid    treble\n";
  for step = 1 to 10 do
    let t', states = next_step !t in
    t := t';
    Stdio.printf
      "%4d %s\n"
      step
      (String.concat
         (List.map states ~f:(fun s ->
            Printf.sprintf "  %3d%s" s.note (if s.due then "*" else " "))))
  done;
  [%expect
    {|
    step   bass    mid    treble
       1    45*   57*   67*
       2    45    57    64*
       3    45    57    64*
       4    45    57    79*
       5    45    57    84*
       6    45    57    81*
       7    45    57    84*
       8    45    57*   60*
       9    45    57    60*
      10    45    57    67*
    |}]
;;

let%expect_test "the articulation grid is the ctz schedule" =
  let t = ref (create ~seed:7) in
  let due_steps = Array.create ~len:3 [] in
  for step = 1 to 128 do
    let t', states = next_step !t in
    t := t';
    List.iteri states ~f:(fun k s -> if s.due then due_steps.(k) <- step :: due_steps.(k))
  done;
  let count k = List.length due_steps.(k) in
  Stdio.printf
    "bass %d times (every 64), mid %d times (every 8), treble %d times (every step)\n"
    (count 0)
    (count 1)
    (count 2);
  Stdio.printf
    "bass at steps %s\n"
    (String.concat ~sep:" " (List.rev_map due_steps.(0) ~f:Int.to_string));
  [%expect
    {|
    bass 3 times (every 64), mid 17 times (every 8), treble 128 times (every step)
    bass at steps 1 64 128
    |}]
;;
