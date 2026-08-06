open Core
module Json = Yojson.Safe

type chorale = int list array

type t =
  { train : chorale list
  ; valid : chorale list
  ; test : chorale list
  }

let default_path = "corpus/JSB-Chorales-dataset/jsb-chorales-16th.json"
let bar_steps = 16

(* the pitch-0 rule of the design document; the JSB corpus never takes this path *)
let escape_zero_pitch pitch = if pitch = 0 then 1 else pitch

let pitch_of_json = function
  | `Int pitch -> escape_zero_pitch pitch
  | `Float pitch -> escape_zero_pitch (Int.of_float pitch)
  | json -> invalid_argf "a pitch is not a number: %s" (Json.to_string json) ()
;;

let step_of_json json =
  json
  |> Json.Util.to_list
  |> List.map ~f:pitch_of_json
  |> List.dedup_and_sort ~compare:Int.compare
;;

let chorale_of_json json =
  json |> Json.Util.to_list |> List.map ~f:step_of_json |> Array.of_list
;;

let load ~path =
  let json = Json.from_file path in
  let chorales name =
    json |> Json.Util.member name |> Json.Util.to_list |> List.map ~f:chorale_of_json
  in
  { train = chorales "train"; valid = chorales "valid"; test = chorales "test" }
;;

let transpose ~by chorale =
  let transpose_pitch pitch =
    match pitch + by with
    | moved when moved < 1 || moved > 127 ->
      invalid_argf "pitch %d moved by %d leaves 1 to 127" pitch by ()
    | transposed -> transposed
  in
  let transpose_step step = List.map step ~f:transpose_pitch in
  Array.map chorale ~f:transpose_step
;;

(* the lowest and the highest sounding pitch, or [None] for silence *)
let chorale_pitch_range chorale =
  let aux_pitch acc pitch =
    match acc with
    | None -> Some (pitch, pitch)
    | Some (low, high) -> Some (min low pitch, max high pitch)
  in
  let aux_step acc step = List.fold step ~init:acc ~f:aux_pitch in
  Array.fold chorale ~init:None ~f:aux_step
;;

let pitch_range chorales =
  let widen acc chorale =
    match acc, chorale_pitch_range chorale with
    | range, None | None, range -> range
    | Some (low, high), Some (low', high') -> Some (min low low', max high high')
  in
  match List.fold chorales ~init:None ~f:widen with
  | None -> invalid_arg "pitch_range: no pitch sounds in the chorales"
  | Some range -> range
;;

let legal_shifts ~within:(low, high) chorale =
  match chorale_pitch_range chorale with
  | None -> [ 0 ]
  | Some (lowest, highest) -> List.range ~stop:`inclusive (low - lowest) (high - highest)
;;

(* One step, one sentence: the OFFs of the pitches that stop, the ONs of the pitches that
   start, then [End]. A pitch in both neighbour steps is a held note and takes no token.
   [Set.to_list] gives the ascending order of the canonical sentence, thus one chord has
   one sentence and not a permutation family. *)
let tokenize chorale =
  let sentence ~previous ~current =
    let on pitch = Token.On pitch in
    let off pitch = Token.Off pitch in
    let to_set data = Set.of_list (module Int) data in
    let to_tokens ~from ~minus type_ =
      Set.diff (to_set from) (to_set minus) |> Set.to_list |> List.map ~f:type_
    in
    to_tokens ~from:previous ~minus:current off
    @ to_tokens ~from:current ~minus:previous on
    @ [ Token.End ]
  in
  let aux previous current = current, sentence ~previous ~current in
  chorale |> Array.to_list |> List.folding_map ~init:[] ~f:aux |> List.concat
;;

(* The cadential holds of one chorale, from the corpus study of 2026-08-06: a sonority
   that rings six steps or more marks a cadence, and the cadences sit on the downbeats.
   The result is the start step of each hold, ascending. *)
let cadential_holds chorale =
  let sonorities = Array.to_list (Array.map chorale ~f:(Set.of_list (module Int))) in
  let runs = List.group sonorities ~break:(fun a b -> not (Set.equal a b)) in
  let (_ : int), holds =
    List.fold_map runs ~init:0 ~f:(fun start run ->
      let cadential = (not (Set.is_empty (List.hd_exn run))) && List.length run >= 6 in
      start + List.length run, if cadential then Some start else None)
  in
  List.filter_opt holds
;;

(* The vote of one bar length: the rotation that puts the most holds on a downbeat, with
   the smallest rotation on a tie. The lift — hits times bar length — normalizes the two
   bar lengths against their chance rates, thus the votes of 12 and 16 compare. *)
let vote holds bar =
  List.map (List.range 0 bar) ~f:(fun rotation ->
    let hits = List.count holds ~f:(fun hold -> (hold - rotation) mod bar = 0) in
    hits * bar, rotation)
  |> List.max_elt ~compare:(fun (lift, _) (lift', _) -> Int.compare lift lift')
  |> Option.value_exn
;;

(* Fewer than three holds gives no signal: the piece keeps the plain sixteen-step grid. *)
let metre chorale =
  let holds = cadential_holds chorale in
  if List.length holds < 3
  then bar_steps, 0
  else (
    let lift12, rotation12 = vote holds 12 in
    let lift16, rotation16 = vote holds 16 in
    if lift16 >= lift12 then 16, rotation16 else 12, rotation12)
;;

(* The leading silence: one bar of empty steps before the music, thus the walk begins with
   bare END sentences and the model learns how a piece starts after silence. The cleared
   context of the sampler then boots inside the training distribution. The phases align to
   the estimated downbeats and run without a seam through the silent bar into the music. *)
let encode ~lead_bars chorale =
  let bar, rotation = metre chorale in
  let padded = Array.append (Array.create ~len:(lead_bars * bar) []) chorale in
  let tokens = tokenize padded in
  let _, phases =
    List.fold_map tokens ~init:0 ~f:(fun step token ->
      let next =
        match token with
        | Token.End -> step + 1
        | Off _ | On _ -> step
      in
      next, (((step - rotation) mod bar) + bar) mod bar)
  in
  ~codes:(Array.of_list_map tokens ~f:Token.to_byte), ~phases:(Array.of_list phases)
;;

let%expect_test "the legal shifts of the range-limited policy" =
  let chorale = [| [ 60; 64; 67 ]; []; [ 59 ] |] in
  print_s ([%sexp_of: int * int] (pitch_range [ chorale ]));
  [%expect {| (59 67) |}];
  print_s ([%sexp_of: int list] (legal_shifts ~within:(55, 70) chorale));
  [%expect {| (-4 -3 -2 -1 0 1 2 3) |}];
  print_s ([%sexp_of: int list] (legal_shifts ~within:(60, 61) chorale));
  [%expect {| () |}]
;;

let%expect_test "the walk of a small chorale" =
  (* a chord, a hold, a move of two voices, a rest, then a unison *)
  let chorale = [| [ 67; 64; 60 ]; [ 67; 64; 60 ]; [ 67; 65; 62 ]; []; [ 60; 60 ] |] in
  print_s ([%sexp_of: Token.t list] (tokenize chorale));
  [%expect
    {|
    ((On 60) (On 64) (On 67) End End (Off 60) (Off 64) (On 62) (On 65) End
     (Off 62) (Off 65) (Off 67) End (On 60) End)
    |}];
  let ~codes, ~phases = encode ~lead_bars:1 chorale in
  print_s ([%sexp_of: int array] codes);
  print_s ([%sexp_of: int array] phases);
  [%expect
    {|
    (0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 188 192 195 0 0 60 64 190 193 0 62 65 67 0
     188 0)
    (0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0 0 1 2 2 2 2 2 3 3 3 3 4 4)
    |}]
;;

let%expect_test "the metre from the cadential holds" =
  (* a bar = a six-step hold on the downbeat, then moving single notes *)
  let hold = Array.create ~len:6 [ 60; 64; 67 ] in
  let motion count = Array.init count ~f:(fun i -> [ 40 + i ]) in
  let bar_44 = Array.append hold (motion 10) in
  let bar_34 = Array.append hold (motion 6) in
  (* three 16-step bars: common time, downbeats at 0, 16, 32 *)
  let common = Array.concat [ bar_44; bar_44; bar_44 ] in
  print_s ([%sexp_of: int * int] (metre common));
  [%expect {| (16 0) |}];
  (* three 12-step bars: a waltz, downbeats at 0, 12, 24 *)
  let waltz = Array.concat [ bar_34; bar_34; bar_34 ] in
  print_s ([%sexp_of: int * int] (metre waltz));
  [%expect {| (12 0) |}];
  (* a four-step pickup moves every downbeat: the rotation follows *)
  let pickup = Array.append (motion 4) common in
  print_s ([%sexp_of: int * int] (metre pickup));
  [%expect {| (16 4) |}];
  (* two holds are no signal: the plain grid *)
  let short = Array.concat [ bar_44; bar_44 ] in
  print_s ([%sexp_of: int * int] (metre short));
  [%expect {| (16 0) |}]
;;
