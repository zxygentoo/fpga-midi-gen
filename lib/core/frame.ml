open Core

module Event = struct
  type t =
    | On of int
    | Off of int
  [@@deriving sexp_of]
end

let voices = 4
let silent = 0
let silent_code = 0x00
let sounds_flag = 0x80
let pitch_mask = 0x7F
let code_bits = 8

let code_of_pitch pitch =
  if pitch < 0
  then silent_code
  else if pitch > pitch_mask
  then invalid_argf "the pitch %d is outside 0 to 127" pitch ()
  else sounds_flag lor pitch
;;

let pitch_of_code code =
  if code land sounds_flag = 0 then None else Some (code land pitch_mask)
;;

let of_codes codes =
  if List.length codes <> voices
  then invalid_argf "a frame takes %d codes, one for each seat" voices ();
  (* seat 0 is the low byte, thus the fold walks the seats downward *)
  List.rev codes
  |> List.fold ~init:0 ~f:(fun frame code -> (frame lsl code_bits) lor code)
;;

let codes frame =
  List.init voices ~f:(fun seat -> (frame lsr (code_bits * seat)) land 0xFF)
;;

let sounding_set frame =
  codes frame |> List.filter_map ~f:pitch_of_code |> Set.of_list (module Int)
;;

let pitches frame = Set.to_list (sounding_set frame)

let events_of_frames frames =
  let step sounding frame =
    let wanted = sounding_set frame in
    ( wanted
    , List.map (Set.to_list (Set.diff sounding wanted)) ~f:(fun p -> Event.Off p)
      @ List.map (Set.to_list (Set.diff wanted sounding)) ~f:(fun p -> Event.On p) )
  in
  Array.to_list frames |> List.folding_map ~init:(Set.empty (module Int)) ~f:step
;;

let%expect_test "a voice code carries its pitch, and a rest carries none" =
  let show pitch = printf "%3d -> %02x -> %s\n" pitch (code_of_pitch pitch) in
  show (-1) (Sexp.to_string ([%sexp_of: int option] (pitch_of_code (code_of_pitch (-1)))));
  show 0 (Sexp.to_string ([%sexp_of: int option] (pitch_of_code (code_of_pitch 0))));
  show 60 (Sexp.to_string ([%sexp_of: int option] (pitch_of_code (code_of_pitch 60))));
  show 127 (Sexp.to_string ([%sexp_of: int option] (pitch_of_code (code_of_pitch 127))));
  [%expect
    {|
     -1 -> 00 -> ()
      0 -> 80 -> (0)
     60 -> bc -> (60)
    127 -> ff -> (127)
    |}];
  (* the pitch field reserves nothing: 0 and 127 are both music the wire can state *)
  printf "%d\n" silent;
  [%expect {| 0 |}]
;;

let%expect_test "a frame packs its seats, seat 0 in the low byte" =
  let frame pitches = of_codes (List.map pitches ~f:code_of_pitch) in
  (* seat 0 first, thus this list reads low to high: bass, tenor, alto, soprano *)
  printf "%08x\n" (frame [ 48; 55; 64; 72 ]);
  [%expect {| c8c0b7b0 |}];
  printf "%08x\n" (frame [ -1; -1; -1; -1 ]);
  [%expect {| 00000000 |}];
  (* the round trip: the codes come back in the order they went in *)
  print_s ([%sexp_of: int list] (codes (frame [ 48; -1; 64; 72 ])));
  [%expect {| (176 0 192 200) |}];
  (* a unison is two seats and one pitch on the wire *)
  print_s ([%sexp_of: int list] (pitches (frame [ 48; 60; 60; 72 ])));
  [%expect {| (48 60 72) |}]
;;

let%expect_test "the frames of a stream become its events" =
  let frame pitches = of_codes (List.map pitches ~f:code_of_pitch) in
  let silent = frame [ -1; -1; -1; -1 ] in
  let show name frames =
    printf
      "%-14s %s\n"
      name
      (Sexp.to_string
         ([%sexp_of: Event.t list list] (events_of_frames (Array.of_list frames))))
  in
  (* the eight cases of the decode table of docs/transformer.md *)
  show "hold" [ frame [ 60; -1; -1; -1 ]; frame [ 60; -1; -1; -1 ] ];
  show "strike" [ silent; frame [ 60; -1; -1; -1 ] ];
  show "release" [ frame [ 60; -1; -1; -1 ]; silent ];
  show "move" [ frame [ 60; -1; -1; -1 ]; frame [ 62; -1; -1; -1 ] ];
  show "exchange" [ frame [ -1; 60; 64; -1 ]; frame [ -1; 64; 60; -1 ] ];
  show "unison in" [ frame [ -1; 60; 64; -1 ]; frame [ -1; 60; 60; -1 ] ];
  show "unison out" [ frame [ -1; 60; 60; -1 ]; frame [ -1; 60; 64; -1 ] ];
  show "seam" [ frame [ 48; 55; 60; 64 ]; silent ];
  [%expect
    {|
    hold           (((On 60))())
    strike         (()((On 60)))
    release        (((On 60))((Off 60)))
    move           (((On 60))((Off 60)(On 62)))
    exchange       (((On 60)(On 64))())
    unison in      (((On 60)(On 64))((Off 64)))
    unison out     (((On 60))((On 64)))
    seam           (((On 48)(On 55)(On 60)(On 64))((Off 48)(Off 55)(Off 60)(Off 64)))
    |}]
;;

let%expect_test "the rule holds its three properties over a drawn stream" =
  (* the properties the legality mask used to carry, now held by the rule itself *)
  let state = ref 12345 in
  let next () =
    state := (!state * 1103515245) + 12345;
    (!state lsr 16) land 0x7FFF
  in
  let drawn =
    Array.init 4096 ~f:(fun (_ : int) ->
      of_codes
        (List.init voices ~f:(fun (_ : int) ->
           let draw = next () % 20 in
           if draw < 4 then silent_code else code_of_pitch (48 + (draw % 16)))))
  in
  let after_event (strikes, sounding) = function
    | Event.On pitch ->
      if Set.mem sounding pitch then failwith "a strike of a pitch that sounds";
      strikes + 1, Set.add sounding pitch
    | Event.Off pitch ->
      if not (Set.mem sounding pitch)
      then failwith "a release of a pitch that does not sound";
      strikes, Set.remove sounding pitch
  in
  let after_step state events =
    let strikes, sounding = List.fold events ~init:state ~f:after_event in
    if Set.length sounding > voices then failwith "five notes sound at the same time";
    strikes, sounding
  in
  let strikes =
    List.fold (events_of_frames drawn) ~init:(0, Set.empty (module Int)) ~f:after_step
    |> fst
  in
  printf "%d strikes, and the three properties hold at every step\n" strikes;
  [%expect {| 9876 strikes, and the three properties hold at every step |}]
;;
