open Hardcaml
open Signal

type t =
  { rd_addr : Signal.t
  ; tx_data : Signal.t
  ; tx_valid : Signal.t
  ; busy : Signal.t
  }

(* the FSM states *)
let s_idle = 0
let s_scan_req = 1
let s_scan_eval = 2
let s_code = 3
let s_data_req = 4
let s_data_latch = 5
let s_data_send = 6
let s_delim = 7

let create ~clock ~clear ~start ~length ~rd_data ~tx_busy =
  let spec = Reg_spec.create ~clock ~clear () in
  let open Always in
  let state = Variable.reg spec ~width:3 in
  let len = Variable.reg spec ~width:7 in
  let pos = Variable.reg spec ~width:7 in
  let scan_j = Variable.reg spec ~width:7 in
  let group_end = Variable.reg spec ~width:7 in
  let hit_zero = Variable.reg spec ~width:1 in
  let byte_r = Variable.reg spec ~width:8 in
  let tx_byte = Variable.wire ~default:(zero 8) () in
  let tx_stb = Variable.wire ~default:gnd () in
  let in_state k = state.value ==:. k in
  let goto k = state <--. k in
  let send byte next =
    proc [ tx_byte <-- byte; when_ ~:tx_busy [ tx_stb <-- vdd; proc next ] ]
  in
  let after_group =
    if_
      hit_zero.value
      [ pos <-- group_end.value +:. 1; scan_j <-- group_end.value +:. 1; goto s_scan_req ]
      [ goto s_delim ]
  in
  compile
    [ when_
        (in_state s_idle &: start)
        [ len <-- length; pos <--. 0; scan_j <--. 0; goto s_scan_req ]
    ; when_ (in_state s_scan_req) [ goto s_scan_eval ]
    ; when_
        (in_state s_scan_eval)
        [ if_
            (scan_j.value ==: len.value |: (rd_data ==:. 0))
            [ group_end <-- scan_j.value
            ; hit_zero <-- (scan_j.value <>: len.value)
            ; goto s_code
            ]
            [ scan_j <-- scan_j.value +:. 1; goto s_scan_req ]
        ]
    ; when_
        (in_state s_code)
        [ send
            (uresize (group_end.value -: pos.value +:. 1) ~width:8)
            [ if_ (pos.value ==: group_end.value) [ after_group ] [ goto s_data_req ] ]
        ]
    ; when_ (in_state s_data_req) [ goto s_data_latch ]
    ; when_ (in_state s_data_latch) [ byte_r <-- rd_data; goto s_data_send ]
    ; when_
        (in_state s_data_send)
        [ send
            byte_r.value
            [ pos <-- pos.value +:. 1
            ; if_
                (pos.value +:. 1 ==: group_end.value)
                [ after_group ]
                [ goto s_data_req ]
            ]
        ]
    ; when_ (in_state s_delim) [ send (zero 8) [ goto s_idle ] ]
    ];
  let rd_addr = mux2 (in_state s_scan_req) scan_j.value pos.value in
  { rd_addr
  ; tx_data = tx_byte.value
  ; tx_valid = tx_stb.value
  ; busy = ~:(in_state s_idle)
  }
;;

let%expect_test "the framer agrees with Cobs.encode" =
  let run payload =
    (* the payload lives in a small ROM with a registered read, per the contract of the
       block *)
    let bytes = List.map Char.code (List.of_seq (String.to_seq payload)) in
    let padded = bytes @ List.init (8 - List.length bytes) (fun _ -> 0) in
    let circuit =
      let clock = input "clock" 1 in
      let clear = input "clear" 1 in
      let spec = Reg_spec.create ~clock ~clear () in
      let rd_addr_w = wire 7 in
      let rd_data =
        reg
          spec
          (mux
             (select rd_addr_w ~high:2 ~low:0)
             (List.map (of_unsigned_int ~width:8) padded))
      in
      let t =
        create
          ~clock
          ~clear
          ~start:(input "start" 1)
          ~length:(input "length" 7)
          ~rd_data
          ~tx_busy:(input "tx_busy" 1)
      in
      assign rd_addr_w t.rd_addr;
      Circuit.create_exn
        ~name:"cobs_tx"
        [ output "tx_data" t.tx_data; output "tx_valid" t.tx_valid; output "busy" t.busy ]
    in
    let sim = Cyclesim.create circuit in
    let start = Cyclesim.in_port sim "start" in
    let length = Cyclesim.in_port sim "length" in
    let tx_data = Cyclesim.out_port ~clock_edge:Before sim "tx_data" in
    let tx_valid = Cyclesim.out_port ~clock_edge:Before sim "tx_valid" in
    length := Bits.of_unsigned_int ~width:7 (String.length payload);
    start := Bits.vdd;
    Cyclesim.cycle sim;
    start := Bits.gnd;
    let out = Buffer.create 16 in
    let stop = ref 200 in
    while !stop > 0 do
      Cyclesim.cycle sim;
      if Bits.to_bool !tx_valid
      then (
        let b = Bits.to_int_trunc !tx_data in
        Buffer.add_string out (Printf.sprintf "%02x " b);
        if b = 0 then stop := 0);
      stop := max 0 (!stop - 1)
    done;
    let sw =
      Cobs.encode (Bytes.of_string payload)
      |> Bytes.to_seq
      |> Seq.map (fun c -> Printf.sprintf "%02x " (Char.code c))
      |> List.of_seq
      |> String.concat ""
    in
    Printf.printf "hw %s| sw %s\n" (Buffer.contents out) sw
  in
  run "\x82\x00";
  [%expect {| hw 02 82 01 00 | sw 02 82 01 00 |}];
  run "\x81\x00\x64";
  [%expect {| hw 02 81 02 64 00 | sw 02 81 02 64 00 |}];
  run "\x11\x22\x33";
  [%expect {| hw 04 11 22 33 00 | sw 04 11 22 33 00 |}];
  run "\x00\x00";
  [%expect {| hw 01 01 01 00 | sw 01 01 01 00 |}];
  run "\x41\x00";
  [%expect {| hw 02 41 01 00 | sw 02 41 01 00 |}]
;;
