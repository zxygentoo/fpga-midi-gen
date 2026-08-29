(* The placement family — see placement.mli. The rules stand HERE, one time for the whole
   repository, rather than once in each unit that obeys them. *)

open Base
open Hardcaml

let no_dsp product = Signal.add_attribute product (Rtl_attribute.Vivado.use_dsp false)
let replica copy = Signal.add_attribute copy (Rtl_attribute.Vivado.dont_touch true)
let slice_rows = 8
let slices_for ~rows = (rows + slice_rows - 1) / slice_rows
let block_ram = Rtl_attribute.Vivado.Ram_style.block

(* Hardcaml's [Rtl_attribute.Vivado] states RAM_STYLE and SRL_STYLE and no ROM_STYLE, thus
   this one is made here. It is the same string Vivado documents for a memory that is
   never written. *)
let block_rom =
  Rtl_attribute.create
    "ROM_STYLE"
    ~applies_to:[ Rtl_attribute.Applies_to.Memories ]
    ~value:(Rtl_attribute.Value.String "block")
;;

let rom ?attributes ~read_addresses image =
  let size = Array.length image in
  if size = 0 then failwith "Placement.rom: the image is empty";
  if Array.is_empty read_addresses
  then failwith "Placement.rom: a ROM wants a read address";
  (* the checks [Signal.rom] takes from [validate] and this path does not: one width
     throughout the image, and an address that reaches every word of it and no more *)
  let data_width = Bits.width image.(0) in
  if Array.exists image ~f:(fun word -> Bits.width word <> data_width)
  then failwith "Placement.rom: the image states more than one width";
  (* [address_bits_for] and not [ceil_log2]: a memory of one word still takes a one-bit
     address, which is the rule every caller here sizes its address by *)
  let address_width = Signal.address_bits_for size in
  Array.iter read_addresses ~f:(fun addr ->
    if Signal.width addr <> address_width
    then
      raise_s
        (Sexp.message
           "Placement.rom: the address does not size on the image"
           [ "address", sexp_of_int (Signal.width addr)
           ; "wanted", sexp_of_int address_width
           ; "words", sexp_of_int size
           ]));
  Signal.Expert.multiport_memory_prim
    ?attributes
    ~initialize_to:image
    size
    ~remove_unused_write_ports:true
    ~data_width
    ~write_ports:[||]
    ~read_addresses
;;

(* ==================================================================== *)
(* The gates *)
(* ==================================================================== *)

let%expect_test "a Placement.rom is a ROM: an image, a read, and no write logic" =
  (* WHAT THIS GATE HOLDS IS THE VERILOG. A memory built with a dead write port — a write
     enable tied low — carries an [always] block that can never fire, and Vivado reads a
     written memory. This form has no write port at all: the image stands in an [initial]
     block, the read is one [assign], and the attribute states ROM_STYLE and not
     RAM_STYLE. *)
  let image = Array.init 8 ~f:(fun k -> Bits.of_unsigned_int ~width:8 (k * 3)) in
  let addr = Signal.input "addr" 3 in
  let circuit =
    Circuit.create_exn
      ~name:"probe"
      [ Signal.output
          "q"
          (rom ~attributes:[ block_rom ] ~read_addresses:[| addr |] image).(0)
      ]
  in
  Rtl.print Verilog circuit;
  [%expect
    {|
    module probe (
        addr,
        q
    );

        input [2:0] addr;
        output [7:0] q;

        (* ROM_STYLE="block" *)
        reg [7:0] signal_multiport_mem[0:7];
        wire [7:0] signal_mem_read_port;
        initial begin
            signal_multiport_mem[0] <= 8'b00000000;
            signal_multiport_mem[1] <= 8'b00000011;
            signal_multiport_mem[2] <= 8'b00000110;
            signal_multiport_mem[3] <= 8'b00001001;
            signal_multiport_mem[4] <= 8'b00001100;
            signal_multiport_mem[5] <= 8'b00001111;
            signal_multiport_mem[6] <= 8'b00010010;
            signal_multiport_mem[7] <= 8'b00010101;
        end
        assign signal_mem_read_port = signal_multiport_mem[addr];
        assign q = signal_mem_read_port;

    endmodule
    |}]
;;

let%expect_test "the address must size on the image" =
  (* [Signal.rom] takes this check from [validate]; this path builds the primitive itself
     and states it here, because an address one bit short reads half a ROM in silence. *)
  let image = Array.init 8 ~f:(fun k -> Bits.of_unsigned_int ~width:8 k) in
  let said width =
    match rom ~read_addresses:[| Signal.input "addr" width |] image with
    | _ -> "no complaint"
    | exception exn -> Exn.to_string exn
  in
  Stdio.print_endline (said 3);
  Stdio.print_endline (said 2);
  [%expect
    {|
    no complaint
    ("Placement.rom: the address does not size on the image" (address 2)
      (wanted 3) (words 8))
    |}]
;;
