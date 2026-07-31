# The host control

## Scope

The host control is the interface between the host drivers and the FPGA. It defines
two things:

1. The control registers: the state that the host can read and write at run
   time.
2. The wire protocol: the way to read and write this state through the UART.

## The control registers

The control registers are the local storage of the control unit in the
FPGA. There are 16 of them, each one 8 bits. They are not a window into a
larger memory: they are the complete state that the host can touch.

The wire protocol gives an address to each register, from `0xFFF0` to
`0xFFFF`. An access that touches an address outside this range gets STATUS
`02`, and the FPGA changes no register.

One read at `0xFFF0` with length 16 gets each register in one transaction.

The registers:

| Address | Name | Content | Default |
|---|---|---|---|
| `0xFFFF` | RUN | the run state, bit 0 | 0 |
| `0xFFFE` | CHANNEL | MIDI channel, 0 to 15. 0 is channel 1 | 2 (= channel 3) |
| `0xFFFC`–`0xFFFD` | STEP_MS | step period in ms, minimum 1 | 250 |
| `0xFFFA`–`0xFFFB` | GATE_MS | gate time in ms | 125 |
| `0xFFF9` | VELOCITY | note velocity, 1 to 127 | 100 |
| `0xFFF5`–`0xFFF8` | SEED | PRNG seed, 32 bits, not 0 | see `lib/control.ml` |
| `0xFFF4` | MIDI_GO | write: bit 0 = 1 sends the test message. Read: 1 while a message waits | 0 |
| `0xFFF3` | MIDI_LEN | length of the test message, 1 to 3 | 0 |
| `0xFFF0`–`0xFFF2` | MIDI_MSG | the test message bytes | 0 |

Semantics:

- RUN holds the run state. A host write sets it, and a push of the board
  button toggles it. A read returns the current state. Therefore RUN can
  change without a host write.
- Power-on is silent. One button push makes the board play, with no host.
- A change of RUN or STEP_MS applies at the next step. When RUN goes to 0,
  the FPGA sends a Note Off for each open note.
- The default CHANNEL is 2, because the S-1 receives on channel 3. CHANNEL
  applies to the model messages only. A test message is raw bytes, and the
  driver sets its channel.
- If GATE_MS is not less than STEP_MS, the FPGA sends the Note Off
  immediately before the subsequent Note On.
- A write applies at one time, at its end. Therefore a value of more than
  one byte never shows a part of one write and a part of the next, and a
  block in the FPGA can look at a cell in each cycle.
- The PRNG loads the seed at the end of a write that covers a SEED cell.
  Thus one write of the four seed bytes loads the PRNG one time.
- The MIDI_MSG cells are the test-message doorbell. The host writes
  MIDI_MSG and MIDI_LEN, then writes a value with bit 0 = 1 to MIDI_GO.
  The FPGA sends the message to the MIDI output. A write with bit 0 = 0
  does not send. Because a write applies its bytes in the sequence of
  increasing addresses, one write of 5 bytes at `0xFFF0` does the complete
  operation: the payload, the length, and the send.
- The FPGA ignores the send bit while a message waits, and also when
  MIDI_LEN is not 1 to 3. MIDI_GO reads 1 from the ring until the MIDI
  transmitter takes the message, and not until the last byte is on the
  line. The driver reads MIDI_GO as 0 to know that the FPGA accepts the
  subsequent ring.
- The FPGA takes a copy of MIDI_MSG and MIDI_LEN at the ring. Therefore a
  write to those cells after MIDI_GO reads 0 cannot damage a message that
  is on the line.

The MIDI output:

- The FPGA merges the model output and the test messages at message
  boundaries only.
- The FPGA does not use running status.
- The MIDI output is 31250 baud. The model and the doorbell together stay
  far below this rate, thus the output queue cannot overflow in correct
  operation.

## The wire protocol

The host link is the USB UART, 115200 baud, 8N1. The host is the master.
The FPGA sends data only as the response to a request. One request gets
exactly one response.

Each frame uses COBS (Consistent Overhead Byte Stuffing). A frame ends with
one zero byte, and the frame body contains no zero byte. Therefore the
receiver finds the frame boundary again after each zero byte.

The request payload, before the COBS encoding:

| Field | Bytes | Content |
|---|---|---|
| OP | 1 | `01`: read, `02`: write |
| ADDR | 2 | the start address |
| LEN | 1 | 1 to 32. Read: the number of bytes. Write: the length of DATA |
| DATA | LEN | write only: the bytes to write |

The response payload, before the COBS encoding:

| Field | Bytes | Content |
|---|---|---|
| OP | 1 | the request OP plus `80` |
| STATUS | 1 | see the table below |
| DATA | LEN | read only, when STATUS is `00`: the bytes |

| STATUS | Meaning |
|---|---|
| `00` | OK |
| `01` | The operation is not known |
| `02` | The address range does not agree with the operation |
| `03` | The length is not correct |

Rules:

- All values with more than one byte are little-endian. This rule applies to
  ADDR on the wire and to the values in the registers.
- A write applies its bytes in the sequence of increasing addresses.
- The largest payload is 36 bytes: the request header of 4 bytes and DATA of
  32 bytes. The FPGA discards a frame with a longer payload, and also a
  frame that does not decode. It sends no response for these frames.
- A driver decides how long it waits for a response, and it can send the
  request again. This is safe, because each read and each write is
  idempotent. The one exception is a write that covers MIDI_GO: it sends the
  test message again. A duplicate test message is acceptable, because test
  messages are a debug tool.
- A write reply comes after the write is complete. Therefore a read-back
  after a write shows the true state. The driver can verify each write with
  a read-back.
