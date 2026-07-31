# The host control

## Scope

The host control is the interface between the host drivers and the FPGA. It defines
two things:

1. The memory map: the state that the host can read and write at run time.
2. The wire protocol: the way to read and write this state through the UART.

## The memory map

The memory has 8-bit cells and one flat 16-bit address space. The map has
two parts:

- Control: `0xFFF0` to `0xFFFF`, read and write. This is the full control
  register file, 16 bytes.
- The model window: from `0x0000`, up. Byte `i` of the blob is at address `i`.

### Control

An access that touches an address outside the control cells gets STATUS
`02`, and the FPGA changes no cell.

One read at `0xFFF0` with length 16 gets the full register file in one
transaction.

Control cells:

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
- The PRNG loads the seed when the host writes address `0xFFF8`, the last
  seed byte. Thus one ascending write of the four seed bytes loads the
  PRNG one time, at the end.
- The MIDI_MSG cells are the test-message doorbell. The host writes
  MIDI_MSG and MIDI_LEN, then writes a value with bit 0 = 1 to MIDI_GO.
  The FPGA sends the message to the MIDI output. A write with bit 0 = 0
  does not send. Because a write applies its bytes in the sequence of
  increasing addresses, one write of 5 bytes at `0xFFF0` does the complete
  operation: the payload, the length, and the send.
- The FPGA ignores the send bit while a message waits, and also when
  MIDI_LEN is not 1 to 3. The driver must read MIDI_GO as 0 before a write
  that covers a cell of MIDI_MSG, MIDI_LEN or MIDI_GO. This rule keeps
  each test message complete on the MIDI output.

The MIDI output:

- The FPGA merges the model output and the test messages at message
  boundaries only.
- The FPGA does not use running status.
- The MIDI output is 31250 baud. The model and the doorbell together stay
  far below this rate, thus the output queue cannot overflow in correct
  operation.

### The model window

In this version the wire protocol does not map the window.

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
  ADDR on the wire and to the cell values in the memory map.
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
