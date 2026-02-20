# Building and Debugging stoplight.c

## Build

```bash
gcc -O2 -o stoplight implementations/stoplight.c -lm
```

Common flags:

| Flag | Purpose |
|------|---------|
| `-O2` | Optimize (faster runtime) |
| `-O0` | Disable optimization (better for debugging) |
| `-g` | Include debug symbols (required for gdb/valgrind) |
| `-Wall -Wextra` | Enable warnings |
| `-fsanitize=address` | Enable AddressSanitizer (catches memory bugs at runtime) |

## Run

```bash
./stoplight
```

Expected output: iterations converging toward low error, e.g.:

```
iter 1, error: 0.4231
iter 2, error: 0.3812
...
iter 312, error: 0.0010
```

## Debugging

### Warnings first

Always build with warnings before reaching for a debugger:

```bash
gcc -O0 -g -Wall -Wextra -o stoplight implementations/stoplight.c
```

### Memory errors — AddressSanitizer (fastest)

Catches heap overflows, use-after-free, double-free:

```bash
gcc -O0 -g -fsanitize=address -o stoplight implementations/stoplight.c
./stoplight
```

ASan prints a detailed report with the exact line if a violation occurs.

### Memory leaks — Valgrind

```bash
gcc -O0 -g -o stoplight implementations/stoplight.c
valgrind --leak-check=full ./stoplight
```

Valgrind is slower than ASan but catches leaks that ASan misses. The `free()` calls at the bottom of each loop iteration must cover every `malloc` from that iteration.

### Step-through — gdb

```bash
gcc -O0 -g -o stoplight implementations/stoplight.c
gdb ./stoplight
```

Useful gdb commands:

```
(gdb) break main          # stop at main
(gdb) break 172           # stop at line 172 (the printf)
(gdb) run                 # start
(gdb) next                # step over
(gdb) step                # step into
(gdb) print idx           # inspect a variable
(gdb) print sum           # inspect sum
(gdb) continue            # continue to next breakpoint
(gdb) quit
```

### Print debugging

The simplest option. Add temporary prints inside the loop to inspect matrix values:

```c
printf("output_0[0]: %f\n", output_0[0]);
printf("pred[0]: %f\n", pred[0]);
```

## Common issues

**Segfault** — usually an out-of-bounds index in `matmul`. Double-check `y`, `x`, `z` arg order: `matmul(left, right, y, x, z)` where left is `(y, x)` and right is `(x, z)`.

**NaN / diverging error** — weights exploding. The learning rate (`0.1` in `scale(...)`) may be too high, or weights are being freed before they're done being read.

**Wrong dimensions** — draw the shapes of each matrix on paper and trace through each `matmul` call. The inner dimensions must match: `(y, x) @ (x, z) → (y, z)`.

**Memory leak** — every `malloc` inside the loop needs a corresponding `free` before the next iteration. If you add a new intermediate matrix, add its `free` to the cleanup block at the bottom of the loop.
