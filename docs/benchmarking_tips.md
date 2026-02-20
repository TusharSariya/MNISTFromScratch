# Benchmarking Tips

## CPU Frequency Scaling

Modern CPUs idle at low clock speeds and ramp up under load. On this machine: 800 MHz idle, 5.1 GHz boost.

Check current governor:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

Switch to `performance` (locks at max frequency) for stable benchmarks:

```bash
sudo cpupower frequency-set -g performance
```

Switch back when done:

```bash
sudo cpupower frequency-set -g powersave
```

If `cpupower` isn't installed:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Why this matters:** Under `powersave`, a fast program can finish before the CPU ramps up. Chaining after a heavy command (like `gcc`) masks this because the CPU is already boosted. This creates misleading results — the same binary appears faster or slower depending on what ran before it.

### intel_pstate gotcha

On Intel CPUs, the `intel_pstate` driver manages frequency independently. Even with the `performance` governor, `intel_pstate` can still scale idle cores down. The governor alone doesn't force max frequency.

To actually lock cores at max, set the **minimum frequency** to match the max:

```bash
sudo cpupower frequency-set -d 5100000
```

Verify it's working:

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | sort -n | head -5
```

Undo when done (let cores idle again):

```bash
sudo cpupower frequency-set -g powersave
sudo cpupower frequency-set -d 800000
```

**Warning:** This forces all cores to max clock constantly. Watch thermals — use `sensors` (from `lm-sensors`) to monitor temps.

### Single-core turbo vs all-core thermal limits

Under `powersave`, only the active core boosts. It can turbo higher because the rest of the chip is cool. Under `performance` (especially with `-d` forcing min frequency), all cores run hot, so the thermal/power budget can force the active core's effective clock *lower* than what `powersave` would allow for a single-threaded burst. For single-threaded benchmarks, `powersave` with a warmup run can actually be faster.

## Compiler Optimization Levels

| Flag | What it does | When to use |
|------|-------------|-------------|
| (none) | No optimization, literal translation of C | Never for benchmarks |
| `-O2` | Inlining, register allocation, loop opts | Default for benchmarks |
| `-O3` | Aggressive: vectorization, unrolling | When `-O2` isn't enough |
| `-O0 -g` | No optimization + debug symbols | Debugging only |

Always specify `-O2` or higher when timing:

```bash
gcc -O2 -o stoplight implementations/stoplight.c -lm
```

Without it, trivial overhead (redundant loads, no inlining) dominates and hides the real algorithmic cost.

## Warmup Runs

The first run of a binary is often slower due to:

- CPU frequency ramp-up (see above)
- Page cache misses (loading the binary and shared libs from disk)
- Branch predictor cold start

Fix: discard the first run and measure the second:

```bash
./stoplight > /dev/null && ./stoplight
```

Or add a warmup loop inside the program before the timed section.

## Random Seeds

`srand(time(NULL))` gives a different seed every second → different weights → different convergence path → different iteration count and timing profile. This makes runs non-comparable.

For reproducible benchmarks, use a fixed seed:

```c
srand(42);  // fixed seed, same weights every run
```

Python already does this (`np.random.seed(1)`).

## Per-Iteration Timing

Wall-clock time (`time ./program`) includes startup, I/O, and shutdown. Per-iteration timing isolates the actual computation.

**C** — use `clock_gettime(CLOCK_MONOTONIC, ...)` for nanosecond resolution:

```c
struct timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);
double us = ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
```

**Python** — use `time.perf_counter()` (highest resolution available):

```python
start = time.perf_counter()
# ... work ...
elapsed = time.perf_counter() - start
```

Avoid `time.time()` — it's wall-clock and can jump due to NTP adjustments.

## Report Percentiles, Not Averages

Averages hide outliers. A single OS interrupt can spike one iteration and inflate the mean. Report p50/p90/p99 instead:

- **p50** — typical iteration (what "usually" happens)
- **p90** — worst case for most iterations
- **p99** — tail latency (OS noise, GC pauses, cache evictions)

If p99 is much higher than p50, the variance is from the system, not your code.

## Checklist

Before comparing two implementations:

1. Same seed (or average over multiple seeds)
2. Same compiler flags (`-O2`)
3. CPU governor set to `performance`
4. Warmup run to prime CPU and caches
5. Report percentiles, not just averages
6. Run multiple times to confirm stability
