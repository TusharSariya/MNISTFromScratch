# Profiling Python with Pyinstrument and Flamegraphs

## Pyinstrument

Pyinstrument is a statistical profiler that records call stacks at intervals and displays a tree view of where time is spent. Unlike cProfile, it only shows the slow parts, so the output is much easier to read.

### Install

```bash
pip install pyinstrument
```

### Usage

Run from the command line (no code changes needed):

```bash
python -m pyinstrument conv2d.py
```

Or wrap a specific section in code:

```python
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()
# ... code to profile ...
profiler.stop()
profiler.print()
```

### Reading the output

Pyinstrument prints a tree where each line shows:
- Time in seconds spent in that function (inclusive of children)
- The function name and file location

Wider branches = more time. Only functions that took meaningful time are shown. This makes it easy to spot the bottleneck without wading through noise.

### Example findings from conv2d.py

```
2.953s total
├─ 1.326s  importing torchvision/torch (45%)
├─ 1.027s  conv2d_forward (35%)
└─ 0.554s  data loading via MNIST.__getitem__ (19%)
```

The import overhead dominates because PyTorch is a massive library. The nested Python loops in conv2d_forward are the second bottleneck — im2col would replace them with a single numpy matmul call.

## Flamegraphs

Flamegraphs visualize profiling data as stacked horizontal bars. The x-axis represents the proportion of time spent, and the y-axis shows the call stack depth. Wider bars = more time in that function.

### Install

```bash
pip install py-spy
```

### Usage

```bash
py-spy record -o flamegraph.svg -- python conv2d.py
```

Open `flamegraph.svg` in a browser. It's interactive — hover for details, click to zoom into a call stack.

### Pyinstrument vs Flamegraphs

| | Pyinstrument | py-spy flamegraph |
|---|---|---|
| Output | Text tree in terminal | Interactive SVG in browser |
| Best for | Quickly finding the bottleneck | Visualizing the full call stack proportionally |
| Filtering | Profile specific code blocks with start/stop | No built-in function filtering |
| Overhead | Low (statistical sampling) | Very low (external sampling, no code changes) |
| Install | `pip install pyinstrument` | `pip install py-spy` |

### Tips

- For flamegraphs, remove print/debug statements before profiling to reduce noise.
- For pyinstrument, wrap only the code you care about with `profiler.start()` / `profiler.stop()` to isolate the function you're investigating.
- Test with a small data subset first — profiling slow code on full datasets means waiting a long time for results.
