## Requirements

CProfilerFlameGraph depends on snakeviz to render flamegraph as html file. Main snakeviz repo requires renders flamegraph by launching webserver and rendering html through that. On some enviornments you want to just get a statically renderable html that can be shared. Snakeviz's author provides utility for this at https://gist.github.com/jiffyclub/6b5e0f0f05ab487ff607. This util just combined the patch from gist and added a util that does not need to launch server.

## Installation

pip install snakeviz

## Usage

```python
    from util.python_profiler import CProfilerFlameGraph
    with CProfilerFlameGraph("my_profiler.html"):
        prog = export_to_exec_prog(model, example_inputs)
```

You should see my_profiler.html generted in the working directory.
