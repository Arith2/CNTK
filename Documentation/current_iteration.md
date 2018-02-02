# CNTK Current release

## Change profiler details output format to be chrome://tracing

## Enable per-node timing. Working example [here](/Examples/Image/Classification/MLP/Python/SimpleMNIST.py)
- usage in Python.
```
import cntk as C
C.debugging.debug.set_node_timing(True)
C.debugging.start_profiler() # optional
C.debugging.enable_profiler() # optional
#<trainer|evaluator|function> executions
<trainer|evaluator|function>.print_node_timing()
C.debugging.stop_profiler()
```

### per-node timing creates items in profiler details when profiler is enabled. 