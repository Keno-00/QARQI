# User Guide

This guide will walk you through the basic usage of QARQI.

## Basic Simulation

To run a basic simulation from the command line:

```bash
qarqi --counts 1000 -n 8
```

This will:
1. Load the default Lenna image.
2. Resize it to 8x8.
3. Encode it into a quantum circuit.
4. Run 1000 shots of simulation.
5. Save the result in the `runs/` directory.

## Advanced Usage

For more control, you can use the Python API:

```python
from qarqi.core.circuit import QARQICircuit
# ... see examples/ directory for full scripts
```
