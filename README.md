# QARQI: Quadrant Amplitude Representation of Quantum Images

[![CI](https://github.com/Keno-00/qarqi/actions/workflows/ci.yml/badge.svg)](https://github.com/Keno-00/qarqi/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

QARQI is a quantum image representation framework that leverages multi-level quantum systems (Qu-Dits) for efficient and high-fidelity image encoding. It maps pixel intensities to rotation angles and uses a polarity-magnitude register structure.

---

## 🚀 Key Features

- **QARQICircuit**: High-level API for quantum image upload and simulation using `mqt.qudits`.
- **Qu-Dit Optimization**: Leverages ternary (3-level) and higher-order qudits to reduce qubit count.
- **QARQIResult**: Structured result processing for automated decoding and reconstruction.
- **CLI-Ready**: Built-in command-line interface for rapid experimentation.
- **Ground Truth Support**: Manual statevector calculation for ideal verification.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Keno-00/qarqi.git
cd qarqi

# Install in editable mode
pip install -e .
```

## 💻 Usage

### Command Line Interface

```bash
# Run a 4x4 simulation with 500 shots
qarqi --counts 500 -n 4

# Run ideal ground truth simulation
qarqi --statevector --img resources/lenna.jpg -n 8
```

### Python API

```python
import cv2
from qarqi.core.circuit import QARQICircuit
from qarqi.core.results import QARQIResult
from qarqi.utils.math import angle_map, compute_register

# 1. Load image
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (8, 8))
theta_map = angle_map(img)

# 2. Build circuit
d = 4 # for 8x8 image
circuit = QARQICircuit(d)

# 3. Simulate
counts, sv = circuit.simulate(shots=1000)

# 4. Results
result = QARQIResult(counts, d, mode='counts')
recon = result.get_probability_map()
```

## 📂 Project Structure

```
qarqi/
├── qarqi/                  # Main package
│   ├── core/               # Circuit & Results logic
│   ├── utils/              # Math & Plotting
│   └── cli/                # CLI implementation
├── docs/                   # Documentation site
├── tests/                  # Pytest suite
├── examples/               # Library usage examples
├── resources/              # Sample images
├── pyproject.toml          # Metadata & configuration
└── .github/                # CI/CD Workflows
```

## 📚 Documentation

For detailed guides, visit the documentation site or view the `docs/` folder:
```bash
pip install -e .[docs]
mkdocs serve
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development workflows.

## 📝 Citation

If you use QARQI in your research, please cite:
```bibtex
@software{QARQI_2026,
  author = {Keno S. Jose},
  title = {Quantum Architecture for Real-time Qu-DIT Imaging (QARQI)},
  url = {https://github.com/Keno-00/qarqi},
  version = {0.1.0},
  year = {2026}
}
```

## 📄 License

Licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.
