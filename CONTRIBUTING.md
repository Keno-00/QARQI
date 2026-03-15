# Contributing to QARQI

Thank you for your interest in contributing to QARQI! We welcome contributions from everyone.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Keno-00/qarqi.git
   cd qarqi
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -e .
   pip install pytest ruff
   ```

## Workflow

1. Fork the repository and create a new branch.
2. Make your changes.
3. Ensure tests pass:
   ```bash
   python -m pytest
   ```
4. Commit your changes and push to your fork.
5. Submit a pull request.

## Code Style

We use `ruff` for linting and formatting. Please ensure your code adheres to these standards before submitting a PR.
