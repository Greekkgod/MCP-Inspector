# Contributing to HF Model Explorer MCP

First off, thank you for considering contributing to the Hugging Face Model Explorer MCP server! It's people like you that make open-source such a great community.

## How to Contribute

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Create a virtual environment** and install the dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b your-feature-branch
   ```
5. **Make your changes**. If you are adding a new tool, be sure to update `src/hf_model_explorer_mcp/server.py`.
6. **Write tests** in the `tests/test_server.py` file to ensure your feature works.
7. **Run the test suite** to ensure everything passes:
   ```bash
   pytest tests
   ```
8. **Commit your changes** and push to your fork.
9. **Submit a Pull Request** to the `main` branch of this repository!

## Code Style
Please ensure your code follows standard Python conventions and uses type hints where applicable.
