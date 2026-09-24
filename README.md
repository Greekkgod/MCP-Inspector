# hf-model-explorer-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI agents search, compare, and
right-size Hugging Face models — including estimating whether a model will actually fit
in your available GPU VRAM before you download 15GB of weights and find out the hard way.

Works with any MCP-compatible client: Claude, Google Antigravity, Cursor, and others.

## Why

Picking a model off Hugging Face usually means checking the model card, guessing at
parameter count, mentally doing GB-per-precision math, and hoping it fits your GPU.
This wraps that into tools an agent can call directly, so you can ask things like:

> "Find me a small instruction-tuned model I can run locally on a 12GB GPU"

and get an actual, sized answer instead of a guess.

## Tools

| Tool | What it does |
|---|---|
| `search_models` | Search HF Hub by query and task, sorted by downloads/likes/recency |
| `get_model_details` | Full details on one model: params, license, tags, task |
| `compare_models` | Side-by-side comparison of 2–4 models |
| `check_hardware_fit` | Estimates VRAM needed at fp32/fp16/int8/int4 and whether it fits your GPU |
| `find_similar_models` | Finds lighter/heavier/similar-sized alternatives for the same task |

## Install

```bash
pip install hf-model-explorer-mcp
```

Or run directly without installing, via `uvx`:

```bash
uvx hf-model-explorer-mcp
```

## Configure

### Claude Desktop / Claude Code

Add to your MCP config (`claude_desktop_config.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "hf-model-explorer": {
      "command": "uvx",
      "args": ["hf-model-explorer-mcp"]
    }
  }
}
```

### Google Antigravity

Add to `~/.gemini/antigravity/mcp_config.json` (path may vary by Antigravity version —
check Settings → MCP Servers in the IDE):

```json
{
  "mcpServers": {
    "hf-model-explorer": {
      "command": "uvx",
      "args": ["hf-model-explorer-mcp"]
    }
  }
}
```

## Example

```
User: I want to run an instruction-tuned 7B-ish model locally. I have a 12GB GPU.

Agent calls: search_models(query="instruct", task="text-generation")
Agent calls: check_hardware_fit(model_id="mistralai/Mistral-7B-Instruct-v0.2", available_vram_gb=12)

→ {
    "parameters": "7.0B",
    "recommended_precision": "fp16",
    "estimated_vram_by_precision_gb": { "fp16": 15.65, "int8": 7.82, "int4": 3.91 },
    "fits": true
  }
```

## How hardware-fit estimation works

Parameter count comes from the model's `safetensors` metadata on the Hub. Memory is
estimated as `params × bytes_per_param × 1.2` (a 20% overhead margin for activations
and KV-cache), using standard bytes-per-parameter figures:

| Precision | Bytes/param |
|---|---|
| fp32 | 4.0 |
| fp16 / bf16 | 2.0 |
| int8 | 1.0 |
| int4 | 0.5 |

This is an estimate, not a guarantee — actual usage varies by framework, batch size,
and context length. Models without safetensors metadata return an explicit "unknown"
rather than a guessed number.

## Development

```bash
git clone https://github.com/YOUR_USERNAME/hf-model-explorer-mcp
cd hf-model-explorer-mcp
pip install -e ".[dev]"
pytest
```

All 23 tests mock the Hugging Face API, so the suite runs offline with no rate limits.

## Publishing (for your own fork)

1. **PyPI**: `python -m build && twine upload dist/*`
2. **MCP Registry**: update `server.json` with your GitHub username and package version,
   then follow the [MCP Registry publishing guide](https://github.com/modelcontextprotocol/registry)
   to submit it — typically via `mcp-publisher` authenticated against your GitHub repo.

## License

MIT
