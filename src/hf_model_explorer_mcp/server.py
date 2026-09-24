"""
Hugging Face Model Explorer — MCP Server
------------------------------------------------------------
Exposes Hugging Face Hub model search, comparison, and hardware-fit
estimation as MCP tools, so any MCP-compatible agent (Claude, Antigravity,
Cursor, etc.) can help a user pick and size a model.

Tools:
  - search_models        find models by task / name, sorted by popularity
  - get_model_details     full info on a single model
  - compare_models        side-by-side comparison of 2-4 models
  - check_hardware_fit    will this model fit in N GB of VRAM?
  - find_similar_models   lighter/heavier alternatives for the same task
"""

from __future__ import annotations

import math
from typing import Any, Literal

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from mcp.server.mcpserver import MCPServer

api = HfApi()
mcp = MCPServer(
    name="hf-model-explorer",
    version="0.1.0",
    instructions=(
        "Tools for searching, comparing, and sizing Hugging Face models. "
        "Use check_hardware_fit before recommending a model for local use, "
        "since a model that looks like a good fit on paper may not fit in "
        "the user's available VRAM at full precision."
    ),
)

# Bytes-per-parameter for common precisions, used to estimate memory footprint.
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Rough multiplier to account for activations/KV-cache/runtime overhead on
# top of raw weight size. This is a simplification, not an exact figure.
RUNTIME_OVERHEAD_MULTIPLIER = 1.2


def _param_count(model_info: Any) -> int | None:
    """Best-effort extraction of total parameter count from a HF ModelInfo."""
    safetensors = getattr(model_info, "safetensors", None)
    if safetensors and getattr(safetensors, "total", None):
        return int(safetensors.total)

    # Fallback: some model cards expose it in config (rare to get without a
    # dedicated config fetch), so we don't guess further here — better to
    # return None and let the caller say "unknown" than to fabricate a number.
    return None


def _model_summary(model_info: Any) -> dict[str, Any]:
    params = _param_count(model_info)
    return {
        "id": model_info.id,
        "pipeline_tag": model_info.pipeline_tag,
        "downloads": getattr(model_info, "downloads", None),
        "likes": getattr(model_info, "likes", None),
        "license": next(
            (t.split("license:")[1] for t in (model_info.tags or []) if t.startswith("license:")),
            None,
        ),
        "gated": getattr(model_info, "gated", False),
        "parameters": params,
        "parameters_display": _format_params(params),
        "last_modified": str(getattr(model_info, "last_modified", "")) or None,
        "url": f"https://huggingface.co/{model_info.id}",
    }


def _format_params(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return str(n)


@mcp.tool()
def search_models(
    query: str,
    task: str | None = None,
    limit: int = 10,
    sort_by: Literal["downloads", "likes", "last_modified"] = "downloads",
) -> dict[str, Any]:
    """
    Search Hugging Face Hub for models matching a query, optionally filtered
    by task (pipeline_tag), sorted by popularity or recency.

    Args:
        query: free-text search, e.g. "llama instruct" or "sentiment analysis"
        task: optional HF pipeline tag filter, e.g. "text-generation",
              "text-classification", "image-classification", "translation"
        limit: max number of results (1-50)
        sort_by: "downloads" (default), "likes", or "last_modified"
    """
    limit = max(1, min(limit, 50))
    try:
        results = list(
            api.list_models(
                search=query,
                pipeline_tag=task,
                sort=sort_by,
                limit=limit,
                cardData=False,
                expand=["safetensors"],
            )
        )
    except HfHubHTTPError as e:
        return {"error": f"Hugging Face API error: {e}"}

    return {
        "query": query,
        "task_filter": task,
        "count": len(results),
        "models": [_model_summary(m) for m in results],
    }


@mcp.tool()
def get_model_details(model_id: str) -> dict[str, Any]:
    """
    Get full details for a single Hugging Face model, including parameter
    count, license, tags, and pipeline task.

    Args:
        model_id: the HF model repo id, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    """
    try:
        info = api.model_info(model_id, files_metadata=False)
    except HfHubHTTPError as e:
        return {"error": f"Could not fetch '{model_id}': {e}"}

    summary = _model_summary(info)
    summary["tags"] = info.tags or []
    summary["library_name"] = getattr(info, "library_name", None)
    return summary


@mcp.tool()
def compare_models(model_ids: list[str]) -> dict[str, Any]:
    """
    Compare 2-4 Hugging Face models side by side: parameters, license,
    downloads, task, and last update.

    Args:
        model_ids: list of 2-4 HF model repo ids to compare
    """
    if not (2 <= len(model_ids) <= 4):
        return {"error": "Provide between 2 and 4 model_ids to compare."}

    comparison = []
    for model_id in model_ids:
        try:
            info = api.model_info(model_id, files_metadata=False)
            comparison.append(_model_summary(info))
        except HfHubHTTPError as e:
            comparison.append({"id": model_id, "error": str(e)})

    return {"models": comparison}


@mcp.tool()
def check_hardware_fit(
    model_id: str,
    available_vram_gb: float,
    precision: Literal["fp32", "fp16", "bf16", "int8", "int4", "auto"] = "auto",
) -> dict[str, Any]:
    """
    Estimate whether a Hugging Face model will fit in a given amount of
    GPU VRAM, and at which precision. Uses the model's parameter count and
    standard bytes-per-parameter figures for each precision, plus a runtime
    overhead margin for activations and KV-cache. This is an estimate, not
    a guarantee — actual usage varies by framework, batch size, and context
    length.

    Args:
        model_id: the HF model repo id, e.g. "mistralai/Mistral-7B-v0.1"
        available_vram_gb: available GPU VRAM in gigabytes, e.g. 24
        precision: specific precision to check, or "auto" to find the
                   lowest-precision option that fits
    """
    try:
        info = api.model_info(model_id, files_metadata=False)
    except HfHubHTTPError as e:
        return {"error": f"Could not fetch '{model_id}': {e}"}

    params = _param_count(info)
    if params is None:
        return {
            "model_id": model_id,
            "error": (
                "Could not determine parameter count for this model "
                "(no safetensors metadata available). Try get_model_details "
                "to inspect the model manually."
            ),
        }

    def estimate_gb(bytes_per_param: float) -> float:
        raw_bytes = params * bytes_per_param
        return (raw_bytes * RUNTIME_OVERHEAD_MULTIPLIER) / (1024**3)

    fits_by_precision = {
        prec: round(estimate_gb(bpp), 2) for prec, bpp in BYTES_PER_PARAM.items()
    }

    if precision == "auto":
        # Prefer the highest-quality precision that still fits.
        precision_order = ["fp32", "fp16", "int8", "int4"]
        best_fit = next(
            (p for p in precision_order if fits_by_precision.get(p, math.inf) <= available_vram_gb),
            None,
        )
        return {
            "model_id": model_id,
            "parameters": _format_params(params),
            "available_vram_gb": available_vram_gb,
            "estimated_vram_by_precision_gb": fits_by_precision,
            "recommended_precision": best_fit,
            "fits": best_fit is not None,
            "note": (
                "Estimates include a ~20% overhead margin for activations/KV-cache. "
                "Real usage depends on framework, batch size, and context length."
            ),
        }

    required = fits_by_precision.get(precision)
    return {
        "model_id": model_id,
        "parameters": _format_params(params),
        "precision": precision,
        "estimated_vram_gb": required,
        "available_vram_gb": available_vram_gb,
        "fits": required is not None and required <= available_vram_gb,
        "note": (
            "Estimate includes a ~20% overhead margin for activations/KV-cache. "
            "Real usage depends on framework, batch size, and context length."
        ),
    }


@mcp.tool()
def find_similar_models(
    model_id: str,
    variant: Literal["lighter", "heavier", "same_size"] = "lighter",
    limit: int = 5,
) -> dict[str, Any]:
    """
    Find alternative models for the same task as the given model, filtered
    by relative size (lighter, heavier, or similar-sized alternatives).

    Args:
        model_id: the HF model repo id to find alternatives for
        variant: "lighter" (fewer params), "heavier" (more params),
                 or "same_size" (roughly comparable)
        limit: max number of alternatives to return (1-20)
    """
    limit = max(1, min(limit, 20))
    try:
        base_info = api.model_info(model_id, files_metadata=False)
    except HfHubHTTPError as e:
        return {"error": f"Could not fetch '{model_id}': {e}"}

    base_params = _param_count(base_info)
    task = base_info.pipeline_tag
    if not task:
        return {"error": f"'{model_id}' has no pipeline_tag; can't search for same-task alternatives."}

    try:
        candidates = list(
            api.list_models(pipeline_tag=task, sort="downloads", limit=limit * 6, cardData=False, expand=["safetensors"])
        )
    except HfHubHTTPError as e:
        return {"error": f"Hugging Face API error: {e}"}

    candidates = [c for c in candidates if c.id != model_id]

    if base_params is None:
        # No size data to filter by — just return top popular alternatives for the task.
        filtered = candidates[:limit]
    else:
        scored = []
        for c in candidates:
            c_params = _param_count(c)
            if c_params is None:
                continue
            if variant == "lighter" and c_params < base_params:
                scored.append((c, c_params))
            elif variant == "heavier" and c_params > base_params:
                scored.append((c, c_params))
            elif variant == "same_size" and 0.5 * base_params <= c_params <= 1.5 * base_params:
                scored.append((c, c_params))

        if variant == "lighter":
            scored.sort(key=lambda pair: -pair[1])  # closest-to-base first
        elif variant == "heavier":
            scored.sort(key=lambda pair: pair[1])
        filtered = [c for c, _ in scored[:limit]]

    return {
        "base_model": model_id,
        "base_parameters": _format_params(base_params),
        "task": task,
        "variant": variant,
        "alternatives": [_model_summary(c) for c in filtered],
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
