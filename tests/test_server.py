"""
Unit tests for hf_model_explorer_mcp.server.

All Hugging Face Hub calls are mocked so these tests run offline and
deterministically — no network access or API rate limits required.
"""

from unittest.mock import MagicMock, patch

import pytest

from hf_model_explorer_mcp import server


def make_api_error(message: str = "404 Not Found") -> server.HfHubHTTPError:
    """Build a valid HfHubHTTPError for tests (it requires a response object)."""
    fake_response = MagicMock()
    fake_response.status_code = 404
    return server.HfHubHTTPError(message, response=fake_response)


def make_fake_model(
    model_id: str,
    params: int | None,
    pipeline_tag: str = "text-generation",
    license_tag: str = "license:apache-2.0",
    downloads: int = 1000,
    likes: int = 10,
    gated: bool = False,
):
    m = MagicMock()
    m.id = model_id
    m.pipeline_tag = pipeline_tag
    m.downloads = downloads
    m.likes = likes
    m.tags = [license_tag]
    m.gated = gated
    m.last_modified = "2024-01-01"
    m.library_name = "transformers"
    m.safetensors = MagicMock(total=params) if params is not None else None
    return m


# ---------------------- _format_params ----------------------

def test_format_params_billions():
    assert server._format_params(7_000_000_000) == "7.0B"


def test_format_params_millions():
    assert server._format_params(350_000_000) == "350.0M"


def test_format_params_small():
    assert server._format_params(500) == "500"


def test_format_params_none():
    assert server._format_params(None) == "unknown"


# ---------------------- _param_count ----------------------

def test_param_count_from_safetensors():
    m = make_fake_model("org/model", params=7_000_000_000)
    assert server._param_count(m) == 7_000_000_000


def test_param_count_missing_returns_none():
    m = make_fake_model("org/model", params=None)
    assert server._param_count(m) is None


# ---------------------- search_models ----------------------

def test_search_models_returns_summaries():
    fake_results = [
        make_fake_model("org/model-a", 7_000_000_000),
        make_fake_model("org/model-b", 13_000_000_000),
    ]
    with patch.object(server.api, "list_models", return_value=fake_results):
        result = server.search_models("test query", task="text-generation", limit=5)

    assert result["count"] == 2
    assert result["models"][0]["id"] == "org/model-a"
    assert result["models"][0]["parameters"] == 7_000_000_000
    assert result["models"][0]["license"] == "apache-2.0"


def test_search_models_clamps_limit():
    with patch.object(server.api, "list_models", return_value=[]) as mock_list:
        server.search_models("query", limit=999)
    _, kwargs = mock_list.call_args
    assert kwargs["limit"] == 50


def test_search_models_handles_api_error():
    with patch.object(server.api, "list_models", side_effect=make_api_error("boom")):
        result = server.search_models("query")
    assert "error" in result


# ---------------------- get_model_details ----------------------

def test_get_model_details_success():
    fake = make_fake_model("org/model", 1_000_000_000)
    with patch.object(server.api, "model_info", return_value=fake):
        result = server.get_model_details("org/model")
    assert result["id"] == "org/model"
    assert result["parameters"] == 1_000_000_000
    assert "tags" in result


def test_get_model_details_not_found():
    with patch.object(server.api, "model_info", side_effect=make_api_error("404")):
        result = server.get_model_details("org/nonexistent")
    assert "error" in result


# ---------------------- compare_models ----------------------

def test_compare_models_requires_two_to_four():
    assert "error" in server.compare_models(["only-one"])
    assert "error" in server.compare_models(["a", "b", "c", "d", "e"])


def test_compare_models_success():
    models = {
        "org/a": make_fake_model("org/a", 7_000_000_000),
        "org/b": make_fake_model("org/b", 13_000_000_000),
    }
    with patch.object(server.api, "model_info", side_effect=lambda model_id, **_: models[model_id]):
        result = server.compare_models(["org/a", "org/b"])
    assert len(result["models"]) == 2
    assert result["models"][0]["id"] == "org/a"


def test_compare_models_partial_failure_does_not_crash():
    def side_effect(model_id, **_):
        if model_id == "org/bad":
            raise make_api_error("404")
        return make_fake_model(model_id, 1_000_000_000)

    with patch.object(server.api, "model_info", side_effect=side_effect):
        result = server.compare_models(["org/good", "org/bad"])
    assert result["models"][0]["id"] == "org/good"
    assert "error" in result["models"][1]


# ---------------------- check_hardware_fit ----------------------

def test_hardware_fit_7b_int8_fits_in_8gb():
    fake = make_fake_model("org/7b-model", 7_000_000_000)
    with patch.object(server.api, "model_info", return_value=fake):
        result = server.check_hardware_fit("org/7b-model", available_vram_gb=8, precision="auto")
    assert result["fits"] is True
    assert result["recommended_precision"] == "int8"


def test_hardware_fit_too_large_for_any_precision():
    fake = make_fake_model("org/huge-model", 175_000_000_000)
    with patch.object(server.api, "model_info", return_value=fake):
        result = server.check_hardware_fit("org/huge-model", available_vram_gb=4, precision="auto")
    assert result["fits"] is False
    assert result["recommended_precision"] is None


def test_hardware_fit_specific_precision():
    fake = make_fake_model("org/7b-model", 7_000_000_000)
    with patch.object(server.api, "model_info", return_value=fake):
        result = server.check_hardware_fit("org/7b-model", available_vram_gb=16, precision="fp16")
    assert result["precision"] == "fp16"
    assert result["fits"] is True


def test_hardware_fit_missing_param_count():
    fake = make_fake_model("org/model", params=None)
    with patch.object(server.api, "model_info", return_value=fake):
        result = server.check_hardware_fit("org/model", available_vram_gb=8)
    assert "error" in result


def test_hardware_fit_model_not_found():
    with patch.object(server.api, "model_info", side_effect=make_api_error("404")):
        result = server.check_hardware_fit("org/nonexistent", available_vram_gb=8)
    assert "error" in result


# ---------------------- find_similar_models ----------------------

def test_find_similar_models_lighter():
    base = make_fake_model("org/13b-model", 13_000_000_000)
    candidates = [
        make_fake_model("org/7b-alt", 7_000_000_000),
        make_fake_model("org/3b-alt", 3_000_000_000),
        make_fake_model("org/20b-alt", 20_000_000_000),  # heavier, should be excluded
    ]
    with patch.object(server.api, "model_info", return_value=base), \
         patch.object(server.api, "list_models", return_value=candidates):
        result = server.find_similar_models("org/13b-model", variant="lighter", limit=5)

    ids = [m["id"] for m in result["alternatives"]]
    assert "org/20b-alt" not in ids
    assert "org/7b-alt" in ids


def test_find_similar_models_requires_pipeline_tag():
    base = make_fake_model("org/model", 1_000_000_000)
    base.pipeline_tag = None
    with patch.object(server.api, "model_info", return_value=base):
        result = server.find_similar_models("org/model")
    assert "error" in result


def test_find_similar_models_base_not_found():
    with patch.object(server.api, "model_info", side_effect=make_api_error("404")):
        result = server.find_similar_models("org/nonexistent")
    assert "error" in result


# ---------------------- tool registration ----------------------

@pytest.mark.asyncio
async def test_all_tools_registered():
    tools = await server.mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "search_models",
        "get_model_details",
        "compare_models",
        "check_hardware_fit",
        "find_similar_models",
    }
