from decimal import Decimal
from types import SimpleNamespace

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
    normalize_usage,
)


def test_normalize_usage_anthropic_keeps_cache_buckets_separate():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=400,
    )

    normalized = normalize_usage(usage, provider="anthropic", api_mode="anthropic_messages")

    assert normalized.input_tokens == 1000
    assert normalized.output_tokens == 500
    assert normalized.cache_read_tokens == 2000
    assert normalized.cache_write_tokens == 400
    assert normalized.prompt_tokens == 3400


def test_normalize_usage_openai_subtracts_cached_prompt_tokens():
    usage = SimpleNamespace(
        prompt_tokens=3000,
        completion_tokens=700,
        prompt_tokens_details=SimpleNamespace(cached_tokens=1800),
    )

    normalized = normalize_usage(usage, provider="openai", api_mode="chat_completions")

    assert normalized.input_tokens == 1200
    assert normalized.cache_read_tokens == 1800
    assert normalized.output_tokens == 700


def test_openrouter_models_api_pricing_is_converted_from_per_token_to_per_million(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "anthropic/claude-opus-4.6": {
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.000025",
                    "input_cache_read": "0.0000005",
                    "input_cache_write": "0.00000625",
                }
            }
        },
    )

    entry = get_pricing_entry(
        "anthropic/claude-opus-4.6",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert float(entry.input_cost_per_million) == 5.0
    assert float(entry.output_cost_per_million) == 25.0
    assert float(entry.cache_read_cost_per_million) == 0.5
    assert float(entry.cache_write_cost_per_million) == 6.25


def test_estimate_usage_cost_marks_subscription_routes_included():
    result = estimate_usage_cost(
        "gpt-5.3-codex",
        CanonicalUsage(input_tokens=1000, output_tokens=500),
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert result.status == "included"
    assert float(result.amount_usd) == 0.0


def test_estimate_usage_cost_refuses_cache_pricing_without_official_cache_rate(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "google/gemini-2.5-pro": {
                "pricing": {
                    "prompt": "0.00000125",
                    "completion": "0.00001",
                }
            }
        },
    )

    result = estimate_usage_cost(
        "google/gemini-2.5-pro",
        CanonicalUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=100),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert result.status == "unknown"


def test_get_pricing_entry_qwen36_plus_prefers_pinned_snapshot(monkeypatch):
    # Even when live metadata returns an entry missing input_cache_read (exactly
    # how OpenRouter reports qwen3.6-plus), the pinned official snapshot must win.
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_model_metadata",
        lambda: {
            "qwen/qwen3.6-plus": {
                "pricing": {
                    "prompt": "0.000000325",
                    "completion": "0.00000195",
                    "input_cache_write": "0.00000040625",
                    # no input_cache_read — this is the metadata gap being patched
                }
            }
        },
    )

    entry = get_pricing_entry(
        "qwen/qwen3.6-plus",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert entry.source == "official_docs_snapshot"
    assert float(entry.input_cost_per_million) == 0.325
    assert float(entry.output_cost_per_million) == 1.95
    assert float(entry.cache_read_cost_per_million) == 0.0325   # 0.10x input
    assert float(entry.cache_write_cost_per_million) == 0.40625  # 1.25x input


def test_estimate_usage_cost_qwen36_plus_prices_cache_reads():
    # Regression for the OpenRouter-metadata gap: with the pinned snapshot a cache
    # hit must be priced (read @ 0.10x), not bailed to status="unknown".
    result = estimate_usage_cost(
        "qwen/qwen3.6-plus",
        CanonicalUsage(input_tokens=1000, output_tokens=500, cache_read_tokens=2000),
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
    )

    assert result.status == "estimated"
    assert result.amount_usd is not None
    expected = (
        Decimal(1000) * Decimal("0.325")
        + Decimal(500) * Decimal("1.95")
        + Decimal(2000) * Decimal("0.0325")
    ) / Decimal(1_000_000)
    assert result.amount_usd == expected


def test_custom_endpoint_models_api_pricing_is_supported(monkeypatch):
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda base_url, api_key=None: {
            "zai-org/GLM-5-TEE": {
                "pricing": {
                    "prompt": "0.0000005",
                    "completion": "0.000002",
                }
            }
        },
    )

    entry = get_pricing_entry(
        "zai-org/GLM-5-TEE",
        provider="custom",
        base_url="https://llm.chutes.ai/v1",
        api_key="test-key",
    )

    assert float(entry.input_cost_per_million) == 0.5
    assert float(entry.output_cost_per_million) == 2.0
