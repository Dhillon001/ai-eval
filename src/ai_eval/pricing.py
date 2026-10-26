"""Per-model token prices and run cost math.

Prices are USD per million tokens. The built-in table only covers the repo's default models and
can drift from provider pricing; override or extend it per suite with a ``pricing:`` block::

    pricing:
      my-finetune:
        input_per_mtok: 0.30
        output_per_mtok: 1.20
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ai_eval.models import CaseOutcome


class ModelPrice(BaseModel):
    """USD per million tokens. Optional long-context tier: a call whose prompt exceeds
    ``long_context_threshold`` tokens is billed entirely at the ``long_*`` rates."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    long_context_threshold: int | None = Field(default=None, gt=0)
    long_input_per_mtok: float | None = Field(default=None, ge=0)
    long_output_per_mtok: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _long_tier_complete(self) -> ModelPrice:
        long_fields = (self.long_context_threshold, self.long_input_per_mtok, self.long_output_per_mtok)
        if any(v is not None for v in long_fields) and any(v is None for v in long_fields):
            raise ValueError(
                "long_context_threshold, long_input_per_mtok and long_output_per_mtok must be set together"
            )
        return self

    def cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float:
        p, c = prompt_tokens or 0, completion_tokens or 0
        if self.long_context_threshold is not None and p > self.long_context_threshold:
            in_rate, out_rate = self.long_input_per_mtok or 0.0, self.long_output_per_mtok or 0.0
        else:
            in_rate, out_rate = self.input_per_mtok, self.output_per_mtok
        return (p * in_rate + c * out_rate) / 1_000_000


# Public list prices as of Oct 2026 for the models the built-in adapters and judge default to.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-4o-mini": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60),
    "gpt-4o": ModelPrice(input_per_mtok=2.50, output_per_mtok=10.00),
    "claude-haiku-5-5": ModelPrice(
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        long_context_threshold=100_000,
        long_input_per_mtok=0.50,
        long_output_per_mtok=2.50,
    ),
}


class PriceTable:
    """Model -> price lookup. Exact names win; otherwise the longest known name that the model
    starts with followed by ``-`` (so ``gpt-4o-mini-2024-07-18`` uses ``gpt-4o-mini``)."""

    def __init__(self, prices: dict[str, ModelPrice] | None = None):
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)

    @classmethod
    def from_config(cls, overrides: dict[str, Any] | None) -> PriceTable:
        table = cls()
        for model, spec in (overrides or {}).items():
            table.prices[str(model)] = spec if isinstance(spec, ModelPrice) else ModelPrice.model_validate(spec)
        return table

    def lookup(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        if model in self.prices:
            return self.prices[model]
        matches = [name for name in self.prices if model.startswith(name + "-")]
        return self.prices[max(matches, key=len)] if matches else None

    def cost(self, model: str | None, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        price = self.lookup(model)
        return None if price is None else price.cost(prompt_tokens, completion_tokens)


def _judge_model(judge: Any) -> str | None:
    # LLM judge details carry "provider:model".
    if not isinstance(judge, str):
        return None
    return judge.split(":", 1)[1] if ":" in judge else judge


def run_cost(outcomes: list[CaseOutcome], table: PriceTable, default_model: str | None = None) -> dict[str, Any]:
    """Cost of a run in USD, split into the system under test and LLM judge calls.

    Judge verdicts served from the cache cost nothing. Calls that report tokens for a model
    missing from the table are listed in ``unpriced_models`` and left out of the totals. Each
    cost is ``None`` when no call contributed a priced token count.
    """
    system: float | None = None
    judge: float | None = None
    unpriced: set[str] = set()

    for o in outcomes:
        run = o.run
        if run.prompt_tokens is not None or run.completion_tokens is not None:
            model = run.trace.get("model") or default_model
            c = table.cost(model, run.prompt_tokens, run.completion_tokens)
            if c is None:
                unpriced.add(model or "<unknown>")
            else:
                system = (system or 0.0) + c
        for s in o.scores:
            d = s.details
            if d.get("cached") is True:
                continue
            p, comp = d.get("prompt_tokens"), d.get("completion_tokens")
            if p is None and comp is None:
                continue
            model = _judge_model(d.get("judge"))
            c = table.cost(model, p, comp)
            if c is None:
                unpriced.add(model or "<unknown>")
            else:
                judge = (judge or 0.0) + c

    total = None if system is None and judge is None else (system or 0.0) + (judge or 0.0)

    def _r(v: float | None) -> float | None:
        return None if v is None else round(v, 6)

    return {
        "cost_usd": _r(total),
        "cost_usd_system": _r(system),
        "cost_usd_judge": _r(judge),
        "unpriced_models": sorted(unpriced),
    }
