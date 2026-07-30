from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _ORBase(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ModelEntry(_ORBase):
    id: str
    canonical_slug: str | None = None


class ModelsResponse(_ORBase):
    data: list[ModelEntry] = Field(default_factory=list[ModelEntry])


class EndpointEntry(_ORBase):
    tag: str
    uptime_last_5m: float | None = None


class EndpointsResponse(_ORBase):
    data: EndpointsBody | None = None


class EndpointsBody(_ORBase):
    endpoints: list[EndpointEntry] = Field(default_factory=list[EndpointEntry])


class StatsDataPolicy(_ORBase):
    retains_prompts: bool | None = Field(default=None, alias="retainsPrompts")


class StatsPricing(_ORBase):
    prompt: float | None = None
    completion: float | None = None


class StatsSample(_ORBase):
    p50_throughput: float | None = None
    request_count: int | None = None


class StatsEndpoint(_ORBase):
    provider_slug: str
    quantization: str | None = None
    context_length: int | None = None
    data_policy: StatsDataPolicy | None = Field(default=None, alias="data_policy")
    stats: StatsSample | None = None
    pricing: StatsPricing | None = None


class StatsResponse(_ORBase):
    data: list[StatsEndpoint] = Field(default_factory=list[StatsEndpoint])
