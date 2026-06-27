"""Acquisition configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hwtools.model.ids import AcqType

# Averaging counts are powers of two; instruments cap the range (Rigol: 2..1024).
_MIN_AVERAGES = 2
_MAX_AVERAGES = 1024


class AcquireConfig(BaseModel):
    """How samples are processed during acquisition.

    ``averages`` is only meaningful when ``type`` is ``AVERAGE``, where it must be
    a power of two within the instrument range. ``memory_depth`` of ``None`` means
    let the instrument choose (its "AUTO").
    """

    model_config = ConfigDict(frozen=True)

    type: AcqType = AcqType.NORMAL
    averages: int = Field(default=1, description="Averaging count (AVERAGE mode only).")
    memory_depth: int | None = Field(
        default=None, gt=0, description="Requested record length; None = instrument AUTO."
    )

    @model_validator(mode="after")
    def _check_averages(self) -> AcquireConfig:
        if self.type is AcqType.AVERAGE:
            n = self.averages
            if not (_MIN_AVERAGES <= n <= _MAX_AVERAGES and _is_power_of_two(n)):
                raise ValueError(
                    f"averages must be a power of two in "
                    f"[{_MIN_AVERAGES}, {_MAX_AVERAGES}] for AVERAGE mode, got {n}"
                )
        return self


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
