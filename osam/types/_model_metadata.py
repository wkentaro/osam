from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    license_name: str
    license_url: str
    source_url: str
    license_spdx: str | None = None
