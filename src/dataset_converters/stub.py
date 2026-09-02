"""Stub converter for datasets not yet fully implemented."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .base import BaseConverter


class StubConverter(BaseConverter):
    """Raises NotImplementedError with manual instructions."""

    def __init__(self, raw_data_dir: Path, output_dir: Path, dataset_name: str, instructions: str = ""):
        super().__init__(raw_data_dir, output_dir)
        self._dataset_name = dataset_name
        self._instructions = instructions

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def check_raw_data(self) -> bool:
        target = self.raw_data_dir / self._dataset_name
        return target.exists() and any(target.iterdir())

    def convert(self) -> Dict[str, pd.DataFrame]:
        msg = (
            f"Converter for {self._dataset_name} is not implemented yet. "
            f"Place raw files under {self.raw_data_dir / self._dataset_name} and implement conversion."
        )
        if self._instructions:
            msg += f" Instructions: {self._instructions}"
        raise NotImplementedError(msg)
