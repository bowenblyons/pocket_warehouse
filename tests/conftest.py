from pathlib import Path

import pytest

from pocket_warehouse.utils.config import load_config


@pytest.fixture
def test_config():
    path = Path(__file__).parent / "fixtures" / "test_config.yaml"
    return load_config(path)
