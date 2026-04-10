from dataclasses import FrozenInstanceError

import pytest

from src.domain.entities import FilterResult


class TestFilterResult:
    def test_filter_true(self):
        result = FilterResult(filter=True)
        assert result.filter is True

    def test_filter_false(self):
        result = FilterResult(filter=False)
        assert result.filter is False

    def test_frozen(self):
        result = FilterResult(filter=True)
        with pytest.raises(FrozenInstanceError):
            result.filter = False
