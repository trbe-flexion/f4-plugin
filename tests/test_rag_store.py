"""Tests for FlagRAGStore using ephemeral ChromaDB (no disk, no model download)."""

import json
import uuid

import pytest

from src.rag.store import FlagRAGStore


@pytest.fixture
def store():
    """Create a FlagRAGStore with ephemeral ChromaDB and a unique collection name."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    return FlagRAGStore(collection_name=name)


@pytest.fixture
def sample_passages():
    return [
        {"flag": "waterfall_methodology", "passage": "The project uses waterfall methodology."},
        {"flag": "waterfall_methodology", "passage": "Sequential phases with phase gates."},
        {"flag": "brownfield", "passage": "Contractor takes over existing codebase."},
        {"flag": "agile_methodology", "passage": "Agile sprints and iterative delivery."},
    ]


class TestFlagRAGStore:
    def test_add_and_count(self, store, sample_passages):
        added = store.add_passages(sample_passages)
        assert added == 4
        assert store.count() == 4

    def test_add_idempotent(self, store, sample_passages):
        store.add_passages(sample_passages)
        added_again = store.add_passages(sample_passages)
        assert added_again == 0
        assert store.count() == 4

    def test_query_returns_results(self, store, sample_passages):
        store.add_passages(sample_passages)
        results = store.query("waterfall development phases", top_k=2)
        assert len(results) == 2
        assert all("passage" in r for r in results)
        assert all("flag" in r for r in results)
        assert all("distance" in r for r in results)

    def test_query_empty_collection(self, store):
        results = store.query("anything", top_k=3)
        assert results == []

    def test_query_top_k_limits_results(self, store, sample_passages):
        store.add_passages(sample_passages)
        results = store.query("some text", top_k=1)
        assert len(results) == 1

    def test_metadata_stored(self, store, sample_passages):
        store.add_passages(sample_passages)
        results = store.query("waterfall", top_k=4)
        flags_found = {r["flag"] for r in results}
        assert "waterfall_methodology" in flags_found

    def test_empty_passages_list(self, store):
        added = store.add_passages([])
        assert added == 0
        assert store.count() == 0

    def test_count_empty(self, store):
        assert store.count() == 0


class TestGetOrInit:
    def test_populates_from_seeds(self, tmp_path):
        seeds = [
            {"flag": "brownfield", "passage": "Taking over legacy code."},
            {"flag": "agile_methodology", "passage": "Sprints and retros."},
        ]
        seeds_path = tmp_path / "seeds.jsonl"
        with open(seeds_path, "w") as f:
            for record in seeds:
                f.write(json.dumps(record) + "\n")

        name = f"test_{uuid.uuid4().hex[:8]}"
        store = FlagRAGStore.get_or_init(
            persist_directory=str(tmp_path / "chromadb"),
            seeds_path=str(seeds_path),
            collection_name=name,
        )
        assert store.count() == 2

    def test_skips_if_already_populated(self, tmp_path):
        seeds = [{"flag": "brownfield", "passage": "Legacy code."}]
        seeds_path = tmp_path / "seeds.jsonl"
        seeds_path.write_text(json.dumps(seeds[0]) + "\n")

        name = f"test_{uuid.uuid4().hex[:8]}"
        db_dir = str(tmp_path / "chromadb")
        store1 = FlagRAGStore.get_or_init(
            persist_directory=db_dir, seeds_path=str(seeds_path), collection_name=name
        )
        assert store1.count() == 1

        store2 = FlagRAGStore.get_or_init(
            persist_directory=db_dir, seeds_path=str(seeds_path), collection_name=name
        )
        assert store2.count() == 1

    def test_missing_seeds_file(self, tmp_path):
        name = f"test_{uuid.uuid4().hex[:8]}"
        store = FlagRAGStore.get_or_init(
            persist_directory=str(tmp_path / "chromadb"),
            seeds_path=str(tmp_path / "nonexistent.jsonl"),
            collection_name=name,
        )
        assert store.count() == 0
