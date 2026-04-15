"""ChromaDB vector store for flag definitions and RFP examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "f4_flag_passages"
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"


class FlagRAGStore:
    """Wrapper around ChromaDB for flag-related passage retrieval."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_passages(self, passages: list[dict]) -> int:
        """Add passages from rag_seeds format (list of {flag, passage} dicts).

        Skips duplicates. Returns number of new documents added.
        """
        existing_ids = set(self.collection.get()["ids"])

        documents = []
        ids = []
        metadatas = []
        flag_counters: dict[str, int] = {}

        for record in passages:
            flag = record["flag"]
            passage = record["passage"]

            count = flag_counters.get(flag, 0)
            doc_id = f"{flag}_{count}"
            flag_counters[flag] = count + 1

            if doc_id in existing_ids:
                continue

            documents.append(passage)
            ids.append(doc_id)
            metadatas.append({"flag": flag})

        if documents:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

        return len(documents)

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        """Query for similar passages. Returns list of {passage, flag, distance}."""
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else []

        for doc, meta, dist in zip(docs, metas, dists, strict=True):
            items.append(
                {
                    "passage": doc,
                    "flag": meta["flag"],
                    "distance": dist,
                }
            )

        return items

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self.collection.count()

    @classmethod
    def get_or_init(
        cls,
        persist_directory: str = "data/chromadb",
        seeds_path: str = "data/rag_exemplars.jsonl",
        **kwargs,
    ) -> FlagRAGStore:
        """Get a populated store, auto-populating from seeds if empty."""
        store = cls(persist_directory=persist_directory, **kwargs)
        if store.count() == 0:
            seeds_file = Path(seeds_path)
            if not seeds_file.exists():
                logger.warning("RAG seeds not found at %s, store will be empty", seeds_path)
                return store
            records = []
            with open(seeds_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            added = store.add_passages(records)
            logger.info("Auto-populated RAG store: %d documents from %s", added, seeds_path)
        return store
