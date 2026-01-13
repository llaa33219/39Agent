import asyncio
from typing import Optional, Any
from pathlib import Path
import json
import hashlib

from .config import DATA_DIR


class MemoryManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._db = None
        self._table = None
        self._embedder = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        self._initialized = True

    def _init_sync(self):
        import lancedb
        from sentence_transformers import SentenceTransformer

        db_path = DATA_DIR / "lancedb"
        db_path.mkdir(parents=True, exist_ok=True)

        self._db = lancedb.connect(str(db_path))
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        table_name = f"memory_{self.session_id}"
        if table_name in self._db.table_names():
            self._table = self._db.open_table(table_name)
        else:
            self._table = None

    def _ensure_table(self):
        """Create table if it doesn't exist (needs at least one record)"""
        if self._table is None:
            return False
        return True

    async def add(self, content: str, metadata: Optional[dict] = None):
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._add_sync, content, metadata or {})

    def _add_sync(self, content: str, metadata: dict):
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        vector = self._embedder.encode(content).tolist()

        data = [
            {
                "id": doc_id,
                "text": content,
                "vector": vector,
                "metadata": json.dumps(metadata),
            }
        ]

        table_name = f"memory_{self.session_id}"
        if self._table is None:
            self._table = self._db.create_table(table_name, data)
        else:
            self._table.add(data)

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        if not self._initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        if self._table is None:
            return []

        query_vector = self._embedder.encode(query).tolist()
        results = self._table.search(query_vector).limit(limit).to_list()

        memories = []
        for row in results:
            memories.append(
                {
                    "content": row["text"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "distance": row.get("_distance", 0),
                }
            )

        return memories

    async def clear(self):
        if not self._initialized:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._clear_sync)

    def _clear_sync(self):
        table_name = f"memory_{self.session_id}"
        if table_name in self._db.table_names():
            self._db.drop_table(table_name)
        self._table = None


class ConversationHistory:
    def __init__(self, max_recent: int = 5):
        self.max_recent = max_recent
        self.messages: list[dict] = []
        self.memory: Optional[MemoryManager] = None

    def set_memory(self, memory: MemoryManager):
        self.memory = memory

    async def add_message(
        self, role: str, content: str, tool_use: Optional[dict] = None
    ):
        message: dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if tool_use:
            message["tool_use"] = tool_use

        self.messages.append(message)

        if len(self.messages) > self.max_recent:
            overflow = self.messages[: -self.max_recent]
            self.messages = self.messages[-self.max_recent :]

            if self.memory:
                for msg in overflow:
                    await self.memory.add(
                        json.dumps(msg, ensure_ascii=False), {"role": msg["role"]}
                    )

    def get_recent(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages = []
