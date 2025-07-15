"""
Modular Vector Backend Interface for MCP
Provides a pluggable interface for vector storage and search backends.
Default: SQLite/FAISS (portable, local)
Planned: Milvus, Annoy, Qdrant, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, TYPE_CHECKING
import json
import sqlite3
import math
import logging

# Optional dependency flags for type safety (see: https://adamj.eu/tech/2021/12/29/python-type-hints-optional-imports/)
try:
    import pymilvus  # type: ignore[import]
    HAVE_MILVUS = True
except ImportError:
    HAVE_MILVUS = False
try:
    import annoy  # type: ignore[import]
    HAVE_ANNOY = True
except ImportError:
    HAVE_ANNOY = False
try:
    import qdrant_client  # type: ignore[import]
    HAVE_QDRANT = True
except ImportError:
    HAVE_QDRANT = False

# For type checking only (no runtime import errors)
if TYPE_CHECKING:
    import pymilvus
    import annoy
    import qdrant_client

class VectorBackend(ABC):
    """
    Abstract base class for pluggable vector storage and search backends.
    Implementations: SQLite/FAISS (default), Milvus, Annoy, etc.
    """
    @abstractmethod
    def add_vector(self, vector: dict, metadata: dict) -> int:
        """Add a vector and associated metadata. Returns unique vector ID."""
        pass

    @abstractmethod
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Search for similar vectors. Returns list of (similarity, metadata) dicts."""
        pass

    @abstractmethod
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Batch search for multiple query vectors. Returns list of search results per query."""
        pass

    @abstractmethod
    def create_index(self):
        """Create or rebuild the vector index (if needed)."""
        pass

    @abstractmethod
    def load_index(self):
        """Load the vector index into memory (if needed)."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Return backend statistics (e.g., index size, memory usage, etc.)."""
        pass

class SQLiteFAISSBackend(VectorBackend):
    """
    Default portable backend using SQLite (and optionally FAISS if available).
    Implements basic vector storage and cosine similarity search in SQLite.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_vector(self, vector: dict, metadata: dict) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        vector_json = json.dumps(vector)
        metadata_json = json.dumps(metadata)
        cursor.execute('''
            INSERT INTO vectors (vector, metadata) VALUES (?, ?)
        ''', (vector_json, metadata_json))
        vector_id = cursor.lastrowid if cursor.lastrowid is not None else 0
        conn.commit()
        conn.close()
        return vector_id

    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, vector, metadata FROM vectors')
        results = []
        query_vec = self._to_list(query_vector)
        for row in cursor.fetchall():
            vec_id, vec_json, meta_json = row
            vec = self._to_list(json.loads(vec_json))
            sim = self._cosine_similarity(query_vec, vec)
            if sim >= min_similarity:
                results.append({'id': vec_id, 'similarity': sim, 'metadata': json.loads(meta_json)})
        results.sort(key=lambda x: -x['similarity'])
        conn.close()
        return results[:limit]

    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]

    def create_index(self):
        # No-op for MVP
        pass

    def load_index(self):
        # No-op for MVP
        pass

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM vectors')
        count = cursor.fetchone()[0]
        conn.close()
        return {"backend": "SQLiteFAISS", "vector_count": count, "status": "ok"}

    def _to_list(self, vector_dict):
        # Accepts dict or list, returns list of floats
        if isinstance(vector_dict, dict):
            return [float(v) for v in vector_dict.values()]
        elif isinstance(vector_dict, list):
            return [float(v) for v in vector_dict]
        return []

    def _cosine_similarity(self, v1, v2):
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a*b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a*a for a in v1))
        norm2 = math.sqrt(sum(b*b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

class MilvusBackend(VectorBackend):
    """Milvus vector backend (remote, scalable). See: https://milvus.io/docs/milvus_and_mcp.md"""
    def __init__(self, config: dict):
        self.config = config
        self.collection_name = config.get("collection_name", "mcp_vectors")
        self.dim = config.get("dim", 128)
        self.pymilvus_available = HAVE_MILVUS
        self._init_collection()

    def _init_collection(self):
        if not HAVE_MILVUS:
            return
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType  # type: ignore[import]
            connections.connect("default", host=self.config.get("host", "localhost"), port=self.config.get("port", "19530"))
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
            ]
            schema = CollectionSchema(fields, description="MCP vector collection")
            if self.collection_name in Collection.list_collections():
                self.collection = Collection(self.collection_name)
            else:
                self.collection = Collection(self.collection_name, schema)
        except ImportError:
            raise ImportError("pymilvus is required for MilvusBackend. Please install it if you want to use this backend.")

    def add_vector(self, vector: dict, metadata: dict) -> int:
        if not HAVE_MILVUS:
            return 0
        try:
            from pymilvus import Collection  # type: ignore[import]
            vec = [float(v) for v in vector.values()] if isinstance(vector, dict) else list(vector)
            meta = json.dumps(metadata)
            data = [[vec], [meta]]
            res = self.collection.insert([data[0], data[1]])
            return int(res.primary_keys[0]) if hasattr(res, 'primary_keys') and res.primary_keys else 0
        except ImportError:
            raise ImportError("pymilvus is required for MilvusBackend. Please install it if you want to use this backend.")

    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        if not HAVE_MILVUS:
            return []
        try:
            vec = [float(v) for v in query_vector.values()] if isinstance(query_vector, dict) else list(query_vector)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            res = self.collection.search([vec], "vector", search_params, limit)
            results = []
            for hits in res:
                for hit in hits:
                    if hit.distance <= (1 - min_similarity):
                        results.append({"id": hit.id, "similarity": 1 - hit.distance, "metadata": json.loads(hit.entity.get("metadata", "{}"))})
            return results[:limit]
        except ImportError:
            raise ImportError("pymilvus is required for MilvusBackend. Please install it if you want to use this backend.")

    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]

    def create_index(self):
        if not HAVE_MILVUS:
            return
        try:
            index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
            self.collection.create_index("vector", index_params)
        except ImportError:
            raise ImportError("pymilvus is required for MilvusBackend. Please install it if you want to use this backend.")

    def load_index(self):
        if not HAVE_MILVUS:
            return
        try:
            self.collection.load()
        except ImportError:
            raise ImportError("pymilvus is required for MilvusBackend. Please install it if you want to use this backend.")

    def get_stats(self) -> dict:
        if not HAVE_MILVUS:
            return {"backend": "Milvus", "status": "unavailable"}
        return {"backend": "Milvus", "status": "ok", "collection": self.collection_name}

class AnnoyBackend(VectorBackend):
    """Annoy vector backend (lightweight, research/embedded). See: https://github.com/spotify/annoy"""
    def __init__(self, config: dict):
        self.config = config
        self.dim = config.get("dim", 128)
        self.index_path = config.get("index_path", "data/annoy_index.ann")
        self.annoy_available = HAVE_ANNOY
        self.index = None
        self.vectors = []
        self.metadata = []
        self._init_index()

    def _init_index(self):
        if not HAVE_ANNOY:
            return
        try:
            from annoy import AnnoyIndex  # type: ignore[import]
            self.index = AnnoyIndex(self.dim, "angular")
            self.index.load(self.index_path)
        except ImportError:
            raise ImportError("annoy is required for AnnoyBackend. Please install it if you want to use this backend.")

    def add_vector(self, vector: dict, metadata: dict) -> int:
        if not HAVE_ANNOY:
            return 0
        try:
            from annoy import AnnoyIndex  # type: ignore[import]
            vec = [float(v) for v in vector.values()] if isinstance(vector, dict) else list(vector)
            idx = len(self.vectors)
            if self.index is not None:
                self.index.add_item(idx, vec)
            self.vectors.append(vec)
            self.metadata.append(metadata)
            return idx
        except ImportError:
            raise ImportError("annoy is required for AnnoyBackend. Please install it if you want to use this backend.")

    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        if not HAVE_ANNOY or self.index is None:
            return []
        try:
            vec = [float(v) for v in query_vector.values()] if isinstance(query_vector, dict) else list(query_vector)
            idxs = self.index.get_nns_by_vector(vec, limit, include_distances=True) if self.index is not None else ([], [])
            results = []
            for i, dist in zip(idxs[0], idxs[1]):
                sim = 1 - dist  # Annoy returns distance, convert to similarity
                if sim >= min_similarity:
                    results.append({"id": i, "similarity": sim, "metadata": self.metadata[i] if i < len(self.metadata) else {}})
            return results[:limit]
        except ImportError:
            raise ImportError("annoy is required for AnnoyBackend. Please install it if you want to use this backend.")

    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]

    def create_index(self):
        if not HAVE_ANNOY or self.index is None:
            return
        try:
            self.index.build(10)
            self.index.save(self.index_path)
        except ImportError:
            raise ImportError("annoy is required for AnnoyBackend. Please install it if you want to use this backend.")

    def load_index(self):
        if not HAVE_ANNOY:
            return
        try:
            from annoy import AnnoyIndex  # type: ignore[import]
            self.index = AnnoyIndex(self.dim, "angular")
            self.index.load(self.index_path)
        except ImportError:
            raise ImportError("annoy is required for AnnoyBackend. Please install it if you want to use this backend.")

    def get_stats(self) -> dict:
        if not HAVE_ANNOY:
            return {"backend": "Annoy", "status": "unavailable"}
        return {"backend": "Annoy", "status": "ok", "vector_count": len(self.vectors)}

class QdrantBackend(VectorBackend):
    """Qdrant vector backend (future-proof, scalable). See: https://qdrant.tech/documentation/"""
    def __init__(self, config: dict):
        self.config = config
        self.collection_name = config.get("collection_name", "mcp_vectors")
        self.dim = config.get("dim", 128)
        self.qdrant_available = HAVE_QDRANT
        self._init_collection()

    def _init_collection(self):
        if not HAVE_QDRANT:
            return
        try:
            from qdrant_client import QdrantClient  # type: ignore[import]
            self.client = QdrantClient(host=self.config.get("host", "localhost"), port=self.config.get("port", 6333))
            self.client.get_collection(self.collection_name)
        except ImportError:
            raise ImportError("qdrant_client is required for QdrantBackend. Please install it if you want to use this backend.")

    def add_vector(self, vector: dict, metadata: dict) -> int:
        if not HAVE_QDRANT:
            return 0
        try:
            from qdrant_client.http import models as rest  # type: ignore[import]
            vec = [float(v) for v in vector.values()] if isinstance(vector, dict) else list(vector)
            res = self.client.upsert(collection_name=self.collection_name, points=[{
                "id": None,
                "vector": vec,
                "payload": metadata
            }])
            return res.result["operation_id"] if hasattr(res, "result") and "operation_id" in res.result else 0
        except ImportError:
            raise ImportError("qdrant_client is required for QdrantBackend. Please install it if you want to use this backend.")

    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        if not HAVE_QDRANT:
            return []
        try:
            vec = [float(v) for v in query_vector.values()] if isinstance(query_vector, dict) else list(query_vector)
            res = self.client.search(collection_name=self.collection_name, query_vector=vec, limit=limit)
            results = []
            for hit in res:
                sim = hit.score
                if sim >= min_similarity:
                    results.append({"id": hit.id, "similarity": sim, "metadata": hit.payload})
            return results[:limit]
        except ImportError:
            raise ImportError("qdrant_client is required for QdrantBackend. Please install it if you want to use this backend.")

    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]

    def create_index(self):
        if not HAVE_QDRANT:
            return
        # Qdrant auto-manages indexes
        pass

    def load_index(self):
        if not HAVE_QDRANT:
            return
        # Qdrant auto-manages indexes
        pass

    def get_stats(self) -> dict:
        if not HAVE_QDRANT:
            return {"backend": "Qdrant", "status": "unavailable"}
        return {"backend": "Qdrant", "status": "ok", "collection": self.collection_name}

# --- Brain-Inspired Computing Backends (2024 Research Stubs) ---
# See: Zolfagharinejad et al., 2024 (EPJ B, Open Access), Ren & Xia, 2024 (arXiv)

class NeuromorphicBackend(VectorBackend):
    """
    Neuromorphic computing backend (RESEARCH STUB).
    Not for production use. See idea.txt and Zolfagharinejad et al., 2024; Ren & Xia, 2024.
    """
    def add_vector(self, vector: dict, metadata: dict) -> int:
        logging.warning(f"[{self.__class__.__name__}] add_vector not implemented. See idea.txt and Zolfagharinejad et al., 2024.")
        return 0
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        logging.warning(f"[{self.__class__.__name__}] search_vector not implemented. See idea.txt and Zolfagharinejad et al., 2024.")
        return []
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        logging.warning(f"[{self.__class__.__name__}] batch_search not implemented. See idea.txt and Zolfagharinejad et al., 2024.")
        return [[] for _ in query_vectors]
    def create_index(self):
        logging.warning(f"[{self.__class__.__name__}] create_index not implemented. See idea.txt and Zolfagharinejad et al., 2024.")
        pass
    def load_index(self):
        logging.warning(f"[{self.__class__.__name__}] load_index not implemented. See idea.txt and Zolfagharinejad et al., 2024.")
        pass
    def get_stats(self) -> dict:
        logging.warning(f"[{self.__class__.__name__}] get_stats not implemented. See idea.txt and Zolfagharinejad et al., 2024.")
        return {"backend": self.__class__.__name__, "status": "stub"}

class InMemoryBackend(VectorBackend):
    """
    In-memory computing backend (stub).
    Performs computation directly in memory arrays to reduce data movement and energy use.
    See: Zolfagharinejad et al., 2024.
    """
    def add_vector(self, vector: dict, metadata: dict) -> int:
        return 0
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        return []
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [[] for _ in query_vectors]
    def create_index(self):
        pass
    def load_index(self):
        pass
    def get_stats(self) -> dict:
        return {"backend": "InMemory", "status": "stub"}

class ReservoirBackend(VectorBackend):
    """
    Reservoir computing backend (RESEARCH STUB).
    Not for production use. See idea.txt and Zolfagharinejad et al., 2024; Ren & Xia, 2024.
    """
    def add_vector(self, vector: dict, metadata: dict) -> int:
        return 0
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        return []
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [[] for _ in query_vectors]
    def create_index(self):
        pass
    def load_index(self):
        pass
    def get_stats(self) -> dict:
        return {"backend": "Reservoir", "status": "stub"}

class HyperdimensionalBackend(VectorBackend):
    """
    Hyperdimensional computing backend (stub).
    Implements high-dimensional vector symbolic architectures for robust, brain-like information encoding and manipulation.
    See: Zolfagharinejad et al., 2024; Ren & Xia, 2024.
    """
    def add_vector(self, vector: dict, metadata: dict) -> int:
        return 0
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        return []
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [[] for _ in query_vectors]
    def create_index(self):
        pass
    def load_index(self):
        pass
    def get_stats(self) -> dict:
        return {"backend": "Hyperdimensional", "status": "stub"}

def get_vector_backend(backend_name: str, config: dict = {}) -> VectorBackend:
    """Factory for selecting vector backend by name."""
    db_path = config.get("db_path")
    if backend_name.lower() == "sqlitefaiss":
        return SQLiteFAISSBackend(db_path or "data/unified_memory.db")
    elif backend_name.lower() == "milvus":
        return MilvusBackend(config)
    elif backend_name.lower() == "annoy":
        return AnnoyBackend(config)
    elif backend_name.lower() == "qdrant":
        return QdrantBackend(config)
    else:
        raise ValueError(f"Unknown vector backend: {backend_name}")

# Future: MilvusBackend, AnnoyBackend, QdrantBackend, etc. 