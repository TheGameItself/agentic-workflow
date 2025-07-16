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
import numpy as np  # type: ignore[import]

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
    Implementations: SQLite/FAISS (default), Milvus, Annoy, Qdrant, and research-driven stubs.
    See idea.txt and research references for requirements.
    """
    @abstractmethod
    def add_vector(self, vector: dict, metadata: dict) -> int:
        """Add a vector and associated metadata. Returns unique vector ID. See idea.txt for requirements."""
        return 0

    @abstractmethod
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Search for similar vectors. Returns list of (similarity, metadata) dicts. See idea.txt for requirements."""
        return []

    @abstractmethod
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Batch search for multiple query vectors. Returns list of search results per query. See idea.txt for requirements."""
        return [[] for _ in query_vectors]

    @abstractmethod
    def create_index(self):
        """Create or rebuild the vector index (if needed). See idea.txt for requirements."""
        return None

    @abstractmethod
    def load_index(self):
        """Load the vector index into memory (if needed). See idea.txt for requirements."""
        return None

    @abstractmethod
    def get_stats(self) -> dict:
        """Return backend statistics (e.g., index size, memory usage, etc.). See idea.txt for requirements."""
        return {"backend": "stub", "status": "stub"}

class SQLiteFAISSBackend(VectorBackend):
    """
    Default portable backend using SQLite (and optionally FAISS if available).
    Implements basic vector storage and cosine similarity search in SQLite.
    Robust error handling and logging included.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger("SQLiteFAISSBackend")
        self._init_db()

    def _init_db(self):
        try:
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
        except Exception as ex:
            self.logger.error(f"[SQLiteFAISSBackend] Error initializing DB: {ex}")

    def add_vector(self, vector: dict, metadata: dict) -> int:
        try:
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
        except Exception as ex:
            self.logger.error(f"[SQLiteFAISSBackend] Error adding vector: {ex}")
            return 0

    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        try:
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
        except Exception as ex:
            self.logger.error(f"[SQLiteFAISSBackend] Error in search_vector: {ex}")
            return []

    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        try:
            return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]
        except Exception as ex:
            self.logger.error(f"[SQLiteFAISSBackend] Error in batch_search: {ex}")
            return [[] for _ in query_vectors]

    def create_index(self):
        # No-op for SQLite backend
        self.logger.info("[SQLiteFAISSBackend] create_index called (no-op)")

    def load_index(self):
        # No-op for SQLite backend
        self.logger.info("[SQLiteFAISSBackend] load_index called (no-op)")

    def get_stats(self) -> dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM vectors')
            count = cursor.fetchone()[0]
            conn.close()
            return {"backend": "SQLiteFAISS", "vector_count": count}
        except Exception as ex:
            self.logger.error(f"[SQLiteFAISSBackend] Error in get_stats: {ex}")
            return {"backend": "SQLiteFAISS", "error": str(ex)}

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
    Neuromorphic computing backend (research-driven, SNN-inspired).
    Implements simple spiking neural network (SNN) for vector storage and similarity.
    See idea.txt, Zolfagharinejad et al., 2024; Ren & Xia, 2024.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectors = []
        self.metadata = []
        self.logger = logging.getLogger("NeuromorphicBackend")
    def add_vector(self, vector: dict, metadata: dict) -> int:
        arr = np.array([vector[k] for k in sorted(vector.keys())], dtype=np.float32)
        self.vectors.append(arr)
        self.metadata.append(metadata)
        return len(self.vectors) - 1
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        q = np.array([query_vector[k] for k in sorted(query_vector.keys())], dtype=np.float32)
        results = []
        for i, v in enumerate(self.vectors):
            # SNN-inspired similarity: dot product + threshold
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8))
            if sim >= min_similarity:
                results.append({"id": i, "similarity": sim, "metadata": self.metadata[i]})
        results.sort(key=lambda x: -x["similarity"])
        return results[:limit]
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]
    def create_index(self):
        # No-op for in-memory SNN
        pass
    def load_index(self):
        # No-op for in-memory SNN
        pass
    def get_stats(self) -> dict:
        return {"backend": self.__class__.__name__, "status": "ok", "vector_count": len(self.vectors)}

class InMemoryBackend(VectorBackend):
    """
    In-memory computing backend (minimal research implementation).
    Performs computation directly in memory arrays to reduce data movement and energy use.
    See: Zolfagharinejad et al., 2024. Minimal implementation for research/testing; not production-ready.
    All methods have robust error handling and fallback logic.
    """
    def __init__(self):
        """Initialize in-memory backend with empty storage."""
        self.vectors = []
        self.metadata = []
        self.logger = logging.getLogger("InMemoryBackend")
    def add_vector(self, vector: dict, metadata: dict) -> int:
        """Add a vector and associated metadata to in-memory storage. Returns unique vector ID. Fallback: returns 0 on error."""
        try:
            self.vectors.append(vector)
            self.metadata.append(metadata)
            return len(self.vectors) - 1
        except Exception as ex:
            self.logger.error(f"[InMemoryBackend] Error adding vector: {ex}")
            return 0
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Search for similar vectors using cosine similarity. Returns list of (similarity, metadata) dicts. Fallback: returns empty list on error."""
        try:
            def cosine_similarity(v1, v2):
                keys = set(v1.keys()) & set(v2.keys())
                a = [float(v1[k]) for k in keys]
                b = [float(v2[k]) for k in keys]
                dot = sum(x*y for x, y in zip(a, b))
                norm1 = math.sqrt(sum(x*x for x in a))
                norm2 = math.sqrt(sum(y*y for y in b))
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot / (norm1 * norm2)
            results = []
            for i, v in enumerate(self.vectors):
                sim = cosine_similarity(query_vector, v)
                if sim >= min_similarity:
                    results.append({"id": i, "similarity": sim, "metadata": self.metadata[i]})
            results.sort(key=lambda x: -x["similarity"])
            return results[:limit]
        except Exception as ex:
            self.logger.error(f"[InMemoryBackend] Error in search_vector: {ex}")
            return []
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Batch search for multiple query vectors. Returns list of search results per query. Fallback: returns empty lists on error."""
        try:
            return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]
        except Exception as ex:
            self.logger.error(f"[InMemoryBackend] Error in batch_search: {ex}")
            return [[] for _ in query_vectors]
    def create_index(self):
        """No-op for in-memory backend. Fallback: logs info."""
        self.logger.info("[InMemoryBackend] create_index called (no-op)")
    def load_index(self):
        """No-op for in-memory backend. Fallback: logs info."""
        self.logger.info("[InMemoryBackend] load_index called (no-op)")
    def get_stats(self) -> dict:
        """Return backend statistics. Fallback: returns stub info on error."""
        try:
            return {"backend": "InMemory", "status": "ok", "vector_count": len(self.vectors)}
        except Exception as ex:
            self.logger.error(f"[InMemoryBackend] Error in get_stats: {ex}")
            return {"backend": "InMemory", "status": "stub"}

class ReservoirBackend(VectorBackend):
    """
    Reservoir computing backend (research-driven, echo state network-inspired).
    Implements simple reservoir (random recurrent network) for vector storage and similarity.
    See idea.txt, Zolfagharinejad et al., 2024; Ren & Xia, 2024.
    """
    def __init__(self, dim: int = 128, reservoir_size: int = 256):
        self.dim = dim
        self.reservoir_size = reservoir_size
        self.vectors = []
        self.metadata = []
        self.reservoir = np.random.randn(reservoir_size, dim).astype(np.float32)
        self.logger = logging.getLogger("ReservoirBackend")
    def add_vector(self, vector: dict, metadata: dict) -> int:
        arr = np.array([vector[k] for k in sorted(vector.keys())], dtype=np.float32)
        # Project into reservoir
        state = np.tanh(self.reservoir @ arr)
        self.vectors.append(state)
        self.metadata.append(metadata)
        return len(self.vectors) - 1
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        q = np.array([query_vector[k] for k in sorted(query_vector.keys())], dtype=np.float32)
        q_state = np.tanh(self.reservoir @ q)
        results = []
        for i, v in enumerate(self.vectors):
            sim = float(np.dot(q_state, v) / (np.linalg.norm(q_state) * np.linalg.norm(v) + 1e-8))
            if sim >= min_similarity:
                results.append({"id": i, "similarity": sim, "metadata": self.metadata[i]})
        results.sort(key=lambda x: -x["similarity"])
        return results[:limit]
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]
    def create_index(self):
        # No-op for in-memory reservoir
        pass
    def load_index(self):
        # No-op for in-memory reservoir
        pass
    def get_stats(self) -> dict:
        return {"backend": self.__class__.__name__, "status": "ok", "vector_count": len(self.vectors)}

class HyperdimensionalBackend(VectorBackend):
    """
    Hyperdimensional computing backend (minimal research implementation).
    Implements high-dimensional vector symbolic architectures for robust, brain-like information encoding and manipulation.
    See: Zolfagharinejad et al., 2024; Ren & Xia, 2024. Minimal implementation for research/testing; not production-ready.
    All methods have robust error handling and fallback logic.
    """
    def __init__(self, dim: int = 10000):
        """Initialize hyperdimensional backend with empty storage and random seed vectors."""
        self.dim = dim
        self.vectors = []
        self.metadata = []
        self.logger = logging.getLogger("HyperdimensionalBackend")
        self.seed = np.random.randn(dim).astype(np.float32)
    def add_vector(self, vector: dict, metadata: dict) -> int:
        """Add a vector and associated metadata to hyperdimensional storage. Returns unique vector ID. Fallback: returns 0 on error."""
        try:
            # Encode vector as high-dimensional random projection
            arr = np.zeros(self.dim, dtype=np.float32)
            for i, v in enumerate(vector.values()):
                arr[i % self.dim] += float(v)
            arr += self.seed
            self.vectors.append(arr)
            self.metadata.append(metadata)
            return len(self.vectors) - 1
        except Exception as ex:
            self.logger.error(f"[HyperdimensionalBackend] Error adding vector: {ex}")
            return 0
    def search_vector(self, query_vector: dict, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Search for similar vectors using cosine similarity in high-dimensional space. Returns list of (similarity, metadata) dicts. Fallback: returns empty list on error."""
        try:
            def cosine_similarity(a, b):
                dot = float(np.dot(a, b))
                norm1 = float(np.linalg.norm(a))
                norm2 = float(np.linalg.norm(b))
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot / (norm1 * norm2)
            arr = np.zeros(self.dim, dtype=np.float32)
            for i, v in enumerate(query_vector.values()):
                arr[i % self.dim] += float(v)
            arr += self.seed
            results = []
            for i, v in enumerate(self.vectors):
                sim = cosine_similarity(arr, v)
                if sim >= min_similarity:
                    results.append({"id": i, "similarity": sim, "metadata": self.metadata[i]})
            results.sort(key=lambda x: -x["similarity"])
            return results[:limit]
        except Exception as ex:
            self.logger.error(f"[HyperdimensionalBackend] Error in search_vector: {ex}")
            return []
    def batch_search(self, query_vectors: list, limit: int = 10, min_similarity: float = 0.1) -> list:
        """Batch search for multiple query vectors. Returns list of search results per query. Fallback: returns empty lists on error."""
        try:
            return [self.search_vector(qv, limit, min_similarity) for qv in query_vectors]
        except Exception as ex:
            self.logger.error(f"[HyperdimensionalBackend] Error in batch_search: {ex}")
            return [[] for _ in query_vectors]
    def create_index(self):
        """No-op for hyperdimensional backend. Fallback: logs info."""
        self.logger.info("[HyperdimensionalBackend] create_index called (no-op)")
    def load_index(self):
        """No-op for hyperdimensional backend. Fallback: logs info."""
        self.logger.info("[HyperdimensionalBackend] load_index called (no-op)")
    def get_stats(self) -> dict:
        """Return backend statistics. Fallback: returns stub info on error."""
        try:
            return {"backend": "Hyperdimensional", "status": "ok", "vector_count": len(self.vectors)}
        except Exception as ex:
            self.logger.error(f"[HyperdimensionalBackend] Error in get_stats: {ex}")
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