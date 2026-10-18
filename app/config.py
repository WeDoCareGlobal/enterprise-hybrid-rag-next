"""Configuration module for Enterprise Hybrid RAG.

Loads configuration from environment variables with sane defaults.
All configuration values are validated at load time.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class QdrantConfig(BaseModel):
    """Qdrant vector store configuration."""

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    collection: str = Field(default="enterprise_documents", description="Qdrant collection name")
    timeout: int = Field(default=10, description="Connection timeout in seconds")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"Qdrant port must be between 1 and 65535, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > 300:
            raise ValueError(f"Qdrant timeout must be between 1 and 300 seconds, got {v}")
        return v


class NimConfig(BaseModel):
    """NVIDIA NIM embedding configuration.

    GPU fraction scheduling (see deploy/nginx/ for Kubernetes examples) allows
    multiple NIMs to share a single GPU with request/limit allocation.
    GPU memory swap keeps model weights in CPU memory when idle, delivering
    44-61x faster time-to-first-token.
    """

    enabled: bool = Field(default=False, description="Enable NVIDIA NIM embedding")
    base_url: str = Field(default="https://integrate.api.nvidia.com/v1", description="NIM base URL")
    api_key: str | None = Field(default=None, description="NVIDIA API key (reads from env if not set)")
    model: str = Field(default="nemotron-3-ultra-550b", description="Embedding model name")
    timeout: int = Field(default=30, description="NIM request timeout in seconds")
    # GPU fraction scheduling
    gpu_request: float = Field(default=0.5, description="Guaranteed GPU fraction (0.0-1.0)")
    gpu_limit: float = Field(default=1.0, description="Burstable GPU fraction limit (0.0-1.0)")
    gpu_swap: bool = Field(default=True, description="Enable GPU memory swap")
    gpu_swap_size: int = Field(default=80, description="Swap size in GB")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization (0.0-1.0)")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1 or v > 300:
            raise ValueError(f"NIM timeout must be between 1 and 300 seconds, got {v}")
        return v

    @field_validator("gpu_request")
    @classmethod
    def validate_gpu_request(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"GPU request must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("gpu_limit")
    @classmethod
    def validate_gpu_limit(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"GPU limit must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_memory_utilization(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"GPU memory utilization must be between 0.0 and 1.0, got {v}")
        return v

    @classmethod
    def from_env(cls) -> NimConfig:
        """Load configuration from environment variables."""
        api_key = os.environ.get("NIM_API_KEY")
        return cls(
            enabled=os.environ.get("NIM_ENABLED", "false").lower() == "true",
            base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=api_key,
            model=os.environ.get("NIM_EMBEDDING_MODEL", "nemotron-3-ultra-550b"),
            timeout=int(os.environ.get("NIM_TIMEOUT", "30")),
            gpu_request=float(os.environ.get("GPU_REQUEST", "0.5")),
            gpu_limit=float(os.environ.get("GPU_LIMIT", "1.0")),
            gpu_swap=os.environ.get("GPU_SWAP", "true").lower() == "true",
            gpu_swap_size=int(os.environ.get("GPU_SWAP_SIZE", "80")),
            gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9")),
        )


class RagasConfig(BaseModel):
    """Ragas evaluation configuration."""

    eval_interval: int = Field(
        default=3600, description="Evaluation interval in seconds"
    )
    min_faithfulness: float = Field(default=0.96, description="Minimum acceptable faithfulness score")
    min_answer_relevancy: float = Field(
        default=0.95, description="Minimum acceptable answer relevancy score"
    )

    @field_validator("eval_interval")
    @classmethod
    def validate_eval_interval(cls, v: int) -> int:
        if v < 60:
            raise ValueError(f"Evaluation interval must be at least 60 seconds, got {v}")
        return v

    @field_validator("min_faithfulness")
    @classmethod
    def validate_min_faithfulness(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Minimum faithfulness must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("min_answer_relevancy")
    @classmethod
    def validate_min_answer_relevancy(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Minimum answer relevancy must be between 0.0 and 1.0, got {v}")
        return v


class AppConfig(BaseModel):
    """Top-level application configuration."""

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    nim: NimConfig = Field(default_factory=NimConfig)
    ragas: RagasConfig = Field(default_factory=RagasConfig)
    # RRF fusion parameters
    rrf_k: int = Field(default=60, description="RRF constant k (higher = more uniform)")
    bm25_weight: float = Field(default=1.0, description="BM25 score weight multiplier")
    vector_weight: float = Field(default=1.0, description="Vector score weight multiplier")
    # Retrieval parameters
    default_top_k: int = Field(default=10, description="Default number of results")
    max_top_k: int = Field(default=50, description="Maximum number of results")
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"RRF k must be positive, got {v}")
        return v

    @field_validator("bm25_weight")
    @classmethod
    def validate_bm25_weight(cls, v: float) -> float:
        if v <= 0 or v > 10:
            raise ValueError(f"BM25 weight must be between 0 and 10, got {v}")
        return v

    @field_validator("vector_weight")
    @classmethod
    def validate_vector_weight(cls, v: float) -> float:
        if v <= 0 or v > 10:
            raise ValueError(f"Vector weight must be between 0 and 10, got {v}")
        return v

    @field_validator("default_top_k", "max_top_k")
    @classmethod
    def validate_top_k(cls, v: int, info: ValidationInfo) -> int:
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1, got {v}")
        if info.field_name == "max_top_k" and info.data.get("default_top_k", 0) > v:
            raise ValueError("max_top_k must be >= default_top_k")
        return v

    @classmethod
    def from_env(cls) -> AppConfig:
        """Load configuration from environment variables."""
        return cls(
            qdrant=QdrantConfig(
                host=os.environ.get("QDRANT_HOST", "localhost"),
                port=int(os.environ.get("QDRANT_PORT", "6333")),
                collection=os.environ.get("QDRANT_COLLECTION", "enterprise_documents"),
                timeout=int(os.environ.get("QDRANT_TIMEOUT", "10")),
            ),
            nim=NimConfig.from_env(),
            ragas=RagasConfig(
                eval_interval=int(os.environ.get("RAGAS_EVAL_INTERVAL", "3600")),
                min_faithfulness=float(os.environ.get("RAGAS_MIN_FAITHFULNESS", "0.96")),
                min_answer_relevancy=float(os.environ.get("RAGAS_MIN_ANSWER_RELEVANCY", "0.95")),
            ),
            rrf_k=int(os.environ.get("RRF_K", "60")),
            bm25_weight=float(os.environ.get("BM25_WEIGHT", "1.0")),
            vector_weight=float(os.environ.get("VECTOR_WEIGHT", "1.0")),
            default_top_k=int(os.environ.get("DEFAULT_TOP_K", "10")),
            max_top_k=int(os.environ.get("MAX_TOP_K", "50")),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
        )


# Global configuration instance
config = AppConfig.from_env()
