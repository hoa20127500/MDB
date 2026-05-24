"""
Embedding service wrapping the sentence-transformers model.

Loads ``all-MiniLM-L6-v2`` once at construction and exposes helpers for
encoding single texts, batches of texts, and building a canonical product
text representation from a :class:`~app.models.ProductRecord`.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger
from app.models import ProductRecord

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps a sentence-transformers model for product and query encoding.

    The model is loaded once at construction time and reused for all
    subsequent encoding operations, ensuring embedding-space consistency
    across products and queries (Requirement 3.4).

    Args:
        model_name: HuggingFace / sentence-transformers model identifier.
            Defaults to ``"all-MiniLM-L6-v2"`` which produces 384-dimensional
            vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        logger.info("Loading embedding model", extra={"model_name": model_name})
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        logger.info("Embedding model loaded", extra={"model_name": model_name})

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def build_product_text(self, record: ProductRecord) -> str:
        """Concatenate ``name``, ``description``, and ``category`` for embedding.

        The three fields are joined with a single space so that the resulting
        string captures the full semantic content of the product.

        Args:
            record: A :class:`~app.models.ProductRecord` instance.

        Returns:
            A single string suitable for passing to :meth:`encode`.
        """
        return f"{record.name} {record.description} {record.category}"

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a vector embedding.

        Args:
            text: The text to encode.

        Returns:
            A list of floats representing the embedding vector (384 dimensions
            for the default model).

        Raises:
            Exception: Re-raises any exception thrown by the underlying model
                so that callers can decide how to handle failures.
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(
        self,
        texts: list[str],
        product_ids: list[str] | None = None,
    ) -> list[list[float] | None]:
        """Encode a list of texts, returning ``None`` for any item that fails.

        The method attempts a single batched model call for efficiency.  If the
        entire batch call fails, every slot is filled with ``None``.  Per-item
        failures (e.g. when iterating over individual results) are also caught
        and logged.

        Callers **must not** store a ``None`` embedding — it signals that
        embedding generation failed for that item and the product document
        should be written without an ``embedding`` field (Requirement 3.5).

        Args:
            texts: Texts to encode, one per product.
            product_ids: Optional list of product identifiers aligned with
                *texts*.  When provided, failed items are logged with their
                corresponding ``product_id``.

        Returns:
            A list of the same length as *texts*.  Each element is either a
            ``list[float]`` embedding or ``None`` if encoding failed for that
            item.
        """
        n = len(texts)
        results: list[list[float] | None] = [None] * n

        if n == 0:
            return results

        # --- Attempt a single batched encode call ---
        try:
            batch_embeddings = self._model.encode(texts, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            # The entire batch call failed — log once and return all-None.
            logger.error(
                "Batch embedding failed for entire batch",
                extra={
                    "batch_size": n,
                    "error": str(exc),
                },
                exc_info=True,
            )
            return results

        # --- Extract per-item results, catching any conversion errors ---
        for i, raw in enumerate(batch_embeddings):
            pid = product_ids[i] if product_ids and i < len(product_ids) else None
            try:
                results[i] = raw.tolist()
            except Exception as exc:  # noqa: BLE001
                extra: dict = {"error": str(exc)}
                if pid is not None:
                    extra["product_id"] = pid
                logger.error(
                    "Failed to convert embedding for item",
                    extra=extra,
                    exc_info=True,
                )
                results[i] = None

        return results
