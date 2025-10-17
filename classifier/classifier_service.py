import numpy as np
from typing import Dict, List, Optional
import logging
import time

from .config import ClassifierConfig
from .model import SigLIPModel
from .image_loader import AsyncImageLoader

fastapi_logger = logging.getLogger(__name__)


class ClassificationResult:
    def __init__(self, category: str, confidence: float, success: bool = True, error: Optional[str] = None):
        self.category = category
        self.confidence = confidence
        self.success = success
        self.error = error

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "category": self.category if self.success else None,
            "confidence": self.confidence if self.success else None,
            "error": self.error
        }


class ClassifierService:
    def __init__(self, config: ClassifierConfig):
        """
        Initialize classifier service with configuration.

        Args:
            config: Configuration object with model settings and categories
        """
        self.config = config
        self.model: Optional[SigLIPModel] = None
        self.image_loader = AsyncImageLoader()

        # Store text prompts and pre-computed embeddings per category
        self.category_prompts: Dict[str, List[str]] = {}
        self.category_embeddings: Dict[str, np.ndarray] = {}

        # Flag to control whether to use pre-computed embeddings or joint processing
        # Pre-computed is faster but joint processing may be more accurate
        self.use_precomputed_embeddings = False

        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the classifier service by loading model and optionally pre-computing category embeddings.
        This should be called once at startup.
        """
        if self._initialized:
            fastapi_logger.warning("Classifier service already initialized")
            return

        fastapi_logger.info("Initializing classifier service...")
        start_time = time.perf_counter()

        # Load model with error handling
        try:
            fastapi_logger.info(f"Loading model: {self.config.model_name}")
            self.model = SigLIPModel(self.config.model_name, self.config.device)
        except (ValueError, RuntimeError, OSError) as e:
            error_msg = f"Failed to load model '{self.config.model_name}': {str(e)}"
            fastapi_logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading model: {str(e)}"
            fastapi_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Load category prompts
        try:
            self.category_prompts = self.config.categories.copy()

            if not self.category_prompts:
                raise ValueError("No categories found in configuration")

            total_prompts = sum(len(prompts) for prompts in self.category_prompts.values())

            if self.use_precomputed_embeddings:
                fastapi_logger.info(f"Pre-computing embeddings for {total_prompts} text prompts across {len(self.category_prompts)} categories...")

                # Log SigLIP's learned parameters
                logit_scale = self.model.model.logit_scale.exp().item()
                logit_bias = self.model.model.logit_bias.item()
                fastapi_logger.info(f"SigLIP learned parameters: logit_scale={logit_scale:.4f}, logit_bias={logit_bias:.4f}")
                fastapi_logger.info("Using: sigmoid(cosine_similarity * logit_scale + logit_bias)")

                # Encode all prompts and store by category
                category_embeddings = {}
                for category, prompts in self.category_prompts.items():
                    embeddings = self.model.encode_text(prompts)

                    # Validate embeddings
                    if embeddings is None or len(embeddings) == 0:
                        raise ValueError(f"Failed to encode text prompts for category '{category}'")
                    if len(embeddings) != len(prompts):
                        raise ValueError(f"Mismatch in embeddings for category '{category}': expected {len(prompts)}, got {len(embeddings)}")

                    # Debug: log embedding statistics
                    for i, (prompt, emb) in enumerate(zip(prompts, embeddings)):
                        emb_norm = float(np.linalg.norm(emb))
                        emb_mean = float(np.mean(emb))
                        fastapi_logger.debug(f"  Text embedding [{i}] '{prompt}': norm={emb_norm:.4f}, mean={emb_mean:.4f}")

                    category_embeddings[category] = embeddings
                    fastapi_logger.info(f"Category '{category}': pre-computed {len(prompts)} text embeddings")

                if not category_embeddings:
                    raise ValueError("No category embeddings were created")

                self.category_embeddings = category_embeddings
            else:
                fastapi_logger.info(f"Loaded {len(self.category_prompts)} categories with {total_prompts} total text prompts")
                fastapi_logger.info("Using joint image-text processing (no pre-computed embeddings)")

        except Exception as e:
            error_msg = f"Failed to initialize category data: {str(e)}"
            fastapi_logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        elapsed = time.perf_counter() - start_time
        fastapi_logger.info(f"Classifier initialized in {elapsed:.2f} seconds")

        self._initialized = True
    
    async def classify_image(self, image_url: str) -> ClassificationResult:
        """
        Classify an image from URL into predefined categories using pre-computed embeddings.

        Args:
            image_url: URL of the image to classify

        Returns:
            ClassificationResult with category, confidence, and success status
        """
        if not self._initialized:
            return ClassificationResult(
                category="",
                confidence=0.0,
                success=False,
                error="Classifier service not initialized"
            )

        # Additional safety check (only if using pre-computed embeddings)
        if hasattr(self, 'category_embeddings') and not self.category_embeddings:
            return ClassificationResult(
                category="",
                confidence=0.0,
                success=False,
                error="No category embeddings available"
            )

        try:
            # Download and load image
            image, error_msg = await self.image_loader.load_image(image_url)
            if image is None:
                return ClassificationResult(
                    category="",
                    confidence=0.0,
                    success=False,
                    error=error_msg
                )

            # Compute similarities
            start_time = time.perf_counter()
            similarities = {}

            if self.use_precomputed_embeddings:
                # Method 1: Pre-computed embeddings (faster)
                fastapi_logger.info("Computing image embedding...")
                image_embedding = self.model.encode_image(image)
                encoding_time = time.perf_counter() - start_time
                fastapi_logger.info(f"Image encoded in {encoding_time:.3f} seconds")

                # Debug: log embedding statistics
                img_norm = float(np.linalg.norm(image_embedding))
                img_mean = float(np.mean(image_embedding))
                img_std = float(np.std(image_embedding))
                img_min = float(np.min(image_embedding))
                img_max = float(np.max(image_embedding))
                fastapi_logger.debug(f"Image embedding stats: norm={img_norm:.4f}, mean={img_mean:.4f}, std={img_std:.4f}, min={img_min:.4f}, max={img_max:.4f}")

                # Reshape once for all comparisons
                img_emb = image_embedding.reshape(1, -1)

                # Compute similarities with pre-computed category embeddings
                for category, text_embeddings in self.category_embeddings.items():
                    # Compute similarity to ALL pre-computed text embeddings at once
                    similarity_scores = self.model.compute_similarity_from_embeddings(img_emb, text_embeddings)

                    # Debug logging: show individual prompt similarities
                    prompts = self.category_prompts[category]
                    fastapi_logger.debug(f"Category '{category}' prompt similarities:")
                    for i, (prompt, score) in enumerate(zip(prompts, similarity_scores)):
                        fastapi_logger.debug(f"  [{i}] '{prompt}': {score:.4f}")

                    # Take the BEST (maximum) similarity for this category
                    max_score = float(np.max(similarity_scores))
                    max_idx = int(np.argmax(similarity_scores))
                    similarities[category] = max_score

                    fastapi_logger.info(f"Category '{category}': best score = {max_score:.4f} (prompt: '{prompts[max_idx]}')")
            else:
                # Method 2: Joint processing (potentially more accurate)
                fastapi_logger.info("Computing similarities using joint image-text processing...")

                for category, prompts in self.category_prompts.items():
                    # Compute similarity to ALL text prompts for this category at once
                    similarity_scores = self.model.compute_similarity(image, prompts)

                    # Debug logging: show individual prompt similarities
                    fastapi_logger.debug(f"Category '{category}' prompt similarities:")
                    for i, (prompt, score) in enumerate(zip(prompts, similarity_scores)):
                        fastapi_logger.debug(f"  [{i}] '{prompt}': {score:.4f}")

                    # Take the BEST (maximum) similarity for this category
                    max_score = float(np.max(similarity_scores))
                    max_idx = int(np.argmax(similarity_scores))
                    similarities[category] = max_score

                    fastapi_logger.info(f"Category '{category}': best score = {max_score:.4f} (prompt: '{prompts[max_idx]}')")

            classification_time = time.perf_counter() - start_time
            fastapi_logger.info(f"Classification completed in {classification_time:.3f} seconds")

            # Find best match
            best_category = max(similarities, key=similarities.get)
            best_confidence = similarities[best_category]

            # Apply threshold
            if best_confidence >= self.config.threshold:
                result_category = best_category
                result_confidence = best_confidence
            else:
                result_category = "other"
                result_confidence = best_confidence

            fastapi_logger.info(f"Classification result: {result_category} (confidence: {result_confidence:.3f})")
            similarities_str = ", ".join([f"{cat}: {score:.3f}" for cat, score in similarities.items()])
            fastapi_logger.info(f"Similarities: {similarities_str}")

            return ClassificationResult(
                category=result_category,
                confidence=result_confidence,
                success=True
            )

        except Exception as e:
            error_msg = f"Classification failed: {str(e)}"
            fastapi_logger.error(error_msg)
            return ClassificationResult(
                category="",
                confidence=0.0,
                success=False,
                error=error_msg
            )
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        return list(self.config.categories.keys())
    
    def get_threshold(self) -> float:
        """Get current classification threshold."""
        return self.config.threshold