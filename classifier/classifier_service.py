import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
from collections import defaultdict

from .config import ClassifierConfig
from .model import SigLIPModel
from .image_loader import AsyncImageLoader

logger = logging.getLogger(__name__)


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
        
        # In-memory storage for category embeddings
        self.category_embeddings: Dict[str, np.ndarray] = {}
        self.text_prompt_mappings: List[Tuple[str, str]] = []  # (prompt, category)
        
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the classifier service by loading model and computing category embeddings.
        This should be called once at startup.
        """
        if self._initialized:
            logger.warning("Classifier service already initialized")
            return
        
        logger.info("Initializing classifier service...")
        start_time = time.perf_counter()
        
        # Load SigLIP model
        logger.info(f"Loading SigLIP model: {self.config.model_name}")
        self.model = SigLIPModel(self.config.model_name, self.config.device)
        
        # Prepare text prompts
        self.text_prompt_mappings = self.config.get_flat_prompts_with_categories()
        all_prompts = [prompt for prompt, _ in self.text_prompt_mappings]
        
        logger.info(f"Computing embeddings for {len(all_prompts)} text prompts across {len(self.config.categories)} categories...")
        
        # Compute text embeddings for all prompts
        text_embeddings = self.model.encode_text(all_prompts)
        
        # Group embeddings by category and compute average
        category_embeddings_lists = defaultdict(list)
        for i, (prompt, category) in enumerate(self.text_prompt_mappings):
            category_embeddings_lists[category].append(text_embeddings[i])
        
        # Average embeddings per category
        for category, embeddings_list in category_embeddings_lists.items():
            # Stack embeddings and compute mean
            stacked = np.stack(embeddings_list)
            mean_embedding = np.mean(stacked, axis=0)
            
            # Normalize the averaged embedding
            normalized = mean_embedding / np.linalg.norm(mean_embedding)
            self.category_embeddings[category] = normalized
            
            logger.info(f"Category '{category}': averaged {len(embeddings_list)} text prompts")
        
        elapsed = time.perf_counter() - start_time
        total_categories = len(self.category_embeddings)
        total_prompts = len(all_prompts)
        
        logger.info(f"Classifier initialized in {elapsed:.2f} seconds")
        logger.info(f"Loaded {total_categories} categories with {total_prompts} total text variations")
        
        self._initialized = True
    
    async def classify_image(self, image_url: str) -> ClassificationResult:
        """
        Classify an image from URL into predefined categories.
        
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
            
            # Encode image
            logger.info("Computing image embedding...")
            start_time = time.perf_counter()
            image_embedding = self.model.encode_image(image)
            encoding_time = time.perf_counter() - start_time
            logger.info(f"Image encoded in {encoding_time:.3f} seconds")
            
            # Compute similarities with all categories
            similarities = {}
            for category, category_embedding in self.category_embeddings.items():
                # Reshape embeddings for similarity computation
                img_emb = image_embedding.reshape(1, -1)
                cat_emb = category_embedding.reshape(1, -1)
                
                similarity = self.model.compute_similarity(img_emb, cat_emb)[0]
                similarities[category] = float(similarity)
            
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
            
            logger.info(f"Classification result: {result_category} (confidence: {result_confidence:.3f})")
            logger.info(f"All similarities: {similarities}")
            
            return ClassificationResult(
                category=result_category,
                confidence=result_confidence,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Classification failed: {str(e)}"
            logger.error(error_msg)
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