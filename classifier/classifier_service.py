import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
from collections import defaultdict

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
            fastapi_logger.warning("Classifier service already initialized")
            return
        
        fastapi_logger.info("Initializing classifier service...")
        start_time = time.perf_counter()
        
        # Load SigLIP model
        fastapi_logger.info(f"Loading SigLIP model: {self.config.model_name}")
        self.model = SigLIPModel(self.config.model_name, self.config.device)
        
        # Prepare text prompts
        self.text_prompt_mappings = self.config.get_flat_prompts_with_categories()
        all_prompts = [prompt for prompt, _ in self.text_prompt_mappings]
        
        fastapi_logger.info(f"Computing embeddings for {len(all_prompts)} text prompts across {len(self.config.categories)} categories...")
        
        # Compute text embeddings for all prompts
        text_embeddings = self.model.encode_text(all_prompts)
        
        # Group embeddings by category (keep individual embeddings, don't average!)
        category_embeddings_lists = defaultdict(list)
        for i, (prompt, category) in enumerate(self.text_prompt_mappings):
            category_embeddings_lists[category].append(text_embeddings[i])
        
        # Store ALL individual embeddings per category
        for category, embeddings_list in category_embeddings_lists.items():
            # Stack embeddings into array but don't average - keep all individual vectors
            stacked = np.stack(embeddings_list)
            self.category_embeddings[category] = stacked
            
            fastapi_logger.info(f"Category '{category}': stored {len(embeddings_list)} individual text prompt embeddings")
        
        elapsed = time.perf_counter() - start_time
        total_categories = len(self.category_embeddings)
        total_prompts = len(all_prompts)
        
        fastapi_logger.info(f"Classifier initialized in {elapsed:.2f} seconds")
        fastapi_logger.info(f"Loaded {total_categories} categories with {total_prompts} total text variations")
        
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
            fastapi_logger.info("Computing image embedding...")
            start_time = time.perf_counter()
            image_embedding = self.model.encode_image(image)
            encoding_time = time.perf_counter() - start_time
            fastapi_logger.info(f"Image encoded in {encoding_time:.3f} seconds")
            
            # ✅ RESHAPE ONCE, before the loop
            img_emb = image_embedding.reshape(1, -1)
            
            # Compute similarities with all categories
            similarities = {}
            for category, text_embeddings in self.category_embeddings.items():
                # Compute similarity to ALL text embeddings at once (batch operation)
                similarity_scores = self.model.compute_similarity(img_emb, text_embeddings)
                
                # Take the BEST (maximum) similarity for this category
                similarities[category] = float(np.max(similarity_scores))
            
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