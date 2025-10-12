import torch
import numpy as np
from typing import List, Union
from PIL import Image
from transformers import AutoProcessor, AutoModel
import logging
import time

logger = logging.getLogger(__name__)


class SigLIPModel:
    def __init__(self, model_name: str = "google/siglip-base-patch16-512", device: str = "cuda"):
        """
        Initialize SigLIP model for zero-shot image classification.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computing device (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading SigLIP model: {model_name}")
        start_time = time.perf_counter()
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        load_time = time.perf_counter() - start_time
        logger.info(f"SigLIP model loaded in {load_time:.2f} seconds on {self.device}")
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text prompts into embeddings.
        
        Args:
            texts: List of text prompts to encode
            
        Returns:
            Normalized text embeddings as numpy array
        """
        with torch.no_grad():
            # Process texts
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text embeddings
            text_outputs = self.model.get_text_features(**inputs)
            
            # Normalize embeddings
            text_embeddings = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)
            
            return text_embeddings.cpu().numpy()
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode image into embedding.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Normalized image embedding as numpy array
        """
        with torch.no_grad():
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image embeddings
            image_outputs = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            image_embeddings = image_outputs / image_outputs.norm(p=2, dim=-1, keepdim=True)
            
            return image_embeddings.cpu().numpy()
    
    def compute_similarity(self, image_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embedding: Single image embedding (1, embedding_dim)
            text_embeddings: Multiple text embeddings (num_texts, embedding_dim)
            
        Returns:
            Similarity scores for each text prompt
        """
        # Ensure embeddings are normalized (they should be from encode methods)
        image_norm = image_embedding / np.linalg.norm(image_embedding, axis=-1, keepdims=True)
        text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(image_norm, text_norm.T)
        
        return similarities.flatten()