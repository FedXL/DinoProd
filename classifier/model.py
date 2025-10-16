import torch
import numpy as np
from typing import List
from PIL import Image
from transformers import AutoProcessor, AutoModel
import logging
import time

fastapi_logger = logging.getLogger(__name__)


class SigLIPModel:
    def __init__(self, model_name: str = "google/siglip2-base-patch16-512", device: str = "cuda"):
        """
        Initialize SigLIP model for zero-shot image classification.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (cuda/cpu)

        Raises:
            ValueError: If model_name is invalid or empty
            RuntimeError: If model fails to load or move to device
            OSError: If model files cannot be downloaded or accessed
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"Invalid model_name: {model_name}. Must be a non-empty string.")

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # Log device info
        if device == "cuda" and not torch.cuda.is_available():
            fastapi_logger.warning(f"CUDA requested but not available, falling back to CPU")
        elif self.device == "cuda":
            fastapi_logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")

        fastapi_logger.info(f"Loading model: {model_name}")
        start_time = time.perf_counter()

        try:
            # Load processor with error handling
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
                fastapi_logger.info("Model processor loaded successfully")
            except Exception as e:
                raise OSError(f"Failed to load model processor for '{model_name}': {str(e)}")

            # Load model with error handling
            try:
                self.model = AutoModel.from_pretrained(model_name)
                fastapi_logger.info("Model weights loaded successfully")
            except Exception as e:
                raise OSError(f"Failed to load model weights for '{model_name}': {str(e)}")

            # Move model to device with error handling
            try:
                self.model = self.model.to(self.device)
                self.model.eval()
                fastapi_logger.info(f"Model moved to {self.device} and set to eval mode")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try to fall back to CPU if CUDA OOM
                    if self.device == "cuda":
                        fastapi_logger.warning("CUDA out of memory, falling back to CPU")
                        try:
                            self.device = "cpu"
                            self.model = self.model.to(self.device)
                            self.model.eval()
                            fastapi_logger.info("Successfully fell back to CPU")
                        except Exception as fallback_e:
                            raise RuntimeError(f"Failed to move model to CPU after CUDA OOM: {str(fallback_e)}")
                    else:
                        raise RuntimeError(f"Out of memory on {self.device}: {str(e)}")
                else:
                    raise RuntimeError(f"Failed to move model to {self.device}: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error moving model to device: {str(e)}")

            load_time = time.perf_counter() - start_time
            fastapi_logger.info(f"{model_name} model loaded successfully in {load_time:.2f} seconds on {self.device}")

        except Exception as e:
            # Log the error and re-raise
            fastapi_logger.error(f"Failed to initialize  model '{model_name}': {str(e)}")
            raise

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
            inputs = self.processor(
                text=texts,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            )
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

    def compute_similarity_from_embeddings(self, image_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between pre-computed image and text embeddings.

        Args:
            image_embedding: Single image embedding (1, embedding_dim) - must be pre-normalized
            text_embeddings: Multiple text embeddings (num_texts, embedding_dim) - must be pre-normalized

        Returns:
            Similarity scores for each text prompt
        """
        # Embeddings are already normalized by encode_image() and encode_text()
        # Direct dot product gives cosine similarity for normalized vectors
        similarities = np.dot(image_embedding, text_embeddings.T)

        return similarities.flatten()

    def compute_similarity(self, image: Image.Image, texts: List[str]) -> np.ndarray:
        """
        Compute similarity between image and text prompts using SigLIP's native method.

        This passes both image and text together to the model, which outputs logits
        that are converted to probabilities using sigmoid. This is the recommended
        approach for SigLIP as it uses the model's learned temperature scaling.

        Args:
            image: PIL Image to classify
            texts: List of text prompts to compare against

        Returns:
            Similarity scores (probabilities) for each text prompt
        """
        with torch.no_grad():
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process image and texts together - this is key for SigLIP!
            # SigLIP uses padding="max_length" and max_length=64 as per documentation
            inputs = self.processor(
                text=texts,
                images=image,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model outputs
            outputs = self.model(**inputs)

            # Get logits and convert to probabilities using sigmoid
            # SigLIP uses sigmoid, not softmax, because it's trained with sigmoid loss
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)

            # Return as numpy array
            return probs.cpu().numpy().flatten()