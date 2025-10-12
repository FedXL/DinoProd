import os
import json
from typing import Dict, List
from pathlib import Path


class ClassifierConfig:
    def __init__(self):
        self.model_name = os.getenv('CLASSIFIER_MODEL', 'google/siglip2-base-patch16-512')
        self.threshold = float(os.getenv('CLASSIFIER_THRESHOLD', '0.35'))
        self.categories_file = os.getenv('CATEGORIES_FILE', 'config/categories.json')
        self.device = "cuda" if os.getenv('CUDA_AVAILABLE', 'true').lower() == 'true' else "cpu"
        
        # Validate threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"CLASSIFIER_THRESHOLD must be between 0.0 and 1.0, got {self.threshold}")
        
        # Load categories
        self.categories = self._load_categories()
    
    def _load_categories(self) -> Dict[str, List[str]]:
        """Load categories from JSON file."""
        categories_path = Path(self.categories_file)
        
        if not categories_path.exists():
            raise FileNotFoundError(f"Categories file not found: {self.categories_file}")
        
        try:
            with open(categories_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            # Validate structure
            if not isinstance(categories, dict):
                raise ValueError("Categories file must contain a JSON object")
            
            for category_name, prompts in categories.items():
                if not isinstance(prompts, list):
                    raise ValueError(f"Category '{category_name}' must have a list of prompts")
                if not prompts:
                    raise ValueError(f"Category '{category_name}' cannot have empty prompts list")
                for prompt in prompts:
                    if not isinstance(prompt, str):
                        raise ValueError(f"All prompts in category '{category_name}' must be strings")
            
            return categories
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in categories file: {e}")
    
    def get_all_text_prompts(self) -> Dict[str, List[str]]:
        """Get all text prompts organized by category."""
        return self.categories.copy()
    
    def get_flat_prompts_with_categories(self) -> List[tuple]:
        """Get flat list of (prompt, category) tuples for embedding computation."""
        prompts = []
        for category, texts in self.categories.items():
            for text in texts:
                prompts.append((text, category))
        return prompts