import os
import json
import torch
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv



class ClassifierConfig:
    def __init__(self):
        self.model_name = os.getenv('CLASSIFIER_MODEL', 'google/siglip2-base-patch16-512')
        self.threshold = float(os.getenv('CLASSIFIER_THRESHOLD', '0.35'))
        self.categories_file = os.getenv('CATEGORIES_FILE', 'config/categories.json')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"CLASSIFIER_THRESHOLD must be between 0.0 and 1.0, got {self.threshold}")
        
        # Load categories
        self.categories = self._load_categories()
    
    def _load_categories(self) -> Dict[str, List[str]]:
        """Load categories from JSON file."""
        categories_path = Path(self.categories_file)
        
        # Validate file existence and readability
        if not categories_path.exists():
            raise FileNotFoundError(f"Categories file not found: {self.categories_file}")
        
        if not categories_path.is_file():
            raise ValueError(f"Categories path is not a file: {self.categories_file}")
        
        try:
            # Check file permissions
            if not os.access(categories_path, os.R_OK):
                raise PermissionError(f"No read permission for categories file: {self.categories_file}")
            
            # Check file size (basic sanity check)
            file_size = categories_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"Categories file is empty: {self.categories_file}")
            if file_size > 1024 * 1024:  # 1MB limit
                raise ValueError(f"Categories file too large (max 1MB): {self.categories_file}")
            
            with open(categories_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            # Validate structure
            if not isinstance(categories, dict):
                raise ValueError("Categories file must contain a JSON object")
            
            if not categories:
                raise ValueError("Categories file cannot be empty")
            
            # Validate each category
            for category_name, prompts in categories.items():
                if not isinstance(category_name, str) or not category_name.strip():
                    raise ValueError(f"Category name must be a non-empty string, got: {category_name}")
                
                if not isinstance(prompts, list):
                    raise ValueError(f"Category '{category_name}' must have a list of prompts")
                
                if not prompts:
                    raise ValueError(f"Category '{category_name}' cannot have empty prompts list")
                
                for i, prompt in enumerate(prompts):
                    if not isinstance(prompt, str):
                        raise ValueError(f"Prompt {i} in category '{category_name}' must be a string, got: {type(prompt)}")
                    if not prompt.strip():
                        raise ValueError(f"Prompt {i} in category '{category_name}' cannot be empty")
            
            return categories
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in categories file '{self.categories_file}': {e}")
        except PermissionError as e:
            raise PermissionError(f"Permission error reading categories file '{self.categories_file}': {e}")
        except OSError as e:
            raise OSError(f"OS error reading categories file '{self.categories_file}': {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading categories file '{self.categories_file}': {e}")
    
    def get_flat_prompts_with_categories(self) -> List[tuple]:
        """Get flat list of (prompt, category) tuples for embedding computation."""
        prompts = []
        for category, texts in self.categories.items():
            for text in texts:
                prompts.append((text, category))
        return prompts