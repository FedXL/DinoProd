# Change Request: Fix Category Embedding Comparison

## Issue

The current implementation averages embeddings per category, which loses information and reduces accuracy. We need to keep individual embeddings and use max similarity instead.

## Required Changes

### 1. In `classifier/classifier_service.py` - Initialization

**CURRENT (WRONG):**
```python
# Computing averaged embeddings per category
for category, prompts in categories.items():
    embeddings = model.encode_text(prompts)
    avg_embedding = np.mean(embeddings, axis=0)  # ❌ WRONG - loses information
    self.category_embeddings[category] = avg_embedding
```

**CHANGE TO:**
```python
# Store ALL individual embeddings per category (don't average!)
for category, prompts in categories.items():
    embeddings = model.encode_text(prompts)  
    self.category_embeddings[category] = embeddings  # ✅ Keep as list/array of vectors
```

### 2. In `classifier/classifier_service.py` - Classification

**CURRENT (WRONG):**
```python
# Single comparison per category
for category, avg_embedding in self.category_embeddings.items():
    similarity = cosine_similarity(image_embedding, avg_embedding)
    category_scores[category] = similarity
```

**CHANGE TO:**
```python
# Compare to ALL embeddings for each category, take MAX
for category, text_embeddings in self.category_embeddings.items():
    # Compute similarity to ALL prompts for this category
    similarities = [
        cosine_similarity(image_embedding, text_emb) 
        for text_emb in text_embeddings
    ]
    # Take the BEST (maximum) similarity for this category
    category_scores[category] = max(similarities)
```

## Why This Change?

1. **Standard approach** for CLIP/SigLIP zero-shot classification
2. **No information loss** - different prompts capture different aspects of the same category
3. **Better accuracy** - one strong match is better than averaged mediocre matches
4. **Minimal performance impact** - comparing 10 vectors instead of 2 takes microseconds

## Example

For an image of a modern glass building:
- Prompt "building facade" → similarity: 0.9 (excellent match!)
- Prompt "old brick building" → similarity: 0.3 (poor match)

**Averaging**: (0.9 + 0.3) / 2 = **0.6** ❌ (loses the strong signal)  
**Max**: max(0.9, 0.3) = **0.9** ✅ (preserves the strong match)

## Implementation Notes

- Keep embeddings as a list/array of vectors, NOT a single averaged vector
- Use `max()` function to find best similarity per category
- Memory impact is negligible: ~45 KB vs ~15 KB (doesn't matter)
- No changes needed to model.py, image_loader.py, or config.py

## Please make these changes to the classifier service code.