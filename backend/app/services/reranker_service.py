"""
Reranker Service
Handles re-ranking of retrieved recipes using cross-encoder model
Improves relevance by considering query-recipe pairs together
"""

import logging
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
from app.config import settings
from app.models.recipe import Recipe

# Setup logger
logger = logging.getLogger(__name__)


class RerankerService:
    """
    Service for re-ranking recipes using cross-encoder model
    Cross-encoders consider query and recipe text together for better relevance
    """
    
    def __init__(self):
        self.model: Optional[CrossEncoder] = None
        self.model_name = settings.RERANKER_MODEL
        self.batch_size = settings.RERANKER_BATCH_SIZE
        self.enabled = settings.RERANKER_ENABLED
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the cross-encoder model (only when needed)"""
        if not self._model_loaded and self.enabled:
            logger.info(f"Loading reranker model: {self.model_name}...")
            try:
                self.model = CrossEncoder(self.model_name)
                self._model_loaded = True
                logger.info(f"Reranker model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading reranker model: {e}", exc_info=True)
                logger.warning("Reranker will be disabled, using FAISS scores only")
                self.enabled = False
                self._model_loaded = False
                raise
    
    def _prepare_recipe_text(self, recipe: Recipe) -> str:
        """
        Prepare recipe text for reranking
        Similar to embedding service but optimized for cross-encoder
        """
        parts = []
        
        # Title (most important)
        if recipe.Title:
            parts.append(recipe.Title)
        
        # Ingredients (cleaned version preferred)
        ingredients_text = recipe.Cleaned_Ingredients or recipe.Ingredients
        if ingredients_text:
            # Remove Python list syntax if present
            ingredients_clean = ingredients_text.replace('[', '').replace(']', '').replace("'", '')
            parts.append(f"Ingredients: {ingredients_clean}")
        
        # Instructions (truncated for efficiency)
        if recipe.Instructions:
            instructions = recipe.Instructions
            words = instructions.split()
            if len(words) > 300:  # Shorter for reranker
                instructions = ' '.join(words[:300]) + '...'
            parts.append(f"Instructions: {instructions}")
        
        return ' '.join(parts)
    
    def _prepare_query_text(self, ingredients: List[str]) -> str:
        """
        Prepare query text from ingredients for reranking
        """
        if not ingredients:
            return ""
        
        # Create a natural language query
        if len(ingredients) == 1:
            query = f"Recipe with {ingredients[0]}"
        elif len(ingredients) <= 3:
            query = f"Recipe with {', '.join(ingredients[:-1])} and {ingredients[-1]}"
        else:
            query = f"Recipe with ingredients: {', '.join(ingredients[:5])} and more"
        
        return query
    
    def is_loaded(self) -> bool:
        """
        Check if reranker model is loaded and ready
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model_loaded and self.model is not None and self.enabled
    
    def rerank(
        self,
        query: str,
        recipes: List[Recipe],
        top_k: int = 10
    ) -> List[Tuple[Recipe, float]]:
        """
        Re-rank recipes based on query relevance using cross-encoder
        
        Args:
            query: Query text (e.g., "Recipe with chicken, pasta, tomato")
            recipes: List of Recipe objects to re-rank
            top_k: Number of top results to return
            
        Returns:
            List of tuples (Recipe, relevance_score) sorted by score (descending)
            Scores are normalized to 0-1 range (higher is better)
        """
        if not self.enabled:
            logger.debug("Reranker is disabled, returning recipes as-is")
            # Return recipes with dummy scores
            return [(recipe, 1.0) for recipe in recipes[:top_k]]
        
        if not recipes:
            logger.warning("Empty recipe list provided for reranking")
            return []
        
        if not self._model_loaded:
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                # Fallback: return recipes with dummy scores
                return [(recipe, 1.0) for recipe in recipes[:top_k]]
        
        try:
            logger.debug(f"Reranking {len(recipes)} recipes with query: '{query[:50]}...'")
            
            # Prepare query-recipe pairs
            pairs = []
            for recipe in recipes:
                recipe_text = self._prepare_recipe_text(recipe)
                pairs.append([query, recipe_text])
            
            # Score pairs using cross-encoder (batch processing)
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Normalize scores to 0-1 range (sigmoid for cross-encoder outputs)
            import numpy as np
            normalized_scores = 1 / (1 + np.exp(-scores))  # Sigmoid normalization
            
            # Create (recipe, score) pairs
            recipe_scores = list(zip(recipes, normalized_scores))
            
            # Sort by score (descending)
            recipe_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            top_results = recipe_scores[:top_k]
            
            logger.debug(f"Reranking completed: {len(top_results)} top results")
            logger.debug(f"Score range: {min(normalized_scores):.3f} - {max(normalized_scores):.3f}")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            logger.warning("Falling back to original order with dummy scores")
            # Fallback: return recipes with dummy scores
            return [(recipe, 1.0) for recipe in recipes[:top_k]]
    
    def rerank_by_ingredients(
        self,
        ingredients: List[str],
        recipes: List[Recipe],
        top_k: int = 10
    ) -> List[Tuple[Recipe, float]]:
        """
        Re-rank recipes based on ingredient list
        
        Args:
            ingredients: List of ingredient names
            recipes: List of Recipe objects to re-rank
            top_k: Number of top results to return
            
        Returns:
            List of tuples (Recipe, relevance_score) sorted by score (descending)
        """
        query = self._prepare_query_text(ingredients)
        return self.rerank(query, recipes, top_k)
    
    def get_model_info(self) -> dict:
        """Get information about the reranker model"""
        return {
            "model_name": self.model_name,
            "loaded": self._model_loaded,
            "enabled": self.enabled,
            "batch_size": self.batch_size
        }


# Singleton instance
reranker_service = RerankerService()

