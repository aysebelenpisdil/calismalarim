"""
RAG Pipeline Service
Coordinates Retriever (FAISS) → Reranker → Generator (LLM) pipeline
Provides end-to-end recipe recommendation with explanations
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from app.services.faiss_service import faiss_service
from app.services.embedding_service import embedding_service
from app.services.reranker_service import reranker_service
from app.services.llm_service import llm_service
from app.services.recipe_service import recipe_service
from app.models.recipe import Recipe, RecipeWithMatch

# Setup logger
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline coordinating 3 components:
    1. Retriever (FAISS) - Vector similarity search
    2. Reranker (Cross-encoder) - Contextual re-ranking
    3. Generator (Gemini LLM) - Explanation generation
    """
    
    def __init__(
        self,
        faiss_service=faiss_service,
        embedding_service=embedding_service,
        reranker_service=reranker_service,
        llm_service=llm_service,
        recipe_service=recipe_service
    ):
        self.retriever = faiss_service
        self.embedder = embedding_service
        self.reranker = reranker_service
        self.generator = llm_service
        self.recipe_service = recipe_service
    
    def _retrieve(
        self,
        user_ingredients: List[str],
        top_k: int = 50
    ) -> List[Recipe]:
        """
        Step 1: Retrieve recipes using FAISS vector search
        
        Args:
            user_ingredients: List of ingredient names
            top_k: Number of recipes to retrieve
            
        Returns:
            List of Recipe objects from FAISS search
        """
        try:
            logger.debug(f"Retrieving top-{top_k} recipes for ingredients: {user_ingredients}")
            
            if not self.retriever.is_loaded():
                logger.warning("FAISS index not loaded, falling back to string matching")
                # Fallback to string matching
                results = self.recipe_service.find_suitable_recipes(
                    user_ingredients=user_ingredients,
                    use_vector_search=False,
                    top_k=top_k
                )
                # Convert RecipeWithMatch to Recipe
                return [Recipe(**recipe.dict()) for recipe in results]
            
            # Use FAISS vector search
            distances, indices = self.retriever.search_by_ingredients(
                ingredients=user_ingredients,
                k=min(top_k, self.recipe_service.get_total_count()),
                embedding_service=self.embedder
            )
            
            # Get recipes from indices
            all_recipes = self.recipe_service.get_all_recipes(
                limit=self.recipe_service.get_total_count()
            )
            
            retrieved_recipes = []
            for idx in indices:
                if idx < len(all_recipes):
                    retrieved_recipes.append(all_recipes[idx])
            
            logger.debug(f"Retrieved {len(retrieved_recipes)} recipes from FAISS")
            return retrieved_recipes
            
        except Exception as e:
            logger.error(f"Error in retrieval step: {e}", exc_info=True)
            logger.warning("Falling back to string matching")
            # Fallback to string matching
            results = self.recipe_service.find_suitable_recipes(
                user_ingredients=user_ingredients,
                use_vector_search=False,
                top_k=top_k
            )
            return [Recipe(**recipe.dict()) for recipe in results]
    
    def _rerank(
        self,
        user_ingredients: List[str],
        recipes: List[Recipe],
        top_k: int = 10
    ) -> List[Tuple[Recipe, float]]:
        """
        Step 2: Re-rank retrieved recipes using cross-encoder
        
        Args:
            user_ingredients: List of ingredient names
            recipes: List of Recipe objects from retrieval
            top_k: Number of top recipes to return
            
        Returns:
            List of tuples (Recipe, relevance_score) sorted by score
        """
        try:
            if not recipes:
                logger.warning("No recipes to rerank")
                return []
            
            logger.debug(f"Reranking {len(recipes)} recipes to top-{top_k}")
            
            # Use reranker service
            reranked_results = self.reranker.rerank_by_ingredients(
                ingredients=user_ingredients,
                recipes=recipes,
                top_k=top_k
            )
            
            logger.debug(f"Reranking completed: {len(reranked_results)} top results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking step: {e}", exc_info=True)
            logger.warning("Falling back to original order")
            # Fallback: return recipes with dummy scores
            return [(recipe, 1.0) for recipe in recipes[:top_k]]
    
    def _generate(
        self,
        user_ingredients: List[str],
        reranked_recipes: List[Tuple[Recipe, float]],
        user_preferences: Optional[Dict[str, Any]] = None,
        excluded_ingredients: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Step 3: Generate explanation using Gemini LLM
        
        Args:
            user_ingredients: List of ingredient names
            reranked_recipes: List of (Recipe, score) tuples
            user_preferences: Dietary preferences dict
            excluded_ingredients: List of excluded ingredients
            
        Returns:
            Explanation text or None if generation fails
        """
        try:
            if not reranked_recipes:
                logger.warning("No recipes provided for explanation generation")
                return None
            
            # Extract recipes from tuples
            recipes = [recipe for recipe, score in reranked_recipes]
            
            logger.debug(f"Generating explanation for {len(recipes)} recipes")
            
            # Use LLM service
            explanation = self.generator.generate_explanation(
                user_ingredients=user_ingredients,
                recommended_recipes=recipes,
                user_preferences=user_preferences,
                excluded_ingredients=excluded_ingredients
            )
            
            if explanation:
                logger.debug(f"Explanation generated: {len(explanation)} characters")
            else:
                logger.debug("No explanation generated (LLM service unavailable)")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in generation step: {e}", exc_info=True)
            return None
    
    def process(
        self,
        user_ingredients: List[str],
        user_preferences: Optional[Dict[str, Any]] = None,
        excluded_ingredients: Optional[List[str]] = None,
        top_k: int = 10,
        explain: bool = True,
        retrieval_top_k: int = 50
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieve → Rerank → Generate
        
        Args:
            user_ingredients: List of ingredient names
            user_preferences: Dietary preferences dict (vegan, glutenFree, etc.)
            excluded_ingredients: List of excluded ingredients
            top_k: Number of final recipes to return (after reranking)
            explain: Whether to generate LLM explanation
            retrieval_top_k: Number of recipes to retrieve before reranking
            
        Returns:
            Dictionary with recipes, explanation, and metadata
        """
        logger.info(f"RAG pipeline started: {len(user_ingredients)} ingredients, top_k={top_k}")
        
        # Step 1: Retrieval (FAISS)
        retrieved_recipes = self._retrieve(
            user_ingredients=user_ingredients,
            top_k=retrieval_top_k
        )
        
        if not retrieved_recipes:
            logger.warning("No recipes retrieved, returning empty result")
            return {
                "recipes": [],
                "explanation": None,
                "metadata": {
                    "retrieval_count": 0,
                    "reranked_count": 0,
                    "pipeline_stages": ["retrieval"]
                }
            }
        
        # Step 2: Reranking (Cross-encoder)
        reranked_results = self._rerank(
            user_ingredients=user_ingredients,
            recipes=retrieved_recipes,
            top_k=top_k
        )
        
        # Convert to RecipeWithMatch format
        final_recipes = []
        for recipe, score in reranked_results:
            # Count matching ingredients (using recipe service's method)
            # Access private method through recipe service instance
            recipe_ingredients_lower = recipe.Ingredients.lower()
            matching_ingredients = []
            for ingredient in user_ingredients:
                if ingredient.lower() in recipe_ingredients_lower:
                    matching_ingredients.append(ingredient)
            
            final_recipes.append(
                RecipeWithMatch(
                    **recipe.dict(),
                    matchingCount=len(matching_ingredients),
                    matchingIngredients=matching_ingredients
                )
            )
        
        # Step 3: Generation (LLM explanation)
        explanation = None
        if explain:
            explanation = self._generate(
                user_ingredients=user_ingredients,
                reranked_recipes=reranked_results,
                user_preferences=user_preferences,
                excluded_ingredients=excluded_ingredients
            )
        
        logger.info(f"RAG pipeline completed: {len(final_recipes)} recipes, explanation={'yes' if explanation else 'no'}")
        
        return {
            "recipes": final_recipes,
            "explanation": explanation,
            "metadata": {
                "retrieval_count": len(retrieved_recipes),
                "reranked_count": len(reranked_results),
                "pipeline_stages": ["retrieval", "reranking"] + (["generation"] if explain else []),
                "retriever_used": self.retriever.is_loaded(),
                "reranker_used": self.reranker.is_loaded(),
                "llm_used": self.generator.is_available()
            }
        }


# Singleton instance
rag_pipeline = RAGPipeline()

