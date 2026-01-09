from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import time
import logging
from app.models.recipe import (
    Recipe,
    RecipeWithMatch,
    RecipeRecommendRequest,
    RecipeRecommendResponse,
    RecipeSearchRequest,
    RecipeSearchResponse,
    RAGRecommendRequest,
    RAGRecommendResponse,
    DietaryPreferences
)
from app.services.recipe_service import recipe_service
from app.services.faiss_service import faiss_service
from app.services.embedding_service import embedding_service
from app.services.rag_pipeline import rag_pipeline

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.get("/", response_model=dict)
async def get_recipes(
    ingredients: Optional[str] = Query(None, description="Comma-separated list of ingredients"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get all recipes with optional filtering by ingredients
    """
    try:
        if ingredients:
            # Filter by ingredients
            ingredient_list = [ing.strip() for ing in ingredients.split(',')]
            filtered_recipes = recipe_service.find_suitable_recipes(ingredient_list)
            recipes = filtered_recipes[offset:offset + limit]
            total = len(filtered_recipes)
        else:
            # Get all recipes
            recipes = recipe_service.get_all_recipes(limit, offset)
            total = recipe_service.get_total_count()
        
        return {
            "recipes": recipes,
            "total": total,
            "count": len(recipes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recipes: {str(e)}")


@router.get("/{title}", response_model=Recipe)
async def get_recipe(title: str):
    """
    Get a specific recipe by title
    """
    try:
        recipe = recipe_service.get_recipe_by_title(title)
        
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")
        
        return recipe
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recipe: {str(e)}")


@router.post("/recommend", response_model=RecipeRecommendResponse)
async def recommend_recipes(request: RecipeRecommendRequest):
    """
    Get recipe recommendations based on fridge ingredients using vector search
    
    Uses FAISS vector search if available, falls back to string matching.
    """
    start_time = time.time()
    
    try:
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="Ingredients list is required")
        
        # Determine search method
        use_vector_search = request.use_vector_search if request.use_vector_search is not None else True
        top_k = request.top_k if request.top_k is not None else 50
        
        # Check if vector search is available
        search_method = "vector" if (use_vector_search and faiss_service.is_loaded()) else "string_matching"
        
        logger.info(f"Recipe recommendation request: {len(request.ingredients)} ingredients, method: {search_method}")
        
        # Get recommendations
        recommendations = recipe_service.find_suitable_recipes(
            user_ingredients=request.ingredients,
            use_vector_search=use_vector_search,
            top_k=top_k
        )
        
        process_time = time.time() - start_time
        logger.info(f"Recommendations generated in {process_time:.3f}s: {len(recommendations)} results")
        
        return RecipeRecommendResponse(
            recommendations=recommendations,
            count=len(recommendations),
            userIngredients=request.ingredients,
            search_method=search_method
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.post("/search", response_model=RecipeSearchResponse)
async def search_recipes(request: RecipeSearchRequest):
    """
    Search recipes by text query using vector similarity search
    
    Example queries:
    - "spicy chicken pasta"
    - "vegetarian dessert"
    - "quick breakfast recipe"
    """
    start_time = time.time()
    
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Search query is required")
        
        top_k = request.top_k if request.top_k is not None else 20
        
        # Check if vector search is available
        if faiss_service.is_loaded():
            try:
                logger.info(f"Text search request: '{request.query}', method: vector")
                
                # Search using FAISS
                distances, indices = faiss_service.search_by_text(
                    text=request.query,
                    k=min(top_k, recipe_service.get_total_count()),
                    embedding_service=embedding_service
                )
                
                # Convert results to RecipeWithMatch
                results = []
                recipes = recipe_service.get_all_recipes(limit=recipe_service.get_total_count())
                
                for idx, dist in zip(indices, distances):
                    if idx < len(recipes):
                        recipe = recipes[idx]
                        # For text search, we don't have ingredient matching, so set empty
                        results.append(
                            RecipeWithMatch(
                                **recipe.dict(),
                                matchingCount=0,
                                matchingIngredients=[]
                            )
                        )
                
                process_time = time.time() - start_time
                logger.info(f"Text search completed in {process_time:.3f}s: {len(results)} results")
                
                return RecipeSearchResponse(
                    recipes=results,
                    count=len(results),
                    query=request.query,
                    search_method="vector"
                )
                
            except Exception as e:
                logger.warning(f"Vector search failed: {e}, falling back to string matching")
                # Fall through to string matching
        
        # Fallback: Simple string matching in recipe titles
        logger.info(f"Text search request: '{request.query}', method: string_matching")
        recipes = recipe_service.get_all_recipes(limit=recipe_service.get_total_count())
        
        query_lower = request.query.lower()
        results = []
        
        for recipe in recipes:
            if query_lower in recipe.Title.lower():
                results.append(
                    RecipeWithMatch(
                        **recipe.dict(),
                        matchingCount=0,
                        matchingIngredients=[]
                    )
                )
                if len(results) >= top_k:
                    break
        
        process_time = time.time() - start_time
        logger.info(f"Text search completed in {process_time:.3f}s: {len(results)} results")
        
        return RecipeSearchResponse(
            recipes=results,
            count=len(results),
            query=request.query,
            search_method="string_matching"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to search recipes: {str(e)}")


@router.post("/rag-recommend", response_model=RAGRecommendResponse)
async def rag_recommend(request: RAGRecommendRequest):
    """
    RAG-based recipe recommendations with explanations
    
    Complete pipeline: Retrieve (FAISS) → Rerank (Cross-encoder) → Generate (Gemini LLM)
    
    Request body:
    {
        "ingredients": ["chicken", "pasta", "tomato"],
        "preferences": {
            "vegan": false,
            "glutenFree": false,
            "dairyFree": false
        },
        "excluded_ingredients": ["mushroom"],
        "explain": true,
        "top_k": 10,
        "retrieval_top_k": 50
    }
    
    Response includes:
    - recipes: Top-k reranked recipes
    - explanation: LLM-generated explanation (if explain=true)
    - metadata: Pipeline execution details
    """
    start_time = time.time()
    
    try:
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="Ingredients list is required")
        
        # Prepare preferences dict
        preferences_dict = None
        if request.preferences:
            preferences_dict = {
                "vegan": request.preferences.vegan or False,
                "vegetarian": request.preferences.vegetarian or False,
                "glutenFree": request.preferences.glutenFree or False,
                "dairyFree": request.preferences.dairyFree or False,
                "nutAllergy": request.preferences.nutAllergy or False
            }
        
        top_k = request.top_k if request.top_k is not None else 10
        retrieval_top_k = request.retrieval_top_k if request.retrieval_top_k is not None else 50
        explain = request.explain if request.explain is not None else True
        
        logger.info(
            f"RAG recommendation request: {len(request.ingredients)} ingredients, "
            f"top_k={top_k}, explain={explain}"
        )
        
        # Process through RAG pipeline
        result = rag_pipeline.process(
            user_ingredients=request.ingredients,
            user_preferences=preferences_dict,
            excluded_ingredients=request.excluded_ingredients or [],
            top_k=top_k,
            explain=explain,
            retrieval_top_k=retrieval_top_k
        )
        
        process_time = time.time() - start_time
        logger.info(
            f"RAG pipeline completed in {process_time:.3f}s: "
            f"{len(result['recipes'])} recipes, "
            f"explanation={'yes' if result['explanation'] else 'no'}"
        )
        
        return RAGRecommendResponse(
            recipes=result['recipes'],
            explanation=result['explanation'],
            metadata=result['metadata'],
            count=len(result['recipes'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in RAG recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate RAG recommendations: {str(e)}"
        )

