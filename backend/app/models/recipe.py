from pydantic import BaseModel, Field
from typing import List, Optional


class Recipe(BaseModel):
    Title: str
    Ingredients: str  # Stored as a stringified list in the source data
    Instructions: Optional[str] = ""  # Some recipes might not have instructions
    Image_Name: str
    Cleaned_Ingredients: str  # Stored as a stringified list


class RecipeWithMatch(Recipe):
    matchingCount: int
    matchingIngredients: List[str]


class RecipeSearchParams(BaseModel):
    ingredients: List[str] = []
    limit: int = 50
    offset: int = 0


class RecipeRecommendRequest(BaseModel):
    ingredients: List[str]
    use_vector_search: Optional[bool] = True
    top_k: Optional[int] = 50


class RecipeRecommendResponse(BaseModel):
    recommendations: List[RecipeWithMatch]
    count: int
    userIngredients: List[str]
    search_method: str  # "vector" or "string_matching"


class RecipeSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 20


class RecipeSearchResponse(BaseModel):
    recipes: List[RecipeWithMatch]
    count: int
    query: str
    search_method: str  # "vector" or "string_matching"


class DietaryPreferences(BaseModel):
    """Dietary preferences for RAG recommendations"""
    vegan: Optional[bool] = False
    vegetarian: Optional[bool] = False
    glutenFree: Optional[bool] = False
    dairyFree: Optional[bool] = False
    nutAllergy: Optional[bool] = False


class RAGRecommendRequest(BaseModel):
    """Request model for RAG-based recommendations"""
    ingredients: List[str]
    preferences: Optional[DietaryPreferences] = None
    excluded_ingredients: Optional[List[str]] = None
    explain: Optional[bool] = True
    top_k: Optional[int] = 10
    retrieval_top_k: Optional[int] = 50


class RAGMetadata(BaseModel):
    """Metadata about RAG pipeline execution"""
    retrieval_count: int
    reranked_count: int
    pipeline_stages: List[str]
    retriever_used: bool
    reranker_used: bool
    llm_used: bool


class RAGRecommendResponse(BaseModel):
    """Response model for RAG-based recommendations"""
    recipes: List[RecipeWithMatch]
    explanation: Optional[str] = None
    metadata: RAGMetadata
    count: int

