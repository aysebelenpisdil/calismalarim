export interface Recipe {
    Title: string;
    Ingredients: string; // Stored as a stringified list in the source data
    Instructions: string;
    Image_Name: string;
    Cleaned_Ingredients: string; // Stored as a stringified list
}

export interface RecipeWithMatch extends Recipe {
    matchingCount: number;
    matchingIngredients: string[];
}

export interface Ingredient {
    name: string;
}

/**
 * Dietary preferences for RAG recommendations
 */
export interface DietaryPreferences {
    vegan?: boolean;
    vegetarian?: boolean;
    glutenFree?: boolean;
    dairyFree?: boolean;
    nutAllergy?: boolean;
}

/**
 * Request model for RAG-based recommendations
 */
export interface RAGRecommendRequest {
    ingredients: string[];
    preferences?: DietaryPreferences;
    excluded_ingredients?: string[];
    explain?: boolean;
    top_k?: number;
    retrieval_top_k?: number;
}

/**
 * Metadata about RAG pipeline execution
 */
export interface RAGMetadata {
    retrieval_count: number;
    reranked_count: number;
    pipeline_stages: string[];
    retriever_used: boolean;
    reranker_used: boolean;
    llm_used: boolean;
}

/**
 * Response model for RAG-based recommendations
 */
export interface RAGRecommendResponse {
    recipes: RecipeWithMatch[];
    explanation: string | null;
    metadata: RAGMetadata;
    count: number;
}
