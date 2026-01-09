import React, { useState, useEffect, useMemo } from 'react';
import { useFridge } from '../store/FridgeContext';
import { getRecipeImageUrl } from '../utils/helpers';
import { getRecommendations, getRAGRecommendations, ApiError } from '../utils/api';
import { Link } from 'react-router-dom';
import { RecipeWithMatch } from '../types';
import { filterRecipes, getActiveFilterLabels } from '../utils/recipeFilter';

const RecipesPage: React.FC = () => {
    const { fridgeIngredients, dietaryPreferences, excludedIngredients } = useFridge();
    const [suitableRecipes, setSuitableRecipes] = useState<RecipeWithMatch[]>([]);
    const [displayedRecipes, setDisplayedRecipes] = useState<RecipeWithMatch[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(1);
    const [responseTime, setResponseTime] = useState<number | null>(null);
    const [useRAG, setUseRAG] = useState(true); // RAG pipeline kullanımı (default: true)
    const [ragExplanation, setRagExplanation] = useState<string | null>(null);
    const [ragMetadata, setRagMetadata] = useState<any>(null);
    
    const RECIPES_PER_PAGE = 12;

    const fetchRecipes = async () => {
        if (fridgeIngredients.length === 0) {
            setSuitableRecipes([]);
            setDisplayedRecipes([]);
            setPage(1);
            setRagExplanation(null);
            setRagMetadata(null);
            return;
        }

        setLoading(true);
        setError(null);
        setResponseTime(null);
        setRagExplanation(null);
        setRagMetadata(null);

        try {
            const startTime = performance.now();
            
            if (useRAG) {
                // RAG Pipeline: Retrieve → Rerank → Generate
                const ragResponse = await getRAGRecommendations({
                    ingredients: fridgeIngredients,
                    preferences: dietaryPreferences,
                    excluded_ingredients: excludedIngredients,
                    explain: true,
                    top_k: 50, // Get more recipes for pagination
                    retrieval_top_k: 100
                });
                
                const endTime = performance.now();
                setResponseTime(Math.round(endTime - startTime));
                
                // RAG response already includes filtered recipes
                const allRecipes = ragResponse.recipes || [];
                
                // Store explanation and metadata
                setRagExplanation(ragResponse.explanation || null);
                setRagMetadata(ragResponse.metadata || null);
                
                setSuitableRecipes(allRecipes);
                setDisplayedRecipes(allRecipes.slice(0, RECIPES_PER_PAGE));
            } else {
                // Traditional recommendation endpoint
                const response = await getRecommendations(fridgeIngredients);
                const endTime = performance.now();
                
                setResponseTime(Math.round(endTime - startTime));
                const allRecipes = response.recommendations || [];
                
                // Apply filters
                const filteredRecipes = filterRecipes(allRecipes, dietaryPreferences, excludedIngredients);
                
                setSuitableRecipes(filteredRecipes);
                setDisplayedRecipes(filteredRecipes.slice(0, RECIPES_PER_PAGE));
            }
            
            setPage(1);
        } catch (err) {
            const apiError = err as ApiError;
            setError(apiError.message);
            setSuitableRecipes([]);
            setDisplayedRecipes([]);
            setRagExplanation(null);
            setRagMetadata(null);
        } finally {
            setLoading(false);
        }
    };

    const loadMore = () => {
        const nextPage = page + 1;
        const startIdx = 0;
        const endIdx = nextPage * RECIPES_PER_PAGE;
        setDisplayedRecipes(suitableRecipes.slice(startIdx, endIdx));
        setPage(nextPage);
    };

    // Get active filter labels
    const activeFilters = useMemo(() => {
        return getActiveFilterLabels(dietaryPreferences, excludedIngredients);
    }, [dietaryPreferences, excludedIngredients]);

    useEffect(() => {
        fetchRecipes();
    }, [fridgeIngredients, dietaryPreferences, excludedIngredients, useRAG]);

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <div className="mb-8">
                <div className="flex items-center justify-between">
                    <div>
                        <div className="flex items-center gap-4">
                            <h1 className="text-3xl font-bold text-gray-900">Recommended Recipes</h1>
                            {/* RAG Toggle */}
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={useRAG}
                                    onChange={(e) => setUseRAG(e.target.checked)}
                                    className="sr-only peer"
                                />
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                                <span className="ml-3 text-sm font-medium text-gray-700">
                                    {useRAG ? '🤖 RAG Pipeline' : '🔍 Standard Search'}
                                </span>
                            </label>
                        </div>
                        <p className="mt-2 text-gray-600">
                            Based on your fridge content: <span className="italic">{fridgeIngredients.join(', ') || "Nothing yet"}</span>
                        </p>
                        {activeFilters.length > 0 && (
                            <div className="mt-3 flex flex-wrap items-center gap-2">
                                <span className="text-sm font-semibold text-gray-700">Filters Active:</span>
                                {activeFilters.map((filter, idx) => (
                                    <span
                                        key={idx}
                                        className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 border border-green-200"
                                    >
                                        {filter}
                                    </span>
                                ))}
                            </div>
                        )}
                        {/* RAG Metadata */}
                        {ragMetadata && useRAG && (
                            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                                <span className="text-gray-500">Pipeline:</span>
                                {ragMetadata.retriever_used && (
                                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">FAISS</span>
                                )}
                                {ragMetadata.reranker_used && (
                                    <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded">Reranker</span>
                                )}
                                {ragMetadata.llm_used && (
                                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded">LLM</span>
                                )}
                                <span className="text-gray-500">
                                    ({ragMetadata.retrieval_count} → {ragMetadata.reranked_count})
                                </span>
                            </div>
                        )}
                    </div>
                    {responseTime && (
                        <div className="text-sm text-gray-500">
                            <span className="font-medium">⚡ {responseTime}ms</span>
                        </div>
                    )}
                </div>
                {suitableRecipes.length > 0 && (
                    <div className="mt-3 text-sm text-gray-600">
                        Showing {displayedRecipes.length} of {suitableRecipes.length} recipes
                    </div>
                )}
                {/* RAG Explanation */}
                {ragExplanation && useRAG && (
                    <div className="mt-4 bg-gradient-to-r from-blue-50 to-purple-50 border-l-4 border-primary rounded-lg p-4 shadow-sm">
                        <div className="flex items-start">
                            <div className="flex-shrink-0">
                                <svg className="h-5 w-5 text-primary" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                </svg>
                            </div>
                            <div className="ml-3 flex-1">
                                <h3 className="text-sm font-semibold text-gray-900 mb-2">🤖 AI Explanation</h3>
                                <p className="text-sm text-gray-700 whitespace-pre-line leading-relaxed">
                                    {ragExplanation}
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {loading ? (
                <div className="flex flex-col items-center justify-center py-16">
                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mb-4"></div>
                    <p className="text-gray-600">Loading recipes...</p>
                </div>
            ) : error ? (
                <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-md">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3 flex-1">
                            <p className="text-sm text-red-700">
                                {error}
                            </p>
                            <button
                                onClick={fetchRecipes}
                                className="mt-2 text-sm font-medium text-red-700 hover:text-red-600 underline"
                            >
                                Try again
                            </button>
                        </div>
                    </div>
                </div>
            ) : fridgeIngredients.length === 0 ? (
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-md">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <p className="text-sm text-yellow-700">
                                Your fridge is empty! <Link to="/" className="font-medium underline hover:text-yellow-600">Go back and add ingredients</Link> to see recipes.
                            </p>
                        </div>
                    </div>
                </div>
            ) : suitableRecipes.length === 0 ? (
                <div className="text-center py-16">
                     <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h3 className="text-lg font-medium text-gray-900">No matching recipes found</h3>
                    <p className="mt-1 text-gray-500">Try adding more versatile ingredients like onions, garlic, or eggs.</p>
                </div>
            ) : (
                <>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                    {displayedRecipes.map((recipe, index) => (
                        <Link 
                            to={`/recipe/${encodeURIComponent(recipe.Title)}`} 
                            state={{ matchingIngredients: recipe.matchingIngredients }} 
                            key={index}
                            className="group flex flex-col bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-100"
                        >
                            <div className="relative h-48 w-full overflow-hidden">
                                <img 
                                    src={getRecipeImageUrl(recipe.Image_Name)} 
                                    alt={recipe.Title}
                                    className="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-500"
                                    onError={(e) => {
                                        (e.target as HTMLImageElement).src = 'https://via.placeholder.com/400x300?text=Recipe';
                                    }}
                                />
                                <div className="absolute top-3 right-3 bg-primary text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg">
                                    {recipe.matchingCount} Matches
                                </div>
                                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                            <div className="flex-1 p-6 flex flex-col">
                                <h3 className="text-xl font-bold text-gray-900 group-hover:text-primary transition-colors line-clamp-2 mb-2">
                                    {recipe.Title}
                                </h3>
                                <div className="mb-2">
                                    <p className="text-sm text-gray-600">
                                        <span className="font-semibold">Matching Ingredients:</span> {recipe.matchingIngredients.join(', ')}
                                    </p>
                                </div>
                                <div className="mt-auto pt-4 border-t border-gray-100 flex items-center justify-between">
                                    <span className="text-sm font-medium text-primary">View Recipe &rarr;</span>
                                    <span className="text-xs text-gray-400 uppercase tracking-wider">
                                        Kitchen Ready
                                    </span>
                                </div>
                            </div>
                        </Link>
                    ))}
                </div>
                
                {displayedRecipes.length < suitableRecipes.length && (
                    <div className="mt-12 text-center">
                        <button
                            onClick={loadMore}
                            className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary transition-all"
                        >
                            Load More Recipes
                            <svg xmlns="http://www.w3.org/2000/svg" className="ml-2 h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                        </button>
                        <p className="mt-2 text-sm text-gray-500">
                            {suitableRecipes.length - displayedRecipes.length} more recipes available
                        </p>
                    </div>
                )}
                </>
            )}
        </div>
    );
};

export default RecipesPage;