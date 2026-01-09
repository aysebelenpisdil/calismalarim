"""
LLM Service
Handles text generation using Google Gemini API
Provides explanations for recipe recommendations
"""

import logging
from typing import List, Optional, Dict, Any
import google.generativeai as genai
from app.config import settings
from app.models.recipe import Recipe

# Setup logger
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for generating explanations using Google Gemini API
    Provides personalized recipe recommendation explanations
    """
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.temperature = settings.GEMINI_TEMPERATURE
        self.enabled = settings.GEMINI_ENABLED
        self.model: Optional[genai.GenerativeModel] = None
        self._model_loaded = False
    
    def _load_model(self):
        """Initialize Gemini API client and load model"""
        if not self.enabled:
            logger.debug("LLM service is disabled")
            return
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            logger.warning("LLM explanations will be disabled")
            self.enabled = False
            return
        
        if not self._model_loaded:
            try:
                logger.info(f"Initializing Gemini API: {self.model_name}")
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self._model_loaded = True
                logger.info(f"Gemini model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Error initializing Gemini API: {e}", exc_info=True)
                logger.warning("LLM explanations will be disabled")
                self.enabled = False
                self._model_loaded = False
                raise
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available and ready
        
        Returns:
            True if model is loaded and API key is available, False otherwise
        """
        return self.enabled and self._model_loaded and self.model is not None
    
    def _build_prompt(
        self,
        user_ingredients: List[str],
        recommended_recipes: List[Recipe],
        user_preferences: Optional[Dict[str, Any]] = None,
        excluded_ingredients: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for Gemini API
        
        Args:
            user_ingredients: List of user's fridge ingredients
            recommended_recipes: List of recommended Recipe objects
            user_preferences: Dietary preferences dict (vegan, gluten_free, etc.)
            excluded_ingredients: List of excluded ingredients
            
        Returns:
            Formatted prompt string
        """
        # System prompt
        system_prompt = """You are a professional chef and culinary advisor. You recommend recipes based on the user's available ingredients and explain why these recipes were selected.

Your task:
1. Explain why the recommended recipes were chosen
2. Specify which ingredients match
3. If any ingredients are missing, mention them and suggest alternatives
4. Pay attention to user preferences (vegan, gluten-free, etc.)
5. Use a concise, clear, and friendly tone (English)

Response format:
- A brief explanation for each recipe (1-2 sentences)
- A general summary (why these recipes were recommended)
- Missing ingredients and alternatives (if any)
"""
        
        # User context
        context_parts = []
        context_parts.append(f"**Available Ingredients:** {', '.join(user_ingredients)}")
        
        if user_preferences:
            active_prefs = []
            if user_preferences.get('vegan'):
                active_prefs.append('Vegan')
            if user_preferences.get('vegetarian') and not user_preferences.get('vegan'):
                active_prefs.append('Vegetarian')
            if user_preferences.get('glutenFree'):
                active_prefs.append('Gluten-Free')
            if user_preferences.get('dairyFree'):
                active_prefs.append('Dairy-Free')
            if user_preferences.get('nutAllergy'):
                active_prefs.append('Nut Allergy')
            
            if active_prefs:
                context_parts.append(f"**Dietary Preferences:** {', '.join(active_prefs)}")
        
        if excluded_ingredients:
            context_parts.append(f"**Excluded Ingredients:** {', '.join(excluded_ingredients)}")
        
        # Recommended recipes
        recipes_text = "\n\n**Recommended Recipes:**\n"
        for i, recipe in enumerate(recommended_recipes[:10], 1):  # Max 10 recipes
            ingredients_text = recipe.Cleaned_Ingredients or recipe.Ingredients
            ingredients_clean = ingredients_text.replace('[', '').replace(']', '').replace("'", '')
            
            recipes_text += f"\n{i}. **{recipe.Title}**\n"
            recipes_text += f"   Ingredients: {ingredients_clean[:200]}...\n"
            if recipe.Instructions:
                instructions_short = recipe.Instructions[:150] + "..." if len(recipe.Instructions) > 150 else recipe.Instructions
                recipes_text += f"   Preparation: {instructions_short}\n"
        
        # Combine all parts
        prompt = f"""{system_prompt}

---

**User Information:**
{chr(10).join(context_parts)}
{recipes_text}

---

Please explain why these recipes were recommended. Use a concise, clear, and friendly tone. Respond in English."""
        
        return prompt
    
    def generate_explanation(
        self,
        user_ingredients: List[str],
        recommended_recipes: List[Recipe],
        user_preferences: Optional[Dict[str, Any]] = None,
        excluded_ingredients: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Generate explanation for recipe recommendations using Gemini API
        
        Args:
            user_ingredients: List of user's fridge ingredients
            recommended_recipes: List of recommended Recipe objects
            user_preferences: Dietary preferences dict
            excluded_ingredients: List of excluded ingredients
            
        Returns:
            Explanation text or None if generation fails
        """
        if not recommended_recipes:
            logger.warning("No recipes provided for explanation generation")
            return None
        
        if not self.enabled:
            logger.debug("LLM service is disabled, skipping explanation generation")
            return None
        
        if not self.api_key:
            logger.debug("GEMINI_API_KEY not found, skipping explanation generation")
            return None
        
        try:
            # Load model if not loaded (lazy loading)
            if not self._model_loaded:
                self._load_model()
            
            # Check if model loaded successfully
            if not self.is_available():
                logger.warning("LLM model could not be loaded, skipping explanation generation")
                return None
            
            # Build prompt
            prompt = self._build_prompt(
                user_ingredients=user_ingredients,
                recommended_recipes=recommended_recipes,
                user_preferences=user_preferences,
                excluded_ingredients=excluded_ingredients
            )
            
            logger.debug(f"Generating explanation for {len(recommended_recipes)} recipes")
            
            # Generate response (Gemini API is synchronous)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            explanation = response.text.strip()
            
            logger.debug(f"Explanation generated: {len(explanation)} characters")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}", exc_info=True)
            logger.warning("Returning None for explanation")
            return None
    
    def get_model_info(self) -> dict:
        """Get information about the LLM service"""
        return {
            "model_name": self.model_name,
            "loaded": self._model_loaded,
            "enabled": self.enabled,
            "has_api_key": bool(self.api_key),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


# Singleton instance
llm_service = LLMService()

