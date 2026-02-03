
import os
import logging
import json
from google import genai
from google.genai import types

logger = logging.getLogger("llm_client")

class GeminiClient:
    """
    Wrapper for google-genai SDK to use gemini-3-flash-preview.
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            
    def generate_response(self, system_instruction: str, user_content: str, model: str = "gemini-2.0-flash-thinking-exp-1219") -> str:
        """
        Generates content using the specified model.
        """
        if not self.client:
            return '{"error": "API Key Missing"}'
            
        try:
            # Construct the prompt
            # The new SDK might handle system instructions differently or via config.
            # User example: client.models.generate_content(model=..., contents=..., config=...)
            
            # Combining system and user for simplicity if config approach is complex, 
            # but usually config argument supports system_instruction.
            
            full_prompt = f"SYSTEM: {system_instruction}\n\nUSER: {user_content}"
            
            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                ),
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Generate Content Failed: {e}")
            return json.dumps({"error": str(e)})

# Singleton instance
_client = GeminiClient()

def run_llm(system_prompt: str, user_prompt: str) -> str:
    return _client.generate_response(system_prompt, user_prompt)
