"""
Unified LLM Client - Abstracts different LLM providers

Supports: Gemini, Anthropic Claude, OpenAI
"""

import json
from typing import Optional, Dict, Any
from config import ModelConfig, get_api_key

class LLMClient:
    """Unified interface for different LLM providers."""

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.provider = model_config.provider

        # Initialize provider-specific client
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            # Try new google.genai library first
            from google import genai

            api_key = get_api_key("gemini")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            client = genai.Client(api_key=api_key)
            self.client = client
            self.use_new_genai = True
            print("[INFO] ✓ Using NEW google.genai library with structured output support")

        except ImportError as e:
            # Fall back to old google.generativeai
            print(f"[WARNING] Could not import google.genai: {e}")
            print("[WARNING] Falling back to OLD google.generativeai library")
            import google.generativeai as genai

            api_key = get_api_key("gemini")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
            self.use_new_genai = False

    def _init_anthropic(self):
        """Initialize Anthropic Claude client."""
        from anthropic import Anthropic

        api_key = get_api_key("anthropic")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)

    def _init_openai(self):
        """Initialize OpenAI client."""
        from openai import OpenAI

        api_key = get_api_key("openai")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text using the configured model.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input/context

        Returns:
            Generated text response
        """
        if self.provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Gemini."""
        # Gemini combines system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        if hasattr(self, 'use_new_genai') and self.use_new_genai:
            config = {"temperature": self.config.temperature}
            if self.config.max_tokens > 0:
                config["max_output_tokens"] = self.config.max_tokens
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=full_prompt,
                config=config,
            )
            return response.text
        else:
            # Old google.generativeai library
            response = self.client.generate_content(full_prompt)
            return response.text

    def _generate_gemini_structured(self, system_prompt: str, user_prompt: str, output_schema) -> str:
        """Generate using Gemini with structured output (JSON schema constraint)."""
        # Gemini combines system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Convert Pydantic schema to JSON schema dict
        if hasattr(output_schema, 'model_json_schema'):
            # Pydantic v2
            schema_dict = output_schema.model_json_schema()
        else:
            # Pydantic v1
            schema_dict = output_schema.schema()

        if hasattr(self, 'use_new_genai') and self.use_new_genai:
            config = {
                "temperature": self.config.temperature,
                "response_mime_type": "application/json",
                "response_json_schema": schema_dict,
            }
            # Only set max_output_tokens if explicitly configured (non-zero)
            if self.config.max_tokens > 0:
                config["max_output_tokens"] = self.config.max_tokens

            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=full_prompt,
                config=config,
            )
            result_text = response.text
            print(f"[DEBUG] Response length: {len(result_text)} characters")

            if hasattr(response, 'candidates') and response.candidates:
                print(f"[DEBUG] Finish reason: {response.candidates[0].finish_reason}")

            return result_text
        else:
            # Old google.generativeai library - fallback to text-based approach
            # The old library has limited/buggy structured output support
            print("[INFO] Using text-based JSON generation (old google.generativeai library)")

            # Add explicit JSON structure to prompt
            json_instruction = "\n\nIMPORTANT: Respond with ONLY a valid JSON object. No markdown, no code blocks, just raw JSON."

            response = self.client.generate_content(f"{full_prompt}\n\n{json_instruction}")
            return response.text

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using Anthropic Claude."""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    def generate_json(self, system_prompt: str, user_prompt: str,
                     output_schema: Any) -> Dict:
        """
        Generate structured JSON output validated against a Pydantic schema.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User input/context
            output_schema: Pydantic model class for validation

        Returns:
            Validated dict matching the schema
        """
        # Use native structured output for Gemini
        if self.provider == "gemini":
            response_text = self._generate_gemini_structured(
                system_prompt,
                user_prompt,
                output_schema
            )
        else:
            # For other providers, add JSON instruction and parse manually
            json_instruction = "\n\nRespond with a valid JSON object matching the schema. Do not include any markdown formatting or code blocks, just the raw JSON."
            response_text = self.generate(
                system_prompt + json_instruction,
                user_prompt
            )

        # Clean up response (remove markdown code blocks if present)
        cleaned = self._extract_json_from_response(response_text)

        print(f"[DEBUG] Cleaned response length: {len(cleaned)} characters")
        print(f"[DEBUG] Starts with: {cleaned[:100]}")
        print(f"[DEBUG] Ends with: ...{cleaned[-100:]}")

        # Parse and validate
        try:
            data = json.loads(cleaned)

            # Handle both Pydantic v1 and v2
            if hasattr(output_schema, 'model_validate'):
                # Pydantic v2
                validated = output_schema.model_validate(data)
            else:
                # Pydantic v1
                validated = output_schema.parse_obj(data)

            return validated
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON response")
            print(f"Response preview: {response_text[:500]}...")
            raise Exception(f"Invalid JSON from {self.provider}") from e
        except Exception as e:
            print(f"[ERROR] Schema validation failed: {e}")
            print(f"[DEBUG] Keys in response: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            print(f"[DEBUG] Full response preview:")
            print(json.dumps(data, indent=2)[:1000])
            raise

    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response, handling markdown code blocks."""
        text = text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            # Try to extract from any code block
            text = text.split("```")[1].split("```")[0].strip()

        return text


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_agent_client(agent_name: str) -> LLMClient:
    """
    Create LLM client for a specific agent using centralized config.

    Args:
        agent_name: One of 'reader', 'methods_extractor', 'positioning', etc.

    Returns:
        Configured LLMClient
    """
    from config import AGENT_MODELS

    if not hasattr(AGENT_MODELS, agent_name):
        raise ValueError(f"Unknown agent: {agent_name}. Check config.py")

    model_config = getattr(AGENT_MODELS, agent_name)
    return LLMClient(model_config)


# ============================================================================
# TESTING
# ============================================================================

def test_client():
    """Test the LLM client with a simple prompt."""
    from config import AGENT_MODELS

    print("Testing Gemini client...")

    client = LLMClient(AGENT_MODELS.reader)

    response = client.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2? Respond with just the number."
    )

    print(f"Response: {response}")
    print("✓ Test successful!")


if __name__ == "__main__":
    test_client()
