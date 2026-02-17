"""
Centralized Configuration for Research Intelligence System

Configure models, API keys, and settings for all agents.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for a specific agent's model."""
    provider: str  # "gemini" | "anthropic" | "openai"
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096

@dataclass
class AgentModels:
    """Model configuration for all agents."""
    # Layer 1: Deep Paper Analysis Agents
    reader: ModelConfig
    methods_extractor: ModelConfig
    positioning: ModelConfig

    # Layer 2: Front Summarization
    front_summarizer: ModelConfig

    # Layer 3: Living Review Updates
    daily_updater: ModelConfig
    weekly_revisor: ModelConfig
    monthly_rewriter: ModelConfig
    email_generator: ModelConfig

# ============================================================================
# DEFAULT CONFIGURATION - Using Gemini Flash
# ============================================================================

DEFAULT_AGENT_MODELS = AgentModels(
    # Layer 1 - Use Gemini Flash for cost efficiency
    reader=ModelConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=0  # 0 = no limit, let model decide
    ),
    methods_extractor=ModelConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=0
    ),
    positioning=ModelConfig(
        provider="gemini",
        model_name="gemini-3-pro-preview",
        temperature=0.0,
        max_tokens=0
    ),

    # Layer 2 - Use stable Gemini Flash for summarization
    front_summarizer=ModelConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.3,
        max_tokens=0  # 0 = no limit; structured output length is schema-driven
    ),

    # Layer 3 - Use Gemini Pro for narrative quality
    daily_updater=ModelConfig(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        temperature=0.3,
        max_tokens=8000
    ),
    weekly_revisor=ModelConfig(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        temperature=0.3,
        max_tokens=8000
    ),
    monthly_rewriter=ModelConfig(
        provider="gemini",
        model_name="gemini-3-flash-preview",
        temperature=0.5,
        max_tokens=8000
    ),
    email_generator=ModelConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.5,
        max_tokens=0
    )
)

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider from environment."""
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY"
    }

    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    return os.getenv(env_var)

# ============================================================================
# COST ESTIMATES (per 1M tokens)
# ============================================================================

COST_PER_MILLION_TOKENS = {
    # Gemini (2026-02-11)
    "gemini-3-flash-preview": {"input": 0.5, "output": 3.0},  
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-3-pro-preview" : {"input": 2.0, "output": 12.00},
    
    # Anthropic Claude
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},

    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a model call."""
    if model_name not in COST_PER_MILLION_TOKENS:
        return 0.0

    rates = COST_PER_MILLION_TOKENS[model_name]
    cost = (input_tokens / 1_000_000 * rates["input"] +
            output_tokens / 1_000_000 * rates["output"])
    return cost

# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

# Can be overridden by environment variables or runtime settings
AGENT_MODELS = DEFAULT_AGENT_MODELS

def override_model(agent_name: str, model_config: ModelConfig):
    """Override model for a specific agent at runtime."""
    global AGENT_MODELS
    if hasattr(AGENT_MODELS, agent_name):
        setattr(AGENT_MODELS, agent_name, model_config)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

# ============================================================================
# DATABASE & PATHS
# ============================================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "src" / "db" / "research_intelligence.db"
PDF_CACHE_DIR = PROJECT_ROOT / "cache" / "pdfs"
LIVING_REVIEWS_DIR = PROJECT_ROOT / "docs" / "living_reviews"
GRAPHS_DIR = LIVING_REVIEWS_DIR / "graphs"

# ============================================================================
# DISPLAY CURRENT CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration."""
    print("="*70)
    print("RESEARCH INTELLIGENCE SYSTEM - CONFIGURATION")
    print("="*70)
    print("\nLayer 1 Agents:")
    print(f"  Reader:           {AGENT_MODELS.reader.provider} / {AGENT_MODELS.reader.model_name}")
    print(f"  Methods:          {AGENT_MODELS.methods_extractor.provider} / {AGENT_MODELS.methods_extractor.model_name}")
    print(f"  Positioning:      {AGENT_MODELS.positioning.provider} / {AGENT_MODELS.positioning.model_name}")

    print("\nLayer 2:")
    print(f"  Front Summarizer: {AGENT_MODELS.front_summarizer.provider} / {AGENT_MODELS.front_summarizer.model_name}")

    print("\nLayer 3:")
    print(f"  Daily Updater:    {AGENT_MODELS.daily_updater.provider} / {AGENT_MODELS.daily_updater.model_name}")
    print(f"  Weekly Revisor:   {AGENT_MODELS.weekly_revisor.provider} / {AGENT_MODELS.weekly_revisor.model_name}")
    print(f"  Monthly Rewriter: {AGENT_MODELS.monthly_rewriter.provider} / {AGENT_MODELS.monthly_rewriter.model_name}")
    print(f"  Email Generator:  {AGENT_MODELS.email_generator.provider} / {AGENT_MODELS.email_generator.model_name}")

    print("\nPaths:")
    print(f"  Database:         {DB_PATH}")
    print(f"  PDF Cache:        {PDF_CACHE_DIR}")
    print(f"  Living Reviews:   {LIVING_REVIEWS_DIR}")
    print("="*70)

if __name__ == "__main__":
    print_config()
