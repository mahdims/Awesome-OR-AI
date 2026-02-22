"""
Centralized Configuration for Research Intelligence System

Configure models, API keys, and settings for all agents.
"""

import os
import yaml
from pathlib import Path
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
# COST ESTIMATES (loaded from research_config/model_config.yaml)
# ============================================================================

# Will be populated by _load_model_config_from_yaml()
COST_PER_MILLION_TOKENS = {}

def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a model call."""
    if model_name not in COST_PER_MILLION_TOKENS:
        return 0.0

    rates = COST_PER_MILLION_TOKENS[model_name]
    cost = (input_tokens / 1_000_000 * rates["input"] +
            output_tokens / 1_000_000 * rates["output"])
    return cost

# ============================================================================
# LOAD MODELS FROM YAML CONFIGURATION
# ============================================================================

def _load_model_config_from_yaml():
    """Load model configuration from research_config/model_config.yaml."""
    PROJECT_ROOT = Path(__file__).parent.parent
    model_config_path = PROJECT_ROOT / "research_config" / "model_config.yaml"

    if not model_config_path.exists():
        print(f"[WARNING] Model config not found at {model_config_path}, using hardcoded defaults")
        return DEFAULT_AGENT_MODELS

    try:
        with open(str(model_config_path), 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        def parse_model_config(layer_config: dict) -> ModelConfig:
            """Parse YAML model config dict into ModelConfig object."""
            return ModelConfig(
                provider=layer_config.get('provider', 'gemini'),
                model_name=layer_config.get('model_name', 'gemini-2.5-flash'),
                temperature=layer_config.get('temperature', 0.0),
                max_tokens=layer_config.get('max_tokens', 0)
            )

        # Parse Layer 1 configs
        layer1 = config.get('layer1', {})
        reader = parse_model_config(layer1.get('reader', {}))
        methods = parse_model_config(layer1.get('methods_extractor', {}))
        positioning = parse_model_config(layer1.get('positioning', {}))

        # Parse Layer 2 config
        layer2 = config.get('layer2', {})
        front_summarizer = parse_model_config(layer2.get('front_summarizer', {}))

        # Parse Layer 3 configs
        layer3 = config.get('layer3', {})
        daily_updater = parse_model_config(layer3.get('daily_updater', {}))
        weekly_revisor = parse_model_config(layer3.get('weekly_revisor', {}))
        monthly_rewriter = parse_model_config(layer3.get('monthly_rewriter', {}))
        email_generator = parse_model_config(layer3.get('email_generator', {}))

        # Load cost estimates from YAML
        global COST_PER_MILLION_TOKENS
        COST_PER_MILLION_TOKENS = config.get('cost_per_million_tokens', {})
        if COST_PER_MILLION_TOKENS:
            print(f"[CONFIG] Loaded cost estimates for {len(COST_PER_MILLION_TOKENS)} models")

        print(f"[CONFIG] Loaded model configuration from {model_config_path}")
        return AgentModels(
            reader=reader,
            methods_extractor=methods,
            positioning=positioning,
            front_summarizer=front_summarizer,
            daily_updater=daily_updater,
            weekly_revisor=weekly_revisor,
            monthly_rewriter=monthly_rewriter,
            email_generator=email_generator
        )

    except Exception as e:
        print(f"[ERROR] Failed to load model config from {model_config_path}: {e}")
        print(f"[WARNING] Using hardcoded defaults")
        return DEFAULT_AGENT_MODELS

# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

# Load models from YAML, fallback to defaults if not available
AGENT_MODELS = _load_model_config_from_yaml()

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
