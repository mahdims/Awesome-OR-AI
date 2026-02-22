# Research Domain Configuration Guide

This directory contains **all configuration needed to adapt the system to a new research domain**. No code changes required!

## Quick Start: Adapting to "AI for Healthcare"

To adapt this system from "OR-AI Intersection" to "AI for Healthcare" (or any other research area), edit only these files in this directory:

### 1. Edit `research_domain.yaml` (Required)

Define your research categories, search strategies, and filters:

```yaml
domain_name: "AI for Healthcare"

github:
  user_name: "your-username"
  repo_name: "AI-Healthcare-Papers"

categories:
  "LLMs for Clinical Decision Support":
    description: "Large language models for diagnosis, treatment planning, and clinical reasoning"
    search_strategy: "keyword"
    filters:
      - "clinical decision support LLM"
      - "medical diagnosis AI"
      - "clinical reasoning language model"
      - "electronic health records NLP"
      - "medical question answering"
    max_results: 500

  "Medical Image Analysis":
    description: "Deep learning for radiology, pathology, and medical imaging"
    search_strategy: "keyword"
    filters:
      - "medical image segmentation"
      - "radiology deep learning"
      - "pathology AI"
      - "CT scan analysis"
      - "MRI deep learning"
    max_results: 500

  "Drug Discovery with AI":
    description: "AI for molecular design, drug-target prediction, and clinical trials"
    search_strategy: "citation"  # Use citation-based discovery
    seed_papers:
      - "2301.12345"  # Replace with key papers in drug discovery AI
      - "2302.67890"
      - "2303.45678"
    max_results: 300
```

**Key fields:**
- `domain_name`: Human-readable name for your research area
- `categories`: Dictionary of research topics (order controls display)
  - `description`: Clear explanation of category scope (used in LLM relevance scoring)
  - `search_strategy`: Either `"keyword"` (ArXiv search) or `"citation"` (Semantic Scholar crawl)
  - `filters`: For keyword strategy — list of search terms (OR'd together)
  - `seed_papers`: For citation strategy — list of ArXiv IDs to crawl citations from
  - `max_results`: Maximum papers to fetch per category

### 2. Edit `researcher_profile.md` (Required)

Add per-category sections describing your research interests. The LLM uses this to score paper relevance (0-10):

```markdown
# Researcher Profile

## Research by Category

### [LLMs for Clinical Decision Support] — PRIMARY FOCUS

**AI-powered diagnostic reasoning and clinical workflows**

Active projects:
- Building LLM-based differential diagnosis systems
- Evaluating safety and hallucination in medical LLMs
- Integrating EHR data with foundation models

Key methodologies:
- Retrieval-augmented generation for medical knowledge
- Fine-tuning on clinical notes and literature
- Human-in-the-loop validation workflows

Reading priorities:
- **Must-read**: Novel architectures for medical reasoning, safety improvements
- **High priority**: Clinical evaluation benchmarks, deployment case studies
- **Medium priority**: General LLM advances applicable to healthcare

### [Medical Image Analysis]

**Deep learning for radiology and pathology**

Active projects:
- Multimodal fusion of imaging and clinical data
- Explainable AI for radiology reports
- Few-shot learning for rare diseases

...

### [Drug Discovery with AI]

...
```

**Structure:**
- Use `### [Category Name]` headings (must match categories in `research_domain.yaml`)
- Describe your active projects, methodologies, and reading priorities
- The LLM scores papers 8-10 if they match your active work, 5-7 if related, 2-4 if tangential

### 3. (Optional) Edit `constants.yaml`

Adjust system thresholds if needed:

```yaml
# For healthcare, you might want to go back further in time
min_year: 2020  # Instead of 2024

# Be more selective with scoring threshold
relevance_threshold: 7  # Instead of 6
```

### 4. (Optional) Edit `model_config.yaml`

Change LLM models if you prefer different providers:

```yaml
layer0:
  relevance_scorer:
    provider: "anthropic"  # Switch from "openai"
    model_name: "claude-haiku-4-5-20251001"
    temperature: 0.0
```

### 5. Run the System

That's it! The system automatically:
- Fetches papers for your new categories
- Applies researcher profile context for scoring
- Generates domain-specific README
- Updates GitHub Pages with new categories

```bash
# Fetch papers for all categories
python src/layer0/fetch_and_score.py

# Or fetch for a single category
python src/layer0/fetch_and_score.py --category "LLMs for Clinical Decision Support"
```

## File Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `research_domain.yaml` | Categories, filters, search strategies, output paths | ✅ **Always** — for new domains |
| `researcher_profile.md` | Per-category research context for LLM scoring | ✅ **Always** — personalize to your interests |
| `constants.yaml` | MIN_YEAR, RELEVANCE_THRESHOLD, API settings | Maybe — if domain needs different thresholds |
| `model_config.yaml` | LLM assignments per agent/layer | Maybe — if you want different models |
| `README.md` | This guide | No — unless improving documentation |

## Configuration Details

### Search Strategies

**1. Keyword-based (`search_strategy: "keyword"`)**
- Uses ArXiv API to search by keywords
- Filters are OR'd together (any match includes the paper)
- Multi-word terms wrapped in quotes for exact match
- Best for: Well-defined topics with clear terminology

**Example:**
```yaml
filters:
  - "medical diagnosis LLM"
  - "clinical reasoning AI"
  - ["deep learning", "radiology"]  # Both terms required (AND)
```

**2. Citation-based (`search_strategy: "citation"`)**
- Uses Semantic Scholar API to crawl citations of seed papers
- Discovers papers through citation network (both citing and cited-by)
- Best for: Emerging fields, niche topics, or when you have key papers

**Example:**
```yaml
seed_papers:
  - "2301.12345"  # Seminal paper in your field
  - "2302.67890"  # Another key paper
```

### Relevance Scoring

The system uses an LLM to score each paper 0-10 for relevance. The scoring prompt includes:
1. **Category description** (from `research_domain.yaml`)
2. **Researcher context** (from `researcher_profile.md`)
3. **Paper title and abstract**

Papers scoring ≥ `relevance_threshold` (default 6) are included.

**Scoring guide used by LLM:**
- **8-10**: Directly addresses your active projects, methods, or stated priorities
- **5-7**: Related to the category and potentially useful
- **2-4**: Tangentially related, different focus or methods
- **0-1**: Completely unrelated

**Tip**: Make your researcher profile specific! The more detail about your active projects and methodologies, the better the LLM can judge relevance.

### Output Configuration

```yaml
output:
  readme:
    enabled: true  # Generate README.md for GitHub
    json_path: './docs/or-llm-daily.json'
    md_path: 'README.md'
  gitpage:
    enabled: true  # Generate docs/index.md for GitHub Pages
    json_path: './docs/or-llm-daily-web.json'
    md_path: './docs/index.md'
```

You can disable either output by setting `enabled: false`.

## Advanced Configuration

### Multi-term AND Filters

For keyword search, you can require multiple terms to appear:

```yaml
filters:
  - ["integer programming", "LLM"]  # Both required (AND)
  - ["scheduling", "optimization", "GPU"]  # All three required
  - "deep learning"  # Single term (no AND)
```

This creates queries like: `("integer programming" AND LLM) OR ("scheduling" AND "optimization" AND GPU) OR "deep learning"`

### Per-category Max Results

Control how many papers to fetch per category:

```yaml
categories:
  "High-volume Topic":
    max_results: 1000  # Fetch more papers

  "Niche Topic":
    max_results: 100  # Fetch fewer papers
```

Global `display.max_results` sets the default.

### API Retry Configuration

Adjust resilience for unreliable networks:

```yaml
arxiv:
  max_retries: 5  # More retries
  retry_delay: 5  # Longer delay between retries
  timeout: 60  # Longer timeout
```

## Examples of Adaptation

### Example 1: "Quantum Computing for Optimization"

```yaml
domain_name: "Quantum Computing for Optimization"

categories:
  "QAOA and Variational Algorithms":
    description: "Quantum Approximate Optimization Algorithm and variational quantum eigensolvers for combinatorial optimization"
    search_strategy: "keyword"
    filters:
      - "QAOA"
      - "variational quantum eigensolver"
      - "quantum approximate optimization"
      - "quantum combinatorial optimization"

  "Quantum Annealing":
    description: "D-Wave and quantum annealing for optimization problems"
    search_strategy: "keyword"
    filters:
      - "quantum annealing"
      - "D-Wave"
      - "adiabatic quantum computation"

  "Quantum Machine Learning":
    description: "Quantum algorithms for machine learning and optimization"
    search_strategy: "citation"
    seed_papers:
      - "1804.03719"  # Quantum Machine Learning: What Quantum Computing Means to Data Mining
```

### Example 2: "Reinforcement Learning for Robotics"

```yaml
domain_name: "Reinforcement Learning for Robotics"

categories:
  "Sim-to-Real Transfer":
    description: "Transfer learning from simulation to real robots"
    search_strategy: "keyword"
    filters:
      - "sim-to-real"
      - "domain randomization robotics"
      - "reality gap"
      - "sim2real"

  "Multi-Agent Robotics":
    description: "Multi-robot coordination and swarm robotics"
    search_strategy: "keyword"
    filters:
      - ["multi-agent", "robotics"]
      - ["swarm robotics", "reinforcement learning"]
      - "multi-robot coordination"

  "Robot Manipulation":
    description: "Grasping, manipulation, and dexterous control"
    search_strategy: "citation"
    seed_papers:
      - "2209.11738"  # Example: RT-1 or similar
```

## Workflow Integration

After editing configuration, the system workflows automatically:

1. **Daily/Bi-daily** (GitHub Actions `bidaily_email.yml`):
   - Fetch papers for each category
   - Deep analysis (Layer 1)
   - Email briefing

2. **Weekly** (GitHub Actions `weekly_report.yml`):
   - Full pipeline including front detection (Layer 2)
   - Per-category weekly emails

3. **Code link updates** (GitHub Actions `update.yml`):
   - Refresh code repository links weekly

You don't need to manually update workflows — they read categories from `research_domain.yaml` dynamically.

## Troubleshooting

### "No papers found for category"

**Problem**: ArXiv search returns no results

**Solutions**:
- Check if filters are too specific — try broader terms
- Verify keywords match ArXiv's terminology
- Switch to `citation` strategy if keyword search isn't working
- Lower `min_year` in `constants.yaml` to include older papers

### "All papers scored below threshold"

**Problem**: Papers are fetched but filtered out by relevance scoring

**Solutions**:
- Lower `relevance_threshold` in `constants.yaml` (from 6 to 4 or 5)
- Make category `description` more specific in `research_domain.yaml`
- Add more context to `researcher_profile.md` so LLM understands your interests better

### "Too many irrelevant papers"

**Problem**: Papers pass relevance threshold but aren't useful

**Solutions**:
- Increase `relevance_threshold` in `constants.yaml` (from 6 to 7 or 8)
- Make filters more specific in `research_domain.yaml`
- Add more detail to `researcher_profile.md` about what you're NOT interested in

## Questions?

For architecture details and layer documentation, see [src/README.md](../src/README.md).

For Claude Code integration and commands, see [.claude/CLAUDE.md](../.claude/CLAUDE.md).

For implementation history and design decisions, see [docs/PLAN.md](../docs/PLAN.md).
