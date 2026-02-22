# Domain Adaptation Notes

This document describes known limitations and domain-specific components when adapting this system to a new research area.

## Fully Configurable (No Code Changes Required)

✅ **Categories and Search Strategies** ([research_domain.yaml](research_domain.yaml))
- Define your research categories
- Specify search keywords or seed papers
- Configure output paths

✅ **Researcher Profile** ([researcher_profile.md](researcher_profile.md))
- Describe your research interests per category
- Specify active projects and methodologies
- Define reading priorities

✅ **System Constants** ([constants.yaml](constants.yaml))
- MIN_YEAR, RELEVANCE_THRESHOLD
- API endpoints and retry settings

✅ **Model Selection** ([model_config.yaml](model_config.yaml))
- Choose LLM providers and models per layer/agent
- Adjust temperature and max_tokens

## Domain-Specific Prompts (May Require Customization)

The following prompts contain references to "operations research", "optimization", and similar domain-specific terminology. If adapting to a significantly different field (e.g., "AI for Healthcare"), you may need to edit these prompts:

### Layer 1 Analysis Prompts

**Location:** `src/layer1/prompts/`

#### reader.txt
**Purpose:** Extract problem, methodology, experiments, results from papers

**Domain-specific content:**
- Line 1: "You are an expert in operations research and AI/ML."
- References to "optimization", "algorithms", "operations research"

**How to adapt:**
Replace domain references with your field. Example for "AI for Healthcare":
```
You are an expert in AI for healthcare and medical applications.

Analyze the following research paper...
[problem]: What clinical/medical challenge does this paper address?
[methodology]: What AI/ML techniques, clinical protocols, or algorithms are used?
[experiments]: What patient datasets, clinical trials, or validation studies are described?
```

#### methods.txt
**Purpose:** Extract method/problem tags and paper lineage

**Domain-specific content:**
- Line 1: "You are an expert in operations research, optimization, and AI/ML."
- References to specific OR/AI methodologies

**How to adapt:**
Update methodology taxonomy for your domain. Example:
```
You are an expert in AI for healthcare, medical imaging, and clinical AI.

Valid method tags: ["deep_learning", "medical_imaging", "clinical_nlp", "diagnosis_ai", ...]
Valid problem tags: ["radiology", "pathology", "clinical_trials", "drug_discovery", ...]
```

#### positioning.txt
**Purpose:** Assess relevance to researcher's active projects

**Domain-specific content:**
- Line 4: "You are an expert in operations research and AI/ML."
- Line 54: References "optimization", "operations research", "multi-agent systems"

**How to adapt:**
Update to reflect your research domain:
```
You are an expert in AI for healthcare.

Category: {category}
Researcher's active projects and interests in this category:
{researcher_profile}
```

### Layer 2 Front Summarization

**Location:** `src/layer2/front_summarizer.py`

**Domain-specific content:**
- Line 31: "You are an expert in operations research and AI/ML."
- Line 38: "Common themes in operations research..."
- Line 42: "specific operations research methods"
- Line 49: "operations research angle"

**How to adapt:**
Edit the `summarize_front()` function prompt to reference your domain:
```python
prompt = f"""You are an expert in AI for healthcare.

This is a research front (cluster of co-cited papers): {front_papers}

Summarize:
1. What clinical/medical problem this front addresses
2. Common AI/ML techniques used across papers
3. What makes this front distinct in healthcare AI
"""
```

### Layer 3 Update Prompts

**Location:** `src/layer3/*.py`

These prompts are more generic and reference "papers", "research", "methods" without heavy domain specificity. They should work across most domains without modification.

**If needed**, update prompts in:
- `src/layer3/daily_update.py` - Daily review updates
- `src/layer3/weekly_revision.py` - Weekly restructuring
- `src/layer3/email_renderer.py` - Email generation

## Adaptation Checklist

When adapting to a new research domain:

### Required Changes
- [ ] Update `research_domain.yaml` - Define categories, filters, seed papers
- [ ] Update `researcher_profile.md` - Describe your research interests

### Recommended Changes
- [ ] Review `src/layer1/prompts/reader.txt` - Update domain expertise statement
- [ ] Review `src/layer1/prompts/methods.txt` - Update method/problem tag taxonomy
- [ ] Review `src/layer1/prompts/positioning.txt` - Update domain expertise statement
- [ ] Review `src/layer2/front_summarizer.py` - Update domain expertise in prompt

### Optional Changes
- [ ] Update `constants.yaml` - Adjust MIN_YEAR, RELEVANCE_THRESHOLD if needed
- [ ] Update `model_config.yaml` - Switch models if desired

## Testing After Adaptation

After editing prompts for a new domain, test each layer:

```bash
# Test Layer 0 (should work without prompt changes)
python src/layer0/fetch_and_score.py --category "Your Category Name"

# Test Layer 1 (requires prompt updates for best results)
python src/scripts/layer1_analyze_new.py --category "Your Category Name" --max 5

# Test Layer 2 (requires prompt updates if domain very different)
python src/scripts/layer2_detect_fronts.py --category "Your Category Name"

# Test Layer 3 (usually works without changes)
python src/scripts/layer3_run.py daily --days 7
```

## Future Improvements

Potential enhancements to make domain adaptation easier:

1. **Prompt Templates:** Move all domain-specific content to `research_config/prompts.yaml` with placeholder variables
2. **Dynamic Taxonomies:** Define valid method/problem tags in `research_domain.yaml` instead of hardcoding in `methods.txt`
3. **Prompt Validation:** Add warnings if prompts contain domain-specific keywords not matching current domain

## Questions?

For architecture details, see [../src/README.md](../src/README.md).

For basic configuration, see [README.md](README.md).
