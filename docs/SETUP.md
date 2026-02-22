# Research Intelligence System - Setup Guide

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `google-generativeai` - Gemini API (FREE during preview!)
- `PyMuPDF` - PDF text extraction
- `networkx`, `python-louvain` - Citation graphs
- `matplotlib`, `seaborn` - Visualization
- `sendgrid` - Email delivery
- Plus: `requests`, `pyyaml`, `openai`, `pydantic` (existing)

### 2. Get Gemini API Key (FREE)

1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key

### 3. Set Environment Variable

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY=your-api-key-here
```

**Permanent (add to your shell profile):**
```bash
# Add to ~/.bashrc or ~/.zshrc
export GEMINI_API_KEY="your-api-key-here"
```

### 4. Verify Setup

```bash
cd e:/Work/Daily_ArXive_retriver/llm-arxiv-daily-main
python src/config.py
```

**Expected output:**
```
======================================================================
RESEARCH INTELLIGENCE SYSTEM - CONFIGURATION
======================================================================

Layer 1 Agents:
  Reader:           gemini / gemini-3-flash-preview
  Methods:          gemini / gemini-3-flash-preview
  Positioning:      gemini / gemini-3-flash-preview
  ...
```

### 5. Test with One Paper

```bash
python src/scripts/layer1_analyze_new.py --max 1
```

**What happens:**
1. Fetches PDF from ArXiv
2. Extracts text (cached for reuse)
3. Runs 3 agents: Reader → Methods → Positioning
4. Stores analysis in database
5. Shows brief summary

**Expected time:** ~30-60 seconds per paper

---

## Configuration

### View Current Config

```bash
python src/config.py
```

### Change Models

Edit `research_config/model_config.yaml`:

```yaml
# Use different model for specific agent
layer1:
  positioning:
    provider: "gemini"
    model_name: "gemini-2.5-pro"  # Upgrade to Pro
    temperature: 0.0
    max_tokens: 4096
```

### Supported Models

**Gemini (Google):**
- `gemini-3-flash-preview` - Default, fastest, cheap
- `gemini-2.5-flash` - Fast, cheap
- `gemini-2.5-pro` - Best quality, slower
- `gemini-3-pro-preview` - Preview, experimental

**Claude (Anthropic):**
- `claude-sonnet-4-5-20250929` - Best quality
- `claude-haiku-4-5-20251001` - Fast, cheap

**OpenAI:**
- `gpt-4o` - High quality
- `gpt-4o-mini` - Fast, cheap

### Mix and Match

```python
DEFAULT_AGENT_MODELS = AgentModels(
    reader=ModelConfig(
        provider="gemini",
        model_name="gemini-3-flash-preview",  # Fast extraction
    ),
    positioning=ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-5-20250929",  # Better reasoning
    ),
    email_generator=ModelConfig(
        provider="openai",
        model_name="gpt-4o",  # Best writing
    )
)
```

---

## Usage

### Analyze All New Papers

```bash
python src/scripts/layer1_analyze_new.py
```

### Analyze First 5 Papers (Testing)

```bash
python src/scripts/layer1_analyze_new.py --max 5
```

### Analyze Specific Category

```bash
python src/scripts/layer1_analyze_new.py --category "LLMs for Algorithm Design"
```

### Dry Run (See What Would Be Analyzed)

```bash
python src/scripts/layer1_analyze_new.py --dry-run
```

---

## Cost Estimates

### Gemini 3 Flash (Current Default)

**Per paper:**
- Input: ~20K tokens × $0.50/1M = $0.01
- Output: ~3K tokens × $3.00/1M = $0.009
- **Total: ~$0.02 per paper**

**One-time analysis (302 papers):**
- ~$6.00

**Daily maintenance (10 papers/day):**
- ~$0.20/day = ~$6/month

### Switching to Gemini 2.5 Flash Lite (Cheaper)

**Per paper:** ~$0.003
**Monthly:** ~$0.90

### Switching to Claude Sonnet (Premium)

**Per paper:** ~$0.08
**Monthly:** ~$24

---

## Troubleshooting

### "No module named 'requests'"

```bash
pip install -r requirements.txt
```

### "GEMINI_API_KEY not set"

```bash
# Check if set
echo $GEMINI_API_KEY  # Linux/Mac
echo %GEMINI_API_KEY%  # Windows CMD
$env:GEMINI_API_KEY  # Windows PowerShell

# Set it
export GEMINI_API_KEY="your-key"
```

### "Import errors"

Make sure you're in the project root:
```bash
cd e:/Work/Daily_ArXive_retriver/llm-arxiv-daily-main
python src/scripts/layer1_analyze_new.py
```

### API Rate Limits

Gemini has generous rate limits:
- Free tier: 15 requests/minute, 1500 requests/day
- Paid tier: 360 requests/minute, 10,000 requests/day

If you hit limits:
```python
# In src/config.py, add delay between papers
import time
time.sleep(2)  # Wait 2 seconds between papers
```

---

## File Locations

**Database:** `src/db/research_intelligence.db`
**PDF Cache:** `cache/pdfs/`
**Config:** `src/config.py`
**Prompts:** `src/layer1/prompts/`

---

## Next Steps

After Layer 1 is working:

1. **Customize Research Profile**
   - Edit: `research_config/researcher_profile.md`
   - Add your actual research interests per category

2. **Implement Layer 2** (Citation graphs, fronts)
   - Run: `python src/scripts/layer2_detect_fronts.py`

3. **Implement Layer 3** (Living reviews, email)
   - Daily/weekly/monthly updates
   - Graph visualizations
   - Email briefings

---

## Support

- Configuration: `src/config.py`
- Progress tracking: `PLAN.md`
- This guide: `SETUP.md`
