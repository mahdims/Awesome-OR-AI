"""
Layer 1 Pipeline - Multi-Agent Deep Paper Analysis

Orchestrates 4 sequential agents to produce comprehensive paper analysis:
1. Reader Agent - Extracts problem, methodology, experiments, results
2. Methods Extractor - Tags methods/problems, identifies lineage
3. Positioning Agent - Assesses relevance to researcher's work
4. Synthesis Controller - Merges outputs and stores in database
"""

from pathlib import Path
import sys
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1.pdf_extractor import fetch_arxiv_pdf_text
from layer1.schemas import PaperAnalysis, ReaderOutput, MethodsOutput, PositioningOutput
from db.database import Database
from llm_client import create_agent_client
from config import AGENT_MODELS

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROJECT_ROOT = Path(__file__).parent.parent.parent  # layer1 -> src -> root
RESEARCHER_PROFILE_PATH = PROJECT_ROOT / "research_config" / "researcher_profile.md"

class PaperAnalysisPipeline:
    """
    Orchestrates 4-agent sequential analysis pipeline.
    Each agent = prompt + LLM call + structured output parsing.
    Uses centralized config for model selection.
    """

    def __init__(self):
        """Initialize pipeline with models from config."""
        self.db = Database()

        # Create LLM clients for each agent from centralized config
        self.reader_client = create_agent_client('reader')
        self.methods_client = create_agent_client('methods_extractor')
        self.positioning_client = create_agent_client('positioning')

        # Load prompts
        self.reader_prompt = (PROMPTS_DIR / "reader.txt").read_text(encoding='utf-8')
        self.methods_prompt = (PROMPTS_DIR / "methods.txt").read_text(encoding='utf-8')
        self.positioning_prompt = (PROMPTS_DIR / "positioning.txt").read_text(encoding='utf-8')
        self.researcher_profile = RESEARCHER_PROFILE_PATH.read_text(encoding='utf-8')

        print("[INIT] Pipeline initialized")
        print(f"  Reader:      {AGENT_MODELS.reader.provider} / {AGENT_MODELS.reader.model_name}")
        print(f"  Methods:     {AGENT_MODELS.methods_extractor.provider} / {AGENT_MODELS.methods_extractor.model_name}")
        print(f"  Positioning: {AGENT_MODELS.positioning.provider} / {AGENT_MODELS.positioning.model_name}")
        print(f"  Database:    {self.db.db_path}")

    def analyze_paper(self, arxiv_id: str, category: str,
                      title: str, authors: list, abstract: str,
                      published_date: str, affiliation: str = "") -> Optional[PaperAnalysis]:
        """
        Run full 4-agent analysis pipeline for a single paper.

        Args:
            arxiv_id: ArXiv identifier (e.g., "2501.12345")
            category: One of the three categories
            title: Paper title
            authors: List of author names
            abstract: Paper abstract
            published_date: Publication date (YYYY-MM-DD)

        Returns:
            PaperAnalysis object or None if analysis fails
        """

        print(f"\n{'='*70}")
        print(f"ANALYZING: {arxiv_id}")
        print(f"Category: {category}")
        print(f"Title: {title[:60]}...")
        print(f"{'='*70}")

        # Check if already analyzed
        with self.db as db:
            if db.has_analysis(arxiv_id):
                existing = db.get_analysis(arxiv_id)
                print(f"[SKIP] Already analyzed on {existing.get('analysis_date', 'unknown')}")
                return None

        try:
            # Fetch PDF
            print(f"\n[STEP 1/4] Fetching PDF...")
            pdf_text, pdf_hash = fetch_arxiv_pdf_text(arxiv_id)
            print(f"  PDF hash: {pdf_hash[:16]}...")
            print(f"  Text length: {len(pdf_text):,} characters")

            # Agent 1: Reader
            print(f"\n[STEP 2/4] Running Reader Agent...")
            reader_output = self._run_agent(
                agent_name="Reader",
                client=self.reader_client,
                prompt=self.reader_prompt,
                context=pdf_text,
                output_schema=ReaderOutput
            )
            print(f"  ✓ Extracted: {reader_output.problem.short} ({reader_output.problem.class_})")
            print(f"  ✓ Method: {reader_output.methodology.core_method}")

            # Agent 2: Methods Extractor
            print(f"\n[STEP 3/4] Running Methods Extractor Agent...")
            methods_output = self._run_agent(
                agent_name="Methods",
                client=self.methods_client,
                prompt=self.methods_prompt,
                context=f"PAPER:\n{pdf_text}\n\nREADER ANALYSIS:\n{reader_output.model_dump_json(indent=2)}",
                output_schema=MethodsOutput
            )
            print(f"  ✓ Methods: {', '.join(methods_output.tags.methods[:3])}")
            print(f"  ✓ Problems: {', '.join(methods_output.tags.problems[:2])}")

            # Agent 3: Positioning
            print(f"\n[STEP 4/4] Running Positioning Agent...")
            positioning_output = self._run_agent(
                agent_name="Positioning",
                client=self.positioning_client,
                prompt=self.positioning_prompt,
                context=f"PAPER:\n{pdf_text}\n\nREADER:\n{reader_output.model_dump_json(indent=2)}\n\nMETHODS:\n{methods_output.model_dump_json(indent=2)}\n\nRESEARCHER PROFILE:\n{self.researcher_profile}",
                output_schema=PositioningOutput
            )
            print(f"  ✓ Relevance: M={positioning_output.relevance_scores.methodological}, "
                  f"P={positioning_output.relevance_scores.problem}, "
                  f"I={positioning_output.relevance_scores.inspirational}")
            print(f"  ✓ Must-read: {positioning_output.significance.must_read}")

            # Agent 4: Synthesis (package it up)
            print(f"\n[SYNTHESIS] Packaging analysis...")
            analysis = PaperAnalysis(
                arxiv_id=arxiv_id,
                category=category,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published_date,
                reader=reader_output,
                methods=methods_output,
                positioning=positioning_output,
                analysis_model=f"{AGENT_MODELS.reader.provider}/{AGENT_MODELS.reader.model_name}",
                pdf_hash=pdf_hash
            )

            # Store in database
            print(f"[DATABASE] Storing analysis...")
            with self.db as db:
                db.insert_analysis(analysis.model_dump())
                # Use S2/JSON affiliation as fallback if reader didn't extract one
                if affiliation:
                    db.update_paper_metadata(arxiv_id, affiliations=affiliation)

            print(f"\n{'='*70}")
            print(f"✓ ANALYSIS COMPLETE: {arxiv_id}")
            print(f"{'='*70}")

            return analysis

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"✗ ERROR ANALYZING {arxiv_id}")
            print(f"{'='*70}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_agent(self, agent_name: str, client, prompt: str, context: str, output_schema):
        """
        Single LLM call with structured output parsing.

        Args:
            agent_name: Name for logging
            client: LLMClient instance
            prompt: System prompt for the agent
            context: User context/input
            output_schema: Pydantic model for validation

        Returns:
            Validated Pydantic model instance
        """
        print(f"  Calling {client.config.provider} API ({client.config.model_name})...")

        try:
            result = client.generate_json(
                system_prompt=prompt,
                user_prompt=context,
                output_schema=output_schema
            )

            print(f"  ✓ {agent_name} agent completed successfully")
            return result

        except Exception as e:
            print(f"  ✗ {agent_name} agent failed: {str(e)}")
            raise

    def analyze_batch(self, papers: list[dict], max_papers: Optional[int] = None) -> dict:
        """
        Analyze multiple papers in batch.

        Args:
            papers: List of paper dicts with keys: arxiv_id, category, title, authors, abstract, published_date
            max_papers: Optional limit on number of papers to process

        Returns:
            Dict with success/failure counts and results
        """
        if max_papers:
            papers = papers[:max_papers]

        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'analyses': []
        }

        total = len(papers)
        for i, paper in enumerate(papers, 1):
            print(f"\n\n{'#'*70}")
            print(f"# PAPER {i}/{total}")
            print(f"{'#'*70}")

            analysis = self.analyze_paper(**paper)

            if analysis:
                results['success'] += 1
                results['analyses'].append(analysis)
            else:
                # Check if it was skipped (already analyzed)
                with self.db as db:
                    if db.has_analysis(paper['arxiv_id']):
                        results['skipped'] += 1
                    else:
                        results['failed'] += 1

        print(f"\n\n{'='*70}")
        print(f"BATCH ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"  Success: {results['success']}")
        print(f"  Skipped (already analyzed): {results['skipped']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Total: {total}")
        print(f"{'='*70}")

        return results


def test_pipeline():
    """Quick test of pipeline with a dummy paper."""
    import os

    # Check for Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        print("Set it with: export GEMINI_API_KEY=your_key_here")
        return

    pipeline = PaperAnalysisPipeline()

    # Test paper (you'll need to replace with a real ArXiv ID)
    test_paper = {
        'arxiv_id': '2501.00001',  # Replace with real ID
        'category': 'LLMs for Algorithm Design',
        'title': 'Test Paper Title',
        'authors': ['Author One', 'Author Two'],
        'abstract': 'This is a test abstract.',
        'published_date': '2025-01-01'
    }

    result = pipeline.analyze_paper(**test_paper)

    if result:
        print("\n✓ Test successful!")
        print(f"Brief: {result.positioning.brief}")
    else:
        print("\n✗ Test failed")


if __name__ == "__main__":
    test_pipeline()
