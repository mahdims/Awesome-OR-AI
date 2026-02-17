from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

# === Agent 1: Reader ===

class ProblemDefinition(BaseModel):
    formal_name: str
    short: str  # Abbreviation like "CVRPTW"
    class_: str = Field(alias="class")  # "routing", "scheduling", etc.
    properties: List[str]  # ["capacitated", "time_windows", "deterministic"]
    scale: str  # "50-1000 customers"

class Methodology(BaseModel):
    core_method: str
    llm_role: str  # "none" | "heuristic_generator" | "evaluator" | "code_writer"
    llm_model_used: Optional[str] = None
    search_type: str  # "constructive" | "improvement" | "hybrid" | "exact"
    novelty_claim: str
    components: List[str]
    training_required: bool

class ExperimentalSetup(BaseModel):
    benchmarks: List[str]
    baselines: List[str]
    hardware: str
    instance_sizes: List[int]

class Results(BaseModel):
    vs_baselines: Dict[str, str]  # {baseline_name: gap_description}
    scalability: str
    statistical_rigor: str
    limitations_acknowledged: List[str]

class Artifacts(BaseModel):
    code_url: Optional[str] = None
    models_released: bool = False
    new_benchmark: bool = False

ConfidenceLevel = Literal["high", "medium", "low"]

class ReaderConfidence(BaseModel):
    problem: ConfidenceLevel = "medium"
    methodology: ConfidenceLevel = "medium"
    experiments: ConfidenceLevel = "medium"
    results: ConfidenceLevel = "medium"
    artifacts: ConfidenceLevel = "medium"
    flags: List[str] = Field(default_factory=list)  # Fields the model is uncertain about

class ReaderOutput(BaseModel):
    affiliations: str = ""  # Comma-separated, sorted by prominence (e.g., "DeepMind, MIT, Tsinghua")
    problem: ProblemDefinition
    methodology: Methodology
    experiments: ExperimentalSetup
    results: Results
    artifacts: Artifacts
    confidence: Optional[ReaderConfidence] = None

# === Agent 2: Methods Extractor ===

class Ancestor(BaseModel):
    paper: str  # arxiv_id or title
    relationship: str  # "extends ALNS framework from"

class Lineage(BaseModel):
    direct_ancestors: List[Ancestor]
    closest_prior_work: str
    novelty_type: str  # "incremental" | "combinatorial_novelty" | "paradigm_shift"

class Tags(BaseModel):
    methods: List[str]
    problems: List[str]
    contribution_type: List[str]  # ["new_method", "new_benchmark", "sota_result"]
    # Fine-grained open-ended differentiators — LLM fills these freely (None = not applicable)
    framework_lineage: Optional[str] = None  # e.g. "alphaevolve", "funsearch", "eoh", "llamea"
    specific_domain: Optional[str] = None    # e.g. "combinatorial_routing", "matrix_multiplication"
    llm_coupling: Optional[str] = None       # e.g. "off_the_shelf", "rl_trained", "fine_tuned"

class Extensions(BaseModel):
    next_steps: List[str]
    transferable_to: List[str]
    open_weaknesses: List[str]

class MethodsConfidence(BaseModel):
    tagging_confidence: ConfidenceLevel = "medium"
    lineage_confidence: ConfidenceLevel = "medium"
    flags: List[str] = Field(default_factory=list)  # Uncertain tags or lineage entries

class MethodsOutput(BaseModel):
    lineage: Lineage
    tags: Tags
    extensions: Extensions
    confidence: Optional[MethodsConfidence] = None

# === Agent 3: Positioning ===

class RelevanceScores(BaseModel):
    methodological: int = Field(ge=0, le=10)
    problem: int = Field(ge=0, le=10)
    inspirational: int = Field(ge=0, le=10)

class Significance(BaseModel):
    must_read: bool
    changes_thinking: bool
    team_discussion: bool
    reasoning: str

class PositioningOutput(BaseModel):
    relevance_scores: RelevanceScores
    significance: Significance
    brief: str  # One-paragraph assessment

# === Full Paper Analysis ===

class PaperAnalysis(BaseModel):
    arxiv_id: str
    category: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str

    # Agent outputs
    reader: ReaderOutput
    methods: MethodsOutput
    positioning: PositioningOutput

    # Metadata
    analysis_model: str
    pdf_hash: str

    def model_dump(self, **kwargs):
        """Override to flatten nested models for database storage."""
        dump = super().model_dump(**kwargs)

        # Convert nested Pydantic models to dicts
        dump['reader'] = self.reader.model_dump()
        dump['methods'] = self.methods.model_dump()
        dump['positioning'] = self.positioning.model_dump()

        # Flatten reader fields
        dump['affiliations'] = self.reader.affiliations
        dump['problem'] = self.reader.problem.model_dump()
        dump['methodology'] = self.reader.methodology.model_dump()
        dump['experiments'] = self.reader.experiments.model_dump()
        dump['results'] = self.reader.results.model_dump()
        dump['artifacts'] = self.reader.artifacts.model_dump()

        # Flatten reader confidence
        dump['reader_confidence'] = self.reader.confidence.model_dump() if self.reader.confidence else None

        # Flatten methods fields
        dump['lineage'] = self.methods.lineage.model_dump()
        dump['tags'] = self.methods.tags.model_dump()
        dump['extensions'] = self.methods.extensions.model_dump()
        dump['methods_confidence'] = self.methods.confidence.model_dump() if self.methods.confidence else None

        # Flatten positioning fields
        dump['relevance'] = self.positioning.relevance_scores.model_dump()
        dump['significance'] = self.positioning.significance.model_dump()
        dump['brief'] = self.positioning.brief

        # Remove the nested structures (keep only flattened)
        del dump['reader']
        del dump['methods']
        del dump['positioning']

        return dump
