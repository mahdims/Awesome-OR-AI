# Living Review: LLMs for Algorithm Design

**Last Updated:** 2026-02-15

---

## Recent Papers

#### 2026-02-13 (44 papers)

### [TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design](https://arxiv.org/abs/2601.21239)

**2026-01-29** |  | M=9 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Nested evolutionary framework with TSED-guided island model and co-evolutionary inner loop (UCB-based LLM logic generation + differential mutation for parameter tuning) | *LLM role:* heuristic_generator

> TIDE introduces a nested evolutionary framework that strictly decouples algorithmic structure generation (via LLM) from numerical parameter tuning (via Differential Evolution), managed by a Tree Similarity Edit Distance (TSED) guided island model. Results on 9 COPs (TSP, BPP, etc.) show it consistently outperforms ReEvo and EoH, primarily because the DE layer optimizes constants at zero token cost, preventing the discard of structurally sound but poorly tuned heuristics. The critical takeaway is the necessity of a gradient-free tuning layer for LLM-generated code; relying on LLMs for numerical constants is inefficient and imprecise. We should immediately implement a similar parameter-tuning inner loop in our AlgoEvo framework.

### [LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm](https://arxiv.org/abs/2512.24077)

**2025-12-30** |  | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Plan-Execute-Summarize (PES) paradigm integrated with Hybrid Evolutionary Memory (Multi-Island, MAP-Elites, Adaptive Boltzmann Selection) | *LLM role:* planner, executor, summarizer

> LoongFlow replaces the standard stochastic mutation operator in LLM evolutionary search with a 'Plan-Execute-Summarize' (PES) cognitive loop. Instead of random code changes, a Planner retrieves the 'intent' and 'summary' of the parent solution's lineage to generate a directed hypothesis, which is then executed and summarized for the next generation. The authors demonstrate a 60% reduction in evaluations and a 100% success rate on AlphaEvolve tasks where standard methods fail or stagnate. The critical takeaway is the 'Lineage-Based Context Retrieval' mechanism: explicitly passing the parent's plan and retrospective summary to the child allows for directed rather than random walks in the search space. We must implement this PES loop in AlgoEvo immediately to fix our sample efficiency issues.

### [Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM](https://arxiv.org/abs/2510.11121)

**2025-10-13** | Nanyang Technological University, Singapore, Singapore Management University, Singapore, Nanjing University of Information Science and Technology, China | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning (DAPO) fine-tuning of LLM for crossover operator generation within Hybrid Genetic Search (HGS) | *LLM role:* heuristic_generator

> Zhu et al. fine-tune a Qwen-14B model using Reinforcement Learning (DAPO) to generate C++ crossover operators for the state-of-the-art HGS solver. Unlike typical prompting papers, they demonstrate that a small, specialized model can improve upon expert-designed components in a highly optimized solver, achieving superior results on CVRPLIB (up to 1000 nodes) where GPT-4o fails. The most stealable insight is their **AST-based anti-plagiarism reward**, which penalizes the model for generating code structurally identical to the prompt examples, effectively forcing exploration and preventing mode collapse—a technique we should immediately adopt for our evolutionary search agents. This confirms we should pivot from pure prompting to RL-finetuning for our code-generation agents.

### [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

**2025-06-16** | Google DeepMind | M=10 P=10 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary algorithm for code superoptimization, orchestrating an autonomous pipeline of LLMs for code generation, critique, and evolution, grounded by code execution and automatic evaluation. | *LLM role:* evolutionary_search

> AlphaEvolve extends FunSearch by evolving entire code files (rather than single functions) using a 'search/replace' diff format and Gemini 2.0, achieving SOTA results across matrix multiplication (beating Strassen), 50+ open math problems, and Google's production scheduling. The results are exceptionally strong and verified, including deployed improvements to Google's Borg scheduler (0.7% resource recovery) and TPU circuits. The critical takeaway is the move to **diff-based full-file evolution** and **meta-prompt evolution** (evolving the prompt instructions alongside the code), which allows the system to modify architecture and logic rather than just heuristics. This is a mandatory blueprint for the next iteration of our AlgoEvo and EvoCut projects.

### [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://arxiv.org/abs/2407.10873)

**2024-07-15** | City University of Hong Kong, Southern University of Science and Technology | M=4 P=10 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based Evolutionary Program Search (EPS) | *LLM role:* evolutionary_search

> Zhang et al. perform a rigorous benchmarking of major LLM-based evolutionary program search (EPS) methods (FunSearch, EoH, ReEvo) against a simple (1+1)-EPS baseline across four problems and nine LLMs. The results are empirically solid and sobering: the simple (1+1)-EPS baseline—iterative improvement via one-shot prompting—frequently matches or outperforms the complex population-based methods, particularly on bin packing, though EoH remains superior on TSP. **Crucial Takeaway:** We are likely over-engineering our search mechanisms; we must implement a (1+1)-EPS baseline in all future experiments (AlgoEvo, EvoCut) because if our multi-agent systems cannot beat this simple hill-climber, our papers will be rejected for unnecessary complexity. Additionally, they find that larger models (GPT-4) do not strictly guarantee better heuristic search performance compared to smaller, code-specialized models like CodeLlama-7B.

### [Landscape-aware Automated Algorithm Design: An Efficient Framework for Real-world Optimization](https://arxiv.org/abs/2602.04529)

**2026-02-04** |  | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining Genetic Programming (GP) for proxy function generation and an LLM-driven Evolutionary Algorithm (LLaMEA) for algorithm discovery, guided by Exploratory Landscape Analysis (ELA) features and Wasserstein distance. | *LLM role:* algorithm_designer

> Yin et al. introduce a framework that decouples algorithm discovery from expensive evaluations by using Genetic Programming to evolve symbolic proxy functions that statistically match the target problem's landscape (via ELA features). Empirical results on photonics problems confirm that algorithms evolved on these cheap proxies transfer successfully to the real tasks, outperforming standard baselines like LSHADE with only 50×D real evaluations. **Key Takeaway:** We can synthesize 'symbolic gyms' that statistically mimic our target problems to run thousands of LLM iterations at near-zero cost. This directly addresses the sample efficiency bottleneck in AlgoEvo and suggests we should move beyond standard neural surrogates to evolved symbolic proxies.

### [Contrastive Concept-Tree Search for LLM-Assisted Algorithm Discovery](https://arxiv.org/abs/2602.03132)

**2026-02-03** |  | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Contrastive Concept-Tree Search (CCTS) using a hierarchical Bernoulli model and Tree-structured Parzen Estimator (TPE) for likelihood-ratio based parent reweighting, combined with cross-entropy updates for concept utility estimation. | *LLM role:* heuristic_generator

> The authors introduce Contrastive Concept-Tree Search (CCTS), which modifies the standard evolutionary loop by prompting the LLM to extract semantic 'concepts' from every generated program, building a dynamic hierarchy. They then apply a Tree-structured Parzen Estimator (TPE) to these concepts to learn a contrastive utility model (p(concept|good)/p(concept|bad)), using this to bias parent selection towards promising algorithmic strategies. Results are rigorous, showing consistent improvements over k-elite baselines on combinatorial tasks like Circle Packing, with a synthetic ablation confirming the model learns ground-truth concept utilities. **Key Takeaway:** We should immediately implement the 'Concept TPE' loop in AlgoEvo—asking the LLM to tag generated heuristics with concepts and maintaining a weight vector over these concepts provides a cheap, interpretable 'process reward model' to guide search.

### [LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI](https://arxiv.org/abs/2601.21511)

**2026-01-29** |  | M=8 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA-SAGE, an LLM-driven evolutionary algorithm that integrates structural feedback from Explainable AI (SHAP) analysis of Abstract Syntax Tree (AST) code features to guide mutations. | *LLM role:* evolutionary_search

> LLaMEA-SAGE augments LLM-based evolutionary search by extracting AST features (complexity, graph metrics) from generated code, training a surrogate model to predict fitness from these features, and using SHAP analysis to generate natural language prompts that guide the LLM to modify specific structural properties (e.g., 'increase cyclomatic complexity'). On the MA-BBOB benchmark, it outperforms state-of-the-art methods (MCTS-AHD, LHNS) and converges faster than vanilla LLaMEA, although the authors honestly report that statistical significance was limited (p=0.44) due to small sample sizes (5 runs). The critical takeaway for us is the pipeline of using static code analysis as a feedback signal—we can immediately steal this 'SAGE' loop to guide AlgoEvo or EvoCut by telling the LLM *how* to structurally mutate code based on surrogate correlations, rather than just hoping for random improvements.

### [Global Optimization for Combinatorial Geometry Problems Revisited in the Era of LLMs](https://arxiv.org/abs/2601.05943)

**2026-01-15** |  | M=4 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Global Nonlinear Programming (NLP) using spatial branch-and-bound | *LLM role:* none

> Berthold et al. demonstrate that standard global NLP solvers (SCIP, Xpress) outperform DeepMind's AlphaEvolve on its own benchmarks (circle/hexagon packing, min-max distance) without any learning or evolution. The results are rigorous, improving on 'newly discovered' solutions within minutes using default solver settings. **CRITICAL TAKEAWAY:** We must validate our AlgoEvo results against classical global solvers to ensure we aren't claiming 'discovery' on problems that are trivial for SCIP; furthermore, it suggests a hybrid path where LLMs generate NLP models for solvers rather than evolving raw heuristic code. This is a necessary reality check for our benchmarking strategy.

### [Automated Algorithmic Discovery for Scientific Computing through LLM-Guided Evolutionary Search: A Case Study in Gravitational-Wave Detection](https://arxiv.org/abs/2508.03661)

**2025-11-16** | Tsinghua University, University of Chinese Academy of Sciences | M=9 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided Evolutionary Monte Carlo Tree Search (Evo-MCTS) with reflective code synthesis and multi-scale evolutionary operations | *LLM role:* code_writer, heuristic_generator, evaluator, evolutionary_search

> Evo-MCTS introduces a hybrid search architecture where MCTS manages the exploration-exploitation balance of an evolutionary process, using LLMs for node expansion via novel operators like 'Path-wise Crossover' (synthesizing code from full root-to-leaf trajectories). The results are empirically strong, outperforming standard LLM-evolution baselines (ReEvo) by ~150% on a complex signal processing task. We learned that structuring the evolutionary lineage as a tree and using MCTS Q-values to select parents—rather than standard population selection—drastically improves sample efficiency and solution quality. This is a blueprint for the 'RL-infused evolution' and 'persistent memory' features we have been planning for our own framework.

### [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)

**2025-10-10** | UC Berkeley | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary search (MAP-Elites and island models) with automated code generation and empirical evaluation | *LLM role:* code_writer, reasoning_agent, feedback_generator

> The authors apply OpenEvolve (an AlphaEvolve-style framework) to 11 computer systems problems, achieving significant gains over human baselines, such as a 5.0x speedup in MoE expert placement and 26% cost reduction in cloud scheduling. The results are empirically rigorous, relying on high-fidelity simulators rather than toy problems. For us, the key takeaway is the engineering recipe: using an ensemble of reasoning models (o3) for exploration and fast models (Gemini) for diversity, combined with a specific 'failure taxonomy' to debug search stagnation. This is immediate proof-of-concept for your 'GPUSched' and 'AlgoEvo' projects; we should adopt their ensemble strategy and simulator-first evaluation pipeline.

### [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://arxiv.org/abs/2508.03082)

**2025-08-20** | Huawei Noah Ark Lab, City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary search framework with complementary population management and diversity-aware memetic search | *LLM role:* heuristic_generator

> EoH-S reformulates Automated Heuristic Design (AHD) to evolve a complementary *set* of heuristics rather than a single robust one, proving the objective is submodular and solvable via a greedy strategy. Results are strong and credible: on TSPLib and CVRPLib, their set of 10 heuristics reduces the optimality gap by ~40-60% compared to the top 10 heuristics from FunSearch or ReEvo. **KEY TAKEAWAY:** We should replace standard elitist selection in AlgoEvo with their 'Complementary Population Management' (CPM). By greedily selecting individuals based on marginal contribution to instance coverage (using instance-wise performance vectors), we can automatically generate diverse operator pools for ALNS instead of relying on hand-crafted diversity metrics.

### [Can Large Language Models Invent Algorithms to Improve Themselves?: Algorithm Discovery for Recursive Self-Improvement through Reinforcement Learning](https://arxiv.org/abs/2410.15639)

**2025-06-10** | NEC Corporation | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Direct Preference Optimization (DPO) for iterative refinement of an algorithm-generating LLM | *LLM role:* evolutionary_search

> Ishibashi et al. propose 'Self-Developing,' a framework where an LLM generates Python code for model merging, evaluates the results, and uses the performance data to fine-tune the generator via DPO in a recursive loop. The results are empirically strong, outperforming human-designed baselines (Task Arithmetic) by 4.3% on GSM8k and demonstrating that the generator explicitly learns better strategies over iterations. **Key Takeaway:** We can replace the static mutation operators in our evolutionary search with a DPO-trained model that learns from the search history—effectively implementing 'learning to search.' This is a direct, actionable upgrade for our AlgoEvo and AlphaEvolve pipelines.

### [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/abs/2506.11057)

**2025-05-22** | Shanghai Key Laboratory of Scalable Computing and Systems, School of Computer Science, Shanghai Jiao Tong University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structure-aware LLM-based algorithm discovery framework combining Graph Neural Network (GNN) for structural embeddings and LLM for solver-specific code generation, refined by an evolutionary algorithm. | *LLM role:* code_writer

> STRCMP introduces a composite architecture where a GNN encodes CO problem instances (MILP/SAT) into embeddings that condition an LLM (fine-tuned via SFT and DPO) to generate solver-specific heuristics within an evolutionary loop. The results are strong and empirically backed, showing significant reductions in convergence time and timeouts compared to text-only evolutionary methods like AutoSAT and LLM4Solver. The key takeaway is the architectural blueprint for fusing instance-specific structural embeddings (via soft prompting) with LLM code generation to drastically improve the sample efficiency of evolutionary search. This is immediately relevant to our EvoCut and AlgoEvo projects, suggesting we should move beyond pure text prompts for topology-heavy problems.

### [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://arxiv.org/abs/2505.12285)

**2025-05-18** | City University of Hong Kong, Southeast University, University of Victoria, Hon Hai Research Institute | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining verbal and numerical guidance for heuristic evolution, achieved by fine-tuning an LLM via reinforcement learning (GRPO) based on heuristic quality, co-evolving the LLM with the search process. | *LLM role:* heuristic_generator_and_fine_tuned_agent

> CALM introduces a hybrid evolutionary framework that fine-tunes the LLM generator *during* the search process using Group Relative Policy Optimization (GRPO), rather than relying solely on prompt evolution. Using a quantized Qwen-7B model on a single consumer GPU, it outperforms GPT-4o-based baselines (FunSearch, EoH) on Bin Packing and VRP benchmarks. The critical takeaway is their reward function design: instead of absolute performance, they reward the *relative improvement* of the generated code over the specific 'parent' heuristics in the prompt, stabilizing the RL signal. We should immediately test this 'online fine-tuning' approach to reduce our API costs and improve sample efficiency in AlgoEvo.

### [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/abs/2409.16867)

**2025-02-04** | City University of Hong Kong, Southern University of Science and Technology | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-based Multi-objective Evolutionary Algorithm with Dominance-Dissimilarity Mechanism | *LLM role:* heuristic_generator

> MEoH extends LLM-based heuristic evolution (like FunSearch/EoH) to multi-objective scenarios (e.g., Gap vs. Runtime) by introducing a 'Dominance-Dissimilarity' mechanism that selects parents based on both Pareto dominance and Abstract Syntax Tree (AST) code distance. The results are credible and strong: on TSP, they find heuristics matching EoH's quality but running 16x faster (1.37s vs 22.4s) by effectively navigating the complexity-performance trade-off. The single most useful takeaway is the **AST-based dissimilarity metric** for population management; we should immediately steal this to prune semantically identical code in our evolutionary loops, thereby forcing exploration and improving sample efficiency. This is a direct upgrade to our current single-objective evolutionary search methods.

### [Evolving Interdependent Operators with Large Language Models for Multi-Objective Combinatorial Optimization](https://arxiv.org/abs/2601.17899)

**2026-02-01** |  | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Monte Carlo Tree Search for progressive design strategy search with operator rotation evolution | *LLM role:* heuristic_generator

> E2OC introduces a hierarchical search framework where MCTS optimizes 'design thoughts' (textual strategies) rather than raw code, subsequently using these strategies to guide a coordinate-descent-style evolution of interdependent operators. While the computational cost is high due to the inner-loop operator rotation, the results on FJSP/TSP (+20% HV vs expert) and comparisons against FunSearch/EoH demonstrate that explicitly modeling operator coupling is superior to isolated evolution. The critical takeaway for us is the **'strategy-first' search layer**: evolving a semantic blueprint for component interaction *before* code generation prevents the local optima trap of independent component optimization, a technique we should immediately test in AlgoEvo.

### [READY: Reward Discovery for Meta-Black-Box Optimization](https://arxiv.org/abs/2601.21847)

**2026-01-29** |  | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* LLM-based program evolution with a multi-task niche-based architecture, fine-grained evolutionary operators, and explicit knowledge transfer | *LLM role:* evolutionary_search

> READY introduces a multi-task evolutionary framework where LLMs evolve reward functions for multiple MetaBBO algorithms simultaneously, utilizing explicit 'Knowledge Transfer' operators to translate successful logic between distinct tasks. The results are robust, demonstrating superior performance over Eureka and EoH on BBOB benchmarks with a 2-4x reduction in search time due to parallelization and shared heuristics. The most stealable insights are the 'History-Reflection' operator—which prompts the LLM to extrapolate trends from the evolutionary trajectory rather than just mutating the current state—and the cross-niche transfer mechanism, both of which should be implemented in our multi-agent optimization stack immediately.

### [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](https://arxiv.org/abs/2601.20539)

**2026-01-29** |  | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent reasoning framework (Policy Agent, World Model Agent, Critic Agents) for Automated Heuristic Design, formulated as a sequential decision process over an entailment graph. | *LLM role:* heuristic_generator

> PathWise reformulates heuristic discovery as a sequential planning problem over an 'Entailment Graph,' where a Policy Agent generates high-level evolutionary directives (rationales) and a World Model executes the code, guided by specific Critic reflections. The results are robust: it outperforms ReEvo, FunSearch, and MCTS-AHD on TSP, CVRP, and Bin Packing while using half the evaluation budget (500 vs 1000), demonstrating genuine sample efficiency. The key takeaway is the **Entailment Graph** structure: explicitly storing the *derivation rationale* and lineage allows the LLM to reason about the search trajectory and avoid redundant failures, a mechanism we should immediately adapt for AlgoEvo to fix our memory bottleneck.

### [Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search](https://arxiv.org/abs/2601.19622)

**2026-01-27** |  | M=7 P=5 I=8 *discuss*

*Method:* Evolutionary Heuristic Design (EoH) framework with Algorithmic-Contextual Prompt Augmentation (A-CEoH) | *LLM role:* heuristic_generator

> This paper introduces 'Algorithmic-Contextual EoH' (A-CEoH), which injects the actual source code of the search algorithm (e.g., the A* driver loop, neighbor generation) into the LLM prompt alongside the problem description. Experiments on the Unit-Load Pre-Marshalling Problem and Sliding Puzzle Problem demonstrate that this algorithmic context allows a 32B parameter model (Qwen2.5-Coder) to generate heuristics superior to those from GPT-4o and human experts. The results are credible and backed by comparisons against optimal baselines. The key takeaway is a transferable 'prompt trick': explicitly showing the LLM the code that *calls* its generated function aligns the heuristic significantly better with the search dynamics than natural language descriptions alone. We should immediately test injecting our ALNS/search driver code into our evolutionary prompt templates.

### [Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer](https://arxiv.org/abs/2601.05770)

**2026-01-09** |  | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Discrete Transformer with functional disentanglement, temperature-annealed sampling, hypothesis testing for attention, and symbolic regression for MLP | *LLM role:* none

> Zhang et al. introduce the 'Discrete Transformer,' a constrained architecture that learns algorithmic tasks via gradient descent and allows for the post-hoc extraction of exact, human-readable Python code. By enforcing functional disentanglement (using attention strictly for routing and MLPs for arithmetic) and employing temperature-annealed sampling, they recover symbolic laws for arithmetic and physics tasks with near-zero error. The critical takeaway is their 'continuous-to-discrete homotopy' strategy—annealing from soft to hard selection during training—which enables differentiable search to converge on discrete, symbolic solutions. This suggests a viable path to discover heuristics via continuous optimization rather than purely stochastic LLM evolution.

### [CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization](https://arxiv.org/abs/2510.14150)

**2026-01-06** | Inter&Co., Worcester Polytechnic Institute, Universidade Federal de Minas Gerais | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Islands-based genetic algorithm with modular LLM orchestration, context-aware recombination, adaptive meta-prompting, and depth-based exploitation | *LLM role:* code_writer

> CodeEvolve couples islands-based genetic algorithms with LLMs, utilizing CVT-MAP-Elites for diversity and a specific 'inspiration-based' crossover operator where the LLM integrates logic from high-ranking peer solutions. The results are strong and backed by numbers: they beat AlphaEvolve on 5/9 benchmarks and demonstrate that Qwen3-Coder-30B matches Gemini-2.5 performance at ~10% of the cost. The single most useful takeaway is the implementation of the 'inspiration' operator and the necessity of MAP-Elites over simple elitism to escape local optima in code space. We should immediately benchmark their open-source framework against our internal AlgoEvo builds.

### [Beyond Algorithm Evolution: An LLM-Driven Framework for the Co-Evolution of Swarm Intelligence Optimization Algorithms and Prompts](https://arxiv.org/abs/2512.09209)

**2025-12-10** | Peking University | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Collaborative evolution of Fireworks Algorithm operators and prompt templates, driven by a single LLM | *LLM role:* evolutionary_search

> The authors introduce a co-evolutionary framework where both the optimization algorithm (Fireworks Algorithm operators) and the prompt templates used to generate them are evolved simultaneously by the LLM. The results demonstrate a massive performance jump on constrained Aircraft Landing problems (from ~56% with FunSearch to 100% with their method), suggesting that static prompts are a primary failure mode for complex OR constraints. The critical takeaway is their prompt fitness function: evaluating a prompt template based on the *performance improvement* (`child - parent`) of the code it generates, rather than absolute performance. We should immediately implement this 'prompt-delta' fitness signal in AlgoEvo to automate our prompt engineering loop.

### [Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research](https://arxiv.org/abs/2510.06056)

**2025-10-07** | MIT-IBM Watson AI Lab, IBM Research, University of Notre Dame | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Agent-based framework integrating deep research (planning, searching, writing) with algorithm evolution (coding, evaluation, evolutionary selection) and iterative debugging. | *LLM role:* heuristic_generator, code_writer, evaluator, decomposition_guide, debugger

> DeepEvolve augments the standard evolutionary coding loop (AlphaEvolve) with two critical components: a 'Deep Research' module that searches the web/literature to generate grounded mutation proposals, and an iterative debugging agent that fixes execution errors. While the '666%' improvement on Circle Packing is likely due to a weak baseline (fixed-size vs. generalized), the engineering results are compelling: the debugging agent raises execution success rates from ~13% to ~99% in complex tasks. The key takeaway for our AlgoEvo work is the architecture of generating a text-based 'research proposal' via RAG before attempting code generation, rather than mutating code directly. We should immediately adopt their debugging loop and consider injecting external literature search into our mutation operators to prevent search stagnation.

### [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://arxiv.org/abs/2509.24509)

**2025-09-30** | Tencent, Renmin University of China, City University of Hong Kong | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Experience-Guided Reflective Co-Evolution of Prompts and Heuristics (EvoPH) with island-based elites selection | *LLM role:* heuristic_generator

> EvoPH introduces a co-evolutionary framework where both the heuristic code and the LLM prompts are evolved, utilizing an island model for diversity and a 'strategy sampling' mechanism that dynamically selects mutation types (e.g., parameter tuning vs. rewrite) based on feedback. They report dominating performance over FunSearch and ReEvo on TSP and BPP (e.g., reducing Christofides gap from ~20% to ~5%), though the static performance of baselines suggests the gain comes largely from automating prompt engineering. The most stealable insight is the **Strategy Sampling** module: explicitly defining a pool of mutation operators and using an 'experience' buffer to select them is a practical implementation of the 'planner' concept we need for AlgoEvo. We should also adopt their island migration topology to improve diversity in our parallelized search.

### [GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models](https://arxiv.org/abs/2509.21593)

**2025-09-25** | Massachusetts Institute of Technology, Stanford University, Technical University of Munich | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Multi-agent LLM framework combining OpenEvolve-based evolutionary search with GeoKnowRAG for geospatial domain knowledge injection | *LLM role:* code_writer, evaluator, prompt_optimizer, evolutionary_search, decomposition_guide

> GeoEvolve augments standard LLM-based evolutionary search (OpenEvolve) with an outer 'researcher' loop that queries a domain-specific RAG (textbooks/papers) to inject theoretical constraints into mutation prompts. On geospatial interpolation tasks, they report 13-21% error reduction over standard evolution, with ablations confirming that retrieved domain knowledge—not just iterative feedback—drives the performance gain. The critical takeaway is the architectural pattern of 'Knowledge-Guided Evolution': instead of relying on the LLM's internal weights for domain theory, they explicitly retrieve and inject theoretical priors (e.g., valid variogram definitions) to steer the search. We should adapt this 'Theory-RAG' outer loop for our AlgoEvo pipeline to force evolved VRP heuristics to respect OR theoretical bounds.

### [LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression](https://arxiv.org/abs/2505.18602)

**2025-08-08** | Victoria University of Wellington, Michigan State University | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven meta-evolutionary framework for designing selection operators, incorporating semantics-aware selection, bloat control, and domain knowledge into prompts | *LLM role:* evolutionary_search

> Zhang et al. develop a meta-evolutionary framework to evolve selection operators for symbolic regression, achieving state-of-the-art results on SRBench by outperforming expert-designed methods like ε-lexicase. The standout contribution is **semantics-aware crossover**: rather than selecting parents based solely on scalar fitness, they compute complementarity scores using performance vectors across instances, explicitly retrieving parents that solve different subsets of the problem. This effectively treats parent selection as a retrieval task based on behavioral signatures, ensuring the LLM combines distinct functional capabilities. We should immediately implement this complementarity-based parent retrieval in AlgoEvo to improve how we merge heuristics.

### [Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning](https://arxiv.org/abs/2504.05108)

**2025-08-04** | EPFL, Apple | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Search with Reinforcement Learning (DPO) fine-tuning | *LLM role:* heuristic_generator

> EvoTune augments LLM-based evolutionary search (FunSearch) by iteratively fine-tuning the LLM weights using Direct Preference Optimization (DPO) on the generated programs. The results are robust, consistently outperforming static FunSearch on Bin Packing, TSP, and Hash Code benchmarks by discovering better heuristics faster. The critical takeaway is the use of **Forward KL regularization** in DPO instead of the standard Reverse KL; this prevents the mode collapse that usually kills evolutionary diversity, allowing the model to learn from high-fitness samples while maintaining exploration. This is a direct blueprint for implementing the 'RL-infused evolution' component of our AlgoEvo project.

### [Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery](https://arxiv.org/abs/2507.03605)

**2025-07-04** | Leiden University, University of Stirling | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA framework with 1+1 elitist evolution strategy and dual mutation prompts (code simplification and random perturbation) | *LLM role:* evolutionary_search

> The authors introduce a behavioral analysis framework for LLM-driven algorithm discovery, mapping the 'behavior space' of generated heuristics using Search Trajectory Networks (STNs) and Code Evolution Graphs (CEGs). Results on BBOB (5D) show that a simple 1+1 elitist strategy alternating between 'simplify code' and 'random new' prompts significantly outperforms population-based approaches, effectively balancing exploitation and exploration while preventing code bloat. The primary takeaway is the critical role of a 'simplify' mutation operator—without it, LLM-generated code tends to drift into complexity without performance gains. We should immediately adopt their visualization metrics to debug our own evolutionary search trajectories and implement their 'simplify' prompt strategy in AlgoEvo.

### [Automatic programming via large language models with population self-evolution for dynamic job shop scheduling problem](https://arxiv.org/abs/2410.22657)

**2024-10-30** | Huazhong University of Science and Technology | M=6 P=8 I=6 *discuss*

*Method:* LLM-based population self-evolutionary (SeEvo) method for automatic heuristic dispatching rules (HDRs) design | *LLM role:* heuristic_generator

> This paper introduces SeEvo, an LLM-based evolutionary search for Dynamic Job Shop Scheduling heuristics that adds an 'individual self-reflection' loop—prompting the LLM to analyze performance differences of a specific rule before and after mutation—alongside standard population-level reflection. While they claim significant improvements over GP/GEP and DRL, the ablation study reveals only a marginal <1% improvement over the existing ReEvo framework on benchmark instances. The primary takeaway for us is the specific prompt engineering technique of injecting an individual's mutation history (previous code vs. current code performance) into the context to guide the next mutation, which could potentially improve sample efficiency in our own evolutionary loops despite their weak empirical validation.

### [ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas](https://arxiv.org/abs/2602.04296)

**2026-02-04** |  | M=5 P=7 I=7 *discuss*

*Method:* ProxyWar framework, a competitive, execution-based evaluation system orchestrating automated code generation, hierarchical testing, iterative repair loops, and multi-agent tournaments with TrueSkill-based ranking. | *LLM role:* code_writer

> ProxyWar introduces a tournament-based evaluation framework for LLM-generated code, using TrueSkill ratings from game simulations (Sudoku, Poker, etc.) instead of static unit tests. The results are robust (10k+ matches) and reveal a low correlation between Pass@1 and actual win rates; notably, 'reasoning' models like DeepSeek-R1 crush 'coding' models like Qwen-Coder in strategic tasks despite lower static scores. For our evolutionary search work, this confirms that we must move beyond static benchmarks to dynamic, competitive evaluation signals to avoid optimizing for syntax over strategy. We should also prioritize reasoning models over code-specialized ones for our agentic logic generation.

### [LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling](https://arxiv.org/abs/2601.15738)

**2026-01-22** |  | M=7 P=6 I=7 *discuss*

*Method:* LLM-assisted evolutionary algorithm for automatic dispatching rule design with dual-expert mechanism and feature-fitting rule evolution | *LLM role:* heuristic_generator, evaluator

> LLM4DRD employs a dual-agent framework (Generator & Evaluator) to evolve priority dispatching rules for dynamic flexible assembly flow shops. The core contribution is the **Hybrid Evaluation** mechanism, where the Evaluator generates qualitative critiques (strengths/weaknesses) that are injected into the Generator's prompts to guide specific operators like 'Dominance-Fusion Crossover' and 'Directed Optimization.' Empirical results show it outperforms FunSearch and EOH, avoiding the premature convergence seen in other methods. The most stealable insight is the prompt structure for crossover: rather than blindly combining code, it uses the Evaluator's analysis of parent strengths to direct the merger, a technique we should implement to improve sample efficiency in our evolutionary search.

### [AlphaResearch: Accelerating New Algorithm Discovery with Language Models](https://arxiv.org/abs/2511.08522)

**2025-11-11** | Yale, NYU, Tsinghua, ByteDance | M=7 P=6 I=7 *discuss*

*Method:* Autonomous research agent with dual research environment combining execution-based verification and simulated real-world peer review | *LLM role:* research_agent

> AlphaResearch introduces a 'dual environment' for algorithm discovery: it generates natural language research ideas, filters them using a reward model fine-tuned on ICLR peer reviews, and then executes the surviving ideas. While it claims to beat human baselines on Packing Circles, the improvement is marginal (<0.1%) and it fails to improve upon baselines in 6/8 benchmark problems. The key takeaway for us is the mechanism of an 'Idea Critic'—using a learned reward model to filter the search space at the prompt level before wasting compute on execution—which directly addresses our sample efficiency goals in evolutionary search.

### [EvoVLMA: Evolutionary Vision-Language Model Adaptation](https://arxiv.org/abs/2508.01558)

**2025-08-03** | Chinese Academy of Sciences | M=7 P=4 I=7 *discuss*

*Method:* LLM-assisted two-stage evolutionary algorithm with crossover and mutation operators for optimizing feature selection and logits computation functions in code space | *LLM role:* code_writer

> This paper proposes EvoVLMA, an LLM-based evolutionary framework that searches for Python code to adapt Vision-Language Models (feature selection and logits computation). They demonstrate that **jointly** evolving two coupled algorithmic components fails (worse than random), whereas a **sequential** two-stage evolution strategy yields SOTA results (beating manual baselines by ~1-2%). For our AlgoEvo work, the key takeaway is the infrastructure design: they wrap code execution in restartable web services with a process monitor to handle the high rate of CUDA errors/timeouts in generated code—a practical 'trick' we should adopt to improve our search stability.

### [Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem](https://arxiv.org/abs/2509.02297)

**2025-09-02** | The University of Manchester | M=3 P=6 I=5 *discuss*

*Method:* Evolution of Heuristics (EoH) framework with constraint scaffolding and iterative self-correction | *LLM role:* heuristic_generator

> Quan et al. apply Evolution of Heuristics (EoH) to the Constrained 3D Packing Problem, finding that naive LLM generation fails completely without 'Constraint Scaffolding' (pre-written geometry libraries) and iterative repair. The results are soberingly realistic: while the scaffolded LLM matches greedy baselines on simple instances, it fails to generalize to complex constraints (stability, separation), significantly trailing human-designed metaheuristics. The key takeaway is their observation that the LLM exclusively optimizes the *scoring function* (weights/priorities) rather than the algorithmic structure, effectively reducing 'code evolution' to 'parameter tuning.' This confirms a critical limitation for our AlgoEvo work: simply asking for code results in local optimization; we must force structural changes or provide better primitives to get true novelty.

### [Data-Driven Discovery of Interpretable Kalman Filter Variants through Large Language Models and Genetic Programming](https://arxiv.org/abs/2508.11703)

**2025-08-25** | Harvard University, KTH Stockholm | M=4 P=6 I=5

*Method:* Hybrid Cartesian Genetic Programming (CGP) and LLM-assisted Evolutionary Search (ES) | *LLM role:* evolutionary_search

> Saketos et al. apply a FunSearch-style loop (using DeepSeek-14B) and Cartesian Genetic Programming (CGP) to evolve Kalman Filter variants, achieving ~3x MSE reduction on non-Gaussian noise scenarios. The results are empirically backed and highlight a critical limitation: LLM-ES failed to reconstruct the exact full Kalman Filter where traditional CGP succeeded, likely due to precision issues in symbolic reconstruction. The main takeaway is that for exact mathematical structure discovery, traditional symbolic mutation (CGP) still holds an edge over 14B-parameter LLM evolution, suggesting we should not fully abandon symbolic operators in our AILS-II control discovery pipeline. It also validates that open-weights 14B models are sufficient for FunSearch-style loops.

### [LLM-Driven Instance-Specific Heuristic Generation and Selection](https://arxiv.org/abs/2506.00490)

**2025-06-03** | Nanyang Technological University, The Hong Kong University of Science and Technology, The Hong Kong Polytechnic University, Southern University of Science and Technology, A*STAR, Zhongguancun Academy, Advanced Micro Devices Inc. | M=3 P=6 I=4

*Method:* LLM-driven instance-specific heuristic generation and selection framework (InstSpecHH) combining LLMs with Evolutionary Algorithms and a neighborhood search strategy | *LLM role:* code_writer, heuristic_selector, feature_description_generator

> Zhang et al. introduce a framework that partitions problem spaces (OBPP, CVRP) into thousands of subclasses and runs LLM-evolution (EoH) on *each* to create a lookup table of heuristics, selected at runtime via k-NN. While they achieve a 5.8% gap reduction on Bin Packing over single-heuristic baselines, the approach requires massive offline compute to generate thousands of scripts. The key takeaway is a negative result: using an LLM to select the best heuristic from candidates yielded negligible gains (0.1%) over simple feature-based distance, suggesting we should avoid LLM-based selector agents for this task. This confirms that 'one-size-fits-all' evolved heuristics struggle with heterogeneity, but we should solve this via adaptive code, not brute-force enumeration.

### [BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics](https://arxiv.org/abs/2504.20183)

**2025-04-28** | LIACS, Leiden University | M=3 P=5 I=5 *discuss*

*Method:* Modular and extensible framework for benchmarking LLM-driven Automated Algorithm Discovery | *LLM role:* algorithm_generator

> BLADE is a benchmarking framework for LLM-driven algorithm discovery (AAD) focused on continuous black-box optimization (BBOB, SBOX), integrating standard logging and analysis tools. The empirical results are standard (LLaMEA variants on BBOB), but the paper introduces **Code Evolution Graphs (CEG)**—a visualization technique that embeds generated code to track lineage and diversity during search. We should steal this visualization method for AlgoEvo to better debug population stagnation and diversity, even though the benchmark suite itself is too focused on continuous toy problems to replace our OR-centric evaluations.

### [Open-Universe Indoor Scene Generation using LLM Program Synthesis and Uncurated Object Databases](https://arxiv.org/abs/2403.09675)

**2024-02-05** | Brown University, UC San Diego, Dymaxion, LLC | M=5 P=1 I=5

*Method:* LLM program synthesis for a declarative Domain-Specific Language (DSL), followed by gradient-based optimization for constraint satisfaction, and VLM/multimodal LLM-based object retrieval and orientation | *LLM role:* program_synthesizer, evaluator, decomposition_guide

> Aguina-Kang et al. generate 3D scenes by prompting an LLM to write declarative Python programs (defining objects and spatial relations) which are then solved via gradient-based optimization. They demonstrate that LLMs perform significantly better at generating relational constraints than direct coordinate prediction, validating the neuro-symbolic architecture where the LLM handles specification and a solver handles instantiation. While the error-handling heuristics for unsatisfiable constraints (backtracking to the LLM) are practically sound, the specific gradient-based solver for spatial constraints is not applicable to our discrete combinatorial problems.

### [Optimizing Photonic Structures with Large Language Model Driven Algorithm Discovery](https://arxiv.org/abs/2503.19742)

**2025-03-25** | LIACS, Leiden University | M=3 P=3 I=4

*Method:* Large Language Model Evolutionary Algorithm (LLaMEA) with structured prompt engineering and dynamic mutation control | *LLM role:* evolutionary_search

> Yin et al. apply the LLaMEA framework to photonic inverse design, experimenting with injecting domain knowledge into prompts and varying evolutionary strategies (e.g., 1+1 vs 5+5). They demonstrate that LLM-generated algorithms can match baselines like DE and CMA-ES on continuous physics benchmarks. The only potentially useful takeaway is their negative result: detailed domain-specific prompts actually *degraded* performance on noisy fitness landscapes by prematurely constraining exploration. Aside from this prompt engineering heuristic, the work is an incremental application with no fundamental methodological contributions.

### [Leveraging Large Language Models to Develop Heuristics for Emerging Optimization Problems](https://arxiv.org/abs/2503.03350)

**2025-03-05** | TU Dortmund University, Karlsruhe Institute of Technology | M=2 P=4 I=3

*Method:* Contextual Evolution of Heuristics (CEoH) framework, an extension of Evolution of Heuristics (EoH) | *LLM role:* heuristic_generator

> Bömer et al. apply the Evolution of Heuristics (EoH) framework to the Unit-load Pre-marshalling Problem, proposing 'CEoH' which merely adds a static problem description to the prompt. Results show that while this context is crucial for smaller open-weights models (enabling Qwen-32B to slightly outperform GPT-4o on specific instances), it actually degrades the performance of GPT-4o compared to the baseline. The only takeaway for our AlgoEvo work is a confirmation that local model scaling requires verbose context injection, whereas frontier models may suffer from over-constrained prompts. This is an application paper with negligible algorithmic contribution.

### [Extending QAOA-GPT to Higher-Order Quantum Optimization Problems](https://arxiv.org/abs/2511.07391)

**2025-11-10** | University of Tennessee, Knoxville | M=2 P=1 I=3

*Method:* Generative Pre-trained Transformer (GPT) based circuit synthesis using FEATHER graph embeddings | *LLM role:* circuit_generator

> Sunny et al. train a nanoGPT model to predict QAOA circuit parameters for higher-order spin glasses by conditioning on FEATHER graph embeddings, effectively distilling the expensive ADAPT-QAOA algorithm into a single forward pass. They report approximation ratios (~0.95) matching the teacher algorithm on 16-qubit simulations. While the use of FEATHER embeddings to provide structural context to a Transformer is a clean implementation of graph-conditional generation, the approach is standard behavior cloning rather than the evolutionary or RL-based discovery methods we prioritize. The work is domain-specific to quantum computing and offers no transferable insights for our VRP or AlgoEvo pipelines.

### [AlgoPilot: Fully Autonomous Program Synthesis Without Human-Written Programs](https://arxiv.org/abs/2501.06423)

**2025-01-11** | Independent Researcher | M=2 P=1 I=3

*Method:* Reinforcement Learning (Transformer agent) guided by a Trajectory Language Model (TLM) | *LLM role:* algorithm_creation

> Yin introduces AlgoPilot, which trains a 'Trajectory Language Model' on traces from randomly generated functions to guide an RL agent in sorting small arrays. While the concept of using a trace-based model as a reward signal (a structural Process Reward Model) to encourage program-like behavior is theoretically interesting, the execution is flawed: the 'random' generator explicitly hardcodes the double-loop structure required for sorting. The results are trivial (sorting arrays of size 14) and the method relies on GPT-4o for the final code synthesis, making it a proof-of-concept with no scalability or genuine autonomy.

### [Automated Heuristic Design for Unit Commitment Using Large Language Models](https://arxiv.org/abs/2506.12495)

**2025-06-14** | Shanghai University of Electric Power, Shanghai Electrical Appliances Research Institute (Group) Co | M=1 P=2 I=0

*Method:* Function Space Search (FunSearch) combining a pre-trained LLM with a system evaluator for program search and evolution | *LLM role:* evolutionary_search

> Lv et al. attempt to apply FunSearch (LLM-based code evolution) to the Unit Commitment problem. The study is critically flawed: it tests on a trivial 10-unit instance (where exact solvers are instantaneous) and compares against an unspecified 'Genetic Algorithm'. The reported 'sampling time' of 6.6s for an LLM evolutionary process is technically implausible unless referring to the final heuristic's execution, indicating a likely confusion in their metrics or methodology. There are no actionable insights or reusable techniques here; it is a low-quality application paper.

<!-- New papers are appended here by the daily updater -->

---

## Research Fronts

*5 fronts detected — snapshot 2026-02-14*

### Front 1 (9 papers) — DECLINING

**Density:** 0.50 | **Methods:** llm_code_generation, llm_evolutionary_search, program_synthesis, llm_as_heuristic, llm_as_evaluator | **Problems:** algorithm_discovery, heuristic_evolution, operator_discovery, bin_packing, dynamic_flexible_assembly_flow_shop_scheduling

*Unique methods:* cartesian_genetic_programming, evo_mcts, evoph, execution_based_verification, island_models, llm_as_heuristic_generator, mcts, mdp_formulation, monte_carlo_tree_search, prompt_engineering, reflective_code_synthesis, reward_model
*Shared methods:* eoh, evolution_of_heuristics, evolutionary_algorithm, evolutionary_algorithms, evolutionary_search, funsearch, genetic_algorithm, genetic_programming, heuristic_evolution, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, llm_research_agent, map_elites, program_synthesis, reevo, self_improving_search

This research front is characterized by advancements in LLM-guided evolutionary search frameworks, primarily building upon FunSearch and ReEvo. The core theme is the development of sophisticated architectures that integrate LLMs into evolutionary loops for automated algorithm and heuristic design. Key innovations include dual-expert mechanisms (LLM4DRD), co-evolution of prompts and heuristics (EvoPH), and the integration of Monte Carlo Tree Search (Evo-MCTS) for enhanced exploration and exploitation. These frameworks are applied across diverse domains such as dynamic flexible assembly flow shop scheduling, unit commitment, vision-language model adaptation, Kalman filter variants, systems performance optimization, and gravitational-wave detection.

Papers in this front demonstrate significant empirical gains. LLM4DRD achieved 16.81% average tardiness reduction in scheduling, while EvoVLMA improved VLM adaptation accuracy by up to +1.91%. Saketos et al. achieved ~3x MSE reduction in Kalman Filter variants. A notable contribution is EvoPH, which reduced the Christofides gap from ~20% to ~5% on TSP by co-evolving prompts and heuristics. Evo-MCTS (ReEvo lineage) significantly outperformed standard LLM-evolution baselines by ~150% AUC on gravitational-wave detection, showcasing the power of MCTS integration. OpenEvolve (an AlphaEvolve-style framework) achieved a 5.0x speedup in MoE expert placement and 26% cost reduction in cloud scheduling. Practical insights include the 'Hybrid Evaluation' mechanism in LLM4DRD, the sequential two-stage evolution in EvoVLMA, and the 'Idea Critic' reward model in AlphaResearch.

This front is maturing, with a clear focus on refining the architectural components of LLM-guided evolutionary search. The trend is towards more complex, hybrid search strategies that combine LLMs with traditional search algorithms (e.g., MCTS, CGP) and advanced prompt engineering. The next papers are likely to focus on developing more robust and generalizable frameworks, addressing limitations like LLM hallucination and overfitting, and exploring human-machine interaction for collaborative algorithm design. The emphasis will be on improving sample efficiency, stability, and the ability to discover exact mathematical structures.

**Papers:**

### [LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling](https://arxiv.org/abs/2601.15738)

**2026-01-22** |  | M=7 P=6 I=7 *discuss*

*Method:* LLM-assisted evolutionary algorithm for automatic dispatching rule design with dual-expert mechanism and feature-fitting rule evolution | *LLM role:* heuristic_generator, evaluator

> LLM4DRD employs a dual-agent framework (Generator & Evaluator) to evolve priority dispatching rules for dynamic flexible assembly flow shops. The core contribution is the **Hybrid Evaluation** mechanism, where the Evaluator generates qualitative critiques (strengths/weaknesses) that are injected into the Generator's prompts to guide specific operators like 'Dominance-Fusion Crossover' and 'Directed Optimization.' Empirical results show it outperforms FunSearch and EOH, avoiding the premature convergence seen in other methods. The most stealable insight is the prompt structure for crossover: rather than blindly combining code, it uses the Evaluator's analysis of parent strengths to direct the merger, a technique we should implement to improve sample efficiency in our evolutionary search.

### [Automated Heuristic Design for Unit Commitment Using Large Language Models](https://arxiv.org/abs/2506.12495)

**2025-06-14** | Shanghai University of Electric Power, Shanghai Electrical Appliances Research Institute (Group) Co | M=1 P=2 I=0 

*Method:* Function Space Search (FunSearch) combining a pre-trained LLM with a system evaluator for program search and evolution | *LLM role:* evolutionary_search

> Lv et al. attempt to apply FunSearch (LLM-based code evolution) to the Unit Commitment problem. The study is critically flawed: it tests on a trivial 10-unit instance (where exact solvers are instantaneous) and compares against an unspecified 'Genetic Algorithm'. The reported 'sampling time' of 6.6s for an LLM evolutionary process is technically implausible unless referring to the final heuristic's execution, indicating a likely confusion in their metrics or methodology. There are no actionable insights or reusable techniques here; it is a low-quality application paper.

### [EvoVLMA: Evolutionary Vision-Language Model Adaptation](https://arxiv.org/abs/2508.01558)

**2025-08-03** | Chinese Academy of Sciences | M=7 P=4 I=7 *discuss*

*Method:* LLM-assisted two-stage evolutionary algorithm with crossover and mutation operators for optimizing feature selection and logits computation functions in code space | *LLM role:* code_writer

> This paper proposes EvoVLMA, an LLM-based evolutionary framework that searches for Python code to adapt Vision-Language Models (feature selection and logits computation). They demonstrate that **jointly** evolving two coupled algorithmic components fails (worse than random), whereas a **sequential** two-stage evolution strategy yields SOTA results (beating manual baselines by ~1-2%). For our AlgoEvo work, the key takeaway is the infrastructure design: they wrap code execution in restartable web services with a process monitor to handle the high rate of CUDA errors/timeouts in generated code—a practical 'trick' we should adopt to improve our search stability.

### [Data-Driven Discovery of Interpretable Kalman Filter Variants through Large Language Models and Genetic Programming](https://arxiv.org/abs/2508.11703)

**2025-08-25** | Harvard University, KTH Stockholm | M=4 P=6 I=5 

*Method:* Hybrid Cartesian Genetic Programming (CGP) and LLM-assisted Evolutionary Search (ES) | *LLM role:* evolutionary_search

> Saketos et al. apply a FunSearch-style loop (using DeepSeek-14B) and Cartesian Genetic Programming (CGP) to evolve Kalman Filter variants, achieving ~3x MSE reduction on non-Gaussian noise scenarios. The results are empirically backed and highlight a critical limitation: LLM-ES failed to reconstruct the exact full Kalman Filter where traditional CGP succeeded, likely due to precision issues in symbolic reconstruction. The main takeaway is that for exact mathematical structure discovery, traditional symbolic mutation (CGP) still holds an edge over 14B-parameter LLM evolution, suggesting we should not fully abandon symbolic operators in our AILS-II control discovery pipeline. It also validates that open-weights 14B models are sufficient for FunSearch-style loops.

### [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)

**2025-10-10** | UC Berkeley | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary search (MAP-Elites and island models) with automated code generation and empirical evaluation | *LLM role:* code_writer, reasoning_agent, feedback_generator

> The authors apply OpenEvolve (an AlphaEvolve-style framework) to 11 computer systems problems, achieving significant gains over human baselines, such as a 5.0x speedup in MoE expert placement and 26% cost reduction in cloud scheduling. The results are empirically rigorous, relying on high-fidelity simulators rather than toy problems. For us, the key takeaway is the engineering recipe: using an ensemble of reasoning models (o3) for exploration and fast models (Gemini) for diversity, combined with a specific 'failure taxonomy' to debug search stagnation. This is immediate proof-of-concept for your 'GPUSched' and 'AlgoEvo' projects; we should adopt their ensemble strategy and simulator-first evaluation pipeline.

### [AlphaResearch: Accelerating New Algorithm Discovery with Language Models](https://arxiv.org/abs/2511.08522)

**2025-11-11** | Yale, NYU, Tsinghua, ByteDance | M=7 P=6 I=7 *discuss*

*Method:* Autonomous research agent with dual research environment combining execution-based verification and simulated real-world peer review | *LLM role:* research_agent

> AlphaResearch introduces a 'dual environment' for algorithm discovery: it generates natural language research ideas, filters them using a reward model fine-tuned on ICLR peer reviews, and then executes the surviving ideas. While it claims to beat human baselines on Packing Circles, the improvement is marginal (<0.1%) and it fails to improve upon baselines in 6/8 benchmark problems. The key takeaway for us is the mechanism of an 'Idea Critic'—using a learned reward model to filter the search space at the prompt level before wasting compute on execution—which directly addresses our sample efficiency goals in evolutionary search.

### [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://arxiv.org/abs/2509.24509)

**2025-09-30** | Tencent, Renmin University of China, City University of Hong Kong | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Experience-Guided Reflective Co-Evolution of Prompts and Heuristics (EvoPH) with island-based elites selection | *LLM role:* heuristic_generator

> EvoPH introduces a co-evolutionary framework where both the heuristic code and the LLM prompts are evolved, utilizing an island model for diversity and a 'strategy sampling' mechanism that dynamically selects mutation types (e.g., parameter tuning vs. rewrite) based on feedback. They report dominating performance over FunSearch and ReEvo on TSP and BPP (e.g., reducing Christofides gap from ~20% to ~5%), though the static performance of baselines suggests the gain comes largely from automating prompt engineering. The most stealable insight is the **Strategy Sampling** module: explicitly defining a pool of mutation operators and using an 'experience' buffer to select them is a practical implementation of the 'planner' concept we need for AlgoEvo. We should also adopt their island migration topology to improve diversity in our parallelized search.

### [Automated Algorithmic Discovery for Scientific Computing through LLM-Guided Evolutionary Search: A Case Study in Gravitational-Wave Detection](https://arxiv.org/abs/2508.03661)

**2025-11-16** | Tsinghua University, University of Chinese Academy of Sciences | M=9 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided Evolutionary Monte Carlo Tree Search (Evo-MCTS) with reflective code synthesis and multi-scale evolutionary operations | *LLM role:* code_writer, heuristic_generator, evaluator, evolutionary_search

> Evo-MCTS introduces a hybrid search architecture where MCTS manages the exploration-exploitation balance of an evolutionary process, using LLMs for node expansion via novel operators like 'Path-wise Crossover' (synthesizing code from full root-to-leaf trajectories). The results are empirically strong, outperforming standard LLM-evolution baselines (ReEvo) by ~150% on a complex signal processing task. We learned that structuring the evolutionary lineage as a tree and using MCTS Q-values to select parents—rather than standard population selection—drastically improves sample efficiency and solution quality. This is a blueprint for the 'RL-infused evolution' and 'persistent memory' features we have been planning for our own framework.

### [Automatic programming via large language models with population self-evolution for dynamic job shop scheduling problem](https://arxiv.org/abs/2410.22657)

**2024-10-30** | Huazhong University of Science and Technology | M=6 P=8 I=6 *discuss*

*Method:* LLM-based population self-evolutionary (SeEvo) method for automatic heuristic dispatching rules (HDRs) design | *LLM role:* heuristic_generator

> This paper introduces SeEvo, an LLM-based evolutionary search for Dynamic Job Shop Scheduling heuristics that adds an 'individual self-reflection' loop—prompting the LLM to analyze performance differences of a specific rule before and after mutation—alongside standard population-level reflection. While they claim significant improvements over GP/GEP and DRL, the ablation study reveals only a marginal <1% improvement over the existing ReEvo framework on benchmark instances. The primary takeaway for us is the specific prompt engineering technique of injecting an individual's mutation history (previous code vs. current code performance) into the context to guide the next mutation, which could potentially improve sample efficiency in our own evolutionary loops despite their weak empirical validation.


### Front 4 (9 papers) — DECLINING

**Density:** 0.75 | **Methods:** llm_as_heuristic, llm_code_generation, evolution_of_heuristics, evolutionary_algorithm, program_synthesis | **Problems:** heuristic_evolution, bin_packing, tsp, cvrp, unit_load_pre_marshalling_problem

*Unique methods:* ast_manipulation, automated_algorithm_design, ceoh, co_evolution, constraint_scaffolding, differential_evolution, direct_preference_optimization, gnn, greedy_constructive_heuristic, iterative_self_correction, memetic_algorithm, multi_armed_bandit, multi_objective_optimization, neighborhood_search, pareto_dominance, population_management, supervised_fine_tuning, ucb
*Shared methods:* eoh, evolution_of_heuristics, evolutionary_algorithm, evolutionary_search, funsearch, in_context_learning, island_model_ea, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_in_the_loop, llm_prompt_optimization, program_synthesis, reevo

This research front focuses on advancing the Evolution of Heuristics (EoH) framework by integrating architectural refinements and sophisticated prompt engineering strategies. Papers introduce methods like TIDE for decoupling structural and parameter optimization, STRCMP for fusing Graph Neural Networks with LLMs, and Algorithmic-Contextual EoH (A-CEoH) for injecting algorithmic context into prompts. The primary goal is to enhance the precision, efficiency, and diversity of LLM-generated heuristics for complex combinatorial optimization problems such as TSP, Bin Packing, CVRP, and the Unit-Load Pre-marshalling Problem.

Key contributions include TIDE's nested evolutionary framework, which achieved a -7.35% gap on Constructive TSP N=50 by optimizing numerical parameters separately. A-CEoH demonstrated that injecting A* algorithm source code into prompts enabled a 32B parameter model to achieve optimal fitness on Unit-Load Pre-marshalling Problem training instances with faster runtime (0.0888s vs 0.2773s). STRCMP showed significant reductions in SAT solver timeouts (e.g., 71.8% better PAR-2 on Zamkeller) by conditioning LLMs with GNN-derived structural embeddings. Furthermore, MEoH introduced an AST-based dissimilarity metric for multi-objective heuristic evolution, yielding heuristics 16x faster on TSP (1.37s vs 22.4s) while maintaining quality, and EoH-S reduced optimality gaps by 40-60% on TSPLib and CVRPLib by evolving complementary sets of heuristics.

This front appears to be maturing, with a "declining" status, indicating a shift from foundational exploration to refinement and specialization. The trajectory suggests a move towards more robust and efficient integration of LLMs into evolutionary search, addressing specific limitations like numerical precision, multi-objective trade-offs, and population diversity. The next likely papers will focus on developing more sophisticated architectural patterns for LLM-EA synergy, potentially exploring adaptive prompt generation or meta-learning for problem-specific architectural choices, rather than broad conceptual breakthroughs.

**Papers:**

### [TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design](https://arxiv.org/abs/2601.21239)

**2026-01-29** |  | M=9 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Nested evolutionary framework with TSED-guided island model and co-evolutionary inner loop (UCB-based LLM logic generation + differential mutation for parameter tuning) | *LLM role:* heuristic_generator

> TIDE introduces a nested evolutionary framework that strictly decouples algorithmic structure generation (via LLM) from numerical parameter tuning (via Differential Evolution), managed by a Tree Similarity Edit Distance (TSED) guided island model. Results on 9 COPs (TSP, BPP, etc.) show it consistently outperforms ReEvo and EoH, primarily because the DE layer optimizes constants at zero token cost, preventing the discard of structurally sound but poorly tuned heuristics. The critical takeaway is the necessity of a gradient-free tuning layer for LLM-generated code; relying on LLMs for numerical constants is inefficient and imprecise. We should immediately implement a similar parameter-tuning inner loop in our AlgoEvo framework.

### [Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search](https://arxiv.org/abs/2601.19622)

**2026-01-27** |  | M=7 P=5 I=8 *discuss*

*Method:* Evolutionary Heuristic Design (EoH) framework with Algorithmic-Contextual Prompt Augmentation (A-CEoH) | *LLM role:* heuristic_generator

> This paper introduces 'Algorithmic-Contextual EoH' (A-CEoH), which injects the actual source code of the search algorithm (e.g., the A* driver loop, neighbor generation) into the LLM prompt alongside the problem description. Experiments on the Unit-Load Pre-Marshalling Problem and Sliding Puzzle Problem demonstrate that this algorithmic context allows a 32B parameter model (Qwen2.5-Coder) to generate heuristics superior to those from GPT-4o and human experts. The results are credible and backed by comparisons against optimal baselines. The key takeaway is a transferable 'prompt trick': explicitly showing the LLM the code that *calls* its generated function aligns the heuristic significantly better with the search dynamics than natural language descriptions alone. We should immediately test injecting our ALNS/search driver code into our evolutionary prompt templates.

### [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/abs/2506.11057)

**2025-05-22** | Shanghai Key Laboratory of Scalable Computing and Systems, School of Computer Science, Shanghai Jiao Tong University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structure-aware LLM-based algorithm discovery framework combining Graph Neural Network (GNN) for structural embeddings and LLM for solver-specific code generation, refined by an evolutionary algorithm. | *LLM role:* code_writer

> STRCMP introduces a composite architecture where a GNN encodes CO problem instances (MILP/SAT) into embeddings that condition an LLM (fine-tuned via SFT and DPO) to generate solver-specific heuristics within an evolutionary loop. The results are strong and empirically backed, showing significant reductions in convergence time and timeouts compared to text-only evolutionary methods like AutoSAT and LLM4Solver. The key takeaway is the architectural blueprint for fusing instance-specific structural embeddings (via soft prompting) with LLM code generation to drastically improve the sample efficiency of evolutionary search. This is immediately relevant to our EvoCut and AlgoEvo projects, suggesting we should move beyond pure text prompts for topology-heavy problems.

### [BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics](https://arxiv.org/abs/2504.20183)

**2025-04-28** | LIACS, Leiden University | M=3 P=5 I=5 *discuss*

*Method:* Modular and extensible framework for benchmarking LLM-driven Automated Algorithm Discovery | *LLM role:* algorithm_generator

> BLADE is a benchmarking framework for LLM-driven algorithm discovery (AAD) focused on continuous black-box optimization (BBOB, SBOX), integrating standard logging and analysis tools. The empirical results are standard (LLaMEA variants on BBOB), but the paper introduces **Code Evolution Graphs (CEG)**—a visualization technique that embeds generated code to track lineage and diversity during search. We should steal this visualization method for AlgoEvo to better debug population stagnation and diversity, even though the benchmark suite itself is too focused on continuous toy problems to replace our OR-centric evaluations.

### [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/abs/2409.16867)

**2025-02-04** | City University of Hong Kong, Southern University of Science and Technology | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-based Multi-objective Evolutionary Algorithm with Dominance-Dissimilarity Mechanism | *LLM role:* heuristic_generator

> MEoH extends LLM-based heuristic evolution (like FunSearch/EoH) to multi-objective scenarios (e.g., Gap vs. Runtime) by introducing a 'Dominance-Dissimilarity' mechanism that selects parents based on both Pareto dominance and Abstract Syntax Tree (AST) code distance. The results are credible and strong: on TSP, they find heuristics matching EoH's quality but running 16x faster (1.37s vs 22.4s) by effectively navigating the complexity-performance trade-off. The single most useful takeaway is the **AST-based dissimilarity metric** for population management; we should immediately steal this to prune semantically identical code in our evolutionary loops, thereby forcing exploration and improving sample efficiency. This is a direct upgrade to our current single-objective evolutionary search methods.

### [Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem](https://arxiv.org/abs/2509.02297)

**2025-09-02** | The University of Manchester | M=3 P=6 I=5 *discuss*

*Method:* Evolution of Heuristics (EoH) framework with constraint scaffolding and iterative self-correction | *LLM role:* heuristic_generator

> Quan et al. apply Evolution of Heuristics (EoH) to the Constrained 3D Packing Problem, finding that naive LLM generation fails completely without 'Constraint Scaffolding' (pre-written geometry libraries) and iterative repair. The results are soberingly realistic: while the scaffolded LLM matches greedy baselines on simple instances, it fails to generalize to complex constraints (stability, separation), significantly trailing human-designed metaheuristics. The key takeaway is their observation that the LLM exclusively optimizes the *scoring function* (weights/priorities) rather than the algorithmic structure, effectively reducing 'code evolution' to 'parameter tuning.' This confirms a critical limitation for our AlgoEvo work: simply asking for code results in local optimization; we must force structural changes or provide better primitives to get true novelty.

### [LLM-Driven Instance-Specific Heuristic Generation and Selection](https://arxiv.org/abs/2506.00490)

**2025-06-03** | Nanyang Technological University, The Hong Kong University of Science and Technology, The Hong Kong Polytechnic University, Southern University of Science and Technology, A*STAR, Zhongguancun Academy, Advanced Micro Devices Inc. | M=3 P=6 I=4 

*Method:* LLM-driven instance-specific heuristic generation and selection framework (InstSpecHH) combining LLMs with Evolutionary Algorithms and a neighborhood search strategy | *LLM role:* code_writer, heuristic_selector, feature_description_generator

> Zhang et al. introduce a framework that partitions problem spaces (OBPP, CVRP) into thousands of subclasses and runs LLM-evolution (EoH) on *each* to create a lookup table of heuristics, selected at runtime via k-NN. While they achieve a 5.8% gap reduction on Bin Packing over single-heuristic baselines, the approach requires massive offline compute to generate thousands of scripts. The key takeaway is a negative result: using an LLM to select the best heuristic from candidates yielded negligible gains (0.1%) over simple feature-based distance, suggesting we should avoid LLM-based selector agents for this task. This confirms that 'one-size-fits-all' evolved heuristics struggle with heterogeneity, but we should solve this via adaptive code, not brute-force enumeration.

### [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://arxiv.org/abs/2508.03082)

**2025-08-20** | Huawei Noah

































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































Ark Lab, City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary search framework with complementary population management and diversity-aware memetic search | *LLM role:* heuristic_generator

> EoH-S reformulates Automated Heuristic Design (AHD) to evolve a complementary *set* of heuristics rather than a single robust one, proving the objective is submodular and solvable via a greedy strategy. Results are strong and credible: on TSPLib and CVRPLib, their set of 10 heuristics reduces the optimality gap by ~40-60% compared to the top 10 heuristics from FunSearch or ReEvo. **KEY TAKEAWAY:** We should replace standard elitist selection in AlgoEvo with their 'Complementary Population Management' (CPM). By greedily selecting individuals based on marginal contribution to instance coverage (using instance-wise performance vectors), we can automatically generate diverse operator pools for ALNS instead of relying on hand-crafted diversity metrics.

### [Leveraging Large Language Models to Develop Heuristics for Emerging Optimization Problems](https://arxiv.org/abs/2503.03350)

**2025-03-05** | TU Dortmund University, Karlsruhe Institute of Technology | M=2 P=4 I=3 

*Method:* Contextual Evolution of Heuristics (CEoH) framework, an extension of Evolution of Heuristics (EoH) | *LLM role:* heuristic_generator

> Bömer et al. apply the Evolution of Heuristics (EoH) framework to the Unit-load Pre-marshalling Problem, proposing 'CEoH' which merely adds a static problem description to the prompt. Results show that while this context is crucial for smaller open-weights models (enabling Qwen-32B to slightly outperform GPT-4o on specific instances), it actually degrades the performance of GPT-4o compared to the baseline. The only takeaway for our AlgoEvo work is a confirmation that local model scaling requires verbose context injection, whereas frontier models may suffer from over-constrained prompts. This is an application paper with negligible algorithmic contribution.


### Front 0 (8 papers) — DECLINING

**Density:** 0.54 | **Methods:** program_synthesis, llm_evolutionary_search, llm_as_heuristic, llm_code_generation, evolution_of_heuristics | **Problems:** algorithm_discovery, tsp, heuristic_evolution, operator_discovery, bin_packing

*Unique methods:* abductive_reflection, adaptive_boltzmann_selection, agent_based_framework, alphaevolve, automated_evaluation, chain_of_thought, code_generation, dpo, evolutionary_computation, fireworks_algorithm, grpo, hybrid_evolutionary_memory, island_model, iterative_debugging, lineage_based_context_retrieval, llm_as_executor, llm_as_planner, llm_as_summarizer, llm_ensemble, local_verification_loop, meta_prompting, multi_island_model, one_plus_one_es, plan_execute_summarize, polymorphic_execution_strategies, reflection, rl_trained, swarm_intelligence
*Shared methods:* evolution_of_heuristics, evolutionary_algorithm, evolutionary_search, genetic_algorithm, heuristic_evolution, island_model_ea, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, llm_research_agent, map_elites, metaheuristics, program_synthesis, reinforcement_learning, self_improving_search

This research front focuses on advancing LLM-driven evolutionary search for automated algorithm and heuristic design, building upon and challenging frameworks like AlphaEvolve, FunSearch, and Evolution of Heuristics (EoH). The core theme is to move beyond simple LLM-based code generation by integrating more sophisticated, directed, and adaptive evolutionary mechanisms, including reinforcement learning for LLM fine-tuning, cognitive planning, and co-evolution of prompts and algorithms.

Key contributions include EvoTune, which uses DPO with Forward KL regularization to fine-tune LLMs, achieving up to 15% better optimality gap on Bin Packing and TSP compared to static FunSearch. CALM introduces online LLM fine-tuning via GRPO with relative improvement rewards, outperforming GPT-4o baselines on Bin Packing and VRP. CodeEvolve, an open-source AlphaEvolve competitor, surpasses AlphaEvolve on 5/9 benchmarks, establishing new SOTA on MinimizeMaxMinDist and CirclePackingSquare (n=32) with an 'inspiration-based' crossover. LoongFlow integrates a Plan-Execute-Summarize (PES) cognitive loop with 'Lineage-Based Context Retrieval', reducing evaluations by 60% and achieving 100% success on AlphaEvolve tasks. DeepEvolve augments AlphaEvolve with a 'Deep Research' module and a debugging agent, significantly improving code execution success rates from ~13% to ~99%. Furthermore, a critical analysis (Paper 1) highlights that simple (1+1)-EPS baselines often match or outperform complex population-based methods, while co-evolving prompt templates with algorithms (Paper 5) yields massive performance gains on constrained problems like Aircraft Landing.

This front is rapidly emerging and maturing, despite its "declining" status, with papers from 2024-2026 demonstrating significant innovation. The trajectory indicates a strong push towards integrating these advanced techniques—RL fine-tuning, cognitive planning, prompt co-evolution, and robust debugging—into unified, computationally efficient, and open-source frameworks. Future work will likely focus on developing more generalizable algorithm discovery across diverse and complex combinatorial optimization problems, while also addressing the computational costs and the implications of the "No Free Lunch" theorem for LLM-based search.

**Papers:**

### [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://arxiv.org/abs/2407.10873)

**2024-07-15** | City University of Hong Kong, Southern University of Science and Technology | M=4 P=10 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based Evolutionary Program Search (EPS) | *LLM role:* evolutionary_search

> Zhang et al. perform a rigorous benchmarking of major LLM-based evolutionary program search (EPS) methods (FunSearch, EoH, ReEvo) against a simple (1+1)-EPS baseline across four problems and nine LLMs. The results are empirically solid and sobering: the simple (1+1)-EPS baseline—iterative improvement via one-shot prompting—frequently matches or outperforms the complex population-based methods, particularly on bin packing, though EoH remains superior on TSP. **Crucial Takeaway:** We are likely over-engineering our search mechanisms; we must implement a (1+1)-EPS baseline in all future experiments (AlgoEvo, EvoCut) because if our multi-agent systems cannot beat this simple hill-climber, our papers will be rejected for unnecessary complexity. Additionally, they find that larger models (GPT-4) do not strictly guarantee better heuristic search performance compared to smaller, code-specialized models like CodeLlama-7B.

### [CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization](https://arxiv.org/abs/2510.14150)

**2026-01-06** | Inter&Co., Worcester Polytechnic Institute, Universidade Federal de Minas Gerais | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Islands-based genetic algorithm with modular LLM orchestration, context-aware recombination, adaptive meta-prompting, and depth-based exploitation | *LLM role:* code_writer

> CodeEvolve couples islands-based genetic algorithms with LLMs, utilizing CVT-MAP-Elites for diversity and a specific 'inspiration-based' crossover operator where the LLM integrates logic from high-ranking peer solutions. The results are strong and backed by numbers: they beat AlphaEvolve on 5/9 benchmarks and demonstrate that Qwen3-Coder-30B matches Gemini-2.5 performance at ~10% of the cost. The single most useful takeaway is the implementation of the 'inspiration' operator and the necessity of MAP-Elites over simple elitism to escape local optima in code space. We should immediately benchmark their open-source framework against our internal AlgoEvo builds.

### [Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning](https://arxiv.org/abs/2504.05108)

**2025-08-04** | EPFL, Apple | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Search with Reinforcement Learning (DPO) fine-tuning | *LLM role:* heuristic_generator

> EvoTune augments LLM-based evolutionary search (FunSearch) by iteratively fine-tuning the LLM weights using Direct Preference Optimization (DPO) on the generated programs. The results are robust, consistently outperforming static FunSearch on Bin Packing, TSP, and Hash Code benchmarks by discovering better heuristics faster. The critical takeaway is the use of **Forward KL regularization** in DPO instead of the standard Reverse KL; this prevents the mode collapse that usually kills evolutionary diversity, allowing the model to learn from high-fitness samples while maintaining exploration. This is a direct blueprint for implementing the 'RL-infused evolution' component of our AlgoEvo project.

### [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://arxiv.org/abs/2505.12285)

**2025-05-18** | City University of Hong Kong, Southeast University, University of Victoria, Hon Hai Research Institute | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining verbal and numerical guidance for heuristic evolution, achieved by fine-tuning an LLM via reinforcement learning (GRPO) based on heuristic quality, co-evolving the LLM with the search process. | *LLM role:* heuristic_generator_and_fine_tuned_agent

> CALM introduces a hybrid evolutionary framework that fine-tunes the LLM generator *during* the search process using Group Relative Policy Optimization (GRPO), rather than relying solely on prompt evolution. Using a quantized Qwen-7B model on a single consumer GPU, it outperforms GPT-4o-based baselines (FunSearch, EoH) on Bin Packing and VRP benchmarks. The critical takeaway is their reward function design: instead of absolute performance, they reward the *relative improvement* of the generated code over the specific 'parent' heuristics in the prompt, stabilizing the RL signal. We should immediately test this 'online fine-tuning' approach to reduce our API costs and improve sample efficiency in AlgoEvo.

### [Beyond Algorithm Evolution: An LLM-Driven Framework for the Co-Evolution of Swarm Intelligence Optimization Algorithms and Prompts](https://arxiv.org/abs/2512.09209)

**2025-12-10** | Peking University | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Collaborative evolution of Fireworks Algorithm operators and prompt templates, driven by a single LLM | *LLM role:* evolutionary_search

> The authors introduce a co-evolutionary framework where both the optimization algorithm (Fireworks Algorithm operators) and the prompt templates used to generate them are evolved simultaneously by the LLM. The results demonstrate a massive performance jump on constrained Aircraft Landing problems (from ~56% with FunSearch to 100% with their method), suggesting that static prompts are a primary failure mode for complex OR constraints. The critical takeaway is their prompt fitness function: evaluating a prompt template based on the *performance improvement* (`child - parent`) of the code it generates, rather than absolute performance. We should immediately implement this 'prompt-delta' fitness signal in AlgoEvo to automate our prompt engineering loop.

### [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

**2025-06-16** | Google DeepMind | M=10 P=10 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary algorithm for code superoptimization, orchestrating an autonomous pipeline of LLMs for code generation, critique, and evolution, grounded by code execution and automatic evaluation. | *LLM role:* evolutionary_search

> AlphaEvolve extends FunSearch by evolving entire code files (rather than single functions) using a 'search/replace' diff format and Gemini 2.0, achieving SOTA results across matrix multiplication (beating Strassen), 50+ open math problems, and Google's production scheduling. The results are exceptionally strong and verified, including deployed improvements to Google's Borg scheduler (0.7% resource recovery) and TPU circuits. The critical takeaway is the move to **diff-based full-file evolution** and **meta-prompt evolution** (evolving the prompt instructions alongside the code), which allows the system to modify architecture and logic rather than just heuristics. This is a mandatory blueprint for the next iteration of our AlgoEvo and EvoCut projects.

### [Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research](https://arxiv.org/abs/2510.06056)

**2025-10-07** | MIT-IBM Watson AI Lab, IBM Research, University of Notre Dame | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Agent-based framework integrating deep research (planning, searching, writing) with algorithm evolution (coding, evaluation, evolutionary selection) and iterative debugging. | *LLM role:* heuristic_generator, code_writer, evaluator, decomposition_guide, debugger

> DeepEvolve augments the standard evolutionary coding loop (AlphaEvolve) with two critical components: a 'Deep Research' module that searches the web/literature to generate grounded mutation proposals, and an iterative debugging agent that fixes execution errors. While the '666%' improvement on Circle Packing is likely due to a weak baseline (fixed-size vs. generalized), the engineering results are compelling: the debugging agent raises execution success rates from ~13% to ~99% in complex tasks. The key takeaway for our AlgoEvo work is the architecture of generating a text-based 'research proposal' via RAG before attempting code generation, rather than mutating code directly. We should immediately adopt their debugging loop and consider injecting external literature search into our mutation operators to prevent search stagnation.

### [LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm](https://arxiv.org/abs/2512.24077)

**2025-12-30** |  | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Plan-Execute-Summarize (PES) paradigm integrated with Hybrid Evolutionary Memory (Multi-Island, MAP-Elites, Adaptive Boltzmann Selection) | *LLM role:* planner, executor, summarizer

> LoongFlow replaces the standard stochastic mutation operator in LLM evolutionary search with a 'Plan-Execute-Summarize' (PES) cognitive loop. Instead of random code changes, a Planner retrieves the 'intent' and 'summary' of the parent solution's lineage to generate a directed hypothesis, which is then executed and summarized for the next generation. The authors demonstrate a 60% reduction in evaluations and a 100% success rate on AlphaEvolve tasks where standard methods fail or stagnate. The critical takeaway is the 'Lineage-Based Context Retrieval' mechanism: explicitly passing the parent's plan and retrospective summary to the child allows for directed rather than random walks in the search space. We must implement this PES loop in AlgoEvo immediately to fix our sample efficiency issues.


### Front 6 (5 papers) — GROWING

**Density:** 0.70 | **Methods:** llm_evolutionary_search, llamea, llm_code_generation, evolutionary_algorithm, llm_as_heuristic | **Problems:** black_box_optimization, heuristic_evolution, automated_algorithm_design, expensive_continuous_optimization, operator_discovery

*Unique methods:* abstract_syntax_tree_analysis, algorithm_analysis, elitist_strategy, evolution_strategy, evolutionary_strategies, evolutionary_strategy, explainable_ai, exploratory_landscape_analysis, llamea, search_space_analysis, shap_analysis, static_code_analysis, surrogate_modeling, wasserstein_distance, xgboost
*Shared methods:* evolution_of_heuristics, evolutionary_algorithm, genetic_programming, in_context_learning, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_prompt_optimization, metaheuristics, program_synthesis

This research front focuses on advancing the LLaMEA (Large Language Model Evolutionary Algorithm) framework for automated algorithm design, particularly targeting expensive continuous and black-box optimization problems. The core theme is to enhance the efficiency, robustness, and explainability of LLM-driven algorithm discovery by integrating sophisticated feedback mechanisms and novel evolutionary strategies, moving beyond basic prompt-based code generation. Key innovations include decoupling algorithm discovery from high-cost evaluations, leveraging structural code analysis, and designing advanced meta-evolutionary operators.

Key contributions include a hybrid framework that uses Genetic Programming to evolve symbolic proxy functions, enabling efficient algorithm discovery for real-world problems like meta-surface design by outperforming baselines with significantly reduced real evaluations [1]. LLaMEA-SAGE integrates Explainable AI (SHAP) analysis of Abstract Syntax Tree (AST) features to guide LLM mutations, achieving faster convergence on MA-BBOB benchmarks [3]. Another paper demonstrates the critical role of a 'simplify' mutation operator within a 1+1 elitist LLaMEA strategy, preventing code bloat and improving performance on BBOB functions [4]. Furthermore, an LLM-driven meta-evolutionary framework for symbolic regression introduces 'semantics-aware crossover' to evolve selection operators, outperforming expert-designed methods on SRBench [2].

This front is rapidly emerging and growing, characterized by a shift towards more intelligent and sample-efficient LLM-driven algorithm design. The trajectory suggests a strong focus on integrating the various advanced feedback loops—such as proxy-based evaluation, structural code analysis, and behavioral insights—into a more unified and adaptive LLaMEA architecture. Future work will likely explore the application of these enhanced frameworks to higher-dimensional, multi-objective, and noisy real-world optimization challenges, while also refining the interpretability and robustness of the generated algorithms.

**Papers:**

### [Landscape-aware Automated Algorithm Design: An Efficient Framework for Real-world Optimization](https://arxiv.org/abs/2602.04529)

**2026-02-04** |  | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining Genetic Programming (GP) for proxy function generation and an LLM-driven Evolutionary Algorithm (LLaMEA) for algorithm discovery, guided by Exploratory Landscape Analysis (ELA) features and Wasserstein distance. | *LLM role:* algorithm_designer

> Yin et al. introduce a framework that decouples algorithm discovery from expensive evaluations by using Genetic Programming to evolve symbolic proxy functions that statistically match the target problem's landscape (via ELA features). Empirical results on photonics problems confirm that algorithms evolved on these cheap proxies transfer successfully to the real tasks, outperforming standard baselines like LSHADE with only 50×D real evaluations. **Key Takeaway:** We can synthesize 'symbolic gyms' that statistically mimic our target problems to run thousands of LLM iterations at near-zero cost. This directly addresses the sample efficiency bottleneck in AlgoEvo and suggests we should move beyond standard neural surrogates to evolved symbolic proxies.

### [LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression](https://arxiv.org/abs/2505.18602)

**2025-08-08** | Victoria University of Wellington, Michigan State University | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven meta-evolutionary framework for designing selection operators, incorporating semantics-aware selection, bloat control, and domain knowledge into prompts | *LLM role:* evolutionary_search

> Zhang et al. develop a meta-evolutionary framework to evolve selection operators for symbolic regression, achieving state-of-the-art results on SRBench by outperforming expert-designed methods like ε-lexicase. The standout contribution is **semantics-aware crossover**: rather than selecting parents based solely on scalar fitness, they compute complementarity scores using performance vectors across instances, explicitly retrieving parents that solve different subsets of the problem. This effectively treats parent selection as a retrieval task based on behavioral signatures, ensuring the LLM combines distinct functional capabilities. We should immediately implement this complementarity-based parent retrieval in AlgoEvo to improve how we merge heuristics.

### [LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI](https://arxiv.org/abs/2601.21511)

**2026-01-29** |  | M=8 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA-SAGE, an LLM-driven evolutionary algorithm that integrates structural feedback from Explainable AI (SHAP) analysis of Abstract Syntax Tree (AST) code features to guide mutations. | *LLM role:* evolutionary_search

> LLaMEA-SAGE augments LLM-based evolutionary search by extracting AST features (complexity, graph metrics) from generated code, training a surrogate model to predict fitness from these features, and using SHAP analysis to generate natural language prompts that guide the LLM to modify specific structural properties (e.g., 'increase cyclomatic complexity'). On the MA-BBOB benchmark, it outperforms state-of-the-art methods (MCTS-AHD, LHNS) and converges faster than vanilla LLaMEA, although the authors honestly report that statistical significance was limited (p=0.44) due to small sample sizes (5 runs). The critical takeaway for us is the pipeline of using static code analysis as a feedback signal—we can immediately steal this 'SAGE' loop to guide AlgoEvo or EvoCut by telling the LLM *how* to structurally mutate code based on surrogate correlations, rather than just hoping for random improvements.

### [Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery](https://arxiv.org/abs/2507.03605)

**2025-07-04** | Leiden University, University of Stirling | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA framework with 1+1 elitist evolution strategy and dual mutation prompts (code simplification and random perturbation) | *LLM role:* evolutionary_search

> The authors introduce a behavioral analysis framework for LLM-driven algorithm discovery, mapping the 'behavior space' of generated heuristics using Search Trajectory Networks (STNs) and Code Evolution Graphs (CEGs). Results on BBOB (5D) show that a simple 1+1 elitist strategy alternating between 'simplify code' and 'random new' prompts significantly outperforms population-based approaches, effectively balancing exploitation and exploration while preventing code bloat. The primary takeaway is the critical role of a 'simplify' mutation operator—without it, LLM-generated code tends to drift into complexity without performance gains. We should immediately adopt their visualization metrics to debug our own evolutionary search trajectories and implement their 'simplify' prompt strategy in AlgoEvo.

### [Optimizing Photonic Structures with Large Language Model Driven Algorithm Discovery](https://arxiv.org/abs/2503.19742)

**2025-03-25** | LIACS, Leiden University | M=3 P=3 I=4 

*Method:* Large Language Model Evolutionary Algorithm (LLaMEA) with structured prompt engineering and dynamic mutation control | *LLM role:* evolutionary_search

> Yin et al. apply the LLaMEA framework to photonic inverse design, experimenting with injecting domain knowledge into prompts and varying evolutionary strategies (e.g., 1+1 vs 5+5). They demonstrate that LLM-generated algorithms can match baselines like DE and CMA-ES on continuous physics benchmarks. The only potentially useful takeaway is their negative result: detailed domain-specific prompts actually *degraded* performance on noisy fitness landscapes by prematurely constraining exploration. Aside from this prompt engineering heuristic, the work is an incremental application with no fundamental methodological contributions.


### Front 7 (2 papers) — EMERGING

**Density:** 1.00 | **Methods:** program_synthesis, reinforcement_learning, transformer, language_model, llm_code_generation | **Problems:** program_synthesis, sorting_algorithms, algorithm_discovery, reward_discovery, meta_black_box_optimization

*Unique methods:* eureka, guided_reinforcement_learning, knowledge_transfer, language_model, multi_task_learning, population_based_optimization, random_search, transformer
*Shared methods:* eoh, evolutionary_algorithms, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, program_synthesis, reevo, reinforcement_learning

This research front explores novel methodologies for automated discovery of computational components, specifically focusing on LLM-guided autonomous program synthesis and evolutionary reward discovery. AlgoPilot introduces a Trajectory Language Model (TLM) to guide a Transformer-based RL agent in synthesizing sorting algorithms, aiming for full autonomy. Concurrently, READY presents a multi-task evolutionary framework where LLMs evolve reward functions for Meta-Black-Box Optimization (MetaBBO) algorithms.

AlgoPilot's key contribution is its attempt at fully autonomous program synthesis, employing a Trajectory Language Model to guide a reinforcement learning agent in creating sorting algorithms, albeit with limited success on small arrays (e.g., fewer operations for array sizes 6, 8, 10 for QuickSort). In contrast, READY introduces a robust multi-task evolutionary framework that leverages LLMs to evolve reward functions for MetaBBO. This framework, featuring multi-task niches and explicit knowledge transfer, significantly outperforms prior methods like Eureka and EoH on the BBOB test suite, achieving average ranks of 2.00 for DEDQN and 2.38 for RLEPSO, alongside a 2-4x reduction in search time.

This front is clearly emerging, marked by foundational but imperfect attempts at full autonomy (AlgoPilot) and more sophisticated, performance-driven evolutionary approaches (READY). The next generation of research will likely focus on overcoming the scalability and genuine autonomy challenges highlighted by AlgoPilot, potentially by replacing LLM-based creation with fully trained sequence-to-sequence models. Simultaneously, further work will expand READY's multi-task evolutionary framework to a broader range of MetaBBO tasks, investigate the impact of diverse LLM architectures, and refine its fine-grained evolutionary operators for even greater efficiency and generalization.

**Papers:**

### [AlgoPilot: Fully Autonomous Program Synthesis Without Human-Written Programs](https://arxiv.org/abs/2501.06423)

**2025-01-11** | Independent Researcher | M=2 P=1 I=3 

*Method:* Reinforcement Learning (Transformer agent) guided by a Trajectory Language Model (TLM) | *LLM role:* algorithm_creation

> Yin introduces AlgoPilot, which trains a 'Trajectory Language Model' on traces from randomly generated functions to guide an RL agent in sorting small arrays. While the concept of using a trace-based model as a reward signal (a structural Process Reward Model) to encourage program-like behavior is theoretically interesting, the execution is flawed: the 'random' generator explicitly hardcodes the double-loop structure required for sorting. The results are trivial (sorting arrays of size 14) and the method relies on GPT-4o for the final code synthesis, making it a proof-of-concept with no scalability or genuine autonomy.

### [READY: Reward Discovery for Meta-Black-Box Optimization](https://arxiv.org/abs/2601.21847)

**2026-01-29** |  | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* LLM-based program evolution with a multi-task niche-based architecture, fine-grained evolutionary operators, and explicit knowledge transfer | *LLM role:* evolutionary_search

> READY introduces a multi-task evolutionary framework where LLMs evolve reward functions for multiple MetaBBO algorithms simultaneously, utilizing explicit 'Knowledge Transfer' operators to translate successful logic between distinct tasks. The results are robust, demonstrating superior performance over Eureka and EoH on BBOB benchmarks with a 2-4x reduction in search time due to parallelization and shared heuristics. The most stealable insights are the 'History-Reflection' operator—which prompts the LLM to extrapolate trends from the evolutionary trajectory rather than just mutating the current state—and the cross-niche transfer mechanism, both of which should be implemented in our multi-agent optimization stack immediately.



## Bridge Papers

Papers connecting multiple research fronts:

### [LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression](https://arxiv.org/abs/2505.18602)

**TRUE SYNTHESIS** | score=0.68 | Front 6 → Front 0, Front 1

> Zhang et al. develop a meta-evolutionary framework to evolve selection operators for symbolic regression, achieving state-of-the-art results on SRBench by outperforming expert-designed methods like ε-

### [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://arxiv.org/abs/2407.10873)

**TRUE SYNTHESIS** | score=0.58 | Front 0 → Front 4, Front 1, Front 6

> Zhang et al. perform a rigorous benchmarking of major LLM-based evolutionary program search (EPS) methods (FunSearch, EoH, ReEvo) against a simple (1+1)-EPS baseline across four problems and nine LLMs

### [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/abs/2506.11057)

**TRUE SYNTHESIS** | score=0.57 | Front 4 → Front 0, Front 1

> STRCMP introduces a composite architecture where a GNN encodes CO problem instances (MILP/SAT) into embeddings that condition an LLM (fine-tuned via SFT and DPO) to generate solver-specific heuristics

### [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/abs/2409.16867)

**TRUE SYNTHESIS** | score=0.50 | Front 4 → Front 0, Front 6, Front 1

> MEoH extends LLM-based heuristic evolution (like FunSearch/EoH) to multi-objective scenarios (e.g., Gap vs. Runtime) by introducing a 'Dominance-Dissimilarity' mechanism that selects parents based on 

### [AlphaResearch: Accelerating New Algorithm Discovery with Language Models](https://arxiv.org/abs/2511.08522)

**TRUE SYNTHESIS** | score=0.50 | Front 1 → Front 0

> AlphaResearch introduces a 'dual environment' for algorithm discovery: it generates natural language research ideas, filters them using a reward model fine-tuned on ICLR peer reviews, and then execute


---

*Generated by Research Intelligence System*
