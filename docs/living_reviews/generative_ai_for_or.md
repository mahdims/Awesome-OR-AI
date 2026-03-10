# Living Review: Generative AI for OR

**Last Updated:** 2026-03-10

---

## Recent Papers

#### 2026-03-01 (1 papers)

### [OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents](https://arxiv.org/abs/2602.19439)

**2026-02-23** | Massachusetts Institute of Technology, Alibaba Group | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-phase closed-loop LLM agent with IIS-guided diagnosis, domain-specific rationality oracle, iterative STaR, and GRPO refinement | *LLM role:* diagnosis_and_repair

> Ao et al. introduce OptiRepair, a closed-loop framework that repairs infeasible LPs using solver IIS feedback (Phase 1) and validates them with a 'Rationality Oracle' based on domain theory (Phase 2). Results are exceptionally strong: fine-tuned 8B models trained via iterative STaR and GRPO achieve 81.7% success, outperforming GPT-5.2 (42.2%) by a massive margin. **Key Takeaway:** We should steal the 'Rationality Oracle' concept—evaluating solution *properties* (e.g., monotonicity, variance bounds) rather than just raw fitness—to serve as a dense signal for our Process Reward Models in AlgoEvo. Additionally, their success with solver-verified GRPO confirms we should prioritize training specialized operators over prompting general LLMs.


#### 2026-02-26 (1 papers)

### [OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents](https://arxiv.org/abs/2602.19439)

**2026-02-23** | Massachusetts Institute of Technology, Alibaba Group | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-phase closed-loop LLM agent with IIS-guided diagnosis, domain-specific rationality oracle, iterative STaR, and GRPO refinement | *LLM role:* diagnosis_and_repair

> Ao et al. introduce OptiRepair, a closed-loop framework that repairs infeasible LPs using solver IIS feedback (Phase 1) and validates them with a 'Rationality Oracle' based on domain theory (Phase 2). Results are exceptionally strong: fine-tuned 8B models trained via iterative STaR and GRPO achieve 81.7% success, outperforming GPT-5.2 (42.2%) by a massive margin. **Key Takeaway:** We should steal the 'Rationality Oracle' concept—evaluating solution *properties* (e.g., monotonicity, variance bounds) rather than just raw fitness—to serve as a dense signal for our Process Reward Models in AlgoEvo. Additionally, their success with solver-verified GRPO confirms we should prioritize training specialized operators over prompting general LLMs.


#### 2026-02-24 (1 papers)

### [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983)

**2026-02-17** | National University of Singapore, Northwestern University, City University of Hong Kong, Wenzhou University, Wenzhou Buyi Pharmacy Chain Co., Ltd. | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structured generation (understand, formalize, synthesize, verify) with two-layer behavioral verification (L1 execution recovery, L2 solver-based perturbation testing) and diagnosis-guided repair. | *LLM role:* code_writer

> ReLoop proposes a verification pipeline for LLM-generated optimization models that detects 'silent failures' (code that runs but solves the wrong problem) by perturbing input parameters and checking for expected solver objective shifts. They demonstrate that standard execution feasibility is a poor proxy for correctness (90% gap) on their new RetailOpt-190 benchmark, and that this perturbation testing significantly improves reliability. The critical takeaway is the use of sensitivity analysis as a ground-truth-free process reward signal: we can validate evolved algorithms in AlgoEvo by asserting that specific input perturbations *must* trigger output changes, filtering out semantically invalid candidates before expensive evaluation.


#### 2026-02-22 (1 papers)

### [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983)

**2026-02-17** | National University of Singapore, Northwestern University, City University of Hong Kong, Wenzhou University, Wenzhou Buyi Pharmacy Chain Co., Ltd. | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structured generation (understand, formalize, synthesize, verify) with two-layer behavioral verification (L1 execution recovery, L2 solver-based perturbation testing) and diagnosis-guided repair. | *LLM role:* code_writer

> ReLoop proposes a verification pipeline for LLM-generated optimization models that detects 'silent failures' (code that runs but solves the wrong problem) by perturbing input parameters and checking for expected solver objective shifts. They demonstrate that standard execution feasibility is a poor proxy for correctness (90% gap) on their new RetailOpt-190 benchmark, and that this perturbation testing significantly improves reliability. The critical takeaway is the use of sensitivity analysis as a ground-truth-free process reward signal: we can validate evolved algorithms in AlgoEvo by asserting that specific input perturbations *must* trigger output changes, filtering out semantically invalid candidates before expensive evaluation.


#### 2026-02-22 (1 papers)

### [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983)

**2026-02-17** | National University of Singapore, Northwestern University, City University of Hong Kong, Wenzhou University, Wenzhou Buyi Pharmacy Chain Co., Ltd. | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structured generation (understand, formalize, synthesize, verify) with two-layer behavioral verification (L1 execution recovery, L2 solver-based perturbation testing) and diagnosis-guided repair. | *LLM role:* code_writer

> ReLoop proposes a verification pipeline for LLM-generated optimization models that detects 'silent failures' (code that runs but solves the wrong problem) by perturbing input parameters and checking for expected solver objective shifts. They demonstrate that standard execution feasibility is a poor proxy for correctness (90% gap) on their new RetailOpt-190 benchmark, and that this perturbation testing significantly improves reliability. The critical takeaway is the use of sensitivity analysis as a ground-truth-free process reward signal: we can validate evolved algorithms in AlgoEvo by asserting that specific input perturbations *must* trigger output changes, filtering out semantically invalid candidates before expensive evaluation.


#### 2026-02-22 (1 papers)

### [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983)

**2026-02-17** | National University of Singapore, Northwestern University, City University of Hong Kong, Wenzhou University, Wenzhou Buyi Pharmacy Chain Co., Ltd. | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structured generation (understand, formalize, synthesize, verify) with two-layer behavioral verification (L1 execution recovery, L2 solver-based perturbation testing) and diagnosis-guided repair. | *LLM role:* code_writer

> ReLoop proposes a verification pipeline for LLM-generated optimization models that detects 'silent failures' (code that runs but solves the wrong problem) by perturbing input parameters and checking for expected solver objective shifts. They demonstrate that standard execution feasibility is a poor proxy for correctness (90% gap) on their new RetailOpt-190 benchmark, and that this perturbation testing significantly improves reliability. The critical takeaway is the use of sensitivity analysis as a ground-truth-free process reward signal: we can validate evolved algorithms in AlgoEvo by asserting that specific input perturbations *must* trigger output changes, filtering out semantically invalid candidates before expensive evaluation.


<!-- New papers are appended here by the daily updater -->

---

## Research Fronts

*5 fronts detected — snapshot 2026-03-10*

### Front 2 (14 papers) — STABLE

**Density:** 0.29 | **Methods:** llm_code_generation, llm_in_the_loop, llm_as_heuristic, program_synthesis, llm_evolutionary_search | **Problems:** bin_packing, scheduling, tsp, knapsack, job_shop_scheduling

*Unique methods:* ADMM, actor_critic, adaptive_algorithms, agentic_framework, agentic_workflow, aide, asymmetric_validation, attention_mechanism, automated_algorithm_design, automated_experimentation, autonomous_coding_agents, bestofn_sampling, bin_packing_heuristics, black_box_optimization, branch_and_bound, centralized_training_decentralized_execution, chain_of_experts, competitive_analysis, compositional_prompting, continual_learning, crossover, cutting_planes, deep_reinforcement_learning, differential_attention, dynamic_weight_adjustment, eoh, evolutionary_algorithm, evolutionary_algorithms, evolutionary_search, execution_aware_modeling, greedy_algorithm, greedy_refinement, heuristic_design, hierarchical_reinforcement_learning, hyper_heuristics, hyperparameter_optimization, imitation_learning, ipython_kernel, k_means_clustering, karp_reductions, knowledge_graphs, lagrangian_relaxation, llm_agent, llm_as_code_generator, llm_as_expert, llm_research_agent, lookahead_mechanism, lower_bound_estimation, mathematical_optimization, memory_compression, meta_optimization, metaheuristics, minimum_bayes_risk_decoding, minizinc_modeling, model_context_protocol, mstc_ahd, multi_agent_reinforcement_learning, multi_head_attention, mutation, mutation_testing, neurosymbolic_ai, nsga_ii, online_scheduling, optimization_model_validation, options_framework, ordered_eviction, prioritized_experience_replay, problem_formulation, react_framework, reevo, reflection, reflection_mechanism, semi_markov_decision_process, software_testing, test_case_generation, tool_use, tree_search, vanilla_prompting, vector_embedding, wasserstein_metric, weighted_sum_method
*Shared methods:* chain_of_thought, constraint_programming, cpmpy, dynamic_programming, evolution_of_heuristics, funsearch, in_context_learning, iterative_refinement, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, milp_solver, multi_agent_system, program_synthesis, prompt_engineering, reinforcement_learning, retrieval_augmented_generation, robust_optimization, self_consistency

This front centers on the development of agentic evolutionary frameworks that leverage Large Language Models (LLMs) for the automated discovery, synthesis, and validation of algorithms and models in Operations Research (OR). Key approaches include FunSearch-inspired methods, hierarchical agent architectures like MiCo, and execution-aware validation loops. The focus is on moving beyond simple code generation to iterative, reflective, and often multi-agent systems that can autonomously explore and refine solutions for complex combinatorial optimization problems.

Notable contributions include OR-Agent [1], which unifies evolutionary search with a tree-structured research workflow, achieving significant improvements over FunSearch and AEL on TSP and CVRP. EvoCut [3] demonstrates LLM-powered evolutionary generation of MILP acceleration cuts, yielding 17-57% gap reductions on benchmarks like TSPLIB. For automated modeling, NEMO [6] introduces an execution-aware agentic framework with an asymmetric simulator-optimizer validation loop, achieving SOTA on 8/9 optimization benchmarks. EquivaMap [8] uses LLMs to discover linear mappings for rigorous equivalence checking of optimization formulations, reaching 100% accuracy on a new dataset. Benchmarking efforts like CO-Bench [2] and HeuriGym [12] provide standardized evaluation for LLM-generated algorithms and heuristics, revealing current LLM limitations in handling feasibility constraints and complex heuristic design.

This front is rapidly emerging and maturing, characterized by a shift from basic LLM prompting to sophisticated multi-agent, evolutionary, and reflective architectures. The next wave of papers will likely focus on integrating formal verification methods (e.g., automated proof systems for cuts [3], theoretical competitive ratios for scheduling [13]), enhancing LLM agents' creative reasoning and problem comprehension for complex constraints [2], and developing more efficient, scalable frameworks that can handle larger problem instances and reduce computational overhead [6, 7, 9]. There's also a clear trajectory towards more robust validation mechanisms, moving beyond simple execution accuracy to mutation testing [10] and equivalence checking [8].

**Papers:**

### [OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery](https://arxiv.org/abs/2602.13769)

**2026-02-14** | Tongji University | M=9 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent research framework with evolutionary-systematic ideation, tree-structured research workflow, and hierarchical optimization-inspired reflection system | *LLM role:* heuristic_generator, code_writer, evaluator, decomposition_guide, evolutionary_search, prompt_optimizer

> OR-Agent replaces flat evolutionary loops with a tree-structured research workflow that prioritizes deep iterative refinement and debugging over broad population sampling. The results are compelling, showing a ~2x improvement in normalized scores over ReEvo and FunSearch across 12 OR benchmarks (TSP, CVRP, etc.). The single most actionable takeaway is the **Experiment Agent's environment probing**: instead of relying on scalar fitness, the agent writes custom callbacks to log intermediate states (e.g., 'lane change attempts' in SUMO), enabling genuine diagnosis of failure modes. We should immediately implement this 'instrumentation-via-code' pattern in our own evaluation pipelines to improve signal quality.

### [CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization](https://arxiv.org/abs/2504.04310)

**2025-04-06** | Carnegie Mellon University | M=4 P=9 I=7 **MUST-READ** *discuss*

*Method:* LLM-based algorithm search using agentic frameworks with iterative refinement and evolutionary search | *LLM role:* evolutionary_search

> Sun et al. introduce CO-Bench, a suite of 36 diverse combinatorial optimization problems (packing, scheduling, routing) designed specifically to benchmark LLM agents in generating algorithms (code), not just solutions. They evaluate 9 frameworks (including FunSearch, ReEvo, AIDE), finding that FunSearch combined with reasoning models (o3-mini) yields the most robust performance, though agents still struggle significantly with strict feasibility constraints (valid solution rates often <60%). **Takeaway:** We should immediately integrate CO-Bench into our pipeline to benchmark AlgoEvo against ReEvo and FunSearch; this saves us months of data curation and provides a standardized metric to prove our method's superiority.

### [EvoCut: Strengthening Integer Programs via Evolution-Guided Language Models](https://arxiv.org/abs/2508.11850)

**2025-08-16** | Huawei Technologies Canada, University of British Columbia, University of Toronto | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary algorithm powered by multiple LLM-based agents for iterative generation and refinement of acceleration cuts | *LLM role:* heuristic_generator

> Yazdani et al. introduce EvoCut, an evolutionary framework where LLMs generate Python code for MILP cuts, filtered by a 'usefulness check' (does it cut the current LP relaxation?) and an 'empirical validity check' (does it preserve known integer optima?). They report 17-57% gap reductions on TSPLIB and JSSP compared to Gurobi defaults, backed by strong ablation studies on the evolutionary operators. **Key Takeaway:** The reliance on 'acceleration cuts'—constraints verified empirically on small datasets rather than formally proven—bypasses the bottleneck of automated theorem proving while still delivering valid speedups. We should immediately adopt their 'LP separation' check as a cheap, high-signal reward for our own evolutionary search loops.

### [Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc](https://arxiv.org/abs/2503.10642)

**2025-02-22** | Brown University, Fidelity Investments | M=3 P=8 I=4 **MUST-READ** *discuss*

*Method:* LLM-based MiniZinc model generation using various prompting strategies (Vanilla, Chain-of-Thought, Compositional) | *LLM role:* code_writer

> Singirikonda et al. introduce TEXT2ZINC, a dataset of 110 Natural Language-to-MiniZinc problems, and benchmark GPT-4 using Vanilla, CoT, and Compositional prompting. Their results are poor (max ~25% solution accuracy), confirming that off-the-shelf LLMs struggle significantly with MiniZinc syntax and logical translation. Crucially, they attempt using Knowledge Graphs as an intermediate representation, but report that it actually *reduced* solution accuracy compared to basic CoT—a valuable negative result for our symbolic modeling work. We should examine their dataset for inclusion in OR-Bench, but their prompting methods are rudimentary baselines we should easily outperform.

### [Learning Virtual Machine Scheduling in Cloud Computing through Language Agents](https://arxiv.org/abs/2505.10117)

**2025-05-15** | Shanghai Jiao Tong University, East China Normal University, Tongji University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hierarchical Language Agent Framework (MiCo) for LLM-driven heuristic design, formulated as Semi-Markov Decision Process with Options (SMDP-Option), using LLM-based function optimization for policy discovery and composition. | *LLM role:* heuristic_generator, evolutionary_search, decomposition_guide

> Wu et al. introduce MiCo, a hierarchical framework that uses LLMs to evolve both a library of scenario-specific scheduling heuristics ('Options') and a master policy ('Composer') that dynamically switches between them based on system state. Tested on large-scale Huawei/Azure VM traces, it achieves a 96.9% competitive ratio against Gurobi, significantly outperforming Deep RL (SchedRL) by ~11% in dynamic scenarios. **Key Insight:** Instead of evolving a single robust heuristic (which often fails in non-stationary environments), explicitly evolve a *portfolio* of specialized heuristics and a separate *selector* function. This SMDP-based decomposition is a concrete architectural pattern we should adopt in AlgoEvo to handle diverse problem instances and non-stationary distributions effectively.

### [NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](https://arxiv.org/abs/2601.21372)

**2026-01-29** | Carnegie Mellon University, C3 AI | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Execution-Aware Optimization Modeling via Autonomous Coding Agents (ACAs) with asymmetric simulator-optimizer validation loop | *LLM role:* code_writer

> NEMO achieves SOTA on 8/9 optimization benchmarks by deploying autonomous coding agents that generate both a declarative optimizer (solver code) and an imperative simulator (verification code). The key innovation is using the simulator to validate the optimizer's results in a closed loop, detecting logical errors without ground truth—a technique that beats fine-tuned models like SIRL by up to 28%. The most stealable insight is this asymmetric validation: imperative Python simulation is often less error-prone than declarative constraint formulation, making it a robust 'critic' for generated solvers. This is immediately applicable to our OR-Bench and AlgoEvo projects for generating reliable reward signals.

### [REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models](https://arxiv.org/abs/2506.07759)

**2025-06-09** | Vicomtech Foundation, University of the Basque Country, Universidad EAFIT, HiTZ Basque Center for Language Technology | M=7 P=6 I=7 **MUST-READ** *discuss*

*Method:* Hybrid framework integrating NSGA-II with LLM-based heuristic generation and a reflection mechanism | *LLM role:* evolutionary_search

> Forniés-Tabuenca et al. propose REMoH, an LLM-driven evolutionary framework for multi-objective FJSSP that uses K-Means to cluster the population by objective performance before generating reflections. While their optimality gaps (~12%) trail behind state-of-the-art CP solvers (~1.5%), the ablation study confirms that their reflection mechanism significantly improves Pareto front diversity (Hypervolume). **The killer feature is the phenotypic clustering step:** instead of reflecting on a random or elitist subset, they group solutions by trade-offs (e.g., 'low makespan' vs 'balanced') to generate targeted prompts. We should implement this clustering-based context construction in AlgoEvo to improve diversity maintenance in multi-objective search without exploding token costs.

### [EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](https://arxiv.org/abs/2502.14760)

**2025-02-20** | Stanford University, The University of Texas at Austin | M=7 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based discovery of linear mapping functions between decision variables, followed by MILP solver-based verification of feasibility and optimality | *LLM role:* heuristic_generator

> Zhai et al. propose EquivaMap, a framework that evaluates whether two MILP formulations are equivalent by using an LLM to discover a linear mapping between their decision variables, which is then rigorously verified by a solver. Unlike 'execution accuracy' (which fails on unit scaling) or 'canonical accuracy' (which fails on variable permutation), they achieve 100% accuracy on a new dataset of equivalent formulations including cuts and slack variables. The core insight is replacing output comparison with a 'propose-mapping-and-verify' loop, effectively using the LLM to construct a proof of equivalence. We must adopt this methodology for the OR-Bench evaluation pipeline immediately, as it eliminates the false negatives currently plaguing our generation benchmarks.

### [Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows](https://arxiv.org/abs/2505.04354)

**2025-05-07** | University of Minnesota, Tongji University, East China Normal University | M=5 P=9 I=6 *discuss*

*Method:* Evolutionary Agentic Workflow combining Foundation Agents (Memory, Reasoning, World Modeling, Action modules) and Evolutionary Search (Distributed Population Management, Solution Diversity Preservation, Knowledge-Guided Evolution) | *LLM role:* evolutionary_search

> Li et al. propose an 'Evolutionary Agentic Workflow' that combines LLMs (DeepSeek) with evolutionary search to automate algorithm design, demonstrating it on VM scheduling and ADMM parameter tuning. The empirical rigor is low; they compare against weak baselines (BestFit for bin packing, a 2000-era heuristic for ADMM) and frame it as a position paper. However, the application of LLM-evolution to discover symbolic mathematical update rules (for ADMM step sizes) rather than just procedural code is a concrete use case we should consider for our EvoCut work. This serves primarily as competitor intelligence—validating our AlgoEvo direction—rather than a source of novel methodology.

### [An Agent-Based Framework for the Automatic Validation of Mathematical Optimization Models](https://arxiv.org/abs/2511.16383)

**2025-11-20** | IBM Research | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-agent LLM framework for automatic validation of optimization models using problem-level API generation, unit test generation, and optimization-specific mutation testing | *LLM role:* code_writer

> Zadorojniy et al. introduce a multi-agent framework for validating LLM-generated optimization models by generating a test suite and verifying the suite's quality via mutation testing (ensuring tests detect deliberate errors injected into the model). On 100 NLP4LP instances, they achieve a 76% mutation kill ratio and successfully classify external models where simple objective value comparisons fail. The critical takeaway is the 'bootstrapped validation' workflow: using mutation analysis to validate the generated unit tests themselves before using them to score the model. We should steal this mutation-based verification loop to create a robust, ground-truth-free fitness signal for our evolutionary search and OR benchmarking pipelines.

### [LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading](https://arxiv.org/abs/2507.14995)

**2025-07-20** | China Agricultural University, University of Glasgow, Guangdong University of Foreign Studies | M=7 P=6 I=7 *discuss*

*Method:* LLM-Enhanced Multi-Agent Reinforcement Learning (MARL) with CTDE-based imitative expert MARL algorithm, using a differential multi-head attention-based critic network and Wasserstein metric for imitation | *LLM role:* heuristic_generator

> This paper proposes a neurosymbolic MARL framework for P2P energy trading where LLMs generate CVXPY optimization models to act as 'experts' for RL agents to imitate via Wasserstein distance. They introduce a 'Differential Attention' mechanism in the critic that subtracts attention maps to filter noise, enabling scalability to 100 agents where standard baselines fail. **Takeaway:** We should steal the Differential Attention architecture for our multi-agent critics to handle irrelevant interactions in large-scale optimization. The workflow of using LLMs to write the *solver* (generating reliable synthetic data) rather than the *solution* is a transferable strategy for bootstrapping RL in our OR domains.

### [HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization](https://arxiv.org/abs/2506.07972)

**2025-06-09** | Cornell University, Harvard University, NVIDIA | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Agentic framework for evaluating and iteratively refining LLM-generated heuristic algorithms via code execution feedback | *LLM role:* heuristic_generator

> The authors introduce HeuriGym, a benchmark suite of 9 hard combinatorial optimization problems (including PDPTW, EDA scheduling, and routing) coupled with an agentic evaluation loop. Results are backed by extensive experiments showing that SOTA LLMs saturate at ~60% of expert performance and, significantly, that existing evolutionary frameworks (ReEvo, EoH) perform *worse* than simple prompting on these large-context tasks (300+ lines of code). The key takeaway is the failure mode of current evolutionary methods: they cannot handle the context fragmentation and feedback integration required for complex heuristic design. We should immediately adopt this benchmark to demonstrate AlgoEvo's superiority, as the current baselines are weak and the problem set aligns perfectly with our focus.

### [Adaptively Robust LLM Inference Optimization under Prediction Uncertainty](https://arxiv.org/abs/2508.14544)

**2025-08-20** | Stanford University, Peking University, HKUST | M=7 P=9 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* Adaptive online scheduling with dynamic lower bound estimation, ordered eviction, and greedy batch formation (Amin algorithm) | *LLM role:* none

> Chen et al. propose $A_{min}$, an online scheduling algorithm for LLM inference that handles unknown output lengths by optimistically assuming the lower bound and evicting jobs (based on accumulated length) if memory overflows. They prove a logarithmic competitive ratio and show via simulations on LMSYS-Chat-1M that this approach nearly matches hindsight-optimal scheduling, vastly outperforming conservative upper-bound baselines. **Key Takeaway:** For our **GPUSched** project, we should abandon conservative memory reservation for output tokens; instead, implement an optimistic scheduler that oversubscribes memory and handles overflows via their ordered eviction policy, as the cost of restart is theoretically bounded and empirically negligible compared to the throughput gains.

### [CP-Agent: Agentic Constraint Programming](https://arxiv.org/abs/2508.07468)

**2025-08-10** | TU Wien | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Agentic Python coding agent using ReAct framework with persistent IPython kernel for iterative refinement of CPMpy constraint models | *LLM role:* code_writer

> Szeider implements a standard ReAct agent with a persistent IPython kernel to iteratively generate and refine CPMpy models, claiming 100% accuracy on CP-Bench. However, this perfect score is achieved on a *modified* version of the benchmark where the author manually fixed 31 ambiguous problem statements and 19 ground-truth errors—making the '100%' result an artifact of dataset cleaning rather than pure model capability. The most actionable takeaways are the negative result for explicit 'task management' tools (which hurt performance on hard problems) and the effectiveness of a minimal (<50 lines) domain prompt over complex scaffolding. We should review their clarified benchmark for our OR-Bench work.


### Front 4 (10 papers) — STABLE

**Density:** 0.42 | **Methods:** llm_as_evaluator, llm_in_the_loop, llm_code_generation, llm_as_heuristic, program_synthesis | **Problems:** linear_programming, optimization_modeling, mixed_integer_linear_programming, combinatorial_optimization, milp_general

*Unique methods:* benchmark_creation, code_interpreter_feedback, compiler_in_the_loop, cp_sat, distributionally_robust_optimization, empirical_study, few_shot_learning, generative_process_supervision, hierarchical_retrieval_augmented_generation, iterative_adaptive_revision, iterative_correction, lagrangian_duality, literate_programming, minizinc, multi_agent_llm_system, multi_armed_bandit, optimization_solver, reflexion, repeated_sampling, retrieval_augmented_in_context_learning, rsome, sac_opt, sample_average_approximation, self_improving_search, self_verification, semantic_alignment, sentence_embedding, solution_majority_voting, stochastic_optimization, systematic_literature_review, weighted_direct_preference_optimization
*Shared methods:* constraint_programming, cpmpy, data_cleaning, evolution_of_heuristics, group_relative_policy_optimization, in_context_learning, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, multi_agent_system, process_reward_model, program_synthesis, prompt_engineering, reinforcement_learning, retrieval_augmented_generation, robust_optimization, supervised_fine_tuning, supervised_learning

This research front focuses on developing advanced LLM-driven frameworks for automated optimization modeling, emphasizing iterative self-correction, semantic verification, and robust generation of executable Operations Research (OR) models from natural language descriptions. Key approaches include grammar-aware synthesis (SyntAGM), generative process supervision (StepORLM), corrective adaptation with expert hints (CALM), and multi-agent systems with sophisticated debugging and retrieval mechanisms (OptimAI, MIRROR, DAOpt, AlphaOPT). The overarching goal is to overcome limitations of direct LLM prompting, such as syntax errors, logical flaws, and overfitting to specific examples, to produce reliable and deployable OR solutions.

Several papers present significant contributions: SyntAGM achieved 61.6% accuracy on NL4Opt by using a compiler-in-the-loop for PyOPL model synthesis. StepORLM, with its Generative Process Reward Model, boosted GPT-4o's Pass@1 accuracy by 29.6% on OR benchmarks. CALM enabled a 4B model to match DeepSeek-R1's performance, improving GPT-3.5-Turbo's Macro AVG by 23.6% by injecting corrective hints. DAOpt demonstrated robust modeling under uncertainty, achieving >70% out-of-sample feasibility with the RSOME library. OptimAI introduced UCB-based debug scheduling, reaching 88.1% accuracy on NLP4LP, while AlphaOPT's self-improving experience library refined applicability conditions, outperforming fine-tuned models by ~13% on OOD benchmarks. The front also highlights critical issues, such as the high error rates (16-54%) in existing benchmarks (NL4Opt, IndustryOR) and LLM overfitting to example instances, leading to ~30% performance drops on hidden instances (DCP-Bench-Open).

This front is rapidly maturing, with a strong emphasis on building more autonomous and reliable OR model generation systems. The trajectory indicates a shift from basic code generation to sophisticated frameworks that learn from execution feedback, refine their internal knowledge, and proactively identify and correct errors. Future work will likely focus on scaling these systems to tackle more complex, real-world industrial problems, integrating advanced reasoning capabilities, and ensuring the generated models are not only syntactically correct but also semantically robust and performant in dynamic, uncertain environments.

**Papers:**

### [Grammar-Aware Literate Generative Mathematical Programming with Compiler-in-the-Loop](https://arxiv.org/abs/2601.17670)

**2026-01-25** | University of Edinburgh, University College Cork | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Iterative generate–compile–assess–revise loop with compiler-in-the-loop and LLM-based alignment judge | *LLM role:* generator, evaluator, revision policy

> SyntAGM is a framework for translating natural language into Algebraic Modeling Language (PyOPL) code using a 'compiler-in-the-loop' approach, where the LLM is constrained by an in-context BNF grammar and iteratively repairs code based on compiler diagnostics. They demonstrate that this approach matches the accuracy of expensive multi-agent systems (like Chain-of-Experts) while being significantly faster and cheaper. The immediate takeaways for us are the **StochasticOR benchmark** (which we should adopt for RobustMAS) and the technique of **injecting explicit BNF grammars** into prompts to enforce syntax in evolutionary search without fine-tuning. The 'literate modeling' approach—embedding reasoning as comments directly next to code constraints—is also a clever memory mechanism we could steal for AlgoEvo.

### [StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models](https://arxiv.org/abs/2509.22558)

**2025-09-26** | Shanghai Jiao Tong University | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Self-evolving framework with generative process supervision using Weighted Direct Preference Optimization (W-DPO) and Supervised Fine-Tuning (SFT) | *LLM role:* code_writer, evaluator, decomposition_guide, evolutionary_search

> Zhou et al. propose StepORLM, a framework where an 8B policy and a **Generative Process Reward Model (GenPRM)** co-evolve. Unlike standard discriminative PRMs that score steps in isolation, their GenPRM generates a reasoning trace to evaluate the full trajectory's logic before assigning credit, addressing the interdependency of OR constraints. They align the policy using **Weighted DPO**, where preference weights are derived from the GenPRM's process scores. They claim to beat GPT-4o and DeepSeek-V3 on 6 OR benchmarks (e.g., NL4Opt, MAMO) with an 8B model. **Key Takeaway:** We should test **Generative PRMs** immediately for AlgoEvo; asking the critic to 'explain then score' (generative) rather than just 'score' (discriminative) likely fixes the credit assignment noise in our long-horizon search.

### [CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling](https://arxiv.org/abs/2510.04204)

**2025-10-05** | Qwen Team, Alibaba Inc., The Chinese University of Hong Kong, Shenzhen, Southern University of Science and Technology, Shanghai University of Finance and Economics, Shenzhen Loop Area Institute (SLAI) | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Corrective Adaptation with Lightweight Modification (CALM) framework with two-stage training (SFT + RL) | *LLM role:* generates optimization models and solver code, performs reflective reasoning, and receives corrective hints from an expert LLM (Intervener)

> Tang et al. propose CALM, a framework that uses an expert 'Intervener' model to inject corrective hints into a small LRM's reasoning trace (e.g., forcing it to use Python instead of manual calculation), followed by SFT and RL (GRPO). Results are strong and verified: a 4B model matches DeepSeek-R1 (671B) on OR benchmarks, specifically fixing the 'Code Utilization Distrust' we see in our own agents. The key takeaway is the 'Intervener' loop: instead of discarding failed traces, they repair them with hints to create a 'golden' reasoning dataset that preserves the 'thinking' process while enforcing tool use. This is a direct, actionable method for improving our AlgoEvo agents' reliability in generating executable heuristics without massive human annotation.

### [DCP-Bench-Open: Evaluating LLMs for Constraint Modelling of Discrete Combinatorial Problems](https://arxiv.org/abs/2506.06052)

**2025-06-06** | KU Leuven, University of Western Macedonia | M=5 P=8 I=7 *changes-thinking* *discuss*

*Method:* LLM-driven constraint model generation | *LLM role:* code_writer, decomposition_guide, evaluator

> This paper introduces DCP-Bench-Open, a benchmark of 164 discrete combinatorial problems, to evaluate LLMs on translating natural language into constraint models (CPMpy, MiniZinc, OR-Tools). The results are rigorous and highlight a critical failure mode: LLMs overfit to the specific data values in the prompt's example instance, causing a ~30% performance drop when evaluated on hidden instances (Multi-Instance Accuracy). Crucially for our pipeline design, they find that Retrieval-Augmented In-Context Learning (RAICL) is ineffective or harmful compared to simply including library documentation in the system prompt. We should adopt their 'Multi-Instance Accuracy' metric immediately for OR-Bench and switch any MiniZinc generation efforts to Python-based frameworks like CPMpy or OR-Tools, which LLMs handle much better.

### [DAOpt: Modeling and Evaluation of Data-Driven Optimization under Uncertainty with LLMs](https://arxiv.org/abs/2511.11576)

**2025-09-24** | Zhejiang University, University of Toronto, Peking University | M=6 P=8 I=7 **MUST-READ** *discuss*

*Method:* LLM-based multi-agent framework for optimization modeling, integrating few-shot learning with OR domain knowledge (RSOME toolbox) and a Reflexion-based checker | *LLM role:* code_writer

> Zhu et al. propose DAOpt, a framework for modeling optimization under uncertainty that integrates LLMs with the RSOME library to handle robust and stochastic formulations. Their experiments on a new dataset (OptU) convincingly demonstrate that standard LLM-generated deterministic models suffer from the 'optimizer's curse,' achieving only ~27% out-of-sample feasibility, whereas their robust approach achieves >70%. The critical takeaway for us is to **stop asking LLMs to derive mathematical duals or robust counterparts**; instead, we should train them to use high-level DSLs (like RSOME) that handle the duality internally. This is an immediate action item for our RobustMAS project to ensure generated solutions are actually executable in stochastic environments.

### [OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents](https://arxiv.org/abs/2504.16918)

**2025-04-23** | University of Maryland at College Park | M=7 P=7 I=8 *discuss*

*Method:* LLM-powered multi-agent system (formulator, planner, coder, code critic, decider, verifier) with UCB-based debug scheduling for adaptive plan selection and iterative code refinement. | *LLM role:* decomposition_guide, code_writer, evaluator, evolutionary_search

> OptimAI introduces a multi-agent framework for translating natural language to optimization models, featuring a 'plan-before-code' stage and a novel **UCB-based debug scheduler**. Instead of linearly debugging a single solution, it treats debugging as a multi-armed bandit problem, dynamically allocating compute to different solution strategies based on a 'Decider' score and exploration term. While the combinatorial results (TSP a280) are trivial, the bandit mechanism is a highly effective heuristic for search control. We should steal this UCB scheduling logic for AlgoEvo to prevent agents from wasting tokens debugging fundamentally flawed heuristics.

### [AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library](https://arxiv.org/abs/2510.18428)

**2025-10-21** | Massachusetts Institute of Technology, London School of Economics and Political Science, University of Florida, Northeastern University, Singapore Management University, Singapore-MIT Alliance for Research and Technology | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Self-improving experience library framework with a continual two-phase cycle: Library Learning (insight extraction and consolidation) and Library Evolution (applicability condition refinement) | *LLM role:* Research agent for insight extraction, condition refinement, and program generation, operating within an evolutionary library learning framework

> AlphaOPT introduces a 'Library Evolution' mechanism that iteratively refines the *applicability conditions* of cached optimization insights based on solver feedback, allowing it to learn from answers alone (no gold programs). On OOD benchmarks like OptiBench, it beats fine-tuned models (ORLM) by ~13% and shows consistent scaling with data size. **Key Takeaway:** The specific mechanism of diagnosing 'unretrieved' vs. 'negative' tasks to rewrite retrieval triggers is a transferable technique for our AlgoEvo memory; it solves the problem of heuristic misapplication in long-term search. We should implement this 'condition refinement' loop immediately to improve our multi-agent memory systems.

### [A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions](https://arxiv.org/abs/2508.10047)

**2024-08-01** | Zhejiang University, Huawei Noah’s Ark Lab, Singapore University of Social Sciences, Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security | M=5 P=9 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* Systematic Literature Review and Empirical Re-evaluation | *LLM role:* evaluator

> This survey and empirical audit reveals that standard optimization modeling benchmarks (NL4Opt, IndustryOR) suffer from critical error rates ranging from 16% to 54%, rendering prior leaderboards unreliable. The authors manually cleaned these datasets and re-evaluated methods, finding that Chain-of-Thought (CoT) often degrades performance compared to standard prompting, while fine-tuned models (ORLM) and multi-agent systems (Chain-of-Experts) perform best. The immediate takeaway is that we must adopt their cleaned datasets for our OR-Bench project; using the original open-source versions is no longer defensible. Additionally, the failure of CoT on these tasks suggests we should prioritize multi-agent or fine-tuned approaches for symbolic formulation tasks.

### [SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling](https://arxiv.org/abs/2510.05115)

**2025-09-28** | Huawei Noah’s Ark Lab, Huawei’s Supply Chain Management Department, City University of Hong Kong | M=7 P=8 I=6 *discuss*

*Method:* Backward-guided iterative semantic alignment and correction using LLM agents | *LLM role:* code_writer, evaluator, decomposition_guide

> SAC-Opt introduces a verification loop where generated Gurobi code is back-translated into natural language ('semantic anchors') to check for alignment with the original problem description. Empirical results are strong, demonstrating a ~22% accuracy improvement on the ComplexLP dataset over OptiMUS-0.3 by catching logic errors that solver feedback misses. The primary takeaway is the utility of granular, constraint-level back-translation as a process reward signal, which we should adopt to improve the reliability of our automated modeling agents.

### [MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research](https://arxiv.org/abs/2602.03318)

**2026-02-03** | Xi'an Jiaotong University, Northwestern Polytechnical University | M=5 P=9 I=6 *discuss*

*Method:* Multi-Agent Framework with Iterative Adaptive Revision (IAR) and Hierarchical Retrieval-Augmented Generation (HRAG) | *LLM role:* code_writer, decomposition_guide, evaluator

> MIRROR is a multi-agent framework that translates natural language OR problems into Gurobi code using Hierarchical RAG (metadata filtering + semantic search) and an iterative repair loop. It achieves ~72% pass@1 across five benchmarks, outperforming Chain-of-Experts and fine-tuned models like LLMOPT without task-specific training. The key takeaway is their **structured revision tip mechanism**: upon execution failure, the agent generates a JSON object explicitly isolating the `error_statement`, `incorrect_code_snippet`, and `correct_code_snippet`, which serves as a precise memory artifact for subsequent retries. This structured reflection pattern is superior to raw error logs and could be immediately adopted in our own code generation pipelines.


### Front 0 (9 papers) — STABLE

**Density:** 0.89 | **Methods:** llm_in_the_loop, llm_code_generation, llm_as_evaluator, llm_fine_tuned, llm_as_heuristic | **Problems:** optimization_modeling, linear_programming, mixed_integer_programming, program_synthesis, milp_general

*Unique methods:* ai_agents, backward_generation, bipartite_graphs, callbacks, data_synthesis, deterministic_parser, dual_reward_system, dualreflect, forward_generation, graph_isomorphism, graph_theory, gurobi, hashing, hierarchical_decomposition, indicator_variables, instance_generation, kahneman_tversky_optimization, llm_as_data_synthesizer, llm_as_solver, mathematical_modeling, mixed_integer_linear_programming, multi_agent_coordination, partial_kl_divergence, piecewise_linear_constraints, pyscipopt, reflected_cot, reinforce_plus_plus, reinforcement_learning_alignment, rejection_sampling, reverse_socratic_method, self_correction, self_instruct, self_refinement, self_reflective_error_correction, sifting, smt_solvers, special_ordered_sets, structure_detection, symbolic_pruning, symmetric_decomposable_graphs, two_stage_reward_system, uct, weisfeiler_lehman_test
*Shared methods:* chain_of_thought, data_augmentation, debugging, direct_preference_optimization, dynamic_programming, group_relative_policy_optimization, in_context_learning, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, milp_solver, monte_carlo_tree_search, multi_agent_system, program_synthesis, prompt_engineering, reinforcement_learning, retrieval_augmented_generation, self_consistency, supervised_fine_tuning, supervised_learning, synthetic_data_generation

This research front focuses on advancing Large Language Models (LLMs) for automated optimization modeling, emphasizing sophisticated agentic systems, scalable data synthesis, and rigorous evaluation methods. Key frameworks include OptiMUS, which employs modular LLM agents and a connection graph for robust problem formulation; ReSocratic, OptMATH, and DualReflect, which introduce solver-verified bidirectional data synthesis pipelines; and Autoformulator, which integrates LLM-enhanced Monte-Carlo Tree Search with SMT-based symbolic pruning. The overarching theme is to enable LLMs to accurately translate natural language into executable optimization models, addressing challenges in mathematical formulation, code generation, and scalability.

Significant contributions include OptiMUS-0.3 achieving up to +40.5% accuracy on NLP4LP and +39.3% on NL4OPT benchmarks, outperforming standard GPT-4o prompting. ReSocratic and OptMATH demonstrate substantial improvements through fine-tuning on synthetic data, with Llama-3-8B improving from 13.6% to 51.1% on OPTIBENCH and OptMATH-Qwen2.5-32B surpassing GPT-4 on NL4Opt. Autoformulator sets new SOTA on NL4OPT and IndustryOR by leveraging symbolic pruning for efficiency. LLMOPT introduces multi-instruction SFT and KTO alignment for general optimization problems, yielding ~11% accuracy gains over GPT-4o. Furthermore, ORGEval provides a novel graph-theoretic evaluation framework, achieving 100% consistency in model isomorphism detection in seconds, a significant improvement over hours for solver-based checks on hard MIPLIB instances. SIRL applies Reinforcement Learning with Verifiable Reward and a novel 'Partial KL' surrogate objective to achieve SOTA on OptMATH and IndustryOR.

This front is rapidly emerging and maturing, characterized by a shift from basic LLM prompting to highly structured, agentic, and data-driven approaches. The trajectory indicates a strong focus on improving the reliability, scalability, and verifiability of LLM-generated optimization models. Future work will likely integrate these advanced data synthesis, symbolic reasoning, and structural evaluation techniques into unified frameworks, pushing towards more robust and interpretable automated optimization modeling systems capable of handling real-world complexity.

**Papers:**

### [OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://arxiv.org/abs/2407.19633)

**2024-07-29** |  | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Modular LLM-based agent (OptiMUS-0.3) employing a connection graph and self-reflective error correction for sequential optimization model formulation and code generation using Gurobi API | *LLM role:* optimization_model_synthesis

> OptiMUS-0.3 is a modular multi-agent system that translates natural language into Gurobi code, utilizing a 'connection graph' to manage variable-constraint relationships in long contexts and specialized agents to detect solver-specific structures (SOS, indicators) or implement sifting. The results are rigorous, introducing a new hard benchmark (NLP4LP) where they outperform GPT-4o by ~40% and beat Chain-of-Experts. The most stealable insight is the 'Structure Detection Agent': instead of relying on the LLM to write generic constraints, we should explicitly prompt for and map high-level structures to efficient solver APIs (like SOS constraints) to improve performance in our EvoCut and AlgoEvo pipelines. This is a necessary read for the OR-Bench team.

### [OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling](https://arxiv.org/abs/2407.09887)

**2024-07-13** | The Hong Kong University of Science and Technology, ETH Zurich, Huawei Noah’s Ark Lab, City University of Hong Kong, Sun Yat-sen University, MBZUAI, University of California Merced, Chongqing University | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* ReSocratic data synthesis for optimization problems, followed by Supervised Fine-Tuning of LLMs for Python code generation using PySCIPOpt solver | *LLM role:* evolutionary_search

> The authors propose OptiBench, a benchmark of 605 optimization problems (linear/nonlinear, tabular/text), and ReSocratic, a data synthesis method that generates formal models first and back-translates them into natural language questions. Results are strong: fine-tuning Llama-3-8B on their 29k synthetic samples improves accuracy from 13.6% to 51.1%, validating the data quality. **Key Takeaway:** The 'Reverse Socratic' synthesis pipeline (Formal Model → Code → NL Question) is the superior strategy for generating synthetic OR datasets because it guarantees solvability and ground truth by construction, unlike forward generation. We should steal this pipeline for generating robust test instances for OR-Bench and potentially for training our OR agents.

### [OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling](https://arxiv.org/abs/2502.11102)

**2025-02-16** | Peking University | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Scalable bidirectional data synthesis framework integrating feedback-driven PD generation, LLM-based backtranslation with self-criticism/refinement, and AutoFormulator with rejection sampling. | *LLM role:* data synthesizer

> The authors introduce OptMATH, a framework for generating synthetic optimization datasets by creating mathematical instances from seed generators, back-translating them to natural language via LLMs, and validating the pairs using a solver-based rejection sampling loop (checking if the re-generated model yields the same optimal value). They demonstrate that a Qwen-32B model fine-tuned on this data beats GPT-4 on NL4Opt and MAMO benchmarks. The critical takeaway is the **solver-verified reverse generation pipeline**: we should immediately steal this workflow to populate OR-Bench and generate diverse, verified training environments for AlgoEvo, replacing manual curation with scalable synthesis.

### [Autoformulation of Mathematical Optimization Models Using LLMs](https://arxiv.org/abs/2411.01679)

**2024-11-03** | University of Cambridge, University of Hawaii at Manoa | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-enhanced Monte-Carlo Tree Search with symbolic pruning and LLM-based evaluation | *LLM role:* conditional_hypothesis_generator, evaluator

> Astorga et al. frame optimization modeling as a hierarchical Monte-Carlo Tree Search (MCTS) problem, using LLMs to generate components and—crucially—employing SMT solvers to prune mathematically equivalent branches (e.g., recognizing `x+y` and `y+x` as identical). They achieve SOTA results on NL4OPT and IndustryOR, outperforming fine-tuned models like ORLM while using significantly fewer samples than naive approaches. **Key Takeaway:** The integration of symbolic equivalence checking (SMT) to prune the search tree is a technique we should immediately steal; implementing this in AlgoEvo would allow us to discard functionally identical code/math mutants before expensive evaluation, directly addressing our sample efficiency bottleneck.

### [LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch](https://arxiv.org/abs/2410.13213)

**2024-10-17** | Ant Group, East China Normal University, Nanjing University | M=5 P=7 I=6 *discuss*

*Method:* Multi-instruction supervised fine-tuning and KTO model alignment with self-correction | *LLM role:* Generates problem formulations, writes solver code, and performs error analysis for self-correction

> The authors fine-tune Qwen1.5-14B to translate natural language optimization problems into Pyomo code via a structured 'five-element' intermediate representation (Sets, Parameters, Variables, Objective, Constraints) and KTO alignment. They achieve ~11% accuracy gains over GPT-4o and ORLM on benchmarks like NL4Opt and IndustryOR, primarily by reducing formulation hallucinations through the structured intermediate step and preference optimization. For our OR-Bench work, the key takeaway is the concrete recipe for using KTO to align symbolic modeling agents, which appears more effective than standard SFT for enforcing constraints in smaller models. While not an evolutionary search paper, it provides a strong, locally runnable baseline for our OR modeling evaluations.

### [ORGEval: Graph-Theoretic Evaluation of LLMs in Optimization Modeling](https://arxiv.org/abs/2510.27610)

**2025-10-31** | The Chinese University of Hong Kong, Shenzhen, Shenzhen Research Institute of Big Data, Shenzhen International Center for Industrial and Applied Mathematics, Shenzhen Loop Area Institute | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Graph-theoretic evaluation framework using Weisfeiler-Lehman (WL) test with Symmetric Decomposable (SD) graph condition for model isomorphism detection | *LLM role:* none

> Wang et al. propose ORGEval, a framework that evaluates LLM-generated optimization models by converting them into bipartite graphs and using the Weisfeiler-Lehman (WL) test to detect isomorphism with a ground truth, rather than solving the instances. They prove that for 'symmetric decomposable' graphs, this method is guaranteed to detect equivalence correctly, achieving 100% consistency and running in seconds compared to hours for solver-based checks on hard MIPLIB instances. The critical takeaway is the shift from execution-based to **structural evaluation**: we can validate model logic via graph topology ($O(k(m+n)^2)$) without incurring the cost of solving NP-hard problems. This is immediately actionable for our OR benchmarking pipelines and could serve as a rapid 'pre-solve' filter in our evolutionary search loops to reject structurally invalid candidates instantly.

### [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://arxiv.org/abs/2402.10172)

**2024-02-15** | Stanford University | M=7 P=8 I=8 **MUST-READ** *discuss*

*Method:* Modular LLM-based multi-agent system (OptiMUS) with connection graph | *LLM role:* multi-agent orchestration for model formulation, code generation, evaluation, and debugging

> OptiMUS is a multi-agent framework for translating natural language into Gurobi code, achieving SOTA performance by using a 'Connection Graph' to map variables and parameters to specific constraints. This graph allows the agents to dynamically filter context and construct minimal prompts, enabling success on problems with long descriptions where baselines like Chain-of-Experts fail. They release NLP4LP, a hard benchmark of 67 complex instances, which we must immediately compare against our OR-Bench efforts. The **Connection Graph** is the key stealable insight: a structured dependency tracking mechanism that solves context pollution in iterative code generation, directly applicable to our AlgoEvo and HERMES memory designs.

### [Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling](https://arxiv.org/abs/2505.11792)

**2025-05-17** | Stanford University, Shanghai Jiao Tong University, The University of Hong Kong, Shanghai University of Finance and Economics, Cardinal Operations | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning with Verifiable Reward (RLVR) using REINFORCE++ with a Partial KL surrogate function | *LLM role:* code_writer

> Chen et al. introduce SIRL, a framework for training LLMs to generate optimization models using Reinforcement Learning with Verifiable Rewards (RLVR) and a novel 'Partial KL' surrogate objective. By removing the KL penalty from the reasoning (CoT) section while retaining it for the code generation section, they balance exploration with syntactic stability, achieving SOTA on OptMATH and IndustryOR against OpenAI-o3 and DeepSeek-R1. The critical takeaway for us is the Partial KL strategy: it allows the model to 'think' freely outside the reference distribution while adhering to strict coding standards—a technique we should immediately test in AlgoEvo. Furthermore, their method of parsing .lp files to extract structural features (variable counts, constraint types) for 'instance-enhanced self-consistency' provides a much richer signal than our current binary success/failure metrics.

### [Auto-Formulating Dynamic Programming Problems with Large Language Models](https://arxiv.org/abs/2507.11737)

**2025-07-15** | University of Chicago, Cornell University, Shanghai Jiao Tong University, Shanghai University of Finance and Economics, Cardinal Operations | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* DPLM, a 7B-parameter specialized model fine-tuned on Qwen-2.5-7B-Instruct using synthetic data generated by DualReflect, combining Supervised Fine-Tuning (SFT) with Reinforcement Learning (GRPO/DPO) alignment. | *LLM role:* model_formulator, code_writer, synthetic_data_generator, refinement_agent

> Zhou et al. introduce DPLM, a 7B model fine-tuned to formulate Dynamic Programming models, achieving performance comparable to o1 on their new DP-Bench. Their key contribution is 'DualReflect,' a synthetic data pipeline that combines Forward Generation (Problem→Code) for diversity with Backward Generation (Code→Problem) for correctness. **Takeaway:** We should steal the Backward Generation approach for AlgoEvo: instead of relying on noisy forward generation, we can take valid heuristics/OR code (which we have in abundance) and reverse-engineer problem descriptions to create massive, verifiable synthetic datasets for fine-tuning our code generation models. The paper proves this method is superior for 'cold-starting' small models in data-scarce domains.


### Front 10 (7 papers) — GROWING

**Density:** 0.62 | **Methods:** llm_code_generation, llm_in_the_loop, llm_as_evaluator, llm_as_heuristic, llm_fine_tuned | **Problems:** linear_programming, optimization_problem_formulation, mixed_integer_linear_programming, constrained_optimization, natural_language_to_optimization_modeling

*Unique methods:* beam_search, bilevel_optimization, canonical_intermediate_representation, contrastive_learning, diversity_aware_rank_based_sampling, epsilon_greedy_search, genetic_algorithm, greedy_decoding, greedy_search, instruction_tuning, llm_as_optimizer, low_rank_adaptation, marge_loss, pairwise_preference_model, ppo, preference_learning, propen, reinforce, rlhf, self_reflection, temperature_sampling, test_time_scaling, tree_of_thought
*Shared methods:* debugging, direct_preference_optimization, evolution_of_heuristics, funsearch, iterative_refinement, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, majority_voting, monte_carlo_tree_search, multi_agent_system, process_reward_model, program_synthesis, retrieval_augmented_generation, supervised_fine_tuning, supervised_learning, synthetic_data_generation

This research front focuses on developing advanced, structured approaches for leveraging Large Language Models (LLMs) in automated optimization problem formulation and code generation. The unifying theme involves moving beyond simple prompt engineering to employ sophisticated techniques such as Canonical Intermediate Representations (CIR), multi-agent systems like OptiTrust, and search-guided methods like Monte Carlo Tree Search (MCTS) and Tree-of-Thought (ToT). Additionally, specialized fine-tuning strategies, including Direct Preference Optimization (DPO) with Diversity-Aware Rank-based (DAR) sampling and Margin-Aligned Expectation (MargE) loss, are crucial for enhancing LLM performance in this domain.

Key contributions include OptiTrust's verifiable synthetic data generation, achieving 91.6% on NL4Opt, significantly outperforming GPT-4. Liu et al. demonstrated that DPO with DAR sampling can fine-tune a Llama-3.2-1B model to match a Llama-3.1-8B on ASP and CVRP. BPP-Search enhanced Tree-of-Thought reasoning, boosting the correct rate by +47.4% on StructuredOR. Lyu et al.'s CIR framework, combined with RAG, achieved 47.2% accuracy on ORCOpt-Bench, a substantial improvement over baselines. SolverLLM introduced LLM-guided MCTS with Prompt Backpropagation, yielding ~10% gains on complex datasets. LLaMoCo showed that a 350M parameter model, instruction-tuned on synthetic data, could outperform GPT-4 Turbo by +368.3% on Ieval by selecting specialized solvers. Finally, LLOME's MargE loss provided a robust fine-tuning method for constrained biophysical sequence optimization, maintaining diversity and improving sample efficiency.

This front is rapidly emerging and growing, driven by the need for more robust, accurate, and efficient automated optimization tools. The trajectory indicates a shift towards integrating multiple advanced techniques, such as combining structured intermediate representations with sophisticated search algorithms and preference-based fine-tuning. Future work will likely focus on tackling increasingly complex, real-world optimization problems characterized by ambiguous or underspecified natural language descriptions, and incorporating real-time feedback loops for continuous model improvement and adaptation.

**Papers:**

### [Toward a Trustworthy Optimization Modeling Agent via Verifiable Synthetic Data Generation](https://arxiv.org/abs/2508.03117)

**2025-08-05** | IBM Research AI | M=6 P=7 I=7 *discuss*

*Method:* Verifiable Synthetic Data Generation (SDG) pipeline combined with a modular LLM agent (OptiTrust) employing multi-stage translation, multi-language inference, and majority-vote cross-validation | *LLM role:* data_generator, code_writer, decomposition_guide, formulation_generator, evaluator

> Lima et al. introduce a pipeline to generate synthetic optimization datasets by starting with symbolic MILP instances (ground truth) and using LLMs to generate natural language descriptions, ensuring full verifiability. They fine-tune a small model (Granite 8B) that beats GPT-4 on 6/7 benchmarks, largely due to a 'majority vote' mechanism where the agent generates code in 5 different modeling languages (Pyomo, Gurobi, etc.) and checks for result consistency. **Takeaway:** We should steal the multi-language execution voting to boost robustness in our code generation agents. Furthermore, their reverse-generation (Symbolic $\to$ NL) strategy is the correct approach for generating infinite, error-free test cases for our OR-Bench work.

### [Fine-tuning Large Language Model for Automated Algorithm Design](https://arxiv.org/abs/2507.10614)

**2025-07-13** | City University of Hong Kong | M=7 P=10 I=8 **MUST-READ** *discuss*

*Method:* Direct Preference Optimization (DPO) with Diversity-Aware Rank-based (DAR) Sampling for LLM fine-tuning | *LLM role:* code_writer

> Liu et al. introduce a fine-tuning pipeline for LLMs in automated algorithm design, utilizing a 'Diversity-Aware Rank-based' sampling strategy to construct DPO preference pairs from evolutionary search histories. By partitioning the population into ranked subsets and sampling pairs with a guaranteed quality gap (skipping adjacent tiers), they ensure training signals are both clear and diverse. Empirically, they show that a fine-tuned Llama-3.2-1B matches the performance of a base Llama-3.1-8B on ASP and CVRP tasks, effectively compressing the search capability into a much cheaper model. We should implement this sampling strategy to recycle our AlgoEvo run logs into specialized 'mutator' models, potentially allowing us to downscale to 1B/3B models for the inner search loop without losing quality.

### [BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](https://arxiv.org/abs/2411.17404)

**2024-11-26** | Huawei, The University of Hong Kong | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* BPP-Search: Tree-of-Thought with Beam Search, Process Reward Model, and Pairwise Preference Algorithm | *LLM role:* policy_model_for_generation, evaluator_for_search_guidance, data_generator

> Wang et al. propose BPP-Search, combining Beam Search, a Process Reward Model (PRM), and a final Pairwise Preference Model to generate LP/MIP models from natural language. While their new 'StructuredOR' dataset is small (38 test instances), it uniquely provides intermediate modeling labels (sets, parameters, variables) essential for training PRMs in this domain. The key takeaway is their finding that PRMs are effective for pruning but imprecise for final ranking; they solve this by adding a pairwise preference model at the leaf layer—a technique we should immediately steal to improve selection robustness in our MASPRM and evolutionary search pipelines. This is a competent execution of 'LLM + Search' applied specifically to our OR niche.

### [Canonical Intermediate Representation for LLM-based optimization problem formulation and code generation](https://arxiv.org/abs/2602.02029)

**2026-02-02** | The Hong Kong Polytechnic University, InfiX.ai | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-agent pipeline with Canonical Intermediate Representation (CIR) and Retrieval-Augmented Generation (RAG) | *LLM role:* Decomposition guide, paradigm selector, code writer, and verifier

> Lyu et al. propose a 'Canonical Intermediate Representation' (CIR) to decouple natural language operational rules from their mathematical instantiation, explicitly forcing the LLM to select modeling paradigms (e.g., time-indexed vs. continuous flow) before coding. They achieve state-of-the-art accuracy (47.2% vs 22.4% baseline) on a new, complex benchmark (ORCOpt-Bench) by using a multi-agent pipeline that retrieves and adapts constraint templates. The key takeaway is the 'Mapper' agent's paradigm selection logic, which prevents common formulation errors in VRPs and scheduling; we should evaluate CIR as a structured mutation space for AlgoEvo to replace brittle code evolution. The new benchmark is immediately relevant for our OR-Bench evaluation suite.

### [SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search](https://arxiv.org/abs/2510.16916)

**2025-10-19** | NEC Labs America, Baylor University, University of Texas at Dallas, Augusta University, Southern Illinois University | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* LLM-guided Monte Carlo Tree Search (MCTS) with dynamic expansion, prompt backpropagation, and uncertainty backpropagation for optimization problem formulation and code generation | *LLM role:* decomposition_guide, heuristic_generator, evaluator, code_writer, evolutionary_search

> SolverLLM frames optimization problem formulation as a hierarchical Monte Carlo Tree Search (MCTS), decomposing the task into six layers (variables, constraints, etc.) and using test-time compute to beat fine-tuned baselines like LLMOPT. The results appear robust, showing ~10% gains on complex datasets, though inference cost is high. **The critical takeaway for us is the 'Prompt Backpropagation' mechanism:** instead of just updating numerical values, they propagate textual error analysis from leaf nodes back up the tree to dynamically modify the prompts of parent nodes, effectively creating 'short-term memory' for the search. We should immediately test this technique in AlgoEvo to prevent the recurrence of failed code patterns during mutation steps. Additionally, their use of semantic entropy to down-weight uncertain rewards in MCTS is a practical solution to the noisy evaluation problem we face in process reward models.

### [LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation](https://arxiv.org/abs/2403.01131)

**2024-03-02** | Singapore Management University, Nanyang Technological University, South China University of Technology | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Instruction tuning of LLMs with a two-phase learning strategy incorporating contrastive learning-based warm-up and sequence-to-sequence loss | *LLM role:* code_writer

> LLaMoCo fine-tunes small LLMs (down to 350M) to generate executable Python optimization code by training on a synthetic dataset where the 'ground truth' is the empirically best-performing solver identified via exhaustive benchmarking. The results are compelling: the fine-tuned 350M model achieves ~85% normalized performance on benchmarks where GPT-4 Turbo only reaches ~14-30%, largely because the small model learns to select specialized evolutionary strategies (like BIPOP-CMA-ES) while GPT-4 defaults to generic gradient-based solvers. **Key Takeaway:** We can replace the expensive GPT-4 calls in our evolutionary search loop with a specialized, fine-tuned local model (CodeLlama-7B) trained on our historical search successes, significantly improving both sample efficiency and scalability. The paper's 'contrastive warm-up' strategy for aligning diverse problem descriptions is also a transferable technique for our problem encoding work.

### [Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks](https://arxiv.org/abs/2410.22296)

**2024-10-29** | Genentech, New York University | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLOME (Language Model Optimization with Margin Expectation) bilevel optimization with Margin-Aligned Expectation (MargE) loss | *LLM role:* optimization_driver

> The authors propose LLOME, a bilevel optimization framework that fine-tunes an LLM using 'MargE' (Margin-Aligned Expectation), a loss function that weights gradient updates by the magnitude of reward improvement (margin) rather than simple preference rankings. Results are rigorous and demonstrate that while DPO leads to generator collapse and infeasibility in constrained spaces, MargE maintains diversity and significantly improves sample efficiency, matching specialized solvers like LaMBO-2 on medium-difficulty tasks. The critical takeaway is that standard alignment methods (DPO/RLHF) are ill-suited for optimization because they discard information about *how much* better a solution is; MargE fixes this by satisfying the Strong Interpolation Criteria. We should immediately evaluate replacing the RL/update component in AlgoEvo with the MargE objective to improve the stability and quality of our evolved heuristics.


### Front 14 (2 papers) — STABLE

**Density:** 1.00 | **Methods:** llm_code_generation, llm_as_evaluator, supervised_learning, llm_fine_tuned, synthetic_data_generation | **Problems:** automated_problem_formulation, expensive_black_box_optimization, antenna_design, constrained_optimization, mixed_integer_linear_programming

*Unique methods:* error_analysis, multi_turn_feedback
*Shared methods:* data_augmentation, data_cleaning, llm_as_evaluator, llm_code_generation, llm_fine_tuned, llm_in_the_loop, majority_voting, self_consistency, supervised_fine_tuning, supervised_learning, synthetic_data_generation

This research front focuses on advancing automated problem formulation (APF) using fine-tuned Large Language Models (LLMs). The core theme revolves around two distinct yet complementary frameworks: APF, which provides a solver-independent method for translating natural language requirements into executable Python optimization functions for high-cost simulation-driven design (e.g., antenna design), and OptiMind, which specializes in formulating Mixed-Integer Linear Programming (MILP) problems from natural language by integrating optimization expertise.

Key contributions include APF's supervised fine-tuning of LLMs on synthetically generated datasets, achieving significantly higher overall alignment scores (+4.58% to +13.25% over baselines) on antenna design tasks, and introducing a 'solver-independent' evaluation metric. OptiMind, on the other hand, employs supervised fine-tuning of a 20B-parameter LLM on a meticulously cleaned dataset, combined with error-aware prompting and multi-turn self-correction. This approach boosted formulation accuracy by +2.7% to +23% across benchmarks like IndustryOR and OptMATH, critically revealing and correcting flaws in existing benchmarks through class-based error analysis.

This front is emerging, with both papers published in late 2025, indicating a rapid development in applying LLMs to automate complex OR problem formulation. The trajectory suggests a move towards more robust, generalizable, and expert-informed LLM-based systems. Future work will likely focus on expanding the scope to new problem domains and addressing current limitations, potentially by integrating the strengths of both frameworks to create more comprehensive and accurate automated modeling tools.

**Papers:**

### [Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](https://arxiv.org/abs/2512.18682)

**2025-12-21** | Xidian University, Victoria University of Wellington, Westlake University | M=7 P=5 I=7 *discuss*

*Method:* Supervised fine-tuning of LLMs on a synthetically generated dataset, created via data augmentation (semantic paraphrasing, order permutation) and LLM-based test instance annotation and selection, to convert natural language requirements into executable Python optimization functions. | *LLM role:* code_writer

> Li et al. propose APF, a framework to fine-tune LLMs for translating engineering requirements into optimization code without running expensive simulations during training. They generate synthetic training data and filter it by checking if the generated code ranks historical data instances similarly to how an LLM 'judge' ranks them based on the text requirements. Results show 7B models outperforming GPT-4o on antenna design tasks, validated by actual simulation. **Key Takeaway:** We can replace expensive ground-truth evaluations in our process reward models by checking consistency between generated code outputs and LLM-predicted rankings on cached historical data—a direct method to improve sample efficiency in AlgoEvo.

### [OptiMind: Teaching LLMs to Think Like Optimization Experts](https://arxiv.org/abs/2509.22979)

**2025-09-26** | Microsoft Research, Stanford University, University of Washington | M=5 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Supervised fine-tuning (SFT) of a 20B-parameter LLM (GPT-OSS-20B variant) on a semi-automatically cleaned, class-specific error-analyzed training dataset, combined with error-aware prompting and multi-turn self-correction at inference. | *LLM role:* code_writer

> The authors fine-tune a 20B model for MILP formulation, but the critical contribution is a rigorous audit of standard benchmarks (IndustryOR, OptMATH), revealing that 30-50% of instances are flawed (missing data, wrong ground truth, infeasible). They introduce a 'class-based error analysis' where the model classifies a problem (e.g., TSP) and retrieves specific, expert-written hints to avoid common pitfalls, boosting accuracy by ~20%. **Takeaway:** We must immediately replace our benchmark versions with their cleaned sets for the OR-Bench project. Additionally, their library of 'error hints' per problem class is a high-value artifact we can scrape and inject into AlgoEvo's prompt templates to improve initial population quality.



## Bridge Papers

Papers connecting multiple research fronts:

### [LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation](https://arxiv.org/abs/2403.01131)

**TRUE SYNTHESIS** | score=0.83 | Front 10 → Front 0, Front 4, Front 2

> LLaMoCo fine-tunes small LLMs (down to 350M) to generate executable Python optimization code by training on a synthetic dataset where the 'ground truth' is the empirically best-performing solver ident

### [EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](https://arxiv.org/abs/2502.14760)

**TRUE SYNTHESIS** | score=0.79 | Front 2 → Front 0, Front 4, Front 10

> Zhai et al. propose EquivaMap, a framework that evaluates whether two MILP formulations are equivalent by using an LLM to discover a linear mapping between their decision variables, which is then rigo

### [Toward a Trustworthy Optimization Modeling Agent via Verifiable Synthetic Data Generation](https://arxiv.org/abs/2508.03117)

**TRUE SYNTHESIS** | score=0.77 | Front 10 → Front 0, Front 2, Front 4

> Lima et al. introduce a pipeline to generate synthetic optimization datasets by starting with symbolic MILP instances (ground truth) and using LLMs to generate natural language descriptions, ensuring 

### [DCP-Bench-Open: Evaluating LLMs for Constraint Modelling of Discrete Combinatorial Problems](https://arxiv.org/abs/2506.06052)

**TRUE SYNTHESIS** | score=0.75 | Front 4 → Front 2, Front 0, Front 10

> This paper introduces DCP-Bench-Open, a benchmark of 164 discrete combinatorial problems, to evaluate LLMs on translating natural language into constraint models (CPMpy, MiniZinc, OR-Tools). The resul

### [Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks](https://arxiv.org/abs/2410.22296)

**TRUE SYNTHESIS** | score=0.71 | Front 10 → Front 0, Front 2, Front 4

> The authors propose LLOME, a bilevel optimization framework that fine-tunes an LLM using 'MargE' (Margin-Aligned Expectation), a loss function that weights gradient updates by the magnitude of reward 


---

*Generated by Research Intelligence System*
