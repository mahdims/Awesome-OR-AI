# Living Review: Generative AI for OR

**Last Updated:** 2026-04-07

---

## Recent Papers

#### 2026-04-02 (1 papers)

### [Execution-Verified Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2604.00442)

**2026-04-01** | Chinese Academy of Sciences, Nanjing University, Nanjing University of Science and Technology | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Execution-Verified Reinforcement Learning (EVOM) with GRPO and DAPO for solver-conditioned code generation | *LLM role:* code_writer

> EVOM trains LLMs for operations research modeling using execution-verified reinforcement learning (GRPO/DAPO) based solely on solver outcomes, bypassing expensive process-level supervision. The results are backed by solid empirical evaluations on OptiBench, NL4OPT, and IndustryOR, demonstrating that it matches or beats process-supervised SFT (ORLM) and enables zero-shot transfer to new solvers (e.g., Gurobi to OR-Tools). The key takeaway is that outcome-only RL prevents the model from overfitting to solver-specific syntax (a major flaw in SFT), forcing it to learn invariant mathematical structures; additionally, their two-stage cold-start trick (LLM-translate 100 samples -> SFT -> RL) is a highly stealable technique for adapting to new environments. This is highly relevant for our OR-Bench project, and we should consider implementing execution-verified RL baselines and leveraging their cold-start adaptation trick when targeting new solvers in our evolutionary search pipelines.


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

*3 fronts detected — snapshot 2026-04-07*

### Front 2 (35 papers) — GROWING

**Density:** 0.03 | **Methods:** llm_in_the_loop, llm_code_generation, llm_as_evaluator, llm_as_heuristic, program_synthesis | **Problems:** linear_programming, combinatorial_optimization, mixed_integer_linear_programming, resource_allocation, milp_general

*Unique methods:* ADMM, absolute_value_linearization, actor_critic, adam_optimizer, agentic_model, agentic_workflow, apollo_milp, asymmetric_validation, attention_mechanism, augmented_lagrangian_method, automated_algorithm_design, automated_experimentation, autonomous_coding_agents, backward_generation, bayesian_optimization, beam_search, behavioral_verification, bi_gcn, bilevel_optimization, bilinear_linearization, bin_packing_heuristics, black_box_optimization, canonical_intermediate_representation, causal_discovery, centralized_training_decentralized_execution, clustergcn, compiler_in_the_loop, conditional_flow_matching, continual_learning, cross_encoder_reranking, curriculum_learning, dapo, dataset_generation, deep_q_networks, deep_reinforcement_learning, diagnosis_guided_repair, differentiable_optimization, differentiable_physics, differential_attention, digital_replicas, direct_preference_optimization, distributionally_robust_optimization, diversity_aware_rank_based_sampling, dualreflect, dynamic_path_reconstruction, dynamic_weight_adjustment, ensemble_methods, epsilon_greedy_search, evolutionary_algorithms, evolutionary_search, exact_linearization, execution_aware_modeling, expert_in_the_loop, expert_prompting, few_shot_learning, fixed_point_relaxation, flow_matching, formal_verification, forward_generation, function_calling, gat, gaussian_reparameterization, gdp_transformation, generative_process_supervision, genetic_algorithm, global_local_search, gradient_based_optimization, gradient_descent, graph_neural_networks, greedy_decoding, greedy_search, grpo, heuristic_design, hierarchical_chunking, hierarchical_reinforcement_learning, hierarchical_retrieval_augmented_generation, human_llm_interaction, hyper_heuristics, hyperparameter_optimization, iis_diagnostics, imitation_learning, intent_classification, irreducible_infeasible_subsystem_analysis, iterative_adaptive_revision, iterative_correction, l1_regularization, lagrangian_duality, lagrangian_relaxation, lexicographical_optimization, linear_fractional_linearization, literate_programming, llm_agents, llm_as_expert, llm_as_meta_optimizer, llm_as_optimizer, llm_as_semantic_generator, llm_objective_formulation, llm_research_agent, logsumexp_approximation, lookahead_mechanism, loop_based_structure_recovery, low_rank_adaptation, marge_loss, mathematical_optimization, mdp_modeling, memory_compression, memory_driven_planning, meta_optimization, metacognition, metadata_augmented_indexing, milp_reformulation, min_max_linearization, minimum_bayes_risk_decoding, mixed_integer_programming, mixture_of_experts, model_data_separation, modeling_language, monotone_transformation_linearization, multi_agent_reinforcement_learning, multi_head_attention, multimodal_ai, multimodal_llm, mutation_testing, natural_language_generation, neural_diving, optimization_model_validation, optimization_solver, options_framework, pairwise_preference_model, particle_swarm, pattern_detection, pmvb, ppo, predict_and_search, preference_learning, prioritized_experience_replay, probabilistic_guarantee, problem_formulation, process_reward_model, progressive_specialization_training, propen, random_forest, reflected_cot, reflection, reflexion, reinforce, reinforcement_learning_alignment, reinforcement_learning_with_verifiable_rewards, reverse_engineering, reward_design, rl_ppo, rl_with_verifiable_rewards, rlhf, rsome, rule_based_reformulation, sac_opt, sample_average_approximation, sampling_and_reweighting, sandboxed_execution, self_instruct, self_reflection, semantic_alignment, semantic_validation, semi_markov_decision_process, sensitivity_analysis, sentence_embedding, sibling_aware_expansion, simulation_environment, simulation_optimization, software_testing, solver_based_perturbation, solver_in_the_loop, sparse_state_storage, star, stochastic_optimization, structure_aware_modeling, structured_output_parsing, symbolic_planning, temperature_sampling, test_case_generation, test_time_scaling, textgrad, topological_sort, tree_of_thought, tree_search, tri_gcn, two_stage_retrieval, uncertainty_estimation, utpc_framework, variable_fixing_heuristic, vector_embedding, virtual_reinforcement_learning, warm_starting, wasserstein_metric, weighted_direct_preference_optimization, weighted_sum_method
*Shared methods:* benchmark_design, chain_of_thought, contrastive_learning, data_augmentation, debugging, dynamic_programming, evolution_of_heuristics, expectation_maximization, funsearch, generative_models, group_relative_policy_optimization, in_context_learning, iterative_refinement, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, majority_voting, metaheuristics, milp_solver, mixed_integer_linear_programming, monte_carlo_tree_search, multi_agent_llm_system, multi_agent_system, neuro_symbolic_ai, neurosymbolic_ai, program_synthesis, prompt_engineering, reinforcement_learning, retrieval_augmented_generation, robust_optimization, self_consistency, self_improving_search, supervised_fine_tuning, supervised_learning, synthetic_data_generation, tool_use, tree_of_thoughts

This front focuses on developing agentic LLM frameworks that achieve verifiable and robust optimization model synthesis and repair from natural language descriptions. Key to this is the integration of iterative self-correction mechanisms and solver-in-the-loop feedback, moving beyond simple code generation to ensure semantic correctness and executability. Frameworks like AlphaOPT, MIRROR, ReLoop, and NEMO exemplify this by employing structured decomposition, multi-agent collaboration, and explicit verification steps.

Researchers are making significant strides by leveraging solver diagnostics (e.g., Gurobi's IIS, compiler errors) as dense reward signals for Reinforcement Learning (RL), as seen in OptiRepair and EVOM, which use GRPO/DAPO to train models that outperform frontier APIs. Behavioral verification (ReLoop) and mutation testing (Zadorojniy et al.) are introduced to detect "silent failures" where models run but solve the wrong problem. Novel architectural patterns include Canonical Intermediate Representations (CIR) (Lyu et al.) for structured generation, Prompt Backpropagation in MCTS (SolverLLM) for dynamic prompt modification, and Generative Process Reward Models (GenPRM) (StepORLM) for comprehensive trajectory evaluation. Benchmarks like MIPLIB-NL (Li et al.) expose the limitations of current LLMs on industrial-scale problems, showing a drop from ~90% to ~18% accuracy, highlighting the need for robust, scalable solutions.

This front is rapidly emerging and maturing, driven by the critical need for reliable and trustworthy AI in Operations Research. The trajectory indicates a shift from basic LLM code generation to sophisticated, closed-loop agentic systems that can autonomously diagnose, repair, and validate their outputs. Future work will likely focus on integrating more advanced domain knowledge, developing theoretical guarantees for correctness, and scaling these verifiable approaches to even larger, more complex real-world problems, potentially through Virtual Reinforcement Learning (VRL) environments (VisionCreator) to reduce computational costs of training. The next papers will likely present hybrid neuro-symbolic architectures that combine LLM flexibility with formal verification methods and specialized OR solvers.

**Papers:**

### [AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library](https://arxiv.org/abs/2510.18428)

**2025-10-21** | Massachusetts Institute of Technology, London School of Economics and Political Science, University of Florida, Northeastern University, Singapore Management University, Singapore-MIT Alliance for Research and Technology | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Self-improving experience library framework with a continual two-phase cycle: Library Learning (insight extraction and consolidation) and Library Evolution (applicability condition refinement) | *LLM role:* Research agent for insight extraction, condition refinement, and program generation, operating within an evolutionary library learning framework

> AlphaOPT introduces a 'Library Evolution' mechanism that iteratively refines the *applicability conditions* of cached optimization insights based on solver feedback, allowing it to learn from answers alone (no gold programs). On OOD benchmarks like OptiBench, it beats fine-tuned models (ORLM) by ~13% and shows consistent scaling with data size. **Key Takeaway:** The specific mechanism of diagnosing 'unretrieved' vs. 'negative' tasks to rewrite retrieval triggers is a transferable technique for our AlgoEvo memory; it solves the problem of heuristic misapplication in long-term search. We should implement this 'condition refinement' loop immediately to improve our multi-agent memory systems.

### [BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](https://arxiv.org/abs/2411.17404)

**2024-11-26** | Huawei, The University of Hong Kong | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* BPP-Search: Tree-of-Thought with Beam Search, Process Reward Model, and Pairwise Preference Algorithm | *LLM role:* policy_model_for_generation, evaluator_for_search_guidance, data_generator

> Wang et al. propose BPP-Search, combining Beam Search, a Process Reward Model (PRM), and a final Pairwise Preference Model to generate LP/MIP models from natural language. While their new 'StructuredOR' dataset is small (38 test instances), it uniquely provides intermediate modeling labels (sets, parameters, variables) essential for training PRMs in this domain. The key takeaway is their finding that PRMs are effective for pruning but imprecise for final ranking; they solve this by adding a pairwise preference model at the leaf layer—a technique we should immediately steal to improve selection robustness in our MASPRM and evolutionary search pipelines. This is a competent execution of 'LLM + Search' applied specifically to our OR niche.

### [LinearizeLLM: An Agent-Based Framework for LLM-Driven Exact Linear Reformulation of Nonlinear Optimization Problems](https://arxiv.org/abs/2510.15969)

**2025-10-12** | Karlsruhe Institute of Technology, Reutlingen University | M=7 P=8 I=7 *discuss*

*Method:* Agent-based LLM framework with specialized reformulation agents and a depth-based processing policy | *LLM role:* decomposition_guide, reformulation_expert

> LinearizeLLM is a multi-agent framework that converts LaTeX nonlinear optimization problems into exact MILP formulations by detecting nonlinear terms and processing them bottom-up based on nesting depth. On 40 benchmark instances, it achieves 73% end-to-end success compared to <15% for one-shot LLMs and Pyomo baselines, demonstrating that structural decomposition is essential for handling complex nested terms. The key takeaway is the 'Structural Policy': rather than letting the LLM plan the reformulation order, they enforce a deterministic bottom-up traversal (linearizing children before parents). We should steal this hybrid approach—using deterministic graph traversal to orchestrate LLM manipulation steps—to improve reliability in our symbolic modeling and EvoCut pipelines.

### [DAOpt: Modeling and Evaluation of Data-Driven Optimization under Uncertainty with LLMs](https://arxiv.org/abs/2511.11576)

**2025-09-24** | Zhejiang University, University of Toronto, Peking University | M=6 P=8 I=7 **MUST-READ** *discuss*

*Method:* LLM-based multi-agent framework for optimization modeling, integrating few-shot learning with OR domain knowledge (RSOME toolbox) and a Reflexion-based checker | *LLM role:* code_writer

> Zhu et al. propose DAOpt, a framework for modeling optimization under uncertainty that integrates LLMs with the RSOME library to handle robust and stochastic formulations. Their experiments on a new dataset (OptU) convincingly demonstrate that standard LLM-generated deterministic models suffer from the 'optimizer's curse,' achieving only ~27% out-of-sample feasibility, whereas their robust approach achieves >70%. The critical takeaway for us is to **stop asking LLMs to derive mathematical duals or robust counterparts**; instead, we should train them to use high-level DSLs (like RSOME) that handle the duality internally. This is an immediate action item for our RobustMAS project to ensure generated solutions are actually executable in stochastic environments.

### [AI4S-SDS: A Neuro-Symbolic Solvent Design System via Sparse MCTS and Differentiable Physics Alignment](https://arxiv.org/abs/2603.03686)

**2026-03-04** | Nanjing University, Suzhou Laboratory, Shanghai Artificial Intelligence Laboratory | M=8 P=4 I=8 **MUST-READ** *discuss*

*Method:* Neuro-symbolic framework integrating Sparse Monte Carlo Tree Search (MCTS) with Sibling-Aware Expansion, Memory-Driven Global Planning, and a Differentiable Physics Engine for continuous ratio optimization. | *LLM role:* semantic_generator

> Chen et al. introduce a neuro-symbolic MCTS framework for mixed discrete-continuous optimization, applying it to solvent design. They solve the LLM context bottleneck via 'Sparse State Storage' (storing only state abstractions and reconstructing paths on-demand) and fix mode collapse using 'Sibling-Aware Expansion' (conditioning the generator on sibling nodes to force orthogonality). While the chemical application is niche, the search architecture is highly relevant: we should steal the sibling-aware conditioning to improve diversity in our evolutionary code generation and adopt their sparse storage pattern to scale our search horizons.

### [Auto-Formulating Dynamic Programming Problems with Large Language Models](https://arxiv.org/abs/2507.11737)

**2025-07-15** | University of Chicago, Cornell University, Shanghai Jiao Tong University, Shanghai University of Finance and Economics, Cardinal Operations | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* DPLM, a 7B-parameter specialized model fine-tuned on Qwen-2.5-7B-Instruct using synthetic data generated by DualReflect, combining Supervised Fine-Tuning (SFT) with Reinforcement Learning (GRPO/DPO) alignment. | *LLM role:* model_formulator, code_writer, synthetic_data_generator, refinement_agent

> Zhou et al. introduce DPLM, a 7B model fine-tuned to formulate Dynamic Programming models, achieving performance comparable to o1 on their new DP-Bench. Their key contribution is 'DualReflect,' a synthetic data pipeline that combines Forward Generation (Problem→Code) for diversity with Backward Generation (Code→Problem) for correctness. **Takeaway:** We should steal the Backward Generation approach for AlgoEvo: instead of relying on noisy forward generation, we can take valid heuristics/OR code (which we have in abundance) and reverse-engineer problem descriptions to create massive, verifiable synthetic datasets for fine-tuning our code generation models. The paper proves this method is superior for 'cold-starting' small models in data-scarce domains.

### [MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research](https://arxiv.org/abs/2602.03318)

**2026-02-03** | Xi'an Jiaotong University, Northwestern Polytechnical University | M=5 P=9 I=6 *discuss*

*Method:* Multi-Agent Framework with Iterative Adaptive Revision (IAR) and Hierarchical Retrieval-Augmented Generation (HRAG) | *LLM role:* code_writer, decomposition_guide, evaluator

> MIRROR is a multi-agent framework that translates natural language OR problems into Gurobi code using Hierarchical RAG (metadata filtering + semantic search) and an iterative repair loop. It achieves ~72% pass@1 across five benchmarks, outperforming Chain-of-Experts and fine-tuned models like LLMOPT without task-specific training. The key takeaway is their **structured revision tip mechanism**: upon execution failure, the agent generates a JSON object explicitly isolating the `error_statement`, `incorrect_code_snippet`, and `correct_code_snippet`, which serves as a precise memory artifact for subsequent retries. This structured reflection pattern is superior to raw error logs and could be immediately adopted in our own code generation pipelines.

### [SOCRATES: Simulation Optimization with Correlated Replicas and Adaptive Trajectory Evaluations](https://arxiv.org/abs/2511.00685)

**2025-11-01** | Columbia, UC Berkeley, Amazon | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Two-stage procedure: Stage 1 constructs an ensemble of Operational AI Replicas (OARs) via LLM-guided causal skeleton inference and EM-type structural learning. Stage 2 employs an LLM as a trajectory-aware meta-optimizer to iteratively revise and compose a hybrid SO algorithm schedule on the OAR ensemble. | *LLM role:* causal_discovery, meta_optimizer, schedule_reviser

> SOCRATES introduces a two-stage framework: first constructing 'Operational AI Replicas' (surrogates) via LLM-guided causal discovery, then using an LLM to analyze optimization trajectories on these surrogates to schedule hybrid algorithms (e.g., running BO then switching to GA). While the benchmarks (inventory, queuing) are simple and the causal inference step seems fragile, the core innovation of **trajectory-based reasoning** is highly transferable. We can steal this mechanism for AlgoEvo: instead of blind evolution, our planner agent should consume the optimization trajectory to dynamically swap operators or restart populations when stagnation is detected, effectively using the LLM as a process reward model.

### [SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling](https://arxiv.org/abs/2510.05115)

**2025-09-28** | Huawei Noah’s Ark Lab, Huawei’s Supply Chain Management Department, City University of Hong Kong | M=7 P=8 I=6 *discuss*

*Method:* Backward-guided iterative semantic alignment and correction using LLM agents | *LLM role:* code_writer, evaluator, decomposition_guide

> SAC-Opt introduces a verification loop where generated Gurobi code is back-translated into natural language ('semantic anchors') to check for alignment with the original problem description. Empirical results are strong, demonstrating a ~22% accuracy improvement on the ComplexLP dataset over OptiMUS-0.3 by catching logic errors that solver feedback misses. The primary takeaway is the utility of granular, constraint-level back-translation as a process reward signal, which we should adopt to improve the reliability of our automated modeling agents.

### [Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows](https://arxiv.org/abs/2505.04354)

**2025-05-07** | University of Minnesota, Tongji University, East China Normal University | M=5 P=9 I=6 *discuss*

*Method:* Evolutionary Agentic Workflow combining Foundation Agents (Memory, Reasoning, World Modeling, Action modules) and Evolutionary Search (Distributed Population Management, Solution Diversity Preservation, Knowledge-Guided Evolution) | *LLM role:* evolutionary_search

> Li et al. propose an 'Evolutionary Agentic Workflow' that combines LLMs (DeepSeek) with evolutionary search to automate algorithm design, demonstrating it on VM scheduling and ADMM parameter tuning. The empirical rigor is low; they compare against weak baselines (BestFit for bin packing, a 2000-era heuristic for ADMM) and frame it as a position paper. However, the application of LLM-evolution to discover symbolic mathematical update rules (for ADMM step sizes) rather than just procedural code is a concrete use case we should consider for our EvoCut work. This serves primarily as competitor intelligence—validating our AlgoEvo direction—rather than a source of novel methodology.

### [An Agent-Based Framework for the Automatic Validation of Mathematical Optimization Models](https://arxiv.org/abs/2511.16383)

**2025-11-20** | IBM Research | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-agent LLM framework for automatic validation of optimization models using problem-level API generation, unit test generation, and optimization-specific mutation testing | *LLM role:* code_writer

> Zadorojniy et al. introduce a multi-agent framework for validating LLM-generated optimization models by generating a test suite and verifying the suite's quality via mutation testing (ensuring tests detect deliberate errors injected into the model). On 100 NLP4LP instances, they achieve a 76% mutation kill ratio and successfully classify external models where simple objective value comparisons fail. The critical takeaway is the 'bootstrapped validation' workflow: using mutation analysis to validate the generated unit tests themselves before using them to score the model. We should steal this mutation-based verification loop to create a robust, ground-truth-free fitness signal for our evolutionary search and OR benchmarking pipelines.

### [A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation](https://arxiv.org/abs/2512.11270)

**2025-12-12** | Sejong University | M=5 P=7 I=6 *discuss*

*Method:* Modular multi-agent LLM framework (A-LAMP) decomposing MDP formulation and policy generation into specialized LLM agents | *LLM role:* Automates MDP modeling, environment code generation, and policy training pipeline

> A-LAMP decomposes the translation of natural language task descriptions into executable RL environments via a multi-agent pipeline, separating parameter extraction, variable definition, and constraint formulation before code generation. The results show that this structured approach allows a 27B model to rival GPT-4o on simple tasks, though the benchmarks (e.g., grid-world drone delivery, trivial wireless scheduling) are toy-scale and the RL application is sometimes forced. The primary takeaway is the specific decomposition schema for symbolic modeling: we should steal their granular extraction pipeline (Parameters -> Objectives -> Variables -> Constraints) to improve the reliability of our automated problem instantiation in OR-Bench and AlgoEvo without relying solely on expensive frontier models.

### [RideAgent: An LLM-Enhanced Optimization Framework for Automated Taxi Fleet Operations](https://arxiv.org/abs/2505.06608)

**2025-05-10** | Tsinghua University, McGill University, George Washington University, JD Intelligent Cities Research, Beijing Technology and Business University | M=7 P=6 I=7 *discuss*

*Method:* LLM-guided variable fixing heuristic for Mixed-Integer Programming (MIP) with an embedded Random Forest (RF) objective, solved lexicographically | *LLM role:* objective_formulation

> RideAgent employs an LLM to analyze a small set of historical optimal solutions, identifying and fixing 'low-sensitivity' decision variables to shrink the MIP search space before handing it to Gurobi. The results are empirically solid, showing a ~50% time reduction with <2.5% optimality gap, outperforming standard cutting plane baselines on NYC taxi data. **Key Takeaway:** We should adapt their 'Small-Sample Guided Optimization' strategy—specifically using LLMs to infer *variable fixing constraints* from elite archive solutions—to accelerate the inner solvers in our AlgoEvo and EvoCut pipelines. This offers a concrete, data-driven way to prune search spaces that complements our current evolutionary approaches.

### [Toward a Trustworthy Optimization Modeling Agent via Verifiable Synthetic Data Generation](https://arxiv.org/abs/2508.03117)

**2025-08-05** | IBM Research AI | M=6 P=7 I=7 *discuss*

*Method:* Verifiable Synthetic Data Generation (SDG) pipeline combined with a modular LLM agent (OptiTrust) employing multi-stage translation, multi-language inference, and majority-vote cross-validation | *LLM role:* data_generator, code_writer, decomposition_guide, formulation_generator, evaluator

> Lima et al. introduce a pipeline to generate synthetic optimization datasets by starting with symbolic MILP instances (ground truth) and using LLMs to generate natural language descriptions, ensuring full verifiability. They fine-tune a small model (Granite 8B) that beats GPT-4 on 6/7 benchmarks, largely due to a 'majority vote' mechanism where the agent generates code in 5 different modeling languages (Pyomo, Gurobi, etc.) and checks for result consistency. **Takeaway:** We should steal the multi-language execution voting to boost robustness in our code generation agents. Furthermore, their reverse-generation (Symbolic $\to$ NL) strategy is the correct approach for generating infinite, error-free test cases for our OR-Bench work.

### [Solver-in-the-Loop: MDP-Based Benchmarks for Self-Correction and Behavioral Rationality in Operations Research](https://arxiv.org/abs/2601.21008)

**2026-02-08** | Massachusetts Institute of Technology, Alibaba Group | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Domain-specific Group Relative Policy Optimization (GRPO) with composite reward and three-stage curriculum learning | *LLM role:* agent_for_debugging_and_decision_making

> Ao et al. introduce a framework for iterative OR model debugging that trains an 8B model using Group Relative Policy Optimization (GRPO) and a Process Reward Model (PRM) to outperform GPT-4o-mini. They utilize Gurobi's Irreducible Infeasible Subsystem (IIS) not just as text feedback, but as a dense reward signal (IIS size reduction) for the PRM, achieving a 95.3% recovery rate versus 86.2% for frontier APIs. **Key Takeaway:** We should steal their PRM construction method—specifically using solver diagnostics (like IIS reduction or compiler error counts) as dense step-level rewards—and their 'faithfulness penalty' to prevent overfitting in our evolutionary search. This is a direct validation of RLVR (Reinforcement Learning with Verifiable Rewards) for OR, proving it superior to large-scale prompting.

### [LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading](https://arxiv.org/abs/2507.14995)

**2025-07-20** | China Agricultural University, University of Glasgow, Guangdong University of Foreign Studies | M=7 P=6 I=7 *discuss*

*Method:* LLM-Enhanced Multi-Agent Reinforcement Learning (MARL) with CTDE-based imitative expert MARL algorithm, using a differential multi-head attention-based critic network and Wasserstein metric for imitation | *LLM role:* heuristic_generator

> This paper proposes a neurosymbolic MARL framework for P2P energy trading where LLMs generate CVXPY optimization models to act as 'experts' for RL agents to imitate via Wasserstein distance. They introduce a 'Differential Attention' mechanism in the critic that subtracts attention maps to filter noise, enabling scalability to 100 agents where standard baselines fail. **Takeaway:** We should steal the Differential Attention architecture for our multi-agent critics to handle irrelevant interactions in large-scale optimization. The workflow of using LLMs to write the *solver* (generating reliable synthetic data) rather than the *solution* is a transferable strategy for bootstrapping RL in our OR domains.

### [Canonical Intermediate Representation for LLM-based optimization problem formulation and code generation](https://arxiv.org/abs/2602.02029)

**2026-02-02** | The Hong Kong Polytechnic University, InfiX.ai | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-agent pipeline with Canonical Intermediate Representation (CIR) and Retrieval-Augmented Generation (RAG) | *LLM role:* Decomposition guide, paradigm selector, code writer, and verifier

> Lyu et al. propose a 'Canonical Intermediate Representation' (CIR) to decouple natural language operational rules from their mathematical instantiation, explicitly forcing the LLM to select modeling paradigms (e.g., time-indexed vs. continuous flow) before coding. They achieve state-of-the-art accuracy (47.2% vs 22.4% baseline) on a new, complex benchmark (ORCOpt-Bench) by using a multi-agent pipeline that retrieves and adapts constraint templates. The key takeaway is the 'Mapper' agent's paradigm selection logic, which prevents common formulation errors in VRPs and scheduling; we should evaluate CIR as a structured mutation space for AlgoEvo to replace brittle code evolution. The new benchmark is immediately relevant for our OR-Bench evaluation suite.

### [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983)

**2026-02-17** | National University of Singapore, Northwestern University, City University of Hong Kong, Wenzhou University, Wenzhou Buyi Pharmacy Chain Co., Ltd. | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structured generation (understand, formalize, synthesize, verify) with two-layer behavioral verification (L1 execution recovery, L2 solver-based perturbation testing) and diagnosis-guided repair. | *LLM role:* code_writer

> ReLoop proposes a verification pipeline for LLM-generated optimization models that detects 'silent failures' (code that runs but solves the wrong problem) by perturbing input parameters and checking for expected solver objective shifts. They demonstrate that standard execution feasibility is a poor proxy for correctness (90% gap) on their new RetailOpt-190 benchmark, and that this perturbation testing significantly improves reliability. The critical takeaway is the use of sensitivity analysis as a ground-truth-free process reward signal: we can validate evolved algorithms in AlgoEvo by asserting that specific input perturbations *must* trigger output changes, filtering out semantically invalid candidates before expensive evaluation.

### [Constructing Industrial-Scale Optimization Modeling Benchmark](https://arxiv.org/abs/2602.10450)

**2026-02-11** | Peking University, Huawei Technologies Co., Ltd., Great Bay University | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structure-aware reverse construction methodology from MIPLIB 2017 | *LLM role:* linguistic_polisher, interactive_assistant

> Li et al. introduce MIPLIB-NL, a benchmark of 223 industrial-scale MILP instances (up to 10^7 variables) reverse-engineered from MIPLIB 2017, enforcing strict model-data separation. Results are sobering: SOTA models like GPT-4 and fine-tuned OR-LLMs drop from ~90% accuracy on existing toy benchmarks to ~18% here, failing primarily on structural consistency and index handling at scale. For us, the key takeaway is their "Loop-Based Structural Scaffold" taxonomy—a method to compress massive industrial formulations into compact LLM prompts via model-data separation. This is a mandatory read for our OR-Bench project, as it demonstrates that current evaluations are effectively measuring overfitting to toy problems rather than genuine modeling capability.

### [SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search](https://arxiv.org/abs/2510.16916)

**2025-10-19** | NEC Labs America, Baylor University, University of Texas at Dallas, Augusta University, Southern Illinois University | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* LLM-guided Monte Carlo Tree Search (MCTS) with dynamic expansion, prompt backpropagation, and uncertainty backpropagation for optimization problem formulation and code generation | *LLM role:* decomposition_guide, heuristic_generator, evaluator, code_writer, evolutionary_search

> SolverLLM frames optimization problem formulation as a hierarchical Monte Carlo Tree Search (MCTS), decomposing the task into six layers (variables, constraints, etc.) and using test-time compute to beat fine-tuned baselines like LLMOPT. The results appear robust, showing ~10% gains on complex datasets, though inference cost is high. **The critical takeaway for us is the 'Prompt Backpropagation' mechanism:** instead of just updating numerical values, they propagate textual error analysis from leaf nodes back up the tree to dynamically modify the prompts of parent nodes, effectively creating 'short-term memory' for the search. We should immediately test this technique in AlgoEvo to prevent the recurrence of failed code patterns during mutation steps. Additionally, their use of semantic entropy to down-weight uncertain rewards in MCTS is a practical solution to the noisy evaluation problem we face in process reward models.

### [OptiRepair: Closed-Loop Diagnosis and Repair of Supply Chain Optimization Models with LLM Agents](https://arxiv.org/abs/2602.19439)

**2026-02-23** | Massachusetts Institute of Technology, Alibaba Group | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-phase closed-loop LLM agent with IIS-guided diagnosis, domain-specific rationality oracle, iterative STaR, and GRPO refinement | *LLM role:* diagnosis_and_repair

> Ao et al. introduce OptiRepair, a closed-loop framework that repairs infeasible LPs using solver IIS feedback (Phase 1) and validates them with a 'Rationality Oracle' based on domain theory (Phase 2). Results are exceptionally strong: fine-tuned 8B models trained via iterative STaR and GRPO achieve 81.7% success, outperforming GPT-5.2 (42.2%) by a massive margin. **Key Takeaway:** We should steal the 'Rationality Oracle' concept—evaluating solution *properties* (e.g., monotonicity, variance bounds) rather than just raw fitness—to serve as a dense signal for our Process Reward Models in AlgoEvo. Additionally, their success with solver-verified GRPO confirms we should prioritize training specialized operators over prompting general LLMs.

### [Grammar-Aware Literate Generative Mathematical Programming with Compiler-in-the-Loop](https://arxiv.org/abs/2601.17670)

**2026-01-25** | University of Edinburgh, University College Cork | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Iterative generate–compile–assess–revise loop with compiler-in-the-loop and LLM-based alignment judge | *LLM role:* generator, evaluator, revision policy

> SyntAGM is a framework for translating natural language into Algebraic Modeling Language (PyOPL) code using a 'compiler-in-the-loop' approach, where the LLM is constrained by an in-context BNF grammar and iteratively repairs code based on compiler diagnostics. They demonstrate that this approach matches the accuracy of expensive multi-agent systems (like Chain-of-Experts) while being significantly faster and cheaper. The immediate takeaways for us are the **StochasticOR benchmark** (which we should adopt for RobustMAS) and the technique of **injecting explicit BNF grammars** into prompts to enforce syntax in evolutionary search without fine-tuning. The 'literate modeling' approach—embedding reasoning as comments directly next to code constraints—is also a clever memory mechanism we could steal for AlgoEvo.

### [Learning Virtual Machine Scheduling in Cloud Computing through Language Agents](https://arxiv.org/abs/2505.10117)

**2025-05-15** | Shanghai Jiao Tong University, East China Normal University, Tongji University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hierarchical Language Agent Framework (MiCo) for LLM-driven heuristic design, formulated as Semi-Markov Decision Process with Options (SMDP-Option), using LLM-based function optimization for policy discovery and composition. | *LLM role:* heuristic_generator, evolutionary_search, decomposition_guide

> Wu et al. introduce MiCo, a hierarchical framework that uses LLMs to evolve both a library of scenario-specific scheduling heuristics ('Options') and a master policy ('Composer') that dynamically switches between them based on system state. Tested on large-scale Huawei/Azure VM traces, it achieves a 96.9% competitive ratio against Gurobi, significantly outperforming Deep RL (SchedRL) by ~11% in dynamic scenarios. **Key Insight:** Instead of evolving a single robust heuristic (which often fails in non-stationary environments), explicitly evolve a *portfolio* of specialized heuristics and a separate *selector* function. This SMDP-based decomposition is a concrete architectural pattern we should adopt in AlgoEvo to handle diverse problem instances and non-stationary distributions effectively.

### [FMIP: Joint Continuous-Integer Flow For Mixed-Integer Linear Programming](https://arxiv.org/abs/2507.23390)

**2025-09-29** | Stanford University, Princeton University, National University of Singapore, Shanghai Jiao Tong University, Shanghai University of Finance and Economics | M=6 P=6 I=7 *discuss*

*Method:* Generative framework based on conditional flow matching for joint continuous-integer variable distribution modeling | *LLM role:* none

> FMIP introduces a flow matching framework that jointly generates integer and continuous variables for MILP, utilizing a tripartite graph and inference-time guidance. Empirical results on MIPLIB and other benchmarks show a ~40% reduction in primal gap compared to integer-only neural baselines (DIFUSCO), though it remains a heuristic warm-start for solvers. The most valuable takeaway is the **hybrid guidance mechanism** (Eq. 6 & 7): it combines gradient descent for continuous variables with a sampling-and-reweighting scheme for discrete variables based on constraint violations. We should consider stealing this reweighting logic for guiding hybrid evolutionary operators or multi-agent action spaces where gradients are available for only part of the state.

### [NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](https://arxiv.org/abs/2601.21372)

**2026-01-29** | Carnegie Mellon University, C3 AI | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Execution-Aware Optimization Modeling via Autonomous Coding Agents (ACAs) with asymmetric simulator-optimizer validation loop | *LLM role:* code_writer

> NEMO achieves SOTA on 8/9 optimization benchmarks by deploying autonomous coding agents that generate both a declarative optimizer (solver code) and an imperative simulator (verification code). The key innovation is using the simulator to validate the optimizer's results in a closed loop, detecting logical errors without ground truth—a technique that beats fine-tuned models like SIRL by up to 28%. The most stealable insight is this asymmetric validation: imperative Python simulation is often less error-prone than declarative constraint formulation, making it a robust 'critic' for generated solvers. This is immediately applicable to our OR-Bench and AlgoEvo projects for generating reliable reward signals.

### [OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery](https://arxiv.org/abs/2602.13769)

**2026-02-14** | Tongji University | M=9 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent research framework with evolutionary-systematic ideation, tree-structured research workflow, and hierarchical optimization-inspired reflection system | *LLM role:* heuristic_generator, code_writer, evaluator, decomposition_guide, evolutionary_search, prompt_optimizer

> OR-Agent replaces flat evolutionary loops with a tree-structured research workflow that prioritizes deep iterative refinement and debugging over broad population sampling. The results are compelling, showing a ~2x improvement in normalized scores over ReEvo and FunSearch across 12 OR benchmarks (TSP, CVRP, etc.). The single most actionable takeaway is the **Experiment Agent's environment probing**: instead of relying on scalar fitness, the agent writes custom callbacks to log intermediate states (e.g., 'lane change attempts' in SUMO), enabling genuine diagnosis of failure modes. We should immediately implement this 'instrumentation-via-code' pattern in our own evaluation pipelines to improve signal quality.

### [StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models](https://arxiv.org/abs/2509.22558)

**2025-09-26** | Shanghai Jiao Tong University | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Self-evolving framework with generative process supervision using Weighted Direct Preference Optimization (W-DPO) and Supervised Fine-Tuning (SFT) | *LLM role:* code_writer, evaluator, decomposition_guide, evolutionary_search

> Zhou et al. propose StepORLM, a framework where an 8B policy and a **Generative Process Reward Model (GenPRM)** co-evolve. Unlike standard discriminative PRMs that score steps in isolation, their GenPRM generates a reasoning trace to evaluate the full trajectory's logic before assigning credit, addressing the interdependency of OR constraints. They align the policy using **Weighted DPO**, where preference weights are derived from the GenPRM's process scores. They claim to beat GPT-4o and DeepSeek-V3 on 6 OR benchmarks (e.g., NL4Opt, MAMO) with an 8B model. **Key Takeaway:** We should test **Generative PRMs** immediately for AlgoEvo; asking the critic to 'explain then score' (generative) rather than just 'score' (discriminative) likely fixes the credit assignment noise in our long-horizon search.

### [Fine-tuning Large Language Model for Automated Algorithm Design](https://arxiv.org/abs/2507.10614)

**2025-07-13** | City University of Hong Kong | M=7 P=10 I=8 **MUST-READ** *discuss*

*Method:* Direct Preference Optimization (DPO) with Diversity-Aware Rank-based (DAR) Sampling for LLM fine-tuning | *LLM role:* code_writer

> Liu et al. introduce a fine-tuning pipeline for LLMs in automated algorithm design, utilizing a 'Diversity-Aware Rank-based' sampling strategy to construct DPO preference pairs from evolutionary search histories. By partitioning the population into ranked subsets and sampling pairs with a guaranteed quality gap (skipping adjacent tiers), they ensure training signals are both clear and diverse. Empirically, they show that a fine-tuned Llama-3.2-1B matches the performance of a base Llama-3.1-8B on ASP and CVRP tasks, effectively compressing the search capability into a much cheaper model. We should implement this sampling strategy to recycle our AlgoEvo run logs into specialized 'mutator' models, potentially allowing us to downscale to 1B/3B models for the inner search loop without losing quality.

### [Execution-Verified Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2604.00442)

**2026-04-01** | Chinese Academy of Sciences, Nanjing University, Nanjing University of Science and Technology | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Execution-Verified Reinforcement Learning (EVOM) with GRPO and DAPO for solver-conditioned code generation | *LLM role:* code_writer

> EVOM trains LLMs for operations research modeling using execution-verified reinforcement learning (GRPO/DAPO) based solely on solver outcomes, bypassing expensive process-level supervision. The results are backed by solid empirical evaluations on OptiBench, NL4OPT, and IndustryOR, demonstrating that it matches or beats process-supervised SFT (ORLM) and enables zero-shot transfer to new solvers (e.g., Gurobi to OR-Tools). The key takeaway is that outcome-only RL prevents the model from overfitting to solver-specific syntax (a major flaw in SFT), forcing it to learn invariant mathematical structures; additionally, their two-stage cold-start trick (LLM-translate 100 samples -> SFT -> RL) is a highly stealable technique for adapting to new environments. This is highly relevant for our OR-Bench project, and we should consider implementing execution-verified RL baselines and leveraging their cold-start adaptation trick when targeting new solvers in our evolutionary search pipelines.

### [Foundation Models for Logistics: Toward Certifiable, Conversational Planning Interfaces](https://arxiv.org/abs/2507.11352)

**2026-01-30** | Neurosymbolic Intelligence, The University of Texas at Austin, University of Colorado Boulder | M=6 P=5 I=7 *discuss*

*Method:* Neurosymbolic Vision-Language Logistics (VLL) agent with uncertainty-aware intent-verification loop, using latent space learning, probabilistic guarantees, DPO, and TextGrad for refinement. | *LLM role:* goal_interpreter_and_refiner

> Yang et al. introduce a neurosymbolic agent that translates natural language into PDDL goals, using a learned latent space to estimate 'intent uncertainty' (distance to class centroids) which gates downstream execution. They use this uncertainty signal to drive both Direct Preference Optimization (DPO) and prompt optimization (TextGrad), achieving higher accuracy than GPT-5 on a lightweight model. **Takeaway:** The concept of deriving a 'probabilistic guarantee' from latent embeddings to serve as a cheap proxy reward or filter is a concrete technique we should test in AlgoEvo to reduce expensive evaluations. However, be skeptical of the topline results as they rely on a simplistic 3-class classification task rather than complex reasoning.

### [CHORUS: Zero-shot Hierarchical Retrieval and Orchestration for Generating Linear Programming Code](https://arxiv.org/abs/2505.01485)

**2025-05-02** | Queen's University | M=5 P=7 I=6 *discuss*

*Method:* Retrieval-Augmented Generation (RAG) framework with hierarchical chunking, metadata-augmented indexing, two-stage retrieval, cross-encoder reranking, expert prompting, and structured output parsing | *LLM role:* code_writer

> CHORUS introduces a RAG framework for generating Gurobi code that replaces standard code retrieval with a metadata-based approach, indexing code examples by generated keywords and summaries rather than raw syntax. On the NL4Opt-Code benchmark, this allows open-source models like Llama-3-70B to match GPT-4 performance (improving accuracy from ~23% to ~57%). The key takeaway for us is the effectiveness of 'metadata-augmented indexing'—bridging the semantic gap between natural language problem descriptions and rigid solver APIs by retrieving based on functional descriptions rather than code embeddings. We should apply this metadata indexing strategy to the code retrieval modules in our OR-Bench and AlgoEvo agents.

### [Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks](https://arxiv.org/abs/2410.22296)

**2024-10-29** | Genentech, New York University | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLOME (Language Model Optimization with Margin Expectation) bilevel optimization with Margin-Aligned Expectation (MargE) loss | *LLM role:* optimization_driver

> The authors propose LLOME, a bilevel optimization framework that fine-tunes an LLM using 'MargE' (Margin-Aligned Expectation), a loss function that weights gradient updates by the magnitude of reward improvement (margin) rather than simple preference rankings. Results are rigorous and demonstrate that while DPO leads to generator collapse and infeasibility in constrained spaces, MargE maintains diversity and significantly improves sample efficiency, matching specialized solvers like LaMBO-2 on medium-difficulty tasks. The critical takeaway is that standard alignment methods (DPO/RLHF) are ill-suited for optimization because they discard information about *how much* better a solution is; MargE fixes this by satisfying the Strong Interpolation Criteria. We should immediately evaluate replacing the RL/update component in AlgoEvo with the MargE objective to improve the stability and quality of our evolved heuristics.

### [GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization](https://arxiv.org/abs/2602.20427)

**2026-02-23** | Cornell University, University of Maryland, College Park | M=7 P=5 I=7 *discuss*

*Method:* Differentiable Scheduling Optimization via Gaussian Reparameterization with Augmented Lagrangian Method | *LLM role:* none

> GauS replaces the standard categorical (Gumbel-Softmax) relaxation in differentiable scheduling with Gaussian variables defined by mean and variance, reducing parameter space from O(N*D) to O(N). Results are strong: it scales to 57k nodes where previous differentiable methods OOM and exact solvers timeout, while maintaining near-100% GPU utilization. The key takeaway is a specific modeling technique: using Gaussian distributions to represent discrete ordinal values (like time steps) naturally captures temporal proximity and provides smoother gradients than categorical buckets. We should test this representation in our continuous latent-space optimization work to replace categorical relaxations for ordered parameters.

### [Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](https://arxiv.org/abs/2512.18682)

**2025-12-21** | Xidian University, Victoria University of Wellington, Westlake University | M=7 P=5 I=7 *discuss*

*Method:* Supervised fine-tuning of LLMs on a synthetically generated dataset, created via data augmentation (semantic paraphrasing, order permutation) and LLM-based test instance annotation and selection, to convert natural language requirements into executable Python optimization functions. | *LLM role:* code_writer

> Li et al. propose APF, a framework to fine-tune LLMs for translating engineering requirements into optimization code without running expensive simulations during training. They generate synthetic training data and filter it by checking if the generated code ranks historical data instances similarly to how an LLM 'judge' ranks them based on the text requirements. Results show 7B models outperforming GPT-4o on antenna design tasks, validated by actual simulation. **Key Takeaway:** We can replace expensive ground-truth evaluations in our process reward models by checking consistency between generated code outputs and LLM-predicted rankings on cached historical data—a direct method to improve sample efficiency in AlgoEvo.

### [VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation](https://arxiv.org/abs/2603.02681)

**2026-03-03** | Tencent Hunyuan, Hong Kong University of Science and Technology | M=8 P=2 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Native visual-generation agentic model (VisionCreator) unifying Understanding, Thinking, Planning, and Creation (UTPC) capabilities, optimized via Progressive Specialization Training (PST) and Virtual Reinforcement Learning (VRL) with LtrReward in VisGenEnv. | *LLM role:* agentic_model

> This paper introduces VisionCreator, an agent trained via 'Virtual Reinforcement Learning' (VRL) where tool outputs and logic are simulated to train long-horizon planning policies without incurring expensive real-world execution costs. They employ a 'Plan-Driven Reward' model (combining LLM-based plan verification with rule-based execution checks) and prove theoretical bounds for the sim-to-real transfer, achieving performance superior to GPT-5 on visual tasks. **Key Takeaway:** We should steal the VRL architecture for AlgoEvo. By constructing a 'Virtual OR Environment' that simulates code validity and approximate heuristic performance, we can train our evolutionary search policies (RL-infused evolution) at a fraction of the current compute cost, bypassing the bottleneck of running full benchmarks during the search policy optimization phase.


### Front 0 (17 papers) — DECLINING

**Density:** 0.53 | **Methods:** llm_code_generation, llm_as_evaluator, llm_in_the_loop, llm_fine_tuned, llm_as_heuristic | **Problems:** optimization_modeling, linear_programming, automated_optimization_modeling, milp_general, integer_programming

*Unique methods:* ai_agents, automatic_symbolic_dualization, benchmark_creation, bipartite_graphs, bootstrapping, callbacks, canonical_graph_edit_distance, canonicalization, code_interpreter_feedback, data_cleaning, data_synthesis, deterministic_parser, dual_reward_system, dynamic_supervised_fine_tuning_policy_optimization, empirical_study, error_analysis, error_driven_learning, evaluation_framework, exact_optimization, graph_edit_distance, graph_isomorphism, graph_theory, gurobi, hashing, heuristic_filtering, hierarchical_decomposition, indicator_variables, instance_generation, kahneman_tversky_optimization, llm_as_data_synthesizer, llm_as_model_generator, llm_as_solver, llm_evaluation, mathematical_modeling, multi_agent_coordination, multi_armed_bandit, multi_turn_feedback, nonlinear_programming, partial_kl_divergence, piecewise_linear_constraints, proximal_policy_optimization, pyscipopt, reinforce_plus_plus, rejection_sampling, reverse_data_synthesis, reverse_socratic_method, self_refinement, self_reflective_error_correction, sifting, smt_solvers, special_ordered_sets, structure_detection, symbolic_pruning, symmetric_decomposable_graphs, systematic_literature_review, test_time_adaptation, test_time_reinforcement_learning, two_stage_reward_system, uct, weisfeiler_lehman_test
*Shared methods:* benchmark_design, data_augmentation, debugging, group_relative_policy_optimization, in_context_learning, instruction_tuning, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, majority_voting, milp_solver, mixed_integer_linear_programming, monte_carlo_tree_search, multi_agent_llm_system, multi_agent_system, program_synthesis, prompt_engineering, reinforcement_learning, self_consistency, self_correction, self_improving_search, supervised_fine_tuning, supervised_learning, synthetic_data_generation

This research front centers on advancing the reliability and performance of Large Language Models (LLMs) for automated optimization modeling, specifically translating natural language descriptions into executable mathematical programs (e.g., MILP, LP). The unifying theme is the development of robust frameworks and methodologies to overcome challenges in LLM accuracy, consistency, and data scarcity for generating correct and verifiable optimization models. Key approaches involve sophisticated data synthesis pipelines, novel evaluation metrics, and advanced self-correction and refinement mechanisms.

Key contributions include sophisticated data synthesis frameworks like OptMATH and ReSocratic, which generate high-quality, solver-verified datasets by back-translating formal models to natural language, significantly improving fine-tuned LLMs (e.g., Llama-3-8B accuracy from 13.6% to 51.1% on OptiBench). Advanced modeling agents such as OptiMUS-0.3 utilize modular structures and connection graphs to achieve SOTA performance, outperforming GPT-4o by up to 40% on benchmarks like NLP4LP. Evaluation methodologies have also seen significant advancements, with ORGEval introducing graph-theoretic isomorphism detection for structural validation, offering 100% consistency in seconds compared to hours for solver-based checks. Furthermore, several papers highlight critical flaws in existing benchmarks, with error rates up to 54%, leading to the release of cleaned datasets and new rigorous benchmarks like ProOPF-B, where even GPT-5.2 achieves only 14.05% accuracy. Reinforcement learning techniques, such as SIRL's Partial KL and MIND's DFPO, enhance LLM code generation by leveraging solver feedback and error-driven learning, while Autoformulator integrates SMT solvers for symbolic pruning in MCTS, drastically improving search efficiency.

Despite being labeled as 'declining' in overall activity, this front is rapidly maturing, with a strong emphasis on building robust and reliable LLM-based systems for OR modeling. The trajectory indicates a shift from foundational LLM integration to advanced techniques for verifiable model generation, efficient search, and continuous self-improvement. The next wave of research will likely focus on integrating these disparate advancements—such as graph-theoretic evaluation, error-driven RL, and symbolic pruning—into unified, scalable multi-agent frameworks capable of tackling real-world, large-scale, and ambiguous optimization problems, potentially with human-in-the-loop feedback for critical applications.

**Papers:**

### [DualSchool: How Reliable are LLMs for Optimization Education?](https://arxiv.org/abs/2505.21775)

**2025-05-27** | Georgia Institute of Technology | M=7 P=5 I=6 *discuss*

*Method:* DUALSCHOOL framework for generating and verifying P2DC instances using Canonical Graph Edit Distance (CGED) | *LLM role:* problem_solver

> This paper evaluates LLMs on Primal-to-Dual Conversion (P2DC), introducing a 'Canonical Graph Edit Distance' (CGED) to verify structural correctness while ignoring benign differences like variable ordering or slack conventions. Results show that even strong LLMs often fail (<50% accuracy) and, crucially, that standard execution-based evaluation (checking objective values) produces frequent false positives by missing errors in redundant constraints. The primary takeaway for us is the CGED methodology: a robust way to score symbolic OR model generation that captures structural validity better than execution alone, which we could steal for our benchmarking and evolutionary search fitness functions.

### [OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://arxiv.org/abs/2407.19633)

**2024-07-29** |  | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Modular LLM-based agent (OptiMUS-0.3) employing a connection graph and self-reflective error correction for sequential optimization model formulation and code generation using Gurobi API | *LLM role:* optimization_model_synthesis

> OptiMUS-0.3 is a modular multi-agent system that translates natural language into Gurobi code, utilizing a 'connection graph' to manage variable-constraint relationships in long contexts and specialized agents to detect solver-specific structures (SOS, indicators) or implement sifting. The results are rigorous, introducing a new hard benchmark (NLP4LP) where they outperform GPT-4o by ~40% and beat Chain-of-Experts. The most stealable insight is the 'Structure Detection Agent': instead of relying on the LLM to write generic constraints, we should explicitly prompt for and map high-level structures to efficient solver APIs (like SOS constraints) to improve performance in our EvoCut and AlgoEvo pipelines. This is a necessary read for the OR-Bench team.

### [OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling](https://arxiv.org/abs/2502.11102)

**2025-02-16** | Peking University | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Scalable bidirectional data synthesis framework integrating feedback-driven PD generation, LLM-based backtranslation with self-criticism/refinement, and AutoFormulator with rejection sampling. | *LLM role:* data synthesizer

> The authors introduce OptMATH, a framework for generating synthetic optimization datasets by creating mathematical instances from seed generators, back-translating them to natural language via LLMs, and validating the pairs using a solver-based rejection sampling loop (checking if the re-generated model yields the same optimal value). They demonstrate that a Qwen-32B model fine-tuned on this data beats GPT-4 on NL4Opt and MAMO benchmarks. The critical takeaway is the **solver-verified reverse generation pipeline**: we should immediately steal this workflow to populate OR-Bench and generate diverse, verified training environments for AlgoEvo, replacing manual curation with scalable synthesis.

### [LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch](https://arxiv.org/abs/2410.13213)

**2024-10-17** | Ant Group, East China Normal University, Nanjing University | M=5 P=7 I=6 *discuss*

*Method:* Multi-instruction supervised fine-tuning and KTO model alignment with self-correction | *LLM role:* Generates problem formulations, writes solver code, and performs error analysis for self-correction

> The authors fine-tune Qwen1.5-14B to translate natural language optimization problems into Pyomo code via a structured 'five-element' intermediate representation (Sets, Parameters, Variables, Objective, Constraints) and KTO alignment. They achieve ~11% accuracy gains over GPT-4o and ORLM on benchmarks like NL4Opt and IndustryOR, primarily by reducing formulation hallucinations through the structured intermediate step and preference optimization. For our OR-Bench work, the key takeaway is the concrete recipe for using KTO to align symbolic modeling agents, which appears more effective than standard SFT for enforcing constraints in smaller models. While not an evolutionary search paper, it provides a strong, locally runnable baseline for our OR modeling evaluations.

### [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://arxiv.org/abs/2405.17743)

**2025-04-04** | Columbia University, Duke University, Shanghai Jiao Tong University, The Chinese University of Hong Kong, Shenzhen, Shenzhen Research Institute of Big Data, Shanghai University of Finance and Economics, Cardinal Operations | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Instruction tuning of open-source LLMs using semi-automated synthetic data generated by OR-Instruct framework | *LLM role:* data_synthesis, model_generator, code_writer

> The authors propose OR-Instruct, a framework that uses GPT-4 to synthesize over 32k optimization modeling pairs (natural language to COPT code) to fine-tune 7B-scale models (ORLM). They demonstrate that these fine-tuned models outperform GPT-4 on their new 'IndustryOR' benchmark, a result that appears robust given the specialized nature of the task. The most valuable takeaway is their specific data augmentation strategy—iteratively altering constraints and injecting specific modeling techniques (e.g., Big M)—which provides a concrete recipe we can steal to generate diverse instances for our OR-Bench project. While the methodology is standard instruction tuning, the resulting artifacts (benchmark and model) establish a new baseline for automated OR modeling that we cannot ignore.

### [OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling](https://arxiv.org/abs/2407.09887)

**2024-07-13** | The Hong Kong University of Science and Technology, ETH Zurich, Huawei Noah’s Ark Lab, City University of Hong Kong, Sun Yat-sen University, MBZUAI, University of California Merced, Chongqing University | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* ReSocratic data synthesis for optimization problems, followed by Supervised Fine-Tuning of LLMs for Python code generation using PySCIPOpt solver | *LLM role:* evolutionary_search

> The authors propose OptiBench, a benchmark of 605 optimization problems (linear/nonlinear, tabular/text), and ReSocratic, a data synthesis method that generates formal models first and back-translates them into natural language questions. Results are strong: fine-tuning Llama-3-8B on their 29k synthetic samples improves accuracy from 13.6% to 51.1%, validating the data quality. **Key Takeaway:** The 'Reverse Socratic' synthesis pipeline (Formal Model → Code → NL Question) is the superior strategy for generating synthetic OR datasets because it guarantees solvability and ground truth by construction, unlike forward generation. We should steal this pipeline for generating robust test instances for OR-Bench and potentially for training our OR agents.

### [ORGEval: Graph-Theoretic Evaluation of LLMs in Optimization Modeling](https://arxiv.org/abs/2510.27610)

**2025-10-31** | The Chinese University of Hong Kong, Shenzhen, Shenzhen Research Institute of Big Data, Shenzhen International Center for Industrial and Applied Mathematics, Shenzhen Loop Area Institute | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Graph-theoretic evaluation framework using Weisfeiler-Lehman (WL) test with Symmetric Decomposable (SD) graph condition for model isomorphism detection | *LLM role:* none

> Wang et al. propose ORGEval, a framework that evaluates LLM-generated optimization models by converting them into bipartite graphs and using the Weisfeiler-Lehman (WL) test to detect isomorphism with a ground truth, rather than solving the instances. They prove that for 'symmetric decomposable' graphs, this method is guaranteed to detect equivalence correctly, achieving 100% consistency and running in seconds compared to hours for solver-based checks on hard MIPLIB instances. The critical takeaway is the shift from execution-based to **structural evaluation**: we can validate model logic via graph topology ($O(k(m+n)^2)$) without incurring the cost of solving NP-hard problems. This is immediately actionable for our OR benchmarking pipelines and could serve as a rapid 'pre-solve' filter in our evolutionary search loops to reject structurally invalid candidates instantly.

### [Automated Optimization Modeling via a Localizable Error-Driven Perspective](https://arxiv.org/abs/2602.11164)

**2026-01-17** | Huawei Noah’s Ark Lab, Fudan University, University of Science and Technology of China | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Error-driven learning framework (MIND) combining Dynamic Supervised Fine-Tuning Policy Optimization (DFPO) with an error-driven reverse data synthesis pipeline | *LLM role:* code_writer, decomposition_guide, evolutionary_search, prompt_optimizer

> This paper introduces MIND, a framework for automated optimization modeling that combines error-driven data synthesis with a novel post-training method called DFPO. Instead of standard RLVR which suffers from sparse rewards on hard problems, DFPO uses a teacher model to minimally correct the student's *failed* rollouts, converting them into on-policy(ish) positive samples for SFT/RL. Results show a 7B model outperforming GPT-4 on IndustryOR and OptMATH benchmarks. **Key Takeaway:** We should steal the DFPO mechanism for AlgoEvo: rather than wasting failed evolutionary samples, use a stronger model (or oracle) to fix the code and feed it back as a reward signal, drastically improving sample efficiency in our RL loops.

### [Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling](https://arxiv.org/abs/2505.11792)

**2025-05-17** | Stanford University, Shanghai Jiao Tong University, The University of Hong Kong, Shanghai University of Finance and Economics, Cardinal Operations | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning with Verifiable Reward (RLVR) using REINFORCE++ with a Partial KL surrogate function | *LLM role:* code_writer

> Chen et al. introduce SIRL, a framework for training LLMs to generate optimization models using Reinforcement Learning with Verifiable Rewards (RLVR) and a novel 'Partial KL' surrogate objective. By removing the KL penalty from the reasoning (CoT) section while retaining it for the code generation section, they balance exploration with syntactic stability, achieving SOTA on OptMATH and IndustryOR against OpenAI-o3 and DeepSeek-R1. The critical takeaway for us is the Partial KL strategy: it allows the model to 'think' freely outside the reference distribution while adhering to strict coding standards—a technique we should immediately test in AlgoEvo. Furthermore, their method of parsing .lp files to extract structural features (variable counts, constraint types) for 'instance-enhanced self-consistency' provides a much richer signal than our current binary success/failure metrics.

### [Autoformulation of Mathematical Optimization Models Using LLMs](https://arxiv.org/abs/2411.01679)

**2024-11-03** | University of Cambridge, University of Hawaii at Manoa | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-enhanced Monte-Carlo Tree Search with symbolic pruning and LLM-based evaluation | *LLM role:* conditional_hypothesis_generator, evaluator

> Astorga et al. frame optimization modeling as a hierarchical Monte-Carlo Tree Search (MCTS) problem, using LLMs to generate components and—crucially—employing SMT solvers to prune mathematically equivalent branches (e.g., recognizing `x+y` and `y+x` as identical). They achieve SOTA results on NL4OPT and IndustryOR, outperforming fine-tuned models like ORLM while using significantly fewer samples than naive approaches. **Key Takeaway:** The integration of symbolic equivalence checking (SMT) to prune the search tree is a technique we should immediately steal; implementing this in AlgoEvo would allow us to discard functionally identical code/math mutants before expensive evaluation, directly addressing our sample efficiency bottleneck.

### [A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions](https://arxiv.org/abs/2508.10047)

**2024-08-01** | Zhejiang University, Huawei Noah’s Ark Lab, Singapore University of Social Sciences, Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security | M=5 P=9 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* Systematic Literature Review and Empirical Re-evaluation | *LLM role:* evaluator

> This survey and empirical audit reveals that standard optimization modeling benchmarks (NL4Opt, IndustryOR) suffer from critical error rates ranging from 16% to 54%, rendering prior leaderboards unreliable. The authors manually cleaned these datasets and re-evaluated methods, finding that Chain-of-Thought (CoT) often degrades performance compared to standard prompting, while fine-tuned models (ORLM) and multi-agent systems (Chain-of-Experts) perform best. The immediate takeaway is that we must adopt their cleaned datasets for our OR-Bench project; using the original open-source versions is no longer defensible. Additionally, the failure of CoT on these tasks suggests we should prioritize multi-agent or fine-tuned approaches for symbolic formulation tasks.

### [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://arxiv.org/abs/2402.10172)

**2024-02-15** | Stanford University | M=7 P=8 I=8 **MUST-READ** *discuss*

*Method:* Modular LLM-based multi-agent system (OptiMUS) with connection graph | *LLM role:* multi-agent orchestration for model formulation, code generation, evaluation, and debugging

> OptiMUS is a multi-agent framework for translating natural language into Gurobi code, achieving SOTA performance by using a 'Connection Graph' to map variables and parameters to specific constraints. This graph allows the agents to dynamically filter context and construct minimal prompts, enabling success on problems with long descriptions where baselines like Chain-of-Experts fail. They release NLP4LP, a hard benchmark of 67 complex instances, which we must immediately compare against our OR-Bench efforts. The **Connection Graph** is the key stealable insight: a structured dependency tracking mechanism that solves context pollution in iterative code generation, directly applicable to our AlgoEvo and HERMES memory designs.

### [ProOPF: Benchmarking and Improving LLMs for Professional-Grade Power Systems Optimization Modeling](https://arxiv.org/abs/2602.03070)

**2026-02-03** |  | M=7 P=6 I=7 *discuss*

*Method:* LLM-based code synthesis for optimization modeling from natural language | *LLM role:* code_writer

> Shen et al. propose a benchmark (ProOPF) for translating natural language into Optimal Power Flow (OPF) models, treating instances as parametric or structural modifications to a canonical base model rather than generating code from scratch. They introduce a rigorous data synthesis pipeline using 'scenario trees' to map qualitative descriptions (e.g., 'heatwave') to quantitative parameter deltas, and define structural extensions (e.g., adding security constraints) as modular patches. Results are sobering: SOTA models (GPT-4, Claude 3.5) score 0% on the hardest level (semantic inference + structural change), though SFT recovers ~11-35%. **Key Takeaway:** We should steal their 'Base + Delta' synthesis approach for our VRP variant generation and OR-Bench work; it allows for scalable, physically valid data generation without requiring an LLM to hallucinate full solvers, and effectively benchmarks 'ambiguity' handling.

### [CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling](https://arxiv.org/abs/2510.04204)

**2025-10-05** | Qwen Team, Alibaba Inc., The Chinese University of Hong Kong, Shenzhen, Southern University of Science and Technology, Shanghai University of Finance and Economics, Shenzhen Loop Area Institute (SLAI) | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Corrective Adaptation with Lightweight Modification (CALM) framework with two-stage training (SFT + RL) | *LLM role:* generates optimization models and solver code, performs reflective reasoning, and receives corrective hints from an expert LLM (Intervener)

> Tang et al. propose CALM, a framework that uses an expert 'Intervener' model to inject corrective hints into a small LRM's reasoning trace (e.g., forcing it to use Python instead of manual calculation), followed by SFT and RL (GRPO). Results are strong and verified: a 4B model matches DeepSeek-R1 (671B) on OR benchmarks, specifically fixing the 'Code Utilization Distrust' we see in our own agents. The key takeaway is the 'Intervener' loop: instead of discarding failed traces, they repair them with hints to create a 'golden' reasoning dataset that preserves the 'thinking' process while enforcing tool use. This is a direct, actionable method for improving our AlgoEvo agents' reliability in generating executable heuristics without massive human annotation.

### [OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning](https://arxiv.org/abs/2511.09092)

**2025-11-12** | The Hong Kong University of Science and Technology, Arizona State University, University of North Carolina at Chapel Hill | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Supervised Fine-tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) with a composite reward function | *LLM role:* code_writer, heuristic_generator, evaluator

> OR-R1 introduces a data-efficient framework that fine-tunes Qwen3-8B using Supervised Fine-Tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) on unlabeled data. The results are empirically strong: it outperforms ORLM and LLMOPT while using only 1/10th of the synthetic training data, specifically narrowing the consistency gap between Pass@1 and Pass@8. The key takeaway for us is the effectiveness of GRPO (normalizing rewards within a sampled group to estimate baselines) combined with majority-voting rewards; this eliminates the need for a separate critic model while significantly improving code generation consistency. We should immediately evaluate GRPO as a lightweight alternative to PPO for the 'RL-infused' components of our evolutionary search methods.

### [OptiMind: Teaching LLMs to Think Like Optimization Experts](https://arxiv.org/abs/2509.22979)

**2025-09-26** | Microsoft Research, Stanford University, University of Washington | M=5 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Supervised fine-tuning (SFT) of a 20B-parameter LLM (GPT-OSS-20B variant) on a semi-automatically cleaned, class-specific error-analyzed training dataset, combined with error-aware prompting and multi-turn self-correction at inference. | *LLM role:* code_writer

> The authors fine-tune a 20B model for MILP formulation, but the critical contribution is a rigorous audit of standard benchmarks (IndustryOR, OptMATH), revealing that 30-50% of instances are flawed (missing data, wrong ground truth, infeasible). They introduce a 'class-based error analysis' where the model classifies a problem (e.g., TSP) and retrieves specific, expert-written hints to avoid common pitfalls, boosting accuracy by ~20%. **Takeaway:** We must immediately replace our benchmark versions with their cleaned sets for the OR-Bench project. Additionally, their library of 'error hints' per problem class is a high-value artifact we can scrape and inject into AlgoEvo's prompt templates to improve initial population quality.

### [OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents](https://arxiv.org/abs/2504.16918)

**2025-04-23** | University of Maryland at College Park | M=7 P=7 I=8 *discuss*

*Method:* LLM-powered multi-agent system (formulator, planner, coder, code critic, decider, verifier) with UCB-based debug scheduling for adaptive plan selection and iterative code refinement. | *LLM role:* decomposition_guide, code_writer, evaluator, evolutionary_search

> OptimAI introduces a multi-agent framework for translating natural language to optimization models, featuring a 'plan-before-code' stage and a novel **UCB-based debug scheduler**. Instead of linearly debugging a single solution, it treats debugging as a multi-armed bandit problem, dynamically allocating compute to different solution strategies based on a 'Decider' score and exploration term. While the combinatorial results (TSP a280) are trivial, the bandit mechanism is a highly effective heuristic for search control. We should steal this UCB scheduling logic for AlgoEvo to prevent agents from wasting tokens debugging fundamentally flawed heuristics.


### Front 1 (14 papers) — STABLE

**Density:** 0.33 | **Methods:** llm_code_generation, llm_in_the_loop, llm_as_heuristic, program_synthesis, constraint_programming | **Problems:** job_shop_scheduling, program_synthesis, scheduling, bin_packing, automated_constraint_programming_modeling

*Unique methods:* adaptive_algorithms, agentic_framework, aide, anytime_algorithm, auction_algorithm, bestofn_sampling, branch_and_bound, chain_of_experts, closed_loop_control, competitive_analysis, compositional_prompting, constraint_programming, constraint_programming_solver, cp_sat, cpmpy, crossover, cutting_planes, dirichlet_process_mixture_model, domain_specific_languages, eoh, evolutionary_algorithm, gaussian_process, global_constraints, greedy_algorithm, greedy_refinement, ipython_kernel, k_means_clustering, karp_reductions, knowledge_graphs, large_language_models, llm_agent, llm_as_code_generator, lower_bound_estimation, makespan_minimization, milp_general, minizinc, minizinc_modeling, model_context_protocol, mstc_ahd, multi_robot_task_allocation, mutation, non_parametric_modeling, nsga_ii, online_scheduling, ordered_eviction, parameter_efficient_fine_tuning, pushdown_automaton, qlora, react_framework, reevo, reflection_mechanism, repeated_sampling, retrieval_augmented_in_context_learning, self_verification, solution_majority_voting, tool_use_agents, vanilla_prompting
*Shared methods:* chain_of_thought, contrastive_learning, dynamic_programming, evolution_of_heuristics, expectation_maximization, funsearch, generative_models, in_context_learning, instruction_tuning, iterative_refinement, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, metaheuristics, milp_solver, multi_agent_llm_system, neuro_symbolic_ai, neurosymbolic_ai, program_synthesis, prompt_engineering, retrieval_augmented_generation, robust_optimization, self_correction, supervised_fine_tuning, supervised_learning, tool_use, tree_of_thoughts

This research front explores advanced applications of Large Language Models (LLMs) for the automated synthesis of formal optimization models and novel heuristics in Operations Research. A central theme is the development of specialized LLM frameworks, such as ConstraintLLM for industrial-level Constraint Programming (CP) modeling and EvoCut for generating MILP acceleration cuts. Other prominent approaches include agentic systems like GALA and CP-Agent for text-to-MiniZinc/CPMpy translation, and evolutionary frameworks like REMoH for multi-objective heuristic design, all leveraging LLMs to generate or refine OR-specific artifacts.

Key contributions include the ConstraintLLM framework (Paper 1), which fine-tunes a 32B LLM for Constraint Programming, achieving 51% accuracy on the IndusCP benchmark through constraint-aware retrieval and self-correction. For model generation, GALA (Paper 2) and CP-Agent (Paper 10) introduce agentic systems for text-to-MiniZinc/CPMpy translation, with CP-Agent claiming 100% accuracy on a clarified CP-Bench. In heuristic generation, HeuriGym (Paper 3) and CO-Bench (Paper 8) establish benchmarks, showing LLMs reach ~60% of expert performance, while LLaMoCo (Paper 9) demonstrates that fine-tuned 350M models can significantly outperform larger LLMs in generating specialized optimization code. Furthermore, EvoCut (Paper 11) leverages LLMs for generating MILP acceleration cuts, yielding 17-57% gap reductions on benchmarks like TSPLIB and JSSP, and EquivaMap (Paper 12) provides a robust method for verifying formulation equivalence with 100% accuracy.

This research front is rapidly maturing, characterized by a proliferation of specialized frameworks and benchmarks. The trajectory indicates a shift from rudimentary LLM code generation to sophisticated agentic and evolutionary systems that deeply integrate LLMs into the OR problem-solving pipeline. Future work will likely focus on enhancing the robustness and generalization capabilities of LLM-generated models and heuristics, particularly in handling strict feasibility constraints and scaling to larger, more complex industrial problems. We can expect the next wave of papers to integrate advanced verification techniques, optimize LLM interaction efficiency, and explore dynamic adaptation of intermediate representations or search strategies.

**Papers:**

### [ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming](https://arxiv.org/abs/2510.05774)

**2025-10-07** | University of Oxford, University of Chinese Academy of Sciences, Hangzhou Institute for Advanced Study, ISCAS, University of Science and Technology Beijing | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Neuro-Symbolic Framework integrating Multi-Instruction Supervised Fine-Tuning (SFT) of an open-source LLM, Constraint-Aware Retrieval Module (CARM), Tree-of-Thoughts (ToT) exploration, and Iterative Self-Correction with Guided Retrieval. | *LLM role:* code_writer

> ConstraintLLM fine-tunes a 32B model for Constraint Programming (CP) modeling, utilizing a "Constraint-Aware Retrieval Module" (CARM) that retrieves few-shot examples based on extracted constraint signatures (e.g., `AllDifferent`, `Cumulative`) rather than text embeddings. They also employ a Tree-of-Thoughts search pruned by test case execution and an iterative self-correction mechanism that retrieves "correction paths" (error-to-fix trajectories). Results are strong: on their new industrial benchmark (IndusCP), they achieve ~51% accuracy with a 32B model, matching or beating GPT-4o and DeepSeek-V3. **Key Takeaway:** The shift from semantic retrieval to *structural* retrieval (matching constraint profiles) is the "stealable" insight; we should implement this for our OR modeling tasks immediately, ignoring surface-level problem descriptions in favor of logical signatures. This directly impacts our OR-Bench and automated formulation work.

### [Gala: Global LLM Agents for Text-to-Model Translation](https://arxiv.org/abs/2509.08970)

**2025-09-10** | University of Southern California, Brown University, Fidelity Investments | M=5 P=8 I=6 *discuss*

*Method:* Multi-agent LLM framework for global constraint detection and assembly | *LLM role:* code_writer

> GALA decomposes text-to-MiniZinc translation into a multi-agent system where specialized agents detect specific Constraint Programming global constraints (e.g., all_different, cumulative) before an assembler unifies them. Results on 110 TEXT2ZINC instances show a modest improvement over CoT (57% vs 52% execution rate with o3-mini), though the sample size is small and lacks statistical rigor. The key takeaway is the architectural shift from generic 'coder/reviewer' roles to 'primitive-specific' agents, which aligns LLM reasoning with the target formalism's structure. We should test this 'primitive-based decomposition' in our OR-Bench pipeline to see if it reduces hallucination of complex constraints better than our current methods.

### [HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization](https://arxiv.org/abs/2506.07972)

**2025-06-09** | Cornell University, Harvard University, NVIDIA | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Agentic framework for evaluating and iteratively refining LLM-generated heuristic algorithms via code execution feedback | *LLM role:* heuristic_generator

> The authors introduce HeuriGym, a benchmark suite of 9 hard combinatorial optimization problems (including PDPTW, EDA scheduling, and routing) coupled with an agentic evaluation loop. Results are backed by extensive experiments showing that SOTA LLMs saturate at ~60% of expert performance and, significantly, that existing evolutionary frameworks (ReEvo, EoH) perform *worse* than simple prompting on these large-context tasks (300+ lines of code). The key takeaway is the failure mode of current evolutionary methods: they cannot handle the context fragmentation and feedback integration required for complex heuristic design. We should immediately adopt this benchmark to demonstrate AlgoEvo's superiority, as the current baselines are weak and the problem set aligns perfectly with our focus.

### [Adaptively Robust LLM Inference Optimization under Prediction Uncertainty](https://arxiv.org/abs/2508.14544)

**2025-08-20** | Stanford University, Peking University, HKUST | M=7 P=9 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* Adaptive online scheduling with dynamic lower bound estimation, ordered eviction, and greedy batch formation (Amin algorithm) | *LLM role:* none

> Chen et al. propose $A_{min}$, an online scheduling algorithm for LLM inference that handles unknown output lengths by optimistically assuming the lower bound and evicting jobs (based on accumulated length) if memory overflows. They prove a logarithmic competitive ratio and show via simulations on LMSYS-Chat-1M that this approach nearly matches hindsight-optimal scheduling, vastly outperforming conservative upper-bound baselines. **Key Takeaway:** For our **GPUSched** project, we should abandon conservative memory reservation for output tokens; instead, implement an optimistic scheduler that oversubscribes memory and handles overflows via their ordered eviction policy, as the cost of restart is theoretically bounded and empirically negligible compared to the throughput gains.

### [Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc](https://arxiv.org/abs/2503.10642)

**2025-02-22** | Brown University, Fidelity Investments | M=3 P=8 I=4 **MUST-READ** *discuss*

*Method:* LLM-based MiniZinc model generation using various prompting strategies (Vanilla, Chain-of-Thought, Compositional) | *LLM role:* code_writer

> Singirikonda et al. introduce TEXT2ZINC, a dataset of 110 Natural Language-to-MiniZinc problems, and benchmark GPT-4 using Vanilla, CoT, and Compositional prompting. Their results are poor (max ~25% solution accuracy), confirming that off-the-shelf LLMs struggle significantly with MiniZinc syntax and logical translation. Crucially, they attempt using Knowledge Graphs as an intermediate representation, but report that it actually *reduced* solution accuracy compared to basic CoT—a valuable negative result for our symbolic modeling work. We should examine their dataset for inclusion in OR-Bench, but their prompting methods are rudimentary baselines we should easily outperform.

### [Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation](https://arxiv.org/abs/2510.02679)

**2025-10-03** | Peking University, The Hong Kong University of Science and Technology, Huazhong University of Science and Technology, University of Science and Technology of China | M=7 P=6 I=7 *discuss*

*Method:* Constraint-centric architecture regulating LLMs with Domain-Specific Languages (DSLs) and an automated DSL adaptation algorithm | *LLM role:* constraint_generator

> This paper proposes a constraint-centric architecture that translates natural language manufacturing descriptions into Job Shop Scheduling (JSP) constraints by mediating through a learned Domain-Specific Language (DSL). Unlike standard prompting, they implement an automated DSL adaptation algorithm using non-parametric modeling (DPMM) and Expectation-Maximization to learn the syntax and semantics of the intermediate representation from data, which is then verified via a Pushdown Automaton. While the experiments rely on synthetic data augmented from standard benchmarks (a weakness), the methodology for **automatically deriving the intermediate representation** rather than hand-coding it is a transferable insight. We could steal this 'automated DSL design' approach to dynamically construct search spaces for AlgoEvo or to improve the robustness of NL-to-OR translation in OR-Bench.

### [DCP-Bench-Open: Evaluating LLMs for Constraint Modelling of Discrete Combinatorial Problems](https://arxiv.org/abs/2506.06052)

**2025-06-06** | KU Leuven, University of Western Macedonia | M=5 P=8 I=7 *changes-thinking* *discuss*

*Method:* LLM-driven constraint model generation | *LLM role:* code_writer, decomposition_guide, evaluator

> This paper introduces DCP-Bench-Open, a benchmark of 164 discrete combinatorial problems, to evaluate LLMs on translating natural language into constraint models (CPMpy, MiniZinc, OR-Tools). The results are rigorous and highlight a critical failure mode: LLMs overfit to the specific data values in the prompt's example instance, causing a ~30% performance drop when evaluated on hidden instances (Multi-Instance Accuracy). Crucially for our pipeline design, they find that Retrieval-Augmented In-Context Learning (RAICL) is ineffective or harmful compared to simply including library documentation in the system prompt. We should adopt their 'Multi-Instance Accuracy' metric immediately for OR-Bench and switch any MiniZinc generation efforts to Python-based frameworks like CPMpy or OR-Tools, which LLMs handle much better.

### [CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization](https://arxiv.org/abs/2504.04310)

**2025-04-06** | Carnegie Mellon University | M=4 P=9 I=7 **MUST-READ** *discuss*

*Method:* LLM-based algorithm search using agentic frameworks with iterative refinement and evolutionary search | *LLM role:* evolutionary_search

> Sun et al. introduce CO-Bench, a suite of 36 diverse combinatorial optimization problems (packing, scheduling, routing) designed specifically to benchmark LLM agents in generating algorithms (code), not just solutions. They evaluate 9 frameworks (including FunSearch, ReEvo, AIDE), finding that FunSearch combined with reasoning models (o3-mini) yields the most robust performance, though agents still struggle significantly with strict feasibility constraints (valid solution rates often <60%). **Takeaway:** We should immediately integrate CO-Bench into our pipeline to benchmark AlgoEvo against ReEvo and FunSearch; this saves us months of data curation and provides a standardized metric to prove our method's superiority.

### [LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation](https://arxiv.org/abs/2403.01131)

**2024-03-02** | Singapore Management University, Nanyang Technological University, South China University of Technology | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Instruction tuning of LLMs with a two-phase learning strategy incorporating contrastive learning-based warm-up and sequence-to-sequence loss | *LLM role:* code_writer

> LLaMoCo fine-tunes small LLMs (down to 350M) to generate executable Python optimization code by training on a synthetic dataset where the 'ground truth' is the empirically best-performing solver identified via exhaustive benchmarking. The results are compelling: the fine-tuned 350M model achieves ~85% normalized performance on benchmarks where GPT-4 Turbo only reaches ~14-30%, largely because the small model learns to select specialized evolutionary strategies (like BIPOP-CMA-ES) while GPT-4 defaults to generic gradient-based solvers. **Key Takeaway:** We can replace the expensive GPT-4 calls in our evolutionary search loop with a specialized, fine-tuned local model (CodeLlama-7B) trained on our historical search successes, significantly improving both sample efficiency and scalability. The paper's 'contrastive warm-up' strategy for aligning diverse problem descriptions is also a transferable technique for our problem encoding work.

### [CP-Agent: Agentic Constraint Programming](https://arxiv.org/abs/2508.07468)

**2025-08-10** | TU Wien | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Agentic Python coding agent using ReAct framework with persistent IPython kernel for iterative refinement of CPMpy constraint models | *LLM role:* code_writer

> Szeider implements a standard ReAct agent with a persistent IPython kernel to iteratively generate and refine CPMpy models, claiming 100% accuracy on CP-Bench. However, this perfect score is achieved on a *modified* version of the benchmark where the author manually fixed 31 ambiguous problem statements and 19 ground-truth errors—making the '100%' result an artifact of dataset cleaning rather than pure model capability. The most actionable takeaways are the negative result for explicit 'task management' tools (which hurt performance on hard problems) and the effectiveness of a minimal (<50 lines) domain prompt over complex scaffolding. We should review their clarified benchmark for our OR-Bench work.

### [EvoCut: Strengthening Integer Programs via Evolution-Guided Language Models](https://arxiv.org/abs/2508.11850)

**2025-08-16** | Huawei Technologies Canada, University of British Columbia, University of Toronto | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary algorithm powered by multiple LLM-based agents for iterative generation and refinement of acceleration cuts | *LLM role:* heuristic_generator

> Yazdani et al. introduce EvoCut, an evolutionary framework where LLMs generate Python code for MILP cuts, filtered by a 'usefulness check' (does it cut the current LP relaxation?) and an 'empirical validity check' (does it preserve known integer optima?). They report 17-57% gap reductions on TSPLIB and JSSP compared to Gurobi defaults, backed by strong ablation studies on the evolutionary operators. **Key Takeaway:** The reliance on 'acceleration cuts'—constraints verified empirically on small datasets rather than formally proven—bypasses the bottleneck of automated theorem proving while still delivering valid speedups. We should immediately adopt their 'LP separation' check as a cheap, high-signal reward for our own evolutionary search loops.

### [EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](https://arxiv.org/abs/2502.14760)

**2025-02-20** | Stanford University, The University of Texas at Austin | M=7 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based discovery of linear mapping functions between decision variables, followed by MILP solver-based verification of feasibility and optimality | *LLM role:* heuristic_generator

> Zhai et al. propose EquivaMap, a framework that evaluates whether two MILP formulations are equivalent by using an LLM to discover a linear mapping between their decision variables, which is then rigorously verified by a solver. Unlike 'execution accuracy' (which fails on unit scaling) or 'canonical accuracy' (which fails on variable permutation), they achieve 100% accuracy on a new dataset of equivalent formulations including cuts and slack variables. The core insight is replacing output comparison with a 'propose-mapping-and-verify' loop, effectively using the LLM to construct a proof of equivalence. We must adopt this methodology for the OR-Bench evaluation pipeline immediately, as it eliminates the false negatives currently plaguing our generation benchmarks.

### [FLEET: Formal Language-Grounded Scheduling for Heterogeneous Robot Teams](https://arxiv.org/abs/2510.07417)

**2025-10-08** | JHU APL, JHU, DEVCOM ARL | M=5 P=7 I=6 *discuss*

*Method:* Hybrid generative–formal framework combining LLM-based task decomposition and fitness estimation with Mixed-Integer Linear Programming (MILP) for makespan minimization. | *LLM role:* decomposition_guide

> FLEET implements a hybrid pipeline where an LLM extracts a task dependency graph and a 'fitness matrix' (capability scores) from natural language, which then populate a standard MILP for multi-robot scheduling. Results on the PARTNR benchmark show it outperforms pure LLM planners (SMART-LLM) by ~7% on heterogeneous tasks, though overall gains are modest. The actionable takeaway is the **fitness matrix extraction**: using the LLM to generate dense cost coefficients ($c_{ij}$) for the optimization model rather than just binary constraints. We should adopt this technique for handling soft semantic preferences in our heterogeneous VRP formulations.

### [REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models](https://arxiv.org/abs/2506.07759)

**2025-06-09** | Vicomtech Foundation, University of the Basque Country, Universidad EAFIT, HiTZ Basque Center for Language Technology | M=7 P=6 I=7 **MUST-READ** *discuss*

*Method:* Hybrid framework integrating NSGA-II with LLM-based heuristic generation and a reflection mechanism | *LLM role:* evolutionary_search

> Forniés-Tabuenca et al. propose REMoH, an LLM-driven evolutionary framework for multi-objective FJSSP that uses K-Means to cluster the population by objective performance before generating reflections. While their optimality gaps (~12%) trail behind state-of-the-art CP solvers (~1.5%), the ablation study confirms that their reflection mechanism significantly improves Pareto front diversity (Hypervolume). **The killer feature is the phenotypic clustering step:** instead of reflecting on a random or elitist subset, they group solutions by trade-offs (e.g., 'low makespan' vs 'balanced') to generate targeted prompts. We should implement this clustering-based context construction in AlgoEvo to improve diversity maintenance in multi-objective search without exploding token costs.



## Bridge Papers

Papers connecting multiple research fronts:

### [StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models](https://arxiv.org/abs/2509.22558)

**TRUE SYNTHESIS** | score=0.85 | Front 2 → Front 0, Front 1

> Zhou et al. propose StepORLM, a framework where an 8B policy and a **Generative Process Reward Model (GenPRM)** co-evolve. Unlike standard discriminative PRMs that score steps in isolation, their GenP

### [EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](https://arxiv.org/abs/2502.14760)

**TRUE SYNTHESIS** | score=0.83 | Front 1 → Front 0, Front 2

> Zhai et al. propose EquivaMap, a framework that evaluates whether two MILP formulations are equivalent by using an LLM to discover a linear mapping between their decision variables, which is then rigo

### [BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](https://arxiv.org/abs/2411.17404)

**TRUE SYNTHESIS** | score=0.82 | Front 2 → Front 0, Front 1

> Wang et al. propose BPP-Search, combining Beam Search, a Process Reward Model (PRM), and a final Pairwise Preference Model to generate LP/MIP models from natural language. While their new 'StructuredO

### [LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation](https://arxiv.org/abs/2403.01131)

**TRUE SYNTHESIS** | score=0.82 | Front 1 → Front 0, Front 2

> LLaMoCo fine-tunes small LLMs (down to 350M) to generate executable Python optimization code by training on a synthetic dataset where the 'ground truth' is the empirically best-performing solver ident

### [Auto-Formulating Dynamic Programming Problems with Large Language Models](https://arxiv.org/abs/2507.11737)

**TRUE SYNTHESIS** | score=0.78 | Front 2 → Front 0, Front 1

> Zhou et al. introduce DPLM, a 7B model fine-tuned to formulate Dynamic Programming models, achieving performance comparable to o1 on their new DP-Bench. Their key contribution is 'DualReflect,' a synt


---

*Generated by Research Intelligence System*
