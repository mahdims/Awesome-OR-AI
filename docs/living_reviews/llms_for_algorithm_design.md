# Living Review: LLMs for Algorithm Design

**Last Updated:** 2026-04-14

---

## Recent Papers

#### 2026-04-12 (1 papers)

### [RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets](https://arxiv.org/abs/2604.04347)

**2026-04-06** | Independent Researchers | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Elo tournament selection for evolutionary optimization with comparative error reports and Deep Focus refinement | *LLM role:* evolutionary_search

> RoboPhD optimizes LLM agent evolution under tight evaluation budgets by replacing traditional validation sets with an Elo-based tournament on training data and allowing agents to evolve their own diagnostic instrumentation. The results are backed by solid empirical comparisons, outperforming GEPA and Autoresearch on 3 out of 4 benchmarks (ARC-AGI, Text2SQL, DocFinQA) under a strict 1,500 evaluation budget. The single most useful takeaway we can steal is 'self-instrumenting agents'—seeding the initial agent with print() statements and letting the evolutionary process grow its own logging to provide richer Actionable Side Information (ASI) to the LLM optimizer. This paper matters immensely for AlgoEvo; we should immediately test dropping our validation splits in favor of Elo tracking and implement self-instrumenting diagnostics to improve our evolutionary signal without increasing API costs.


#### 2026-04-02 (3 papers)

### [CliffSearch: Structured Agentic Co-Evolution over Theory and Code for Scientific Algorithm Discovery](https://arxiv.org/abs/2604.01210)

**2026-04-01** | IBM Research | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-agent-instantiated evolutionary computation over structured scientific artifacts (theory+code or code_only) | *LLM role:* evolutionary_search

> CliffSearch is an LLM-based evolutionary framework that co-evolves algorithm theory and code, using specialized agents for crossover, two-path mutation (exploration vs. repair), and explicit reviewer gating. The results are backed by concrete empirical runs on nanoGPT optimizer discovery and transformer hyper-connection search, demonstrating the discovery of genuinely novel geometric routing and optimizer variants rather than trivial hyperparameter tweaks. The single most useful takeaway is the 'reviewer-gated selection' where an LLM explicitly scores candidates on originality and correctness as a hard survival gate before benchmark scores are considered. This is highly relevant for our AlgoEvo project; we should immediately steal the two-path mutation (novelty vs repair) and the originality hard-gate to prevent our populations from converging on unoriginal benchmark-hacking.

### [COvolve: Adversarial Co-Evolution of Large-Language-Model-Generated Policies and Environments via Two-Player Zero-Sum Game](https://arxiv.org/abs/2603.28386)

**2026-03-30** | Örebro University | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Co-evolutionary framework leveraging LLMs to generate environments and policies as executable Python code, modeled as a two-player zero-sum game, and solved using Policy Space Response Oracles (PSRO) to compute a mixed-strategy Nash equilibrium (MSNE) over policy populations. | *LLM role:* llm_evolutionary_agent

> COvolve uses LLMs to adversarially co-evolve Python code for both environments (tasks) and policies (agents), using Policy Space Response Oracles (PSRO) to compute a mixed-strategy Nash equilibrium (MSNE) that prevents catastrophic forgetting. The results are backed by solid empirical data across MiniGrid, PyGame, and CARLA, demonstrating that the MSNE approach maintains robust performance across a growing historical archive of environments much better than greedy retention. WHAT WE LEARNED: The use of an empirical payoff matrix and MSNE to evaluate new code against an archive of past opponents/environments is a powerful memory and evaluation mechanism for LLM evolution. This matters immensely for us: we can steal this exact architecture for AlgoEvo or EvoCut by co-evolving hard OR instances (environments) alongside our solver heuristics (policies), using MSNE to ensure our heuristics do not overfit to specific instance types.

### [Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models](https://arxiv.org/abs/2603.28416)

**2026-03-30** | Machine Perception and Interaction Lab, Örebro University, Sweden | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Evolutionary search over executable learning update rules using LLM-guided macro mutation and diversity-aware crossover | *LLM role:* evolutionary_search

> This paper evolves executable reinforcement learning update rules using LLMs as macro-mutation and crossover operators, explicitly forbidding standard RL mechanisms to force the discovery of novel algorithms. The results are backed by solid empirical evaluations on Gymnasium benchmarks, showing the evolved algorithms match or beat standard baselines like PPO and SAC on several tasks, though they struggle on a few complex continuous control environments. The single most useful takeaway for us is their diversity-aware crossover, which uses normalized Levenshtein distance to penalize recombining near-duplicate parents, alongside a post-evolution step where the LLM proposes bounds for internal scalar parameters before a final sweep. We should immediately test the Levenshtein-penalized crossover in AlgoEvo to prevent diversity collapse, and adopt the LLM-bounded HPO step to ensure we aren't discarding good heuristics simply because of bad default scalar parameters.


#### 2026-03-22 (2 papers)

### [CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad](https://arxiv.org/abs/2603.14575)

**2026-03-15** | Carnegie Mellon University, MBZUAI, Hong Kong Baptist University, The University of Sydney | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* CausalEvolve with causal scratchpad leveraging LLMs to identify and reason about guiding factors for evolution, incorporating outcome-level and procedure-level factors, multi-arm bandit for intervention, and abductive reasoning. | *LLM role:* heuristic_generator

> CausalEvolve enhances LLM evolutionary search frameworks (like AlphaEvolve and ShinkaEvolve) by introducing a causal scratchpad that extracts outcome- and procedure-level factors to explicitly guide program mutations via a Multi-Armed Bandit. The results are backed by solid empirical numbers, showing it outperforms the state-of-the-art ShinkaEvolve across four algorithmic and mathematical tasks (Hadamard, Autocorrelation, Circle Packing, AIME) by up to 9.1% in best-found scores. The single most useful takeaway for us is their 'surprise detection' module: using LLMs to perform abductive reasoning when a seemingly good combination of factors yields a score drop, thereby uncovering hidden confounders and generating new search directions. This is highly relevant to our work; we should immediately evaluate their MAB-driven causal intervention strategy and procedure-level factor extraction to improve the sample efficiency and memory mechanisms in AlgoEvo.

### [Procedural Generation of Algorithm Discovery Tasks in Machine Learning](https://arxiv.org/abs/2603.17863)

**2026-03-18** | University of Oxford, University College London, University of California, Santa Barbara, University of Wisconsin–Madison, Delft University of Technology | M=6 P=8 I=8 **MUST-READ** *discuss*

*Method:* Procedural generation of algorithm discovery tasks using configurable parameters for domains, modules, and datasets | *LLM role:* research_agent, prompt_optimizer

> This paper introduces DiscoGen, a procedural generator that combinatorially creates millions of algorithm discovery tasks (varying domains, editable modules, and datasets) with strict meta-train/meta-test splits to evaluate and train Algorithm Discovery Agents (ADAs). The results are backed by extensive empirical evaluation of open-source LLMs on a fixed subset (DiscoBench), demonstrating that current ADAs struggle with multi-module discovery and that prompt-tuning over a diverse set of procedurally generated tasks significantly improves generalization. The single most useful takeaway is the combinatorial task generation approach (toggling which modules are editable vs. fixed), which provides a brilliant blueprint for creating an autocurriculum to train our 'evolver' agents. This matters immensely for us; we should immediately consider using DiscoGen to evaluate AlgoEvo, and adapt their procedural task generation strategy to create diverse OR/routing environments for our own RL-infused evolutionary search training.


#### 2026-03-19 (4 papers)

### [CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad](https://arxiv.org/abs/2603.14575)

**2026-03-15** | Carnegie Mellon University, MBZUAI, Hong Kong Baptist University, The University of Sydney | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* CausalEvolve with causal scratchpad leveraging LLMs to identify and reason about guiding factors for evolution, incorporating outcome-level and procedure-level factors, multi-arm bandit for intervention, and abductive reasoning. | *LLM role:* heuristic_generator

> CausalEvolve enhances LLM evolutionary search frameworks (like AlphaEvolve and ShinkaEvolve) by introducing a causal scratchpad that extracts outcome- and procedure-level factors to explicitly guide program mutations via a Multi-Armed Bandit. The results are backed by solid empirical numbers, showing it outperforms the state-of-the-art ShinkaEvolve across four algorithmic and mathematical tasks (Hadamard, Autocorrelation, Circle Packing, AIME) by up to 9.1% in best-found scores. The single most useful takeaway for us is their 'surprise detection' module: using LLMs to perform abductive reasoning when a seemingly good combination of factors yields a score drop, thereby uncovering hidden confounders and generating new search directions. This is highly relevant to our work; we should immediately evaluate their MAB-driven causal intervention strategy and procedure-level factor extraction to improve the sample efficiency and memory mechanisms in AlgoEvo.

### [CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges](https://arxiv.org/abs/2603.11863)

**2026-03-12** | Tsinghua University, Peking University, Southern University of Science and Technology, University of Bristol, The Hong Kong University of Science and Technology (Guangzhou), Xi’an Jiaotong University | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Automated benchmark construction via reverse engineering and self-play; EvoRePE: Inference-time representation engineering for latent space steering. | *LLM role:* code_writer, constraint_generator, evaluator, prompt_optimizer, decomposition_guide, evolutionary_search

> Wang et al. introduce CreativeBench to evaluate LLM code generation creativity and propose EvoRePE, a representation engineering technique that extracts a 'creativity vector' from AlphaEvolve search trajectories to steer model activations at inference time. The results are backed by solid empirical evaluations, showing that injecting this vector improves novelty and correctness even without running the full evolutionary search. THE SINGLE MOST USEFUL TAKEAWAY: We can run our evolutionary search (e.g., AlgoEvo) offline to collect (base_heuristic, evolved_heuristic) pairs, compute the PCA of their hidden state differences, and inject this vector during standard inference to force the model into an exploratory mode. This is highly relevant to our work as it offers a completely new, training-free mechanism to solve the sample efficiency and scalability bottlenecks in LLM evolutionary search.

### [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440)

**2026-03-12** | Intel Corporation | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* MAP-Elites quality-diversity search with kernel-specific behavioral dimensions, meta-prompt evolution, and template-based parameter optimization | *LLM role:* code_writer, prompt_optimizer

> KernelFoundry is an LLM-based evolutionary framework for GPU kernel optimization that combines MAP-Elites quality-diversity search with meta-prompt co-evolution and gradient-informed mutation hints. The results are rigorously backed by numbers, showing a 2.1x speedup over the AI CUDA Engineer baseline on KernelBench L2 and successful optimization of Llama 3 operations. The single most useful takeaway for us is their 'Gradient-Informed Evolution' technique: they track parent-to-child fitness transitions in the MAP-Elites archive to compute pseudo-gradients across behavioral dimensions, which are then translated into specific natural language mutation hints for the LLM (e.g., 'positive gradient in memory -> hint: add shared memory tiling'). While we do not write low-level GPU kernels, this exact architectural improvement—alongside their meta-prompting to prevent context pollution—is highly transferable and should be immediately tested in AlgoEvo and EvoCut to improve our search signal and sample efficiency.

### [Procedural Generation of Algorithm Discovery Tasks in Machine Learning](https://arxiv.org/abs/2603.17863)

**2026-03-18** | University of Oxford, University College London, University of California, Santa Barbara, University of Wisconsin–Madison, Delft University of Technology | M=6 P=8 I=8 **MUST-READ** *discuss*

*Method:* Procedural generation of algorithm discovery tasks using configurable parameters for domains, modules, and datasets | *LLM role:* research_agent, prompt_optimizer

> This paper introduces DiscoGen, a procedural generator that combinatorially creates millions of algorithm discovery tasks (varying domains, editable modules, and datasets) with strict meta-train/meta-test splits to evaluate and train Algorithm Discovery Agents (ADAs). The results are backed by extensive empirical evaluation of open-source LLMs on a fixed subset (DiscoBench), demonstrating that current ADAs struggle with multi-module discovery and that prompt-tuning over a diverse set of procedurally generated tasks significantly improves generalization. The single most useful takeaway is the combinatorial task generation approach (toggling which modules are editable vs. fixed), which provides a brilliant blueprint for creating an autocurriculum to train our 'evolver' agents. This matters immensely for us; we should immediately consider using DiscoGen to evaluate AlgoEvo, and adapt their procedural task generation strategy to create diverse OR/routing environments for our own RL-infused evolutionary search training.


#### 2026-03-15 (4 papers)

### [CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges](https://arxiv.org/abs/2603.11863)

**2026-03-12** | Tsinghua University, Peking University, Southern University of Science and Technology, University of Bristol, The Hong Kong University of Science and Technology (Guangzhou), Xi’an Jiaotong University | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Automated benchmark construction via reverse engineering and self-play; EvoRePE: Inference-time representation engineering for latent space steering. | *LLM role:* code_writer, constraint_generator, evaluator, prompt_optimizer, decomposition_guide, evolutionary_search

> Wang et al. introduce CreativeBench to evaluate LLM code generation creativity and propose EvoRePE, a representation engineering technique that extracts a 'creativity vector' from AlphaEvolve search trajectories to steer model activations at inference time. The results are backed by solid empirical evaluations, showing that injecting this vector improves novelty and correctness even without running the full evolutionary search. THE SINGLE MOST USEFUL TAKEAWAY: We can run our evolutionary search (e.g., AlgoEvo) offline to collect (base_heuristic, evolved_heuristic) pairs, compute the PCA of their hidden state differences, and inject this vector during standard inference to force the model into an exploratory mode. This is highly relevant to our work as it offers a completely new, training-free mechanism to solve the sample efficiency and scalability bottlenecks in LLM evolutionary search.

### [Reinforced Generation of Combinatorial Structures: Ramsey Numbers](https://arxiv.org/abs/2603.09172)

**2026-03-11** | Google DeepMind, Google, University of California, Berkeley | M=8 P=3 I=9 **MUST-READ** *discuss*

*Method:* AlphaEvolve, an LLM-based code mutation agent | *LLM role:* evolutionary_search

> Nagda et al. (DeepMind) apply the AlphaEvolve framework to discover novel stochastic search algorithms that improve lower bounds for five classical Ramsey numbers and match SoTA on 23 others. The results are mathematically verified and represent genuine SoTA advances in extremal combinatorics, proving the framework's capability to generate highly specialized, non-trivial heuristics. The single most useful takeaway for us is their meta-algorithm's scoring function: instead of only rewarding valid states, they evaluate a larger, infeasible 'prospect' state and provide a dense, continuous reward based on its violation count relative to a random baseline. We should immediately steal this 'prospect evaluation' trick for AlgoEvo to smooth the reward landscape when evolving heuristics for highly constrained OR problems like VRP, where finding strictly feasible intermediate solutions is a bottleneck.

### [Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs](https://arxiv.org/abs/2603.07970)

**2026-03-09** | Nanjing University, Huawei Noah’s Ark Lab | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Stagewise Algorithm Design (EvoStage) with multi-agent system and global-local perspective mechanism | *LLM role:* decomposition_guide, code_writer, reflection_agent, evolutionary_search

> EvoStage enhances LLM-based automated algorithm design by decomposing the generation process into sequential stages, using a multi-agent system (coordinator and coders) to iteratively refine code based on real-time intermediate execution feedback. The results are highly credible and backed by strong empirical numbers; it achieves state-of-the-art HPWL on 16 chip placement benchmarks and beats AlphaEvolve/EoH on Bayesian Optimization tasks using an incredibly small budget of just 9 to 25 evaluations. The single most useful takeaway is the shift from black-box end-to-end evaluation to stagewise intermediate feedback, where a coordinator agent reflects on mid-execution metrics to guide the next stage of heuristic design. This matters immensely for our AlgoEvo and MASPRM projects; we should immediately test pausing our VRP/scheduling environments mid-execution to feed intermediate state metrics to a coordinator LLM, which could drastically reduce the number of LLM samples we need to find optimal heuristics.

### [Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models](https://arxiv.org/abs/2603.10098)

**2026-03-10** | Google DeepMind | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Policy-Space Response Oracles (PSRO) with Large Language Model (LLM) as a code-generating oracle, enhanced by iterative refinement or evolutionary search (AlphaEvolve) | *LLM role:* code_writer

> This paper replaces the deep RL oracle in Policy-Space Response Oracles (PSRO) with an LLM that generates interpretable Python code policies, using AlphaEvolve to iteratively refine the code against opponent meta-strategies. The results are backed by solid empirical metrics, showing that the AlphaEvolve variant achieves competitive exploitability and higher population returns than RL baselines (IMPALA) and CFR+ on Repeated Rock-Paper-Scissors and Leduc Poker. The single most useful takeaway for us is their 'context abstraction' technique—using an LLM to summarize opponent code into natural language to bypass context window limits during evolutionary search. This is highly relevant for our AlgoEvo project; we should immediately discuss implementing their two-level loop (outer meta-game equilibrium, inner AlphaEvolve refinement) and context abstraction for our multi-agent evolutionary search.


#### 2026-03-12 (4 papers)

### [Reinforced Generation of Combinatorial Structures: Ramsey Numbers](https://arxiv.org/abs/2603.09172)

**2026-03-11** | Google DeepMind, Google, University of California, Berkeley | M=8 P=3 I=9 **MUST-READ** *discuss*

*Method:* AlphaEvolve, an LLM-based code mutation agent | *LLM role:* evolutionary_search

> Nagda et al. (DeepMind) apply the AlphaEvolve framework to discover novel stochastic search algorithms that improve lower bounds for five classical Ramsey numbers and match SoTA on 23 others. The results are mathematically verified and represent genuine SoTA advances in extremal combinatorics, proving the framework's capability to generate highly specialized, non-trivial heuristics. The single most useful takeaway for us is their meta-algorithm's scoring function: instead of only rewarding valid states, they evaluate a larger, infeasible 'prospect' state and provide a dense, continuous reward based on its violation count relative to a random baseline. We should immediately steal this 'prospect evaluation' trick for AlgoEvo to smooth the reward landscape when evolving heuristics for highly constrained OR problems like VRP, where finding strictly feasible intermediate solutions is a bottleneck.

### [Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs](https://arxiv.org/abs/2603.07970)

**2026-03-09** | Nanjing University, Huawei Noah’s Ark Lab | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Stagewise Algorithm Design (EvoStage) with multi-agent system and global-local perspective mechanism | *LLM role:* decomposition_guide, code_writer, reflection_agent, evolutionary_search

> EvoStage enhances LLM-based automated algorithm design by decomposing the generation process into sequential stages, using a multi-agent system (coordinator and coders) to iteratively refine code based on real-time intermediate execution feedback. The results are highly credible and backed by strong empirical numbers; it achieves state-of-the-art HPWL on 16 chip placement benchmarks and beats AlphaEvolve/EoH on Bayesian Optimization tasks using an incredibly small budget of just 9 to 25 evaluations. The single most useful takeaway is the shift from black-box end-to-end evaluation to stagewise intermediate feedback, where a coordinator agent reflects on mid-execution metrics to guide the next stage of heuristic design. This matters immensely for our AlgoEvo and MASPRM projects; we should immediately test pausing our VRP/scheduling environments mid-execution to feed intermediate state metrics to a coordinator LLM, which could drastically reduce the number of LLM samples we need to find optimal heuristics.

### [Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models](https://arxiv.org/abs/2603.10098)

**2026-03-10** | Google DeepMind | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Policy-Space Response Oracles (PSRO) with Large Language Model (LLM) as a code-generating oracle, enhanced by iterative refinement or evolutionary search (AlphaEvolve) | *LLM role:* code_writer

> This paper replaces the deep RL oracle in Policy-Space Response Oracles (PSRO) with an LLM that generates interpretable Python code policies, using AlphaEvolve to iteratively refine the code against opponent meta-strategies. The results are backed by solid empirical metrics, showing that the AlphaEvolve variant achieves competitive exploitability and higher population returns than RL baselines (IMPALA) and CFR+ on Repeated Rock-Paper-Scissors and Leduc Poker. The single most useful takeaway for us is their 'context abstraction' technique—using an LLM to summarize opponent code into natural language to bypass context window limits during evolutionary search. This is highly relevant for our AlgoEvo project; we should immediately discuss implementing their two-level loop (outer meta-game equilibrium, inner AlphaEvolve refinement) and context abstraction for our multi-agent evolutionary search.

### [Autonomous Algorithm Discovery for Ptychography via Evolutionary LLM Reasoning](https://arxiv.org/abs/2603.05696)

**2026-03-05** | Argonne National Laboratory, Rice University | M=7 P=3 I=8 *discuss*

*Method:* LLM-guided evolutionary search for regularization algorithms combining LLM-driven code generation with semantically-guided crossover and mutation | *LLM role:* evolutionary_search

> This paper applies LLM-guided evolutionary search (similar to FunSearch/AlphaEvolve) to discover novel regularization algorithms for ptychographic image reconstruction. The results are backed by solid empirical metrics, showing up to +0.26 SSIM improvements over unregularized baselines across multiple datasets. The most useful takeaway for us is twofold: first, their 'semantically-guided crossover' explicitly prompts the LLM to analyze two successful parent algorithms and intentionally merge their complementary mathematical strengths, rather than blindly recombining code. Second, the LLM autonomously discovered the benefit of embedding stateful optimizers (like Adam) and iterative sub-loops directly inside a single heuristic step. This matters for our AlgoEvo and VRP work because we should immediately steal this semantic crossover prompting strategy and ensure our evaluation API permits the LLM to generate stateful, multi-pass operators rather than just stateless, single-pass functions.


#### 2026-03-08 (2 papers)

### [Rethinking Code Similarity for Automated Algorithm Design with LLMs](https://arxiv.org/abs/2603.02787)

**2026-03-03** | City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* BehaveSim, a method for measuring algorithmic similarity based on problem-solving trajectories (PSTrajs) quantified using Dynamic Time Warping (DTW) | *LLM role:* heuristic_generator

> Zhang et al. propose BehaveSim, a metric that measures algorithmic similarity by applying Dynamic Time Warping (DTW) to the sequence of intermediate solutions (trajectories) generated during execution, rather than relying on static code analysis. By integrating this into FunSearch and EoH to enforce behavioral diversity, they achieve significant performance gains, notably reducing the optimality gap on TSP by ~7.8% compared to standard FunSearch. **Key Takeaway:** We must stop using code hashes or embedding cosine similarity for population diversity in AlgoEvo; instead, we should instrument generated heuristics to log intermediate states (e.g., partial VRP routes) and cluster them via DTW to prevent convergence to behaviorally identical local optima. This is a mandatory upgrade for our evolutionary search infrastructure.

### [Learning to Evolve for Optimization via Stability-Inducing Neural Unrolling](https://arxiv.org/abs/2512.11453)

**2026-03-03** | The Hong Kong Polytechnic University, The University of Hong Kong | M=5 P=7 I=7 *discuss*

*Method:* Bilevel meta-optimization with stability-inducing neural unrolling, using a structured Mamba-based neural operator and a gradient-derived composite solver. | *LLM role:* none

> Gao et al. propose L2E, a meta-learned neural optimizer that uses Mamba blocks to parameterize evolutionary operators within a stability-enforcing unrolled loop (Krasnosel'skii-Mann iteration). Results on BBOB and LSGO-1000D are strong, showing it outperforms Transformer-based L2O methods (GLHF) and classical heuristics (DE) in sample efficiency and zero-shot generalization. **Key Takeaway:** We should investigate replacing Transformer-based population encoders in AlgoEvo with Mamba blocks to reduce complexity from quadratic to linear ($O(N)$), enabling larger population sizes in our meta-heuristic search. The theoretical framing of evolution as a fixed-point iteration also offers a rigorous stability constraint we could inject into our RL-guided search policies.


#### 2026-03-05 (2 papers)

### [Rethinking Code Similarity for Automated Algorithm Design with LLMs](https://arxiv.org/abs/2603.02787)

**2026-03-03** | City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* BehaveSim, a method for measuring algorithmic similarity based on problem-solving trajectories (PSTrajs) quantified using Dynamic Time Warping (DTW) | *LLM role:* heuristic_generator

> Zhang et al. propose BehaveSim, a metric that measures algorithmic similarity by applying Dynamic Time Warping (DTW) to the sequence of intermediate solutions (trajectories) generated during execution, rather than relying on static code analysis. By integrating this into FunSearch and EoH to enforce behavioral diversity, they achieve significant performance gains, notably reducing the optimality gap on TSP by ~7.8% compared to standard FunSearch. **Key Takeaway:** We must stop using code hashes or embedding cosine similarity for population diversity in AlgoEvo; instead, we should instrument generated heuristics to log intermediate states (e.g., partial VRP routes) and cluster them via DTW to prevent convergence to behaviorally identical local optima. This is a mandatory upgrade for our evolutionary search infrastructure.

### [Learning to Evolve for Optimization via Stability-Inducing Neural Unrolling](https://arxiv.org/abs/2512.11453)

**2026-03-03** | The Hong Kong Polytechnic University, The University of Hong Kong | M=5 P=7 I=7 *discuss*

*Method:* Bilevel meta-optimization with stability-inducing neural unrolling, using a structured Mamba-based neural operator and a gradient-derived composite solver. | *LLM role:* none

> Gao et al. propose L2E, a meta-learned neural optimizer that uses Mamba blocks to parameterize evolutionary operators within a stability-enforcing unrolled loop (Krasnosel'skii-Mann iteration). Results on BBOB and LSGO-1000D are strong, showing it outperforms Transformer-based L2O methods (GLHF) and classical heuristics (DE) in sample efficiency and zero-shot generalization. **Key Takeaway:** We should investigate replacing Transformer-based population encoders in AlgoEvo with Mamba blocks to reduce complexity from quadratic to linear ($O(N)$), enabling larger population sizes in our meta-heuristic search. The theoretical framing of evolution as a fixed-point iteration also offers a rigorous stability constraint we could inject into our RL-guided search policies.


#### 2026-03-01 (1 papers)

### [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133)

**2026-02-23** | University of California, Berkeley, Bespoke Labs | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary algorithm with hierarchical adaptive optimization using an accumulated improvement signal to dynamically modulate local exploration intensity, global resource allocation via multi-armed bandit, and meta-level solution tactics generation | *LLM role:* semantic_mutation_operator

> AdaEvolve replaces static evolutionary schedules with a three-tier adaptive controller: local exploration intensity based on an 'accumulated improvement signal' (pseudo-gradient), global compute allocation via a normalized bandit, and meta-level 'tactic' generation when stagnation occurs. Results are highly convincing, showing SOTA on Circle Packing (beating AlphaEvolve) and 185 other tasks while using the same LLM backbone as baselines, proving the gains are algorithmic. The most stealable insight is the $G_t$ signal metric—an exponential moving average of squared normalized improvements—which allows auto-tuning exploration rates without manual intervention. This is a direct upgrade for our AlgoEvo architecture, specifically addressing our sample efficiency and stagnation bottlenecks.


#### 2026-02-26 (1 papers)

### [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133)

**2026-02-23** | University of California, Berkeley, Bespoke Labs | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary algorithm with hierarchical adaptive optimization using an accumulated improvement signal to dynamically modulate local exploration intensity, global resource allocation via multi-armed bandit, and meta-level solution tactics generation | *LLM role:* semantic_mutation_operator

> AdaEvolve replaces static evolutionary schedules with a three-tier adaptive controller: local exploration intensity based on an 'accumulated improvement signal' (pseudo-gradient), global compute allocation via a normalized bandit, and meta-level 'tactic' generation when stagnation occurs. Results are highly convincing, showing SOTA on Circle Packing (beating AlphaEvolve) and 185 other tasks while using the same LLM backbone as baselines, proving the gains are algorithmic. The most stealable insight is the $G_t$ signal metric—an exponential moving average of squared normalized improvements—which allows auto-tuning exploration rates without manual intervention. This is a direct upgrade for our AlgoEvo architecture, specifically addressing our sample efficiency and stagnation bottlenecks.


#### 2026-02-24 (5 papers)

### [Discovering Multiagent Learning Algorithms with Large Language Models](https://arxiv.org/abs/2602.16928)

**2026-02-18** | Google DeepMind | M=10 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* AlphaEvolve, an LLM-powered evolutionary coding agent | *LLM role:* code_writer

> DeepMind applies AlphaEvolve to discover new variants of CFR and PSRO by evolving Python code for regret accumulation and meta-strategy solving. They identify VAD-CFR and SHOR-PSRO, which outperform human-designed SOTA (DCFR, PCFR+) on benchmarks like Leduc Poker and Liar's Dice; results are rigorous, using exact exploitability. The critical takeaway is the shift from evolving static functions to evolving **stateful classes** (e.g., tracking volatility via EWMA inside the accumulator), allowing the LLM to discover dynamic, adaptive schedules—a technique we should immediately port to AlgoEvo.

### [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/abs/2602.16038)

**2026-02-17** | Massachusetts Institute of Technology | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Language-Guided Optimization (LaGO) framework decomposing heuristic discovery into forward, backward, and update stages, utilizing LLMs for reasoned evolution, code-writing analysis, co-evolution of constructive and refinement heuristics, and diversity-aware population management. | *LLM role:* evolutionary_search

> LaGO decomposes automated heuristic design into three explicit modules: evaluation, a code-writing 'Analyst' (backward pass), and a diversity-aware 'Generator' (update), while co-evolving constructive and refinement heuristics. The authors demonstrate significant gains (+0.17 QYI) on PDPTW and Crew Pairing against ReEvo and EoH, showing that joint optimization of initialization and improvement prevents local optima. The critical takeaway is the 'Analyst' module: instead of asking the LLM for text critiques, they ask it to write Python feature extraction functions to statistically characterize solution quality—a technique we should immediately adopt to upgrade our fitness signals in AlgoEvo.

### [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133)

**2026-02-23** | University of California, Berkeley, Bespoke Labs | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary algorithm with hierarchical adaptive optimization using an accumulated improvement signal to dynamically modulate local exploration intensity, global resource allocation via multi-armed bandit, and meta-level solution tactics generation | *LLM role:* semantic_mutation_operator

> AdaEvolve replaces static evolutionary schedules with a three-tier adaptive controller: local exploration intensity based on an 'accumulated improvement signal' (pseudo-gradient), global compute allocation via a normalized bandit, and meta-level 'tactic' generation when stagnation occurs. Results are highly convincing, showing SOTA on Circle Packing (beating AlphaEvolve) and 185 other tasks while using the same LLM backbone as baselines, proving the gains are algorithmic. The most stealable insight is the $G_t$ signal metric—an exponential moving average of squared normalized improvements—which allows auto-tuning exploration rates without manual intervention. This is a direct upgrade for our AlgoEvo architecture, specifically addressing our sample efficiency and stagnation bottlenecks.

### [MadEvolve: Evolutionary Optimization of Cosmological Algorithms with Large Language Models](https://arxiv.org/abs/2602.15951)

**2026-02-17** | University of Wisconsin-Madison | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary optimization with nested parameter tuning | *LLM role:* code_writer, mutation_operator, report_generator

> MadEvolve extends AlphaEvolve by embedding a gradient-based optimization loop (via JAX) inside the fitness evaluation, allowing the LLM to focus purely on code structure while an optimizer (Adam) handles continuous parameters. They demonstrate 20-30% performance gains on complex cosmological reconstruction tasks, validated on held-out simulations. The critical takeaway is the architectural pattern: prompt the LLM to write differentiable code rather than tuning constants, and use a UCB1 bandit to dynamically select between cheap and expensive models. We should immediately adopt the differentiable inner-loop strategy for our continuous heuristic search projects.

### [RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution](https://arxiv.org/abs/2602.16932)

**2026-02-18** | Walmart Global Tech, Santa Clara University, Independent Researcher | M=6 P=5 I=7 *discuss*

*Method:* LLM-guided program evolution based on AlphaEvolve | *LLM role:* evolutionary_search

> RankEvolve applies AlphaEvolve with MAP-Elites to evolve Python retrieval functions, achieving significant gains over BM25 on BEIR/BRIGHT by rediscovering concepts like soft stop-words and PMI-based scoring. The results are empirically rigorous, showing that 'Freeform' seeds (defining only I/O contracts) significantly outperform 'Composable' or 'Constrained' seeds, albeit at a 10x latency cost. For our AlgoEvo work, the key takeaway is the concrete evidence that constraining the search space to 'clean' components prematurely caps performance; we should adopt their 'Freeform' approach but add an explicit latency/cost constraint to the fitness function to manage the resulting complexity.


#### 2026-02-22 (4 papers)

### [Discovering Multiagent Learning Algorithms with Large Language Models](https://arxiv.org/abs/2602.16928)

**2026-02-18** | Google DeepMind | M=10 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* AlphaEvolve, an LLM-powered evolutionary coding agent | *LLM role:* code_writer

> DeepMind applies AlphaEvolve to discover new variants of CFR and PSRO by evolving Python code for regret accumulation and meta-strategy solving. They identify VAD-CFR and SHOR-PSRO, which outperform human-designed SOTA (DCFR, PCFR+) on benchmarks like Leduc Poker and Liar's Dice; results are rigorous, using exact exploitability. The critical takeaway is the shift from evolving static functions to evolving **stateful classes** (e.g., tracking volatility via EWMA inside the accumulator), allowing the LLM to discover dynamic, adaptive schedules—a technique we should immediately port to AlgoEvo.

### [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/abs/2602.16038)

**2026-02-17** | Massachusetts Institute of Technology | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Language-Guided Optimization (LaGO) framework decomposing heuristic discovery into forward, backward, and update stages, utilizing LLMs for reasoned evolution, code-writing analysis, co-evolution of constructive and refinement heuristics, and diversity-aware population management. | *LLM role:* evolutionary_search

> LaGO decomposes automated heuristic design into three explicit modules: evaluation, a code-writing 'Analyst' (backward pass), and a diversity-aware 'Generator' (update), while co-evolving constructive and refinement heuristics. The authors demonstrate significant gains (+0.17 QYI) on PDPTW and Crew Pairing against ReEvo and EoH, showing that joint optimization of initialization and improvement prevents local optima. The critical takeaway is the 'Analyst' module: instead of asking the LLM for text critiques, they ask it to write Python feature extraction functions to statistically characterize solution quality—a technique we should immediately adopt to upgrade our fitness signals in AlgoEvo.

### [MadEvolve: Evolutionary Optimization of Cosmological Algorithms with Large Language Models](https://arxiv.org/abs/2602.15951)

**2026-02-17** | University of Wisconsin-Madison | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary optimization with nested parameter tuning | *LLM role:* code_writer, mutation_operator, report_generator

> MadEvolve extends AlphaEvolve by embedding a gradient-based optimization loop (via JAX) inside the fitness evaluation, allowing the LLM to focus purely on code structure while an optimizer (Adam) handles continuous parameters. They demonstrate 20-30% performance gains on complex cosmological reconstruction tasks, validated on held-out simulations. The critical takeaway is the architectural pattern: prompt the LLM to write differentiable code rather than tuning constants, and use a UCB1 bandit to dynamically select between cheap and expensive models. We should immediately adopt the differentiable inner-loop strategy for our continuous heuristic search projects.

### [RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution](https://arxiv.org/abs/2602.16932)

**2026-02-18** | Walmart Global Tech, Santa Clara University, Independent Researcher | M=6 P=5 I=7 *discuss*

*Method:* LLM-guided program evolution based on AlphaEvolve | *LLM role:* evolutionary_search

> RankEvolve applies AlphaEvolve with MAP-Elites to evolve Python retrieval functions, achieving significant gains over BM25 on BEIR/BRIGHT by rediscovering concepts like soft stop-words and PMI-based scoring. The results are empirically rigorous, showing that 'Freeform' seeds (defining only I/O contracts) significantly outperform 'Composable' or 'Constrained' seeds, albeit at a 10x latency cost. For our AlgoEvo work, the key takeaway is the concrete evidence that constraining the search space to 'clean' components prematurely caps performance; we should adopt their 'Freeform' approach but add an explicit latency/cost constraint to the fitness function to manage the resulting complexity.


#### 2026-02-22 (2 papers)

### [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/abs/2602.16038)

**2026-02-17** | Massachusetts Institute of Technology | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Language-Guided Optimization (LaGO) framework decomposing heuristic discovery into forward, backward, and update stages, utilizing LLMs for reasoned evolution, code-writing analysis, co-evolution of constructive and refinement heuristics, and diversity-aware population management. | *LLM role:* evolutionary_search

> LaGO decomposes automated heuristic design into three explicit modules: evaluation, a code-writing 'Analyst' (backward pass), and a diversity-aware 'Generator' (update), while co-evolving constructive and refinement heuristics. The authors demonstrate significant gains (+0.17 QYI) on PDPTW and Crew Pairing against ReEvo and EoH, showing that joint optimization of initialization and improvement prevents local optima. The critical takeaway is the 'Analyst' module: instead of asking the LLM for text critiques, they ask it to write Python feature extraction functions to statistically characterize solution quality—a technique we should immediately adopt to upgrade our fitness signals in AlgoEvo.

### [MadEvolve: Evolutionary Optimization of Cosmological Algorithms with Large Language Models](https://arxiv.org/abs/2602.15951)

**2026-02-17** | University of Wisconsin-Madison | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary optimization with nested parameter tuning | *LLM role:* code_writer, mutation_operator, report_generator

> MadEvolve extends AlphaEvolve by embedding a gradient-based optimization loop (via JAX) inside the fitness evaluation, allowing the LLM to focus purely on code structure while an optimizer (Adam) handles continuous parameters. They demonstrate 20-30% performance gains on complex cosmological reconstruction tasks, validated on held-out simulations. The critical takeaway is the architectural pattern: prompt the LLM to write differentiable code rather than tuning constants, and use a UCB1 bandit to dynamically select between cheap and expensive models. We should immediately adopt the differentiable inner-loop strategy for our continuous heuristic search projects.


#### 2026-02-22 (2 papers)

### [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/abs/2602.16038)

**2026-02-17** | Massachusetts Institute of Technology | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Language-Guided Optimization (LaGO) framework decomposing heuristic discovery into forward, backward, and update stages, utilizing LLMs for reasoned evolution, code-writing analysis, co-evolution of constructive and refinement heuristics, and diversity-aware population management. | *LLM role:* evolutionary_search

> LaGO decomposes automated heuristic design into three explicit modules: evaluation, a code-writing 'Analyst' (backward pass), and a diversity-aware 'Generator' (update), while co-evolving constructive and refinement heuristics. The authors demonstrate significant gains (+0.17 QYI) on PDPTW and Crew Pairing against ReEvo and EoH, showing that joint optimization of initialization and improvement prevents local optima. The critical takeaway is the 'Analyst' module: instead of asking the LLM for text critiques, they ask it to write Python feature extraction functions to statistically characterize solution quality—a technique we should immediately adopt to upgrade our fitness signals in AlgoEvo.

### [MadEvolve: Evolutionary Optimization of Cosmological Algorithms with Large Language Models](https://arxiv.org/abs/2602.15951)

**2026-02-17** | University of Wisconsin-Madison | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary optimization with nested parameter tuning | *LLM role:* code_writer, mutation_operator, report_generator

> MadEvolve extends AlphaEvolve by embedding a gradient-based optimization loop (via JAX) inside the fitness evaluation, allowing the LLM to focus purely on code structure while an optimizer (Adam) handles continuous parameters. They demonstrate 20-30% performance gains on complex cosmological reconstruction tasks, validated on held-out simulations. The critical takeaway is the architectural pattern: prompt the LLM to write differentiable code rather than tuning constants, and use a UCB1 bandit to dynamically select between cheap and expensive models. We should immediately adopt the differentiable inner-loop strategy for our continuous heuristic search projects.


<!-- New papers are appended here by the daily updater -->

---

## Research Fronts

*2 fronts detected — snapshot 2026-04-14*

### Front 1 (56 papers) — STABLE

**Density:** 0.07 | **Methods:** llm_code_generation, program_synthesis, llm_as_heuristic, llm_evolutionary_search, evolution_of_heuristics | **Problems:** heuristic_evolution, operator_discovery, tsp, algorithm_discovery, black_box_optimization

*Unique methods:* a_star_search, abductive_reasoning, abstract_syntax_tree, acquisition_function_design, acquisition_functions, active_learning, actor_critic, adam_optimizer, adaptive_iterated_local_search, adaptive_large_neighborhood_search, adaptive_operator_selection, adaptive_optimization, adversarial_generation, agentic_ai, algorithm_analysis, algorithm_distillation, anisotropic_diffusion, ant_colony_optimization, automated_heuristic_design, automated_testing, automatic_linearization, autoresearch, barzilai_borwein, basin_hopping, bilevel_optimization, black_box_distillation, breadth_first_search, butterworth_filter, catastrophic_forgetting_mitigation, causal_discovery, causal_inference, chain_of_thought, character_role_play, code_validation, collaborative_tree_optimization, comparative_error_analysis, competitive_evaluation, concept_learning, contrastive_learning, convexification, counterfactual_regret_minimization, cross_entropy_method, crossover_operators, cumulative_diversity_index, data_augmentation, data_rehearsal, deduplication, deep_focus_refinement, deep_reinforcement_learning, distributed_computing, diversity_metrics, domain_specific_language, dpo, dynamic_model_selection, dynamic_resource_allocation, elitist_strategy, elo_rating_system, entailment_graph, eureka, evolution_of_thought, evolution_strategies, evoph, execution_guided_program_synthesis, exploitation_exploration_tradeoff, feature_fusion, few_shot_learning, few_shot_prompting, fine_tuning, fixed_point_iteration, gaussian_process, gaussian_processes, generative_agent_based_modeling, geometric_deep_learning, gigaevo, global_optimization, gnn, gpt_4o, gridcoder, guided_filter, guided_local_search, harmony_search, heuristic_evolution, hill_climbing, huber_regularization, hybrid_search, image_quality_metrics, iterated_local_search, iterative_refinement, iterative_repair, knowledge_transfer, l_bfgs_b, lago, language_hyper_heuristics, large_neighborhood_search, learning_to_optimize, llm_as_agent, llm_as_environment_designer, llm_as_heuristic_generator, llm_as_meta_controller, llm_as_variation_operator, llm_driven_automatic_heuristic_design, llm_driven_optimization, llm_guided_evolutionary_search, llm_orchestration_router, llm_proposal, local_refinement, mamba, manifold_optimization, mcts, mdp_formulation, memetic_algorithm, meta_agents, meta_evolution, meta_gradients, meta_guidance, meta_optimization, meta_prompt_evolution, monte_carlo_sampling, multi_agent_learning, multi_agent_llm, multi_agent_reasoning, multi_agent_simulation, multi_arm_bandit, multi_head_self_attention, multi_objective_optimization, multi_population_evolutionary_algorithm, multi_task_learning, mutation_operators, nash_equilibrium, negative_feedback_learning, neural_architecture_search, neural_combinatorial_optimization, neural_networks, neural_operator, neural_program_synthesis, neural_unrolling, nonlinear_programming, nsga2, nucleus_sampling, numerical_linear_algebra, on_policy_learning, one_plus_one_es, optimistic_regret_matching, optimization_algorithms, parameter_efficient_fine_tuning, pareto_dominance, path_credit_propagation, perceiver_attention, phase_unwrapping, population_based_optimization, population_management, ppo, pre_training, preference_alignment, presolving, primal_heuristics, proximal_operators, proximal_policy_optimization, psro, qube, receding_horizon_control, reflection, reflective_evolution, rl_dpo, rl_trained, robophd, ruin_and_recreate, search_space_analysis, selection_mechanisms, self_attention, self_configuring_mas, self_generative_mas, self_instruction, self_instrumenting_agents, self_rectifying_mas, self_reflection, shannon_wiener_diversity_index, spatial_branch_and_bound, spea2, spectral_normalization, structural_variation, structure_tensor_analysis, supervised_fine_tuning, symbolic_distillation, template_based_parameter_optimization, test_time_fine_tuning, text_embeddings, theory_synthesis, total_variation, tree_search, tree_structured_parzen_estimator, tri_agent_architecture, trueskill_ranking, ucb_selection, upper_confidence_bound, upper_confidence_bound_for_trees, validation_free_evolution, value_scaled_optimization, zeroth_order_optimization
*Shared methods:* alphaevolve, bayesian_optimization, black_box_optimization, co_evolution, code_embedding, direct_preference_optimization, eoh, evolution_of_heuristics, evolution_strategy, evolutionary_algorithm, evolutionary_algorithms, evolutionary_computation, evolutionary_search, funsearch, game_theory, genetic_algorithm, genetic_programming, gepa, gradient_based_optimization, gradient_descent, greedy_algorithm, grpo, hyper_heuristics, in_context_learning, island_model, island_model_ea, large_language_models, llamea, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_context_learning, llm_in_the_loop, llm_prompt_optimization, local_search, map_elites, meta_learning, metaheuristics, monte_carlo_tree_search, multi_agent_reinforcement_learning, multi_agent_system, multi_agent_systems, multi_armed_bandit, offline_reinforcement_learning, openevolve, policy_space_response_oracles, program_synthesis, prompt_engineering, quality_diversity, reevo, reinforcement_learning, retrieval_augmented_generation, reward_shaping, self_improving_search, shinkaevolve, simulated_annealing, slsqp, static_code_analysis, supervised_learning, transformer, zero_sum_game

This research front focuses on advancing LLM-driven evolutionary search for automated algorithm design, particularly in combinatorial optimization, scheduling, and mathematical discovery. The core theme is the development of sophisticated, adaptive, and causally-informed strategies that move beyond simple LLM code generation. Key frameworks like FunSearch, AlphaEvolve, EoH, ReEvo, and OpenEvolve are being enhanced with explicit reasoning, multi-agent coordination, and dynamic control mechanisms to improve the efficiency and quality of discovered algorithms.

Significant contributions include PathWise's Entailment Graph for planning heuristic discovery, CausalEvolve's causal scratchpad for guiding program mutations, and DyACE's dynamic co-evolution with Look-Ahead Rollout for non-stationary problems, reducing CVRP optimality gaps from 12.5% to 3.1%. AdaEvolve introduces a hierarchical adaptive controller, achieving state-of-the-art on Circle Packing, while STRCMP fuses GNNs with LLMs to inject structural priors, significantly reducing convergence time for MILP/SAT. Multi-agent systems like RoCo, SS-Logic, and MAS2 demonstrate improved robustness and self-rectification. Furthermore, RL-infused evolution, exemplified by EvoTune and CALM, leverages Direct Preference Optimization (DPO) to fine-tune LLM generators online, leading to faster discovery of better heuristics on benchmarks like Bin Packing and TSP.

This front is rapidly maturing, with a strong emphasis on improving sample efficiency, robustness, and generalizability. The trajectory indicates a shift towards integrating more sophisticated causal reasoning, multi-modal feedback (e.g., execution traces, structural embeddings), and meta-learning to dynamically adapt search strategies. Future work will likely explore co-evolving environments or benchmarks alongside algorithms to ensure robustness and prevent overfitting, moving towards truly open-ended algorithm discovery.

**Papers:**

### [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](https://arxiv.org/abs/2601.20539)

**2026-01-29** |  | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent reasoning framework (Policy Agent, World Model Agent, Critic Agents) for Automated Heuristic Design, formulated as a sequential decision process over an entailment graph. | *LLM role:* heuristic_generator

> PathWise reformulates heuristic discovery as a sequential planning problem over an 'Entailment Graph,' where a Policy Agent generates high-level evolutionary directives (rationales) and a World Model executes the code, guided by specific Critic reflections. The results are robust: it outperforms ReEvo, FunSearch, and MCTS-AHD on TSP, CVRP, and Bin Packing while using half the evaluation budget (500 vs 1000), demonstrating genuine sample efficiency. The key takeaway is the **Entailment Graph** structure: explicitly storing the *derivation rationale* and lineage allows the LLM to reason about the search trajectory and avoid redundant failures, a mechanism we should immediately adapt for AlgoEvo to fix our memory bottleneck.

### [Evolution Transformer: In-Context Evolutionary Optimization](https://arxiv.org/abs/2403.02985)

**2024-03-05** | Google DeepMind, TU Berlin | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolution Transformer, a causal Transformer architecture with self-attention and Perceiver cross-attention for search distribution updates | *LLM role:* evolutionary_search

> Lange et al. introduce the Evolution Transformer, a causal architecture that learns to perform evolutionary strategy updates by attending to optimization history, effectively 'distilling' algorithms like CMA-ES into a neural network. Crucially, they propose 'Self-Referential Algorithm Distillation' (SR-EAD), where the model improves itself by perturbing its own weights, generating trajectories, and filtering for the best ones to retrain on—eliminating the need for a teacher. The results are strong, showing generalization to unseen Brax control tasks and successful (though sometimes unstable) self-bootstrapping. The key takeaway for us is the SR-EAD loop as a mechanism for open-ended optimizer improvement, and their use of Perceiver cross-attention to handle variable population sizes—a technique we should immediately steal for our multi-agent memory architectures.

### [CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad](https://arxiv.org/abs/2603.14575)

**2026-03-15** | Carnegie Mellon University, MBZUAI, Hong Kong Baptist University, The University of Sydney | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* CausalEvolve with causal scratchpad leveraging LLMs to identify and reason about guiding factors for evolution, incorporating outcome-level and procedure-level factors, multi-arm bandit for intervention, and abductive reasoning. | *LLM role:* heuristic_generator

> CausalEvolve enhances LLM evolutionary search frameworks (like AlphaEvolve and ShinkaEvolve) by introducing a causal scratchpad that extracts outcome- and procedure-level factors to explicitly guide program mutations via a Multi-Armed Bandit. The results are backed by solid empirical numbers, showing it outperforms the state-of-the-art ShinkaEvolve across four algorithmic and mathematical tasks (Hadamard, Autocorrelation, Circle Packing, AIME) by up to 9.1% in best-found scores. The single most useful takeaway for us is their 'surprise detection' module: using LLMs to perform abductive reasoning when a seemingly good combination of factors yields a score drop, thereby uncovering hidden confounders and generating new search directions. This is highly relevant to our work; we should immediately evaluate their MAB-driven causal intervention strategy and procedure-level factor extraction to improve the sample efficiency and memory mechanisms in AlgoEvo.

### [LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling](https://arxiv.org/abs/2601.15738)

**2026-01-22** |  | M=7 P=6 I=7 *discuss*

*Method:* LLM-assisted evolutionary algorithm for automatic dispatching rule design with dual-expert mechanism and feature-fitting rule evolution | *LLM role:* heuristic_generator, evaluator

> LLM4DRD employs a dual-agent framework (Generator & Evaluator) to evolve priority dispatching rules for dynamic flexible assembly flow shops. The core contribution is the **Hybrid Evaluation** mechanism, where the Evaluator generates qualitative critiques (strengths/weaknesses) that are injected into the Generator's prompts to guide specific operators like 'Dominance-Fusion Crossover' and 'Directed Optimization.' Empirical results show it outperforms FunSearch and EOH, avoiding the premature convergence seen in other methods. The most stealable insight is the prompt structure for crossover: rather than blindly combining code, it uses the Evaluator's analysis of parent strengths to direct the merger, a technique we should implement to improve sample efficiency in our evolutionary search.

### [Amplifying human performance in combinatorial competitive programming](https://arxiv.org/abs/2411.19744)

**2024-11-29** | Google DeepMind | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* FunSearch, an evolutionary algorithm for program search using LLMs and a systematic evaluator | *LLM role:* evolutionary_search

> DeepMind applies FunSearch (using Gemini 1.5 Flash) to evolve scoring functions within human-written greedy backbones for Hash Code and AtCoder problems, achieving top-1% or rank-1 performance against humans. The results are robust, beating top human teams on 5/8 historical contests using a generic evolutionary setup. The critical takeaway is the 'switching variable' technique: using a single evolved function to handle multiple distinct decision points (e.g., selecting a vehicle vs. selecting a route) by passing a state flag, rather than evolving multiple interacting functions. This validates that generalist models (Flash) are sufficient for high-end OR evolution without code-specific fine-tuning. We should adopt their 'Backbone + Scorer' architecture for our VRP/Scheduling work immediately.

### [Contrastive Concept-Tree Search for LLM-Assisted Algorithm Discovery](https://arxiv.org/abs/2602.03132)

**2026-02-03** |  | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Contrastive Concept-Tree Search (CCTS) using a hierarchical Bernoulli model and Tree-structured Parzen Estimator (TPE) for likelihood-ratio based parent reweighting, combined with cross-entropy updates for concept utility estimation. | *LLM role:* heuristic_generator

> The authors introduce Contrastive Concept-Tree Search (CCTS), which modifies the standard evolutionary loop by prompting the LLM to extract semantic 'concepts' from every generated program, building a dynamic hierarchy. They then apply a Tree-structured Parzen Estimator (TPE) to these concepts to learn a contrastive utility model (p(concept|good)/p(concept|bad)), using this to bias parent selection towards promising algorithmic strategies. Results are rigorous, showing consistent improvements over k-elite baselines on combinatorial tasks like Circle Packing, with a synthetic ablation confirming the model learns ground-truth concept utilities. **Key Takeaway:** We should immediately implement the 'Concept TPE' loop in AlgoEvo—asking the LLM to tag generated heuristics with concepts and maintaining a weight vector over these concepts provides a cheap, interpretable 'process reward model' to guide search.

### [Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning](https://arxiv.org/abs/2602.13218)

**2026-01-23** | Tencent, The Hong Kong University of Science and Technology (Guangzhou) | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Agentic Meta-Synthesis framework using a Generate–Validate–Repair closed loop for Generator–Validator program pairs | *LLM role:* program_synthesis, code_writer, decomposition_guide, evaluator, evolutionary_search

> Liu et al. introduce SS-Logic, an agentic framework that evolves Python 'Generator-Validator' pairs to scale logic task families, using a rigorous 'Code-Augmented Blind Review' where independent agents must write code to solve generated tasks to verify their validity. They expand 400 seed families to over 21k instances, achieving consistent gains on AIME (+3.0) and SynLogic (+5.2) via RLVR. **Crucial Takeaway:** We should steal the 'Blind Review' mechanism for AlgoEvo—using the solvability of a generated problem (by an independent code agent) as a strict fitness filter for the generator itself. This directly addresses our bottleneck in filtering invalid or hallucinated heuristics during evolutionary search.

### [DyACE: Dynamic Algorithm Co-evolution for Online Automated Heuristic Design with Large Language Model](https://arxiv.org/abs/2603.13344)

**2026-03-07** |  | M=8 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dynamic Algorithm Co-evolution (DyACE) with Receding Horizon Control and Look-Ahead Rollout Search | *LLM role:* meta-controller

> DyACE reformulates LLM-driven automated heuristic design from a static optimization task into a dynamic control problem, using a receding horizon architecture to continuously co-evolve phase-specific heuristic operators alongside the solution population. The results are backed by strong empirical numbers on JSSP, TSP, and CVRP, significantly outperforming state-of-the-art static baselines like ReEvo and FunSearch (e.g., reducing the optimality gap on CVRP CMT1 from ReEvo's 12.5% to 3.1%). The single most useful takeaway is the 'Look-Ahead Rollout Search' combined with a decoupled Diagnosis Agent: instead of evaluating an algorithm end-to-end, they run short rollouts to extract 'Search Trajectory Features' (landscape kinematics, operator telemetry) which an LLM translates into 'Verbal Gradients' to guide the next code mutation. This is a must-read for our AlgoEvo project; it directly addresses our bottlenecks in sample efficiency and search signaling by proving that dynamic, state-aware operator adaptation beats static heuristic generation, giving us a concrete architecture to steal.

### [RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design](https://arxiv.org/abs/2512.03762)

**2025-12-04** | South China University of Technology | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-Agent Role-Based System (RoCo) for Automatic Heuristic Design (AHD) integrated into an Evolutionary Program Search (EoH) framework | *LLM role:* evolutionary_search

> RoCo replaces standard evolutionary mutation operators with a 4-agent collaboration loop (Explorer, Exploiter, Critic, Integrator) that iteratively refines heuristics and accumulates long-term reflection memory across generations. While the empirical gains over ReEvo are marginal (often <1%) and likely expensive in token cost, the architecture successfully demonstrates how to embed structured multi-agent reasoning into the evolutionary loop to stabilize black-box search. The key takeaway is their Long-term Reflection mechanism, which aggregates critic feedback into a persistent memory buffer to guide future mutations—a technique we should immediately test to improve sample efficiency in AlgoEvo.

### [Talking to Yourself: Defying Forgetting in Large Language Models](https://arxiv.org/abs/2602.20162)

**2026-01-23** | Zhejiang University, Stanford University, ETH Zürich, Binjiang Institute of Zhejiang University, Om AI Research | M=7 P=6 I=7 *changes-thinking* *discuss*

*Method:* Self-Augmented Supervised Fine-Tuning (SA-SFT) which involves generating self-dialogues with the frozen base model and mixing them with task-specific data for standard supervised fine-tuning. | *LLM role:* heuristic_generator

> Sun et al. propose SA-SFT, a method where the model generates its own 'self-talk' data before fine-tuning, which is then mixed with task data to prevent catastrophic forgetting. The results are compelling: they show that this self-generated data preserves reasoning skills (like GSM8K) better than external datasets, and crucially, ablation studies prove that removing math content from the self-data *still* preserves math skills. The key takeaway is that forgetting is largely driven by style-induced gradient drift rather than knowledge overwriting. This offers a practically free way to stabilize our specialized heuristic-generation models without needing to curate external replay buffers.

### [LLM Guided Evolution -- The Automation of Models Advancing Models](https://arxiv.org/abs/2403.11446)

**2024-03-18** | Georgia Tech Research Institute | M=5 P=6 I=7 *discuss*

*Method:* LLM-guided Genetic Algorithm for Neural Architecture Search (NAS) with multi-objective optimization (accuracy and parameter count), incorporating Evolution of Thought (EoT) and Character Role Play (CRP) for mutation and mating. | *LLM role:* evolutionary_search, code_writer, evaluator

> Morris et al. propose 'Guided Evolution,' an LLM-based NAS framework that introduces 'Evolution of Thought' (EoT) and 'Character Role Play' to guide code mutations. While the results are statistically negligible (single trials, ~0.8% gain on CIFAR-10), the EoT mechanism offers a specific, actionable prompt engineering technique: explicitly prompting the LLM to compare a successful elite individual against its original seed to extract 'reasoning' before applying mutations to new individuals. This serves as a lightweight, prompt-based memory/feedback mechanism that could immediately improve sample efficiency in our evolutionary search agents. The 'Character Role Play' (e.g., asking the LLM to act as 'Dr. MaGoo' for unorthodox ideas) is a gimmicky but potentially useful heuristic for maintaining population diversity.

### [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133)

**2026-02-23** | University of California, Berkeley, Bespoke Labs | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary algorithm with hierarchical adaptive optimization using an accumulated improvement signal to dynamically modulate local exploration intensity, global resource allocation via multi-armed bandit, and meta-level solution tactics generation | *LLM role:* semantic_mutation_operator

> AdaEvolve replaces static evolutionary schedules with a three-tier adaptive controller: local exploration intensity based on an 'accumulated improvement signal' (pseudo-gradient), global compute allocation via a normalized bandit, and meta-level 'tactic' generation when stagnation occurs. Results are highly convincing, showing SOTA on Circle Packing (beating AlphaEvolve) and 185 other tasks while using the same LLM backbone as baselines, proving the gains are algorithmic. The most stealable insight is the $G_t$ signal metric—an exponential moving average of squared normalized improvements—which allows auto-tuning exploration rates without manual intervention. This is a direct upgrade for our AlgoEvo architecture, specifically addressing our sample efficiency and stagnation bottlenecks.

### [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/abs/2506.11057)

**2025-05-22** | Shanghai Key Laboratory of Scalable Computing and Systems, School of Computer Science, Shanghai Jiao Tong University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structure-aware LLM-based algorithm discovery framework combining Graph Neural Network (GNN) for structural embeddings and LLM for solver-specific code generation, refined by an evolutionary algorithm. | *LLM role:* code_writer

> STRCMP introduces a composite architecture where a GNN encodes CO problem instances (MILP/SAT) into embeddings that condition an LLM (fine-tuned via SFT and DPO) to generate solver-specific heuristics within an evolutionary loop. The results are strong and empirically backed, showing significant reductions in convergence time and timeouts compared to text-only evolutionary methods like AutoSAT and LLM4Solver. The key takeaway is the architectural blueprint for fusing instance-specific structural embeddings (via soft prompting) with LLM code generation to drastically improve the sample efficiency of evolutionary search. This is immediately relevant to our EvoCut and AlgoEvo projects, suggesting we should move beyond pure text prompts for topology-heavy problems.

### [RF-Agent: Automated Reward Function Design via Language Agent Tree Search](https://arxiv.org/abs/2602.23876)

**2026-02-27** | Beihang University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Monte Carlo Tree Search (MCTS) for reward function design, guided by LLM multi-stage contextual reasoning | *LLM role:* evolutionary_search

> RF-Agent frames LLM-based reward function design as a Monte Carlo Tree Search problem, using specialized expansion actions (mutation, crossover, path reasoning) and LLM self-verification to efficiently explore the code space. The results are real and backed by strong empirical numbers; it significantly outperforms Eureka and Revolve on 17 IsaacGym and Bi-DexHands tasks, achieving higher success rates with the same number of LLM samples, even when using the cheaper GPT-4o-mini model. The most stealable insights for us are the 'Path Reasoning' operator (prompting the LLM with the historical trajectory of code changes rather than just a single parent) and the 'Self-verify' score (using the LLM to predict code quality to bias MCTS selection before running expensive environment evaluations). Additionally, the 'thought-alignment' trick—regenerating the design thought after the code compiles to prevent hallucination drift—is a brilliant engineering fix. This is highly actionable for our work; we should immediately test the MCTS structure, path reasoning, and self-verify PRM-lite scoring in AlgoEvo to improve our sample efficiency.

### [G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design](https://arxiv.org/abs/2602.08253)

**2026-02-09** | Tsinghua University, University of Chinese Academy of Sciences, Northeastern University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Generative Large Neighborhood Search (G-LNS) with co-evolution of destroy and repair operators | *LLM role:* code_writer

> G-LNS extends LLM-based evolutionary search to ALNS by co-evolving Python code for Destroy and Repair operators rather than constructive priority rules. The authors introduce a 'Synergy Matrix' that tracks the performance of specific operator pairs during evaluation, using this data to guide a 'Synergistic Joint Crossover' where the LLM optimizes the coupling between destroy and repair logic. Results are strong: it significantly outperforms FunSearch and EoH on TSP/CVRP and beats OR-Tools on large-scale instances (N=200) under time constraints. The key takeaway for AlgoEvo is the synergy-aware co-evolution mechanism—explicitly tracking and prompting for component interaction is a concrete technique we can apply to multi-agent optimization systems.

### [Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design](https://arxiv.org/abs/2602.23092)

**2026-02-26** | City University of Hong Kong, Southern University of Science and Technology | M=7 P=9 I=8 **MUST-READ** *discuss*

*Method:* Adaptive Iterated Local Search (AILS) with LLM-driven Evolutionary Computation for Automatic Heuristic Design (AHD) of ruin heuristics | *LLM role:* heuristic_generator

> This paper integrates LLM-driven evolutionary search into the AILS framework to evolve 'ruin' heuristics for CVRP, employing a Chain-of-Thought 'voting' mechanism to filter out poor heuristics before expensive evaluation. The results are empirically strong: they claim 8 new Best-Known Solutions on the CVRPLib large-scale benchmark, outperforming HGS and AILS-II. **Key Takeaway:** We should steal the 'acceleration mechanism'—using the LLM to predict heuristic quality via CoT prior to execution—to address the sample efficiency bottleneck in our own evolutionary search loops. This is a direct proof-of-concept that LLM-evolved components can beat hand-crafted SOTA on hard OR instances.

### [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://arxiv.org/abs/2412.14995)

**2024-12-19** | George Mason University, Hanoi University of Science and Technology | M=7 P=8 I=7 *discuss*

*Method:* Adaptive LLM-based Evolutionary Program Search (LLM-EPS) framework combining Harmony Search and Genetic Algorithm with diversity-driven mechanisms | *LLM role:* evolutionary_search

> HSEvo extends LLM-based evolutionary search (LLM-EPS) by integrating a numerical parameter tuning step (Harmony Search) and a token-efficient 'Flash Reflection' mechanism that batches analysis of parent pairs. They report superior results over ReEvo and FunSearch on Bin Packing and TSP, validated by proposed diversity metrics based on code embeddings. **Key Takeaway:** We should implement the hybrid tuning pattern: explicitly parsing LLM-generated code to extract constants and tuning them with a cheap numerical optimizer (rather than asking the LLM to tune parameters), and adopt batched reflections to reduce inference costs.

### [COvolve: Adversarial Co-Evolution of Large-Language-Model-Generated Policies and Environments via Two-Player Zero-Sum Game](https://arxiv.org/abs/2603.28386)

**2026-03-30** | Örebro University | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Co-evolutionary framework leveraging LLMs to generate environments and policies as executable Python code, modeled as a two-player zero-sum game, and solved using Policy Space Response Oracles (PSRO) to compute a mixed-strategy Nash equilibrium (MSNE) over policy populations. | *LLM role:* llm_evolutionary_agent

> COvolve uses LLMs to adversarially co-evolve Python code for both environments (tasks) and policies (agents), using Policy Space Response Oracles (PSRO) to compute a mixed-strategy Nash equilibrium (MSNE) that prevents catastrophic forgetting. The results are backed by solid empirical data across MiniGrid, PyGame, and CARLA, demonstrating that the MSNE approach maintains robust performance across a growing historical archive of environments much better than greedy retention. WHAT WE LEARNED: The use of an empirical payoff matrix and MSNE to evaluate new code against an archive of past opponents/environments is a powerful memory and evaluation mechanism for LLM evolution. This matters immensely for us: we can steal this exact architecture for AlgoEvo or EvoCut by co-evolving hard OR instances (environments) alongside our solver heuristics (policies), using MSNE to ensure our heuristics do not overfit to specific instance types.

### [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://arxiv.org/abs/2402.01145)

**2024-10-14** | Peking University, KAIST, Singapore Management University, Southeast University, PKU-Wuhan Institute for AI | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* Genetic Programming with LLM-based Reflective Evolutionary Search | *LLM role:* heuristic_generator, decomposition_guide

> ReEvo integrates a 'Reflector LLM' into genetic programming that analyzes pairs of heuristics (better vs. worse) to generate textual 'verbal gradients' for crossover and mutation, maintaining a long-term memory of these insights. The results are strong and relevant: they outperform EoH (Evolution of Heuristics) and NCO baselines on TSP, CVRP, and Bin Packing with significantly higher sample efficiency (only ~100 evaluations). The single most useful takeaway is the 'Short-term Reflection' prompting strategy—explicitly asking the LLM to derive a mutation direction by comparing the logic of high-fitness vs. low-fitness parents—which we should immediately test in our AlgoEvo framework to reduce sample costs. This is a direct methodological upgrade for our current evolutionary search pipelines.

### [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://arxiv.org/abs/2407.10873)

**2024-07-15** | City University of Hong Kong, Southern University of Science and Technology | M=4 P=10 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based Evolutionary Program Search (EPS) | *LLM role:* evolutionary_search

> Zhang et al. perform a rigorous benchmarking of major LLM-based evolutionary program search (EPS) methods (FunSearch, EoH, ReEvo) against a simple (1+1)-EPS baseline across four problems and nine LLMs. The results are empirically solid and sobering: the simple (1+1)-EPS baseline—iterative improvement via one-shot prompting—frequently matches or outperforms the complex population-based methods, particularly on bin packing, though EoH remains superior on TSP. **Crucial Takeaway:** We are likely over-engineering our search mechanisms; we must implement a (1+1)-EPS baseline in all future experiments (AlgoEvo, EvoCut) because if our multi-agent systems cannot beat this simple hill-climber, our papers will be rejected for unnecessary complexity. Additionally, they find that larger models (GPT-4) do not strictly guarantee better heuristic search performance compared to smaller, code-specialized models like CodeLlama-7B.

### [MAS$^2$: Self-Generative, Self-Configuring, Self-Rectifying Multi-Agent Systems](https://arxiv.org/abs/2509.24323)

**2025-09-29** | NTU, NUS, USTC, ZJU, BUAA, PKU | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Recursive self-generation of multi-agent systems using a generator-implementer-rectifier tri-agent team, trained via Collaborative Tree Optimization (CTO) with offline reinforcement learning and value-scaled preference alignment. | *LLM role:* multi_agent_system_orchestration_and_rectification

> MAS2 trains a tri-agent system (Generator, Implementer, Rectifier) using offline RL on decision trees to dynamically construct and repair multi-agent workflows. The results are strong, outperforming ADAS and MaAS on standard benchmarks while maintaining Pareto efficiency. The critical takeaway for us is the **Rectifier agent**: rather than discarding failed evolutionary candidates (as we currently do in AlgoEvo), we should implement a dedicated loop to patch runtime errors (e.g., API failures, dimension mismatches). Additionally, their 'Collaborative Tree Optimization' offers a rigorous method to fine-tune the 'Evolver' model using trajectory data, which could replace our current prompt-based meta-heuristics.

### [Global Optimization for Combinatorial Geometry Problems Revisited in the Era of LLMs](https://arxiv.org/abs/2601.05943)

**2026-01-15** |  | M=4 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Global Nonlinear Programming (NLP) using spatial branch-and-bound | *LLM role:* none

> Berthold et al. demonstrate that standard global NLP solvers (SCIP, Xpress) outperform DeepMind's AlphaEvolve on its own benchmarks (circle/hexagon packing, min-max distance) without any learning or evolution. The results are rigorous, improving on 'newly discovered' solutions within minutes using default solver settings. **CRITICAL TAKEAWAY:** We must validate our AlgoEvo results against classical global solvers to ensure we aren't claiming 'discovery' on problems that are trivial for SCIP; furthermore, it suggests a hybrid path where LLMs generate NLP models for solvers rather than evolving raw heuristic code. This is a necessary reality check for our benchmarking strategy.

### [LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression](https://arxiv.org/abs/2505.18602)

**2025-08-08** | Victoria University of Wellington, Michigan State University | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven meta-evolutionary framework for designing selection operators, incorporating semantics-aware selection, bloat control, and domain knowledge into prompts | *LLM role:* evolutionary_search

> Zhang et al. develop a meta-evolutionary framework to evolve selection operators for symbolic regression, achieving state-of-the-art results on SRBench by outperforming expert-designed methods like ε-lexicase. The standout contribution is **semantics-aware crossover**: rather than selecting parents based solely on scalar fitness, they compute complementarity scores using performance vectors across instances, explicitly retrieving parents that solve different subsets of the problem. This effectively treats parent selection as a retrieval task based on behavioral signatures, ensuring the LLM combines distinct functional capabilities. We should immediately implement this complementarity-based parent retrieval in AlgoEvo to improve how we merge heuristics.

### [Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models](https://arxiv.org/abs/2603.10098)

**2026-03-10** | Google DeepMind | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Policy-Space Response Oracles (PSRO) with Large Language Model (LLM) as a code-generating oracle, enhanced by iterative refinement or evolutionary search (AlphaEvolve) | *LLM role:* code_writer

> This paper replaces the deep RL oracle in Policy-Space Response Oracles (PSRO) with an LLM that generates interpretable Python code policies, using AlphaEvolve to iteratively refine the code against opponent meta-strategies. The results are backed by solid empirical metrics, showing that the AlphaEvolve variant achieves competitive exploitability and higher population returns than RL baselines (IMPALA) and CFR+ on Repeated Rock-Paper-Scissors and Leduc Poker. The single most useful takeaway for us is their 'context abstraction' technique—using an LLM to summarize opponent code into natural language to bypass context window limits during evolutionary search. This is highly relevant for our AlgoEvo project; we should immediately discuss implementing their two-level loop (outer meta-game equilibrium, inner AlphaEvolve refinement) and context abstraction for our multi-agent evolutionary search.

### [ImprovEvolve: Ask AlphaEvolve to Improve the Input Solution and Then Improvise](https://arxiv.org/abs/2602.10233)

**2026-02-10** | MIRIAI, FusionBrain Lab, Institute of Numerical Mathematics | M=8 P=5 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search (MAP-Elites) for local search operators (improve, perturb, generate_config) within a basin-hopping framework | *LLM role:* evolutionary_search, initial_program_generator

> Kravatskiy et al. introduce ImprovEvolve, a framework that restricts the LLM to evolving `improve()` (local search) and `perturb()` (mutation) operators, which are then executed by a fixed basin-hopping algorithm. They achieve new state-of-the-art results on Hexagon Packing and the Second Autocorrelation Inequality, demonstrating that this modular approach generalizes to unseen problem sizes where monolithic AlphaEvolve solutions fail. The critical insight is that LLMs are poor at designing global search logic and tuning hyperparameters (LLM edits actively harmed performance), so we should isolate the LLM to generating local moves while keeping the meta-heuristic framework deterministic. We should immediately apply this 'operator-only' evolution strategy to our ALNS research for VRP.

### [tnGPS: Discovering Unknown Tensor Network Structure Search Algorithms via Large Language Models (LLMs](https://arxiv.org/abs/2402.02456)

**2024-06-01** | RIKEN Center for Advanced Intelligence Project, Tencent Inc., Guangdong University of Technology | M=8 P=3 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven automation framework for algorithm discovery via iterative refinement and enhancement using a prompting pipeline | *LLM role:* evolutionary_search

> The authors propose tnGPS, a FunSearch-style framework that evolves Python code for Tensor Network Structure Search by mimicking human innovation stages (categorization, recombination, diversity injection). While the application (Tensor Networks) is niche, the results outperform standard heuristics like TNGA and TNLS. The critical takeaway for us is the 'Knowledge Categorization' phase: they use the LLM to semantically cluster the population of generated algorithms to manage diversity and guide the 'Diversity Injection' step. We should immediately implement this LLM-based population clustering in AlgoEvo to prevent convergence on similar code patterns.

### [Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning](https://arxiv.org/abs/2504.05108)

**2025-08-04** | EPFL, Apple | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Search with Reinforcement Learning (DPO) fine-tuning | *LLM role:* heuristic_generator

> EvoTune augments LLM-based evolutionary search (FunSearch) by iteratively fine-tuning the LLM weights using Direct Preference Optimization (DPO) on the generated programs. The results are robust, consistently outperforming static FunSearch on Bin Packing, TSP, and Hash Code benchmarks by discovering better heuristics faster. The critical takeaway is the use of **Forward KL regularization** in DPO instead of the standard Reverse KL; this prevents the mode collapse that usually kills evolutionary diversity, allowing the model to learn from high-fitness samples while maintaining exploration. This is a direct blueprint for implementing the 'RL-infused evolution' component of our AlgoEvo project.

### [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://arxiv.org/abs/2509.24509)

**2025-09-30** | Tencent, Renmin University of China, City University of Hong Kong | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Experience-Guided Reflective Co-Evolution of Prompts and Heuristics (EvoPH) with island-based elites selection | *LLM role:* heuristic_generator

> EvoPH introduces a co-evolutionary framework where both the heuristic code and the LLM prompts are evolved, utilizing an island model for diversity and a 'strategy sampling' mechanism that dynamically selects mutation types (e.g., parameter tuning vs. rewrite) based on feedback. They report dominating performance over FunSearch and ReEvo on TSP and BPP (e.g., reducing Christofides gap from ~20% to ~5%), though the static performance of baselines suggests the gain comes largely from automating prompt engineering. The most stealable insight is the **Strategy Sampling** module: explicitly defining a pool of mutation operators and using an 'experience' buffer to select them is a practical implementation of the 'planner' concept we need for AlgoEvo. We should also adopt their island migration topology to improve diversity in our parallelized search.

### [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://arxiv.org/abs/2401.02051)

**2024-06-01** | Huawei Noah’s Ark Lab, City University of Hong Kong, Southern University of Science and Technology | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-assisted evolutionary algorithm for co-evolving natural language heuristic descriptions ('thoughts') and executable code implementations | *LLM role:* heuristic_generator

> EoH introduces a dual-track evolutionary framework that evolves both natural language 'thoughts' (heuristic logic) and their corresponding Python code, rather than code alone. On Online Bin Packing, it claims to outperform DeepMind's FunSearch while using only ~2,000 LLM queries (vs FunSearch's millions), and achieves SOTA gaps on TSP and FSSP via Guided Local Search. The critical takeaway is the 'E2' prompt strategy: explicitly asking the LLM to extract common ideas from parent heuristics into a natural language 'thought' before generating code, which acts as a genetic Chain-of-Thought to stabilize mutation. We should immediately implement this 'Thought-then-Code' mutation operator in our AlgoEvo pipeline to address our sample efficiency bottlenecks.

### [CASTER: Breaking the Cost-Performance Barrier in Multi-Agent Orchestration via Context-Aware Strategy for Task Efficient Routing](https://arxiv.org/abs/2601.19793)

**2026-01-27** | University of Houston, China University of Petroleum (East China), Southwest Jiaotong University | M=6 P=8 I=7 *discuss*

*Method:* Dual-Branch Feature Fusion Network for task difficulty estimation with On-Policy iterative training via negative feedback | *LLM role:* task_generator, evaluator

> CASTER implements a context-aware neural router for multi-agent systems that dynamically selects between weak and strong models, reducing inference costs by ~72% compared to a GPT-4o-only baseline. The authors validate this on a custom benchmark across four domains, showing it outperforms cascading strategies (FrugalGPT) by avoiding the 'double-billing' of failed weak calls. The standout takeaway for us is the 'On-Policy Negative Feedback' mechanism: training the router by explicitly relabeling instances where the weak model failed as 'Strong-Required'. We should adapt this active learning logic to train our proxy reward models in AlgoEvo, allowing us to reliably offload expensive evaluations to cheaper proxies without manual annotation.

### [GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models](https://arxiv.org/abs/2509.21593)

**2025-09-25** | Massachusetts Institute of Technology, Stanford University, Technical University of Munich | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Multi-agent LLM framework combining OpenEvolve-based evolutionary search with GeoKnowRAG for geospatial domain knowledge injection | *LLM role:* code_writer, evaluator, prompt_optimizer, evolutionary_search, decomposition_guide

> GeoEvolve augments standard LLM-based evolutionary search (OpenEvolve) with an outer 'researcher' loop that queries a domain-specific RAG (textbooks/papers) to inject theoretical constraints into mutation prompts. On geospatial interpolation tasks, they report 13-21% error reduction over standard evolution, with ablations confirming that retrieved domain knowledge—not just iterative feedback—drives the performance gain. The critical takeaway is the architectural pattern of 'Knowledge-Guided Evolution': instead of relying on the LLM's internal weights for domain theory, they explicitly retrieve and inject theoretical priors (e.g., valid variogram definitions) to steer the search. We should adapt this 'Theory-RAG' outer loop for our AlgoEvo pipeline to force evolved VRP heuristics to respect OR theoretical bounds.

### [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://arxiv.org/abs/2505.12285)

**2025-05-18** | City University of Hong Kong, Southeast University, University of Victoria, Hon Hai Research Institute | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining verbal and numerical guidance for heuristic evolution, achieved by fine-tuning an LLM via reinforcement learning (GRPO) based on heuristic quality, co-evolving the LLM with the search process. | *LLM role:* heuristic_generator_and_fine_tuned_agent

> CALM introduces a hybrid evolutionary framework that fine-tunes the LLM generator *during* the search process using Group Relative Policy Optimization (GRPO), rather than relying solely on prompt evolution. Using a quantized Qwen-7B model on a single consumer GPU, it outperforms GPT-4o-based baselines (FunSearch, EoH) on Bin Packing and VRP benchmarks. The critical takeaway is their reward function design: instead of absolute performance, they reward the *relative improvement* of the generated code over the specific 'parent' heuristics in the prompt, stabilizing the RL signal. We should immediately test this 'online fine-tuning' approach to reduce our API costs and improve sample efficiency in AlgoEvo.

### [Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning](https://arxiv.org/abs/2507.15877)

**2025-09-21** |  | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Execution-guided multi-step neural program synthesis (EG-NPS) with Encoder-Decoder Transformer and tree search | *LLM role:* none

> Ouellette implements an Execution-Guided Neural Program Synthesis (EG-NPS) system for ARC-AGI that conditions the search on the intermediate execution state of every instruction, achieving 80% success on out-of-distribution tasks where TTFT (10%) and standard AlphaEvolve (0-14%) fail. The results are rigorous, using controlled OOD tasks to prove that TTFT relies on in-distribution priors rather than reasoning. The critical takeaway for our AlgoEvo work is the architecture of the 'state-conditioned decoder': instead of blind code generation, we should inject the tokenized execution result of step $t$ into the context for step $t+1$. This is effectively a dense process reward model that solves the sample efficiency bottleneck we face in evolutionary search.

### [EvoVLMA: Evolutionary Vision-Language Model Adaptation](https://arxiv.org/abs/2508.01558)

**2025-08-03** | Chinese Academy of Sciences | M=7 P=4 I=7 *discuss*

*Method:* LLM-assisted two-stage evolutionary algorithm with crossover and mutation operators for optimizing feature selection and logits computation functions in code space | *LLM role:* code_writer

> This paper proposes EvoVLMA, an LLM-based evolutionary framework that searches for Python code to adapt Vision-Language Models (feature selection and logits computation). They demonstrate that **jointly** evolving two coupled algorithmic components fails (worse than random), whereas a **sequential** two-stage evolution strategy yields SOTA results (beating manual baselines by ~1-2%). For our AlgoEvo work, the key takeaway is the infrastructure design: they wrap code execution in restartable web services with a process monitor to handle the high rate of CUDA errors/timeouts in generated code—a practical 'trick' we should adopt to improve our search stability.

### [Automatic programming via large language models with population self-evolution for dynamic job shop scheduling problem](https://arxiv.org/abs/2410.22657)

**2024-10-30** | Huazhong University of Science and Technology | M=6 P=8 I=6 *discuss*

*Method:* LLM-based population self-evolutionary (SeEvo) method for automatic heuristic dispatching rules (HDRs) design | *LLM role:* heuristic_generator

> This paper introduces SeEvo, an LLM-based evolutionary search for Dynamic Job Shop Scheduling heuristics that adds an 'individual self-reflection' loop—prompting the LLM to analyze performance differences of a specific rule before and after mutation—alongside standard population-level reflection. While they claim significant improvements over GP/GEP and DRL, the ablation study reveals only a marginal <1% improvement over the existing ReEvo framework on benchmark instances. The primary takeaway for us is the specific prompt engineering technique of injecting an individual's mutation history (previous code vs. current code performance) into the context to guide the next mutation, which could potentially improve sample efficiency in our own evolutionary loops despite their weak empirical validation.

### [ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas](https://arxiv.org/abs/2602.04296)

**2026-02-04** |  | M=5 P=7 I=7 *discuss*

*Method:* ProxyWar framework, a competitive, execution-based evaluation system orchestrating automated code generation, hierarchical testing, iterative repair loops, and multi-agent tournaments with TrueSkill-based ranking. | *LLM role:* code_writer

> ProxyWar introduces a tournament-based evaluation framework for LLM-generated code, using TrueSkill ratings from game simulations (Sudoku, Poker, etc.) instead of static unit tests. The results are robust (10k+ matches) and reveal a low correlation between Pass@1 and actual win rates; notably, 'reasoning' models like DeepSeek-R1 crush 'coding' models like Qwen-Coder in strategic tasks despite lower static scores. For our evolutionary search work, this confirms that we must move beyond static benchmarks to dynamic, competitive evaluation signals to avoid optimizing for syntax over strategy. We should also prioritize reasoning models over code-specialized ones for our agentic logic generation.

### [RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets](https://arxiv.org/abs/2604.04347)

**2026-04-06** | Independent Researchers | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Elo tournament selection for evolutionary optimization with comparative error reports and Deep Focus refinement | *LLM role:* evolutionary_search

> RoboPhD optimizes LLM agent evolution under tight evaluation budgets by replacing traditional validation sets with an Elo-based tournament on training data and allowing agents to evolve their own diagnostic instrumentation. The results are backed by solid empirical comparisons, outperforming GEPA and Autoresearch on 3 out of 4 benchmarks (ARC-AGI, Text2SQL, DocFinQA) under a strict 1,500 evaluation budget. The single most useful takeaway we can steal is 'self-instrumenting agents'—seeding the initial agent with print() statements and letting the evolutionary process grow its own logging to provide richer Actionable Side Information (ASI) to the LLM optimizer. This paper matters immensely for AlgoEvo; we should immediately test dropping our validation splits in favor of Elo tracking and implement self-instrumenting diagnostics to improve our evolutionary signal without increasing API costs.

### [Learning to Evolve for Optimization via Stability-Inducing Neural Unrolling](https://arxiv.org/abs/2512.11453)

**2026-03-03** | The Hong Kong Polytechnic University, The University of Hong Kong | M=5 P=7 I=7 *discuss*

*Method:* Bilevel meta-optimization with stability-inducing neural unrolling, using a structured Mamba-based neural operator and a gradient-derived composite solver. | *LLM role:* none

> Gao et al. propose L2E, a meta-learned neural optimizer that uses Mamba blocks to parameterize evolutionary operators within a stability-enforcing unrolled loop (Krasnosel'skii-Mann iteration). Results on BBOB and LSGO-1000D are strong, showing it outperforms Transformer-based L2O methods (GLHF) and classical heuristics (DE) in sample efficiency and zero-shot generalization. **Key Takeaway:** We should investigate replacing Transformer-based population encoders in AlgoEvo with Mamba blocks to reduce complexity from quadratic to linear ($O(N)$), enabling larger population sizes in our meta-heuristic search. The theoretical framing of evolution as a fixed-point iteration also offers a rigorous stability constraint we could inject into our RL-guided search policies.

### [Autonomous Algorithm Discovery for Ptychography via Evolutionary LLM Reasoning](https://arxiv.org/abs/2603.05696)

**2026-03-05** | Argonne National Laboratory, Rice University | M=7 P=3 I=8 *discuss*

*Method:* LLM-guided evolutionary search for regularization algorithms combining LLM-driven code generation with semantically-guided crossover and mutation | *LLM role:* evolutionary_search

> This paper applies LLM-guided evolutionary search (similar to FunSearch/AlphaEvolve) to discover novel regularization algorithms for ptychographic image reconstruction. The results are backed by solid empirical metrics, showing up to +0.26 SSIM improvements over unregularized baselines across multiple datasets. The most useful takeaway for us is twofold: first, their 'semantically-guided crossover' explicitly prompts the LLM to analyze two successful parent algorithms and intentionally merge their complementary mathematical strengths, rather than blindly recombining code. Second, the LLM autonomously discovered the benefit of embedding stateful optimizers (like Adam) and iterative sub-loops directly inside a single heuristic step. This matters for our AlgoEvo and VRP work because we should immediately steal this semantic crossover prompting strategy and ensure our evaluation API permits the LLM to generate stateful, multi-pass operators rather than just stateless, single-pass functions.

### [Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery](https://arxiv.org/abs/2507.03605)

**2025-07-04** | Leiden University, University of Stirling | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA framework with 1+1 elitist evolution strategy and dual mutation prompts (code simplification and random perturbation) | *LLM role:* evolutionary_search

> The authors introduce a behavioral analysis framework for LLM-driven algorithm discovery, mapping the 'behavior space' of generated heuristics using Search Trajectory Networks (STNs) and Code Evolution Graphs (CEGs). Results on BBOB (5D) show that a simple 1+1 elitist strategy alternating between 'simplify code' and 'random new' prompts significantly outperforms population-based approaches, effectively balancing exploitation and exploration while preventing code bloat. The primary takeaway is the critical role of a 'simplify' mutation operator—without it, LLM-generated code tends to drift into complexity without performance gains. We should immediately adopt their visualization metrics to debug our own evolutionary search trajectories and implement their 'simplify' prompt strategy in AlgoEvo.

### [Discovering Multiagent Learning Algorithms with Large Language Models](https://arxiv.org/abs/2602.16928)

**2026-02-18** | Google DeepMind | M=10 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* AlphaEvolve, an LLM-powered evolutionary coding agent | *LLM role:* code_writer

> DeepMind applies AlphaEvolve to discover new variants of CFR and PSRO by evolving Python code for regret accumulation and meta-strategy solving. They identify VAD-CFR and SHOR-PSRO, which outperform human-designed SOTA (DCFR, PCFR+) on benchmarks like Leduc Poker and Liar's Dice; results are rigorous, using exact exploitability. The critical takeaway is the shift from evolving static functions to evolving **stateful classes** (e.g., tracking volatility via EWMA inside the accumulator), allowing the LLM to discover dynamic, adaptive schedules—a technique we should immediately port to AlgoEvo.

### [Can Large Language Models Invent Algorithms to Improve Themselves?: Algorithm Discovery for Recursive Self-Improvement through Reinforcement Learning](https://arxiv.org/abs/2410.15639)

**2025-06-10** | NEC Corporation | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Direct Preference Optimization (DPO) for iterative refinement of an algorithm-generating LLM | *LLM role:* evolutionary_search

> Ishibashi et al. propose 'Self-Developing,' a framework where an LLM generates Python code for model merging, evaluates the results, and uses the performance data to fine-tune the generator via DPO in a recursive loop. The results are empirically strong, outperforming human-designed baselines (Task Arithmetic) by 4.3% on GSM8k and demonstrating that the generator explicitly learns better strategies over iterations. **Key Takeaway:** We can replace the static mutation operators in our evolutionary search with a DPO-trained model that learns from the search history—effectively implementing 'learning to search.' This is a direct, actionable upgrade for our AlgoEvo and AlphaEvolve pipelines.

### [QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution](https://arxiv.org/abs/2412.20694)

**2025-02-21** | Westlake University, Zhejiang University, University of Electronic Science and Technology of China | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Algorithm with LLM as variation operator, guided by Quality-Uncertainty Trade-off Criterion (QUTC) using Uncertainty-Inclusive Quality (UIQ) metric | *LLM role:* variation_operator

> QUBE replaces FunSearch's naive score-based parent selection with a UCB algorithm that selects parents based on the *average quality of their offspring* (exploitation) plus an uncertainty term (exploration). The authors demonstrate that a parent's own score is a poor predictor of its ability to evolve further; treating parents as 'bandit arms' based on their lineage statistics yields significantly better results on Bin Packing and TSP with fewer samples. While they fail to beat DeepMind's massive-scale Cap Set record, the methodological insight regarding 'offspring-aware' selection is statistically validated and immediately transferable to our evolutionary search frameworks.

### [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://arxiv.org/abs/2507.17668)

**2025-09-10** | University of Oxford | M=8 P=7 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Empirical comparison of meta-learning algorithms for reinforcement learning algorithm discovery | *LLM role:* code_writer

> Goldie et al. perform a rigorous empirical benchmark comparing LLM-based algorithm proposal against Black-box Evolution Strategies (ES) and various distillation methods. They find that while LLMs are sample-efficient for simple functions, they catastrophically fail to incorporate high-dimensional input features (e.g., the 20+ inputs in OPEN), where Black-box ES remains superior. The most actionable takeaway is 'Same-Size Distillation': distilling a learned black-box algorithm into a fresh network of identical size using synthetic data consistently improves out-of-distribution generalization with zero additional environment samples. We should implement this distillation step immediately and reconsider using LLMs for feature-heavy heuristic components.

### [READY: Reward Discovery for Meta-Black-Box Optimization](https://arxiv.org/abs/2601.21847)

**2026-01-29** |  | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* LLM-based program evolution with a multi-task niche-based architecture, fine-grained evolutionary operators, and explicit knowledge transfer | *LLM role:* evolutionary_search

> READY introduces a multi-task evolutionary framework where LLMs evolve reward functions for multiple MetaBBO algorithms simultaneously, utilizing explicit 'Knowledge Transfer' operators to translate successful logic between distinct tasks. The results are robust, demonstrating superior performance over Eureka and EoH on BBOB benchmarks with a 2-4x reduction in search time due to parallelization and shared heuristics. The most stealable insights are the 'History-Reflection' operator—which prompts the LLM to extrapolate trends from the evolutionary trajectory rather than just mutating the current state—and the cross-niche transfer mechanism, both of which should be implemented in our multi-agent optimization stack immediately.

### [Persona Generators: Generating Diverse Synthetic Personas at Scale](https://arxiv.org/abs/2602.03545)

**2026-02-03** | Google DeepMind | M=8 P=3 I=8 **MUST-READ** *discuss*

*Method:* AlphaEvolve-driven evolutionary search for Persona Generator code optimization with a two-stage LLM-based generation architecture | *LLM role:* evolutionary_search, code_writer, evaluator, research_agent

> Paglieri et al. (DeepMind) apply AlphaEvolve to optimize Python code that generates synthetic personas, explicitly maximizing diversity metrics (convex hull, coverage) in embedding space rather than just fidelity. They achieve >80% coverage of the behavioral space compared to <50% for baselines, proving that evolving the *generator function* is more effective than prompting for diversity. The key takeaway is their two-stage architecture (autoregressive high-level trait generation $\to$ parallel detail expansion), which we should steal to evolve 'Solution Generators' for VRP/OR that inherently resist mode collapse. This validates our direction with AlgoEvo but offers a concrete architectural pattern for maintaining population diversity.

### [LLM-Guided Search for Deletion-Correcting Codes](https://arxiv.org/abs/2504.00613)

**2025-04-01** | Technical University of Munich, Munich Center for Machine Learning | M=7 P=4 I=8 **MUST-READ** *discuss*

*Method:* LLM-guided evolutionary search (FunSearch adaptation) for priority functions | *LLM role:* evolutionary_search

> Weindel and Heckel adapt FunSearch to discover priority functions for the Maximum Independent Set problem (applied to deletion-correcting codes), achieving new SOTA lower bounds for specific lengths (n=12, 13, 16). The critical takeaway for us is their **functional deduplication** step: they hash function outputs on a small subset of data to discard syntactically unique but logically identical programs, which significantly improves sample efficiency by preventing the evaluator from wasting cycles on 'comment changes' or variable renames. Additionally, they demonstrate that optimizing for the single hardest instance generalizes better than averaging performance across a curriculum—a counter-intuitive finding we should test in our reward modeling.

### [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://arxiv.org/abs/2508.03082)

**2025-08-20** | Huawei Noah

































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































Ark Lab, City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary search framework with complementary population management and diversity-aware memetic search | *LLM role:* heuristic_generator

> EoH-S reformulates Automated Heuristic Design (AHD) to evolve a complementary *set* of heuristics rather than a single robust one, proving the objective is submodular and solvable via a greedy strategy. Results are strong and credible: on TSPLib and CVRPLib, their set of 10 heuristics reduces the optimality gap by ~40-60% compared to the top 10 heuristics from FunSearch or ReEvo. **KEY TAKEAWAY:** We should replace standard elitist selection in AlgoEvo with their 'Complementary Population Management' (CPM). By greedily selecting individuals based on marginal contribution to instance coverage (using instance-wise performance vectors), we can automatically generate diverse operator pools for ALNS instead of relying on hand-crafted diversity metrics.

### [CliffSearch: Structured Agentic Co-Evolution over Theory and Code for Scientific Algorithm Discovery](https://arxiv.org/abs/2604.01210)

**2026-04-01** | IBM Research | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-agent-instantiated evolutionary computation over structured scientific artifacts (theory+code or code_only) | *LLM role:* evolutionary_search

> CliffSearch is an LLM-based evolutionary framework that co-evolves algorithm theory and code, using specialized agents for crossover, two-path mutation (exploration vs. repair), and explicit reviewer gating. The results are backed by concrete empirical runs on nanoGPT optimizer discovery and transformer hyper-connection search, demonstrating the discovery of genuinely novel geometric routing and optimizer variants rather than trivial hyperparameter tweaks. The single most useful takeaway is the 'reviewer-gated selection' where an LLM explicitly scores candidates on originality and correctness as a hard survival gate before benchmark scores are considered. This is highly relevant for our AlgoEvo project; we should immediately steal the two-path mutation (novelty vs repair) and the originality hard-gate to prevent our populations from converging on unoriginal benchmark-hacking.

### [EvoX: Meta-Evolution for Automated Discovery](https://arxiv.org/abs/2602.23413)

**2026-02-26** | UC Berkeley, Stanford University, Bespoke Labs | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-level evolutionary process with an inner loop for solution evolution and an outer loop for meta-evolution of search strategies, using LLMs for both solution and strategy generation. | *LLM role:* strategy_generator

> EvoX introduces a two-level LLM-driven evolutionary framework that jointly evolves candidate solutions and the search strategies (parent selection rules, variation operators) used to generate them. The results are highly rigorous and backed by extensive numbers, demonstrating state-of-the-art performance across nearly 200 real-world optimization tasks (math, systems, Frontier-CS) and outperforming AlphaEvolve, OpenEvolve, and ShinkaEvolve. The single most useful takeaway is their use of a 'population state descriptor' combined with a sliding window stagnation detector; when progress stalls, EvoX prompts the LLM with this descriptor and a history of past strategies to generate a new, state-aware search policy (e.g., switching from free-form variation to UCB-guided structural variation). This paper is a mandatory read for us: it is a direct, high-quality implementation of our 'Evolving the Evolver' concept, and we should immediately steal their meta-evolution loop and state descriptor design to fix stagnation issues in AlgoEvo.

### [Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs](https://arxiv.org/abs/2603.07970)

**2026-03-09** | Nanjing University, Huawei Noah’s Ark Lab | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Stagewise Algorithm Design (EvoStage) with multi-agent system and global-local perspective mechanism | *LLM role:* decomposition_guide, code_writer, reflection_agent, evolutionary_search

> EvoStage enhances LLM-based automated algorithm design by decomposing the generation process into sequential stages, using a multi-agent system (coordinator and coders) to iteratively refine code based on real-time intermediate execution feedback. The results are highly credible and backed by strong empirical numbers; it achieves state-of-the-art HPWL on 16 chip placement benchmarks and beats AlphaEvolve/EoH on Bayesian Optimization tasks using an incredibly small budget of just 9 to 25 evaluations. The single most useful takeaway is the shift from black-box end-to-end evaluation to stagewise intermediate feedback, where a coordinator agent reflects on mid-execution metrics to guide the next stage of heuristic design. This matters immensely for our AlgoEvo and MASPRM projects; we should immediately test pausing our VRP/scheduling environments mid-execution to feed intermediate state metrics to a coordinator LLM, which could drastically reduce the number of LLM samples we need to find optimal heuristics.

### [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/abs/2602.16038)

**2026-02-17** | Massachusetts Institute of Technology | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Language-Guided Optimization (LaGO) framework decomposing heuristic discovery into forward, backward, and update stages, utilizing LLMs for reasoned evolution, code-writing analysis, co-evolution of constructive and refinement heuristics, and diversity-aware population management. | *LLM role:* evolutionary_search

> LaGO decomposes automated heuristic design into three explicit modules: evaluation, a code-writing 'Analyst' (backward pass), and a diversity-aware 'Generator' (update), while co-evolving constructive and refinement heuristics. The authors demonstrate significant gains (+0.17 QYI) on PDPTW and Crew Pairing against ReEvo and EoH, showing that joint optimization of initialization and improvement prevents local optima. The critical takeaway is the 'Analyst' module: instead of asking the LLM for text critiques, they ask it to write Python feature extraction functions to statistically characterize solution quality—a technique we should immediately adopt to upgrade our fitness signals in AlgoEvo.

### [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/abs/2409.16867)

**2025-02-04** | City University of Hong Kong, Southern University of Science and Technology | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-based Multi-objective Evolutionary Algorithm with Dominance-Dissimilarity Mechanism | *LLM role:* heuristic_generator

> MEoH extends LLM-based heuristic evolution (like FunSearch/EoH) to multi-objective scenarios (e.g., Gap vs. Runtime) by introducing a 'Dominance-Dissimilarity' mechanism that selects parents based on both Pareto dominance and Abstract Syntax Tree (AST) code distance. The results are credible and strong: on TSP, they find heuristics matching EoH's quality but running 16x faster (1.37s vs 22.4s) by effectively navigating the complexity-performance trade-off. The single most useful takeaway is the **AST-based dissimilarity metric** for population management; we should immediately steal this to prune semantically identical code in our evolutionary loops, thereby forcing exploration and improving sample efficiency. This is a direct upgrade to our current single-objective evolutionary search methods.

### [FunBO: Discovering Acquisition Functions for Bayesian Optimization with FunSearch](https://arxiv.org/abs/2406.04824)

**2024-07-01** | Google DeepMind | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* FunSearch-based evolutionary algorithm for discovering acquisition functions in Python code | *LLM role:* evolutionary_search

> FunBO applies FunSearch to evolve Python code for Bayesian Optimization acquisition functions, evaluating fitness by running full BO loops on synthetic functions. The results are empirically strong, showing that evolved AFs generalize well to out-of-distribution functions and outperform standard baselines like EI and UCB. The most stealable insight is their 'few-shot' adaptation strategy, where a general-purpose heuristic is rapidly fine-tuned on a small set of target instances—a technique we should immediately test for our VRP heuristics. While the method is computationally expensive (brute-forcing the inner loop), the interpretable code outputs provide concrete ideas for dynamic exploration-exploitation trade-offs.

### [Evolving Interdependent Operators with Large Language Models for Multi-Objective Combinatorial Optimization](https://arxiv.org/abs/2601.17899)

**2026-02-01** |  | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Monte Carlo Tree Search for progressive design strategy search with operator rotation evolution | *LLM role:* heuristic_generator

> E2OC introduces a hierarchical search framework where MCTS optimizes 'design thoughts' (textual strategies) rather than raw code, subsequently using these strategies to guide a coordinate-descent-style evolution of interdependent operators. While the computational cost is high due to the inner-loop operator rotation, the results on FJSP/TSP (+20% HV vs expert) and comparisons against FunSearch/EoH demonstrate that explicitly modeling operator coupling is superior to isolated evolution. The critical takeaway for us is the **'strategy-first' search layer**: evolving a semantic blueprint for component interaction *before* code generation prevents the local optima trap of independent component optimization, a technique we should immediately test in AlgoEvo.

### [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440)

**2026-03-12** | Intel Corporation | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* MAP-Elites quality-diversity search with kernel-specific behavioral dimensions, meta-prompt evolution, and template-based parameter optimization | *LLM role:* code_writer, prompt_optimizer

> KernelFoundry is an LLM-based evolutionary framework for GPU kernel optimization that combines MAP-Elites quality-diversity search with meta-prompt co-evolution and gradient-informed mutation hints. The results are rigorously backed by numbers, showing a 2.1x speedup over the AI CUDA Engineer baseline on KernelBench L2 and successful optimization of Llama 3 operations. The single most useful takeaway for us is their 'Gradient-Informed Evolution' technique: they track parent-to-child fitness transitions in the MAP-Elites archive to compute pseudo-gradients across behavioral dimensions, which are then translated into specific natural language mutation hints for the LLM (e.g., 'positive gradient in memory -> hint: add shared memory tiling'). While we do not write low-level GPU kernels, this exact architectural improvement—alongside their meta-prompting to prevent context pollution—is highly transferable and should be immediately tested in AlgoEvo and EvoCut to improve our search signal and sample efficiency.


### Front 0 (51 papers) — STABLE

**Density:** 0.05 | **Methods:** llm_code_generation, program_synthesis, evolution_of_heuristics, llm_evolutionary_search, llm_as_heuristic | **Problems:** heuristic_evolution, algorithm_discovery, automated_algorithm_design, operator_discovery, circle_packing

*Unique methods:* abductive_reflection, abstract_syntax_tree_analysis, activation_steering, adaptive_boltzmann_selection, adaptive_sampling, adaptive_scaling, adaptive_sliding_window, agent_based_framework, agentic_reinforcement_learning, algebraic_graph_construction, algorithm_discovery, algorithm_space_response_oracles, algorithmic_similarity_measurement, ast_manipulation, autodiff, automated_algorithm_design, automated_evaluation, automatic_differentiation, autotuning, bandit_tuned_uip_depth, bandit_tuned_vivification, batch_sampling, behavesim, best_fit_heuristic, best_response_oracle, binary_search, binary_search_algorithm, branch_and_bound, cma_es, code_generation, code_similarity_metrics, combinatorial_reasoning, compressed_watch_architecture, conflict_driven_clause_learning, continual_learning, convex_programming, crossover, cublas_epilogue, cuda_kernel_optimization, dag_execution, darwin_godel_machine, data_algorithm_co_evolution, data_diffusion, data_layout_optimization, decision_trees, derivative_free_optimization, differentiable_sampling, differential_evolution, directed_acyclic_graph, discrete_transformer, diving_heuristics, dynamic_analysis, dynamic_programming, dynamic_time_warping, edit_distance, embedding_based_similarity, embedding_similarity, empirical_evaluation, ensemble_heuristics, entropic_objective, euclidean_distance, evo_mcts, evolutionary_strategy, execution_based_verification, execution_trace_based_similarity, expectation_maximization, explainable_ai, exploratory_landscape_analysis, exponential_moving_average, fireworks_algorithm, functional_disentanglement, git_based_coordination, gradient_correction, graph_algorithms, heuristic_analysis, hgs, hierarchical_clustering, human_in_the_loop, human_llm_collaboration, hybrid_evolutionary_memory, hydra_configuration, hypothesis_testing, id3_algorithm, importance_sampling, improved_seeding, input_space_partitioning, insight_generation, integration_by_parts_reduction, island_based_evolutionary_algorithm, island_evolution, island_models, iterative_debugging, iterative_rounding, jax_framework, kalman_filter, kd_tree, kernel_fusion, langgraph, laporta_algorithm, large_program_database, lineage_based_context_retrieval, lineage_tracking, linear_programming, llm_as_aggregator, llm_as_evolver, llm_as_executor, llm_as_mutation_operator, llm_as_planner, llm_as_policy, llm_as_research_agent, llm_as_summarizer, llm_ensemble, llm_research_agent, llm_rl_trained, local_verification_loop, lora_fine_tuning, machine_unlearning, magnitude_based_pruning, majority_voting, manual_refinement, maturity_aware_heuristic_critic, memoization, meta_game, meta_level_evolution, meta_prompting, mixed_integer_programming, mixed_precision_computing, monte_carlo_simulation, multi_domain_bandit_control, multi_island_evolution, multi_island_model, multi_level_database, multi_uip_clause_learning, multilayer_perceptron, mutation, neural_cellular_automata, novelty_search, numerical_stability, numpy_library, open_ended_evolution, optimization_based_analyzers, pairwise_comparison, pca, penalized_boltzmann_selection, performance_profiling, phylogenetic_graph, piecewise_linear_encoding, plan_execute_summarize, polymorphic_execution_strategies, priority_function_design, problem_solving_trajectory, procedural_generation, program_evolution, program_search, progressive_disclosure, prompt_optimization, puct_search, quasi_random_sampling, race_detection, random_forest, random_hill_climbing, ray_tracing, react_paradigm, reflective_code_synthesis, representation_engineering, representation_learning, reverse_engineering, reward_design, reward_free_evolution, reward_model, rl_dapo, rubric_reward, sat_solvers, self_consistency, self_improving_ai, self_play, semantic_delta, shap_analysis, shapely_library, sigmoid_calibration, smt_solvers, sparse_solver, spectral_graph_theory, structure_based_similarity, surrogate_modeling, swarm_intelligence, symbolic_regression, symmetry_breaking_preprocessing, sympy_library, tabu_search, temperature_annealing, tensor_cores, test_time_learning, three_class_experience_replay, three_way_crossover, tiling, time_series_forecasting, token_based_similarity, tool_use, trajectory_prediction, truncated_svd, ucb, ucb1_bandit, version_control_system, vivification_sensitivity, wasserstein_distance, xgboost
*Shared methods:* alphaevolve, bayesian_optimization, black_box_optimization, co_evolution, code_embedding, direct_preference_optimization, eoh, evolution_of_heuristics, evolution_strategy, evolutionary_algorithm, evolutionary_algorithms, evolutionary_computation, evolutionary_search, funsearch, game_theory, genetic_algorithm, genetic_programming, gepa, gradient_based_optimization, gradient_descent, greedy_algorithm, grpo, hyper_heuristics, in_context_learning, island_model, island_model_ea, large_language_models, llamea, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_context_learning, llm_in_the_loop, llm_prompt_optimization, local_search, map_elites, meta_learning, metaheuristics, monte_carlo_tree_search, multi_agent_reinforcement_learning, multi_agent_system, multi_agent_systems, multi_armed_bandit, offline_reinforcement_learning, openevolve, policy_space_response_oracles, program_synthesis, prompt_engineering, quality_diversity, reevo, reinforcement_learning, retrieval_augmented_generation, reward_shaping, self_improving_search, shinkaevolve, simulated_annealing, slsqp, static_code_analysis, supervised_learning, transformer, zero_sum_game

This research front is characterized by the development of advanced architectural patterns for LLM-guided evolutionary search, moving beyond simple prompt-and-evaluate loops to enable more robust and efficient automated algorithm and program design. Key frameworks like AlphaEvolve, FunSearch, and EoH are being extended with sophisticated mechanisms such as nested evolution, multi-agent systems, semantic deltas, and co-evolution of algorithm structure, prompts, and data. These innovations target diverse domains, including compiler optimization, cache replacement policies, mathematical discovery, scientific computing, mixed-integer linear programming, and combinatorial optimization.

Significant contributions include TIDE's nested framework, which achieved a -7.35% gap on Constructive TSP N=50 by decoupling structure from parameter tuning. StitchCUDA demonstrated 1.72x speedup on GPU program generation using rubric-based agentic RL. BehaveSim improved FunSearch+BehaveSim Top-1 performance by 7.85% on TSP by enforcing behavioral diversity. AlphaEvolve itself, evolving entire code files via diff-based and meta-prompt evolution, beat Strassen on 4x4 matrices and improved Google's Borg scheduler by 0.7%. DeltaEvolve achieved +557.7% on Blackbox Optimization with 64.4% less token consumption using semantic deltas. ParEVO synthesized parallel algorithms with up to 106x speedup on ParEval using fine-tuned LLMs and MAP-Elites, while SATLUTION evolved C++ SAT solvers that outperformed 2025 human competition winners.

This front is rapidly maturing, demonstrating a clear shift towards creating self-improving AI systems capable of meta-learning and dynamic adaptation. The trajectory indicates a strong focus on enhancing sample efficiency, generalization, and robustness through novel representations and feedback loops. Future work will likely integrate more advanced self-reflection, self-modification, and multi-modal feedback mechanisms, leading to increasingly autonomous algorithm discovery systems that can dynamically learn from their own failures and adapt to new problem classes, ultimately "evolving the evolver."

**Papers:**

### [TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design](https://arxiv.org/abs/2601.21239)

**2026-01-29** |  | M=9 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Nested evolutionary framework with TSED-guided island model and co-evolutionary inner loop (UCB-based LLM logic generation + differential mutation for parameter tuning) | *LLM role:* heuristic_generator

> TIDE introduces a nested evolutionary framework that strictly decouples algorithmic structure generation (via LLM) from numerical parameter tuning (via Differential Evolution), managed by a Tree Similarity Edit Distance (TSED) guided island model. Results on 9 COPs (TSP, BPP, etc.) show it consistently outperforms ReEvo and EoH, primarily because the DE layer optimizes constants at zero token cost, preventing the discard of structurally sound but poorly tuned heuristics. The critical takeaway is the necessity of a gradient-free tuning layer for LLM-generated code; relying on LLMs for numerical constants is inefficient and imprecise. We should immediately implement a similar parameter-tuning inner loop in our AlgoEvo framework.

### [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](https://arxiv.org/abs/2603.02637)

**2026-03-03** | University of Minnesota-Twin Cities | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent framework with rubric-based agentic reinforcement learning (GRPO) | *LLM role:* decomposition_guide, code_writer, evaluator

> StitchCUDA automates end-to-end GPU program generation using a multi-agent framework, but its core contribution is a training recipe that solves reward hacking in code optimization. They decompose expensive multi-turn agentic RL into single-turn 'atomic skills' (generation vs. refinement) and use GRPO with an LLM-evaluated 'Rubric Reward' (e.g., 'Did you use tiling?') rather than just sparse outcome metrics. This prevents the model from gaming the system (e.g., wrapping PyTorch code) and forces actual optimization behavior. We should steal the atomic skill decomposition to drastically reduce training costs for AlgoEvo and implement Rubric Rewards to fix our process reward models.

### [Rethinking Code Similarity for Automated Algorithm Design with LLMs](https://arxiv.org/abs/2603.02787)

**2026-03-03** | City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* BehaveSim, a method for measuring algorithmic similarity based on problem-solving trajectories (PSTrajs) quantified using Dynamic Time Warping (DTW) | *LLM role:* heuristic_generator

> Zhang et al. propose BehaveSim, a metric that measures algorithmic similarity by applying Dynamic Time Warping (DTW) to the sequence of intermediate solutions (trajectories) generated during execution, rather than relying on static code analysis. By integrating this into FunSearch and EoH to enforce behavioral diversity, they achieve significant performance gains, notably reducing the optimality gap on TSP by ~7.8% compared to standard FunSearch. **Key Takeaway:** We must stop using code hashes or embedding cosine similarity for population diversity in AlgoEvo; instead, we should instrument generated heuristics to log intermediate states (e.g., partial VRP routes) and cluster them via DTW to prevent convergence to behaviorally identical local optima. This is a mandatory upgrade for our evolutionary search infrastructure.

### [Automatic Design of Optimization Test Problems with Large Language Models](https://arxiv.org/abs/2602.02724)

**2026-02-02** | AGH University of Krakow, Warsaw University of Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary search for Python function generation | *LLM role:* evolutionary_search

> Achtelik et al. adapt LLM-driven evolutionary search (EoH) to generate interpretable Python functions that match specific landscape features (ELA), effectively creating synthetic benchmarks on demand. Unlike prior neural network approaches that fail to scale, this method performs robustly in higher dimensions (3D-5D) and produces portable code. The key takeaway is the capability to procedurally generate 'hard' or specific-property instances; we should immediately adopt this to create a dynamic training curriculum for AlgoEvo, ensuring our evolved metaheuristics generalize beyond standard libraries like BBOB.

### [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

**2025-06-16** | Google DeepMind | M=10 P=10 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary algorithm for code superoptimization, orchestrating an autonomous pipeline of LLMs for code generation, critique, and evolution, grounded by code execution and automatic evaluation. | *LLM role:* evolutionary_search

> AlphaEvolve extends FunSearch by evolving entire code files (rather than single functions) using a 'search/replace' diff format and Gemini 2.0, achieving SOTA results across matrix multiplication (beating Strassen), 50+ open math problems, and Google's production scheduling. The results are exceptionally strong and verified, including deployed improvements to Google's Borg scheduler (0.7% resource recovery) and TPU circuits. The critical takeaway is the move to **diff-based full-file evolution** and **meta-prompt evolution** (evolving the prompt instructions alongside the code), which allows the system to modify architecture and logic rather than just heuristics. This is a mandatory blueprint for the next iteration of our AlgoEvo and EvoCut projects.

### [From Understanding to Excelling: Template-Free Algorithm Design through Structural-Functional Co-Evolution](https://arxiv.org/abs/2503.10721)

**2025-03-13** | Princeton University, Nanyang Technological University, City University of Hong Kong, University of Science and Technology of China, The Hong Kong University of Science and Technology (Guangzhou) | M=9 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven bi-dimensional structural-functional co-evolutionary algorithm | *LLM role:* code_writer, heuristic_generator, decomposition_guide, prompt_optimizer

> Zhao et al. propose CAE, a framework that co-evolves algorithm structure (workflow/call graphs) alongside function implementations, aiming to eliminate the fixed templates required by SOTA methods like FunSearch and EoH. On TSP benchmarks, they report reducing optimality gaps by ~2-5% compared to ReEvo, and in quadratic optimization, the system autonomously discovered numerical stability fixes (e.g., replacing matrix inversion with solvers) that human baselines missed. The critical takeaway is the 'bi-dimensional co-evolution' strategy: explicitly maintaining and mutating a population of control flow graphs separate from the function bodies, which allows the system to escape the local optima imposed by a fixed human-designed harness. We must evaluate if this structural search approach can be integrated into AlgoEvo to automate our harness design.

### [Mining Generalizable Activation Functions](https://arxiv.org/abs/2602.05688)

**2026-02-05** | Google DeepMind | M=8 P=5 I=8 **MUST-READ** *discuss*

*Method:* Evolutionary search powered by AlphaEvolve framework | *LLM role:* code_writer

> Vitvitskyi et al. (DeepMind) utilize AlphaEvolve to discover novel activation functions by evolving Python code on small, synthetic datasets explicitly designed to test OOD generalization (e.g., polynomials, Feynman equations). The results are credible and backed by downstream transfer: discovered functions like `GELU * (1 + 0.5 sinc(x))` outperform baselines on algorithmic reasoning tasks (CLRS-30) while matching standard vision benchmarks. **Key Takeaway:** The 'Small-Scale Lab' methodology—optimizing on cheap, synthetic proxy tasks to find generalizable logic—is a validated strategy to bypass the computational bottleneck of evaluating evolved candidates on large-scale instances. We should steal this 'proxy evolution' setup for AlgoEvo to drastically reduce evaluation costs while targeting generalization in VRP heuristics.

### [Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research](https://arxiv.org/abs/2510.06056)

**2025-10-07** | MIT-IBM Watson AI Lab, IBM Research, University of Notre Dame | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Agent-based framework integrating deep research (planning, searching, writing) with algorithm evolution (coding, evaluation, evolutionary selection) and iterative debugging. | *LLM role:* heuristic_generator, code_writer, evaluator, decomposition_guide, debugger

> DeepEvolve augments the standard evolutionary coding loop (AlphaEvolve) with two critical components: a 'Deep Research' module that searches the web/literature to generate grounded mutation proposals, and an iterative debugging agent that fixes execution errors. While the '666%' improvement on Circle Packing is likely due to a weak baseline (fixed-size vs. generalized), the engineering results are compelling: the debugging agent raises execution success rates from ~13% to ~99% in complex tasks. The key takeaway for our AlgoEvo work is the architecture of generating a text-based 'research proposal' via RAG before attempting code generation, rather than mutating code directly. We should immediately adopt their debugging loop and consider injecting external literature search into our mutation operators to prevent search stagnation.

### [CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization](https://arxiv.org/abs/2510.14150)

**2026-01-06** | Inter&Co., Worcester Polytechnic Institute, Universidade Federal de Minas Gerais | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Islands-based genetic algorithm with modular LLM orchestration, context-aware recombination, adaptive meta-prompting, and depth-based exploitation | *LLM role:* code_writer

> CodeEvolve couples islands-based genetic algorithms with LLMs, utilizing CVT-MAP-Elites for diversity and a specific 'inspiration-based' crossover operator where the LLM integrates logic from high-ranking peer solutions. The results are strong and backed by numbers: they beat AlphaEvolve on 5/9 benchmarks and demonstrate that Qwen3-Coder-30B matches Gemini-2.5 performance at ~10% of the cost. The single most useful takeaway is the implementation of the 'inspiration' operator and the necessity of MAP-Elites over simple elitism to escape local optima in code space. We should immediately benchmark their open-source framework against our internal AlgoEvo builds.

### [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954)

**2025-09-26** | Sakana AI, Vector Institute, University of British Columbia, Canada CIFAR AI Chair | M=10 P=8 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* Darwin Gödel Machine (DGM) with iterative self-modification, empirical validation, and population-based open-ended exploration | *LLM role:* coding_agent, self_modifier, problem_solver, diagnosis_agent

> DGM implements a population-based evolutionary loop where agents modify their own Python source code (tools, memory, flow) to improve performance on coding benchmarks, rather than just optimizing prompts or parameters. Results are strong and verified: it boosts a base agent from 20% to 50% on SWE-bench Verified, matching handcrafted SoTA, with ablations proving the necessity of the population archive (open-endedness) over single-lineage hill climbing. **Key Takeaway:** The 'self-diagnosis' mechanism—feeding execution logs to a model to propose specific *architectural* code changes (e.g., implementing a 'str_replace' tool to fix granular editing errors)—is the exact mechanism we need to implement for evolving our heuristic searchers. This validates that LLM-driven code evolution is viable for complex logic improvement, not just toy tasks.

### [The Art of Being Difficult: Combining Human and AI Strengths to Find Adversarial Instances for Heuristics](https://arxiv.org/abs/2601.16849)

**2026-01-23** | Google DeepMind, University of Bonn, University of Manitoba | M=5 P=8 I=7 *discuss*

*Method:* Human-LLM collaborative program search (Co-FunSearch) | *LLM role:* code_writer

> This paper applies FunSearch to generate adversarial instances for classical OR heuristics (Knapsack, Bin Packing, k-median), successfully breaking long-standing theoretical lower bounds. The results are rigorous: they disprove the output-polynomial time of the Nemhauser-Ullmann algorithm and improve the Best-Fit bin packing bound to 1.5. The key takeaway for our AlgoEvo work is the workflow: the LLM finds 'messy' structural patterns (e.g., repeated floats) which humans then manually generalize into asymptotic proofs. This validates Program Search over vector search but exposes the 'generalization gap'—we should implement a post-processing agent to automate this manual refinement step.

### [DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution](https://arxiv.org/abs/2602.02919)

**2026-02-02** | Microsoft, The Ohio State University | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Momentum-driven evolutionary framework with semantic delta and progressive disclosure within an Expectation-Maximization (EM) process | *LLM role:* program_synthesizer

> DeltaEvolve replaces the standard full-code history in evolutionary search with 'semantic deltas'—structured text summaries capturing the 'from/to' logic of modifications and their hypotheses. Across 5 domains (including BBOB and Symbolic Regression), they demonstrate superior objective scores over AlphaEvolve while reducing token consumption by ~37%. The critical takeaway is the 'Progressive Disclosure' mechanism: treating history as a momentum vector (deltas) rather than a state archive (snapshots) allows us to fit a deeper evolutionary trajectory into the context window. We should immediately test their 'Delta Plan' prompt structure in AlgoEvo to improve sample efficiency and reduce costs.

### [Mathematical exploration and discovery at scale](https://arxiv.org/abs/2511.02864)

**2025-12-22** | Google DeepMind, UCLA, Brown University, Institute for Advanced Study | M=9 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search for programs (search heuristics) that find mathematical constructions | *LLM role:* Generates and mutates code for search heuristics; guides meta-level evolution of search strategies

> DeepMind applies AlphaEvolve to 67 math problems, formalizing the distinction between 'Search Mode' (evolving heuristics for fixed instances) and 'Generalizer Mode' (evolving algorithms that extrapolate from small to large n). Results are rigorous, establishing new bounds on Kakeya sets and 10+ other problems by exploiting verifier loopholes and heuristic specialization. The most critical takeaway for AlgoEvo is Section 44: evolving code that *calls* other LLMs leads to emergent prompt optimization and injection strategies, suggesting a path for our multi-agent optimization work. We must adopt their 'Generalizer' training curriculum (train on small n, test on large n) to fix our scalability bottlenecks.

### [Automated Algorithmic Discovery for Scientific Computing through LLM-Guided Evolutionary Search: A Case Study in Gravitational-Wave Detection](https://arxiv.org/abs/2508.03661)

**2025-11-16** | Tsinghua University, University of Chinese Academy of Sciences | M=9 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided Evolutionary Monte Carlo Tree Search (Evo-MCTS) with reflective code synthesis and multi-scale evolutionary operations | *LLM role:* code_writer, heuristic_generator, evaluator, evolutionary_search

> Evo-MCTS introduces a hybrid search architecture where MCTS manages the exploration-exploitation balance of an evolutionary process, using LLMs for node expansion via novel operators like 'Path-wise Crossover' (synthesizing code from full root-to-leaf trajectories). The results are empirically strong, outperforming standard LLM-evolution baselines (ReEvo) by ~150% on a complex signal processing task. We learned that structuring the evolutionary lineage as a tree and using MCTS Q-values to select parents—rather than standard population selection—drastically improves sample efficiency and solution quality. This is a blueprint for the 'RL-infused evolution' and 'persistent memory' features we have been planning for our own framework.

### [Robust Heuristic Algorithm Design with LLMs](https://arxiv.org/abs/2510.08755)

**2025-10-09** | Microsoft, MIT, Microsoft Research, University of Southern California, The University of Texas at Austin | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Explanation-guided genetic search for heuristic design | *LLM role:* evolutionary_search, decomposition_guide, code_writer, prompt_optimizer

> Karimi et al. introduce 'Robusta', an enhancement to FunSearch that uses a Heuristic Analyzer (solver-based) to identify adversarial inputs and a Suggester LLM to explain *why* the current heuristic fails before generating new code. They demonstrate a 28x improvement in worst-case performance over FunSearch on traffic engineering tasks, with results backed by rigorous comparison against optimal solvers. The critical takeaway is the 'Suggester' intermediate step: converting raw failure instances into natural language coding strategies significantly improves the LLM's ability to fix logic bugs compared to raw samples alone. We should immediately attempt to replicate this 'Analyzer -> Explainer -> Coder' loop for our VRP work, using small-scale solvers to generate counter-examples for our evolved ALNS operators.

### [ArchAgent: Agentic AI-driven Computer Architecture Discovery](https://arxiv.org/abs/2602.22425)

**2026-02-25** | Google DeepMind, Google, University of California, Berkeley | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based evolutionary search using AlphaEvolve | *LLM role:* code_writer

> Gupta et al. (Google/DeepMind) apply AlphaEvolve to generate C++ cache replacement policies, achieving ~1-5% IPC gains over SOTA (Mockingjay/SHiP) on SPEC and Google traces. The results are rigorous and demonstrate that LLM-driven evolution can outperform human experts in mature hardware domains 3-5x faster. WE LEARNED: The documentation of 'simulator escapes'—where the agent exploited a compiler optimization bug to delete memory writes and artificially boost scores—is a critical warning for our own automated benchmarking; we must harden our OR evaluators against adversarial optimization. Additionally, their 'hyperspecialization' approach (tuning policies per workload) validates our interest in instance-specific heuristic evolution.

### [ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution](https://arxiv.org/abs/2509.19349)

**2025-09-17** | Sakana AI | M=9 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary framework with adaptive parent sampling, code novelty rejection-sampling, and bandit-based LLM ensemble selection | *LLM role:* mutation_operator, evaluator, decomposition_guide

> ShinkaEvolve presents an open-source evolutionary framework that drastically improves sample efficiency (e.g., beating AlphaEvolve on Circle Packing with only 150 evaluations vs. thousands) by integrating embedding-based novelty rejection, adaptive parent sampling, and bandit-based LLM selection. The results are credible, backed by code from Sakana AI, and directly target our primary pain point of high API costs/sample inefficiency in evolutionary search. **Key Takeaway:** We must implement their 'novelty rejection sampling' immediately—using a cheap embedding model to filter out semantically similar code mutations (threshold 0.95) before execution is a trivial but high-impact optimization for our AlgoEvo pipeline. This paper proves that smart filtering is superior to the brute-force compute strategies we have been relying on.

### [ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution](https://arxiv.org/abs/2603.02510)

**2026-03-03** | Google DeepMind, Yale University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Coding Agent (ECA) with fine-tuned LLMs for C++ ParlayLib and Rust RPB, guided by compiler, race detector, and performance profiler feedback | *LLM role:* evolutionary_search

> ParEVO synthesizes high-performance parallel algorithms for irregular data by combining fine-tuned LLMs with an Evolutionary Coding Agent (ECA) that uses compilers, dynamic race detectors, and hardware profilers as absolute fitness functions. The results are highly rigorous and backed by extensive hardware benchmarks, showing up to a 106x speedup on ParEval and matching or beating expert human baselines on complex graph problems (e.g., 4.1x speedup on Maximal Independent Set). The single most useful takeaway for us is their evolutionary architecture: they use MAP-Elites (categorizing by code length, cyclomatic complexity, and sync primitives) to prevent diversity collapse, and they fine-tune their base models using Direct Preference Optimization (DPO) on past evolutionary trajectories to drastically improve sample efficiency. This is a must-read paper that directly advances our primary focus on LLM evolutionary search, providing concrete techniques we should immediately steal for AlgoEvo.

### [Let the Barbarians In: How AI Can Accelerate Systems Performance Research](https://arxiv.org/abs/2512.14806)

**2025-12-22** | UC Berkeley | M=7 P=10 I=9 **MUST-READ** *discuss*

*Method:* Evolutionary search using LLM-based Prompt Generator, Solution Generator, Evaluator, Storage, and Solution Selector components | *LLM role:* solution_generator, prompt_generator, evaluator

> Cheng et al. (UC Berkeley) perform a rigorous empirical evaluation of LLM evolutionary search (ADRS) across 10 systems problems, achieving SOTA results on MoE load balancing (13x speedup via rediscovering Hamilton's Apportionment) and cloud scheduling. The results are real and backed by code, comparing frameworks like OpenEvolve, GEPA, and ShinkaEvolve. **Key Takeaway:** Their 'Best Practices' section offers concrete engineering constraints we should adopt: specifically, that 'moderate' feedback (worst-k cases) outperforms 'detailed' feedback (prevents overfitting), and that restricting mutations to diff-based edits is essential to prevent reward hacking. This paper validates our core research thesis while providing the benchmarks we now need to beat.

### [MadEvolve: Evolutionary Optimization of Cosmological Algorithms with Large Language Models](https://arxiv.org/abs/2602.15951)

**2026-02-17** | University of Wisconsin-Madison | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary optimization with nested parameter tuning | *LLM role:* code_writer, mutation_operator, report_generator

> MadEvolve extends AlphaEvolve by embedding a gradient-based optimization loop (via JAX) inside the fitness evaluation, allowing the LLM to focus purely on code structure while an optimizer (Adam) handles continuous parameters. They demonstrate 20-30% performance gains on complex cosmological reconstruction tasks, validated on held-out simulations. The critical takeaway is the architectural pattern: prompt the LLM to write differentiable code rather than tuning constants, and use a UCB1 bandit to dynamically select between cheap and expensive models. We should immediately adopt the differentiable inner-loop strategy for our continuous heuristic search projects.

### [CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges](https://arxiv.org/abs/2603.11863)

**2026-03-12** | Tsinghua University, Peking University, Southern University of Science and Technology, University of Bristol, The Hong Kong University of Science and Technology (Guangzhou), Xi’an Jiaotong University | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Automated benchmark construction via reverse engineering and self-play; EvoRePE: Inference-time representation engineering for latent space steering. | *LLM role:* code_writer, constraint_generator, evaluator, prompt_optimizer, decomposition_guide, evolutionary_search

> Wang et al. introduce CreativeBench to evaluate LLM code generation creativity and propose EvoRePE, a representation engineering technique that extracts a 'creativity vector' from AlphaEvolve search trajectories to steer model activations at inference time. The results are backed by solid empirical evaluations, showing that injecting this vector improves novelty and correctness even without running the full evolutionary search. THE SINGLE MOST USEFUL TAKEAWAY: We can run our evolutionary search (e.g., AlgoEvo) offline to collect (base_heuristic, evolved_heuristic) pairs, compute the PCA of their hidden state differences, and inject this vector during standard inference to force the model into an exploratory mode. This is highly relevant to our work as it offers a completely new, training-free mechanism to solve the sample efficiency and scalability bottlenecks in LLM evolutionary search.

### [ThetaEvolve: Test-time Learning on Open Problems](https://arxiv.org/abs/2511.23473)

**2025-11-28** | Microsoft, University of Washington, Carnegie Mellon University, University of Wisconsin-Madison, University of California, San Diego | M=10 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Program evolution with test-time Reinforcement Learning (RL) using GRPO algorithm | *LLM role:* code_writer

> ThetaEvolve integrates test-time reinforcement learning (GRPO) directly into an AlphaEvolve-style loop, allowing a single 8B model to learn from its own successful mutations and achieve new SOTA bounds on Circle Packing and Autocorrelation inequalities. The results are rigorous, showing that RL applied to the *dynamic* environment (sampling from the evolving database) vastly outperforms RL on static prompts or pure inference search. The most stealable insight is the 'lazy penalty' mechanism—penalizing semantically equivalent code or stagnation—which forces the RL policy to learn genuine exploration strategies rather than memorization. This is a blueprint for the 'RL-infused evolution' milestone in our AlgoEvo roadmap.

### [AlphaResearch: Accelerating New Algorithm Discovery with Language Models](https://arxiv.org/abs/2511.08522)

**2025-11-11** | Yale, NYU, Tsinghua, ByteDance | M=7 P=6 I=7 *discuss*

*Method:* Autonomous research agent with dual research environment combining execution-based verification and simulated real-world peer review | *LLM role:* research_agent

> AlphaResearch introduces a 'dual environment' for algorithm discovery: it generates natural language research ideas, filters them using a reward model fine-tuned on ICLR peer reviews, and then executes the surviving ideas. While it claims to beat human baselines on Packing Circles, the improvement is marginal (<0.1%) and it fails to improve upon baselines in 6/8 benchmark problems. The key takeaway for us is the mechanism of an 'Idea Critic'—using a learned reward model to filter the search space at the prompt level before wasting compute on execution—which directly addresses our sample efficiency goals in evolutionary search.

### [Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM](https://arxiv.org/abs/2510.11121)

**2025-10-13** | Nanyang Technological University, Singapore, Singapore Management University, Singapore, Nanjing University of Information Science and Technology, China | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning (DAPO) fine-tuning of LLM for crossover operator generation within Hybrid Genetic Search (HGS) | *LLM role:* heuristic_generator

> Zhu et al. fine-tune a Qwen-14B model using Reinforcement Learning (DAPO) to generate C++ crossover operators for the state-of-the-art HGS solver. Unlike typical prompting papers, they demonstrate that a small, specialized model can improve upon expert-designed components in a highly optimized solver, achieving superior results on CVRPLIB (up to 1000 nodes) where GPT-4o fails. The most stealable insight is their **AST-based anti-plagiarism reward**, which penalizes the model for generating code structurally identical to the prompt examples, effectively forcing exploration and preventing mode collapse—a technique we should immediately adopt for our evolutionary search agents. This confirms we should pivot from pure prompting to RL-finetuning for our code-generation agents.

### [Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve](https://arxiv.org/abs/2601.21096)

**2026-01-28** | Google DeepMind, Google, Cornell University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-powered coding agent (AlphaEvolve) with evolutionary search and black-box autotuning (Vizier) | *LLM role:* code_writer

> Magellan couples AlphaEvolve with a black-box autotuner (Vizier) to evolve C++ compiler heuristics, achieving >5% binary size reduction in LLVM and beating both human experts and prior neural policies. The results are rigorous, validated on production workloads and showing temporal generalization. **The critical takeaway is the 'Hierarchical Search' strategy:** rather than asking the LLM to write fully specified code, they prompt it to generate *templates* with exposed parameters (flags), delegating numerical tuning to a cheap external optimizer. This directly addresses the sample efficiency issues we face in AlgoEvo; we should immediately steal this architecture to separate structural evolution from parameter tuning.

### [LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm](https://arxiv.org/abs/2512.24077)

**2025-12-30** |  | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Plan-Execute-Summarize (PES) paradigm integrated with Hybrid Evolutionary Memory (Multi-Island, MAP-Elites, Adaptive Boltzmann Selection) | *LLM role:* planner, executor, summarizer

> LoongFlow replaces the standard stochastic mutation operator in LLM evolutionary search with a 'Plan-Execute-Summarize' (PES) cognitive loop. Instead of random code changes, a Planner retrieves the 'intent' and 'summary' of the parent solution's lineage to generate a directed hypothesis, which is then executed and summarized for the next generation. The authors demonstrate a 60% reduction in evaluations and a 100% success rate on AlphaEvolve tasks where standard methods fail or stagnate. The critical takeaway is the 'Lineage-Based Context Retrieval' mechanism: explicitly passing the parent's plan and retrospective summary to the child allows for directed rather than random walks in the search space. We must implement this PES loop in AlgoEvo immediately to fix our sample efficiency issues.

### [C-Evolve: Consensus-based Evolution for Prompt Groups](https://arxiv.org/abs/2509.23331)

**2025-09-27** | Westlake University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Island-based evolutionary algorithm with Exponential Moving Average (EMA) voting score as fitness, optimizing groups of prompts for consensus via majority voting or LLM-based aggregation. | *LLM role:* evolver, consensus_aggregator

> C-Evolve modifies island-based evolution to optimize a group of prompts that maximize consensus accuracy (majority vote) rather than individual performance. The authors introduce a 'voting score' fitness function—calculated via Exponential Moving Average (EMA) of an individual's contribution to sampled groups—which successfully drives the population toward diverse, complementary strategies that outperform ensembles of individually optimized prompts (beating AlphaEvolve by ~4% on Qwen3-8B). The single most actionable takeaway is the **EMA voting score mechanism**: we can steal this exact fitness formulation to evolve portfolios of complementary VRP heuristics in AlgoEvo, replacing our current focus on converging to a single 'best' solver. While the benchmarks are standard (MATH, HotpotQA), the method offers a robust solution to the 'single heuristic limitation' we face in OR.

### [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)

**2026-01-22** | Stanford University, NVIDIA, UC San Diego, Together AI, Astera Institute | M=9 P=10 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning with Entropic Objective and PUCT-based Reuse at Test Time | *LLM role:* Policy being optimized to generate solutions/code

> TTT-Discover introduces a method to fine-tune an LLM (gpt-oss-120b) *during* inference on a single test problem using RL, replacing the frozen-model evolutionary search of AlphaEvolve. They employ a novel 'entropic objective' that optimizes for the single best solution (discovery) rather than expected return, combined with PUCT-based state reuse. The results are empirically rigorous, setting new SOTA on Erdős’ problem, GPU kernel optimization, and AtCoder contests, directly beating AlphaEvolve and ShinkaEvolve. The critical takeaway is that for hard discovery tasks, shifting the model's distribution via online updates is superior to context-based search; we should immediately test their entropic objective in our AlgoEvo pipeline.

### [DHEvo: Data-Algorithm Based Heuristic Evolution for Generalizable MILP Solving](https://arxiv.org/abs/2507.15615)

**2025-07-21** | Harbin Institute of Technology, Huawei Noah’s Ark Lab, Nanyang Technological University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Data-Algorithm Co-evolution Framework (DHEvo) with LLM-based Multi-Agent Evolution System (MA-Evolution System) | *LLM role:* code_writer

> DHEvo introduces a 'data-algorithm co-evolution' framework that iteratively evolves heuristic code while simultaneously filtering the training instance set to retain only 'representative' instances (those where current heuristics perform well/stably). Empirical results on SCIP diving heuristics show it outperforms FunSearch and EoH by ~60% on Setcover while significantly reducing performance variance, validating the claim that dynamic data curation prevents overfitting. The key takeaway is the counter-intuitive curriculum strategy: rather than training on the hardest instances, filtering for instances with 'regular' feasible regions (high fitness) stabilizes the evolutionary search for code. We should immediately test this dynamic instance filtering in AlgoEvo to improve sample efficiency and generalization.

### [Programmatic Representation Learning with Language Models](https://arxiv.org/abs/2510.14825)

**2025-10-16** | Harvard University, Stanford University | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-synthesized programmatic feature functions combined with decision tree predictors, using Features FunSearch (F2) and Dynamic ID3 (D-ID3) algorithms | *LLM role:* code_writer

> The authors propose two algorithms, F2 (Features FunSearch) and D-ID3 (Dynamic ID3), to learn programmatic features for decision trees. D-ID3 is particularly novel: instead of evolving a global heuristic, it calls the LLM at *each split node* to generate a feature that discriminates the specific data subset at that leaf. Results are strong on Chess (matching Transformers trained on 250x more data) and Text, though the Image results (MNIST) are trivial. **Key Takeaway:** The D-ID3 architecture—using the solver's current state (leaf node data) to prompt the LLM for *local* code generation—is a powerful pattern we should steal for our VRP solvers (e.g., evolving local repair operators for specific route bottlenecks) and EvoCut work.

### [EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery](https://arxiv.org/abs/2512.13857)

**2025-12-17** | aiXplain Inc | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search on a multi-alternative Quality-Diversity Directed Acyclic Graph (DAG) representation with alternative-level performance statistics and deterministic self-repair | *LLM role:* evolutionary_search

> EvoLattice replaces the standard 'overwrite-based' evolution of monolithic programs with a persistent DAG where each node holds multiple alternative implementations, evaluating all valid combinatorial paths to compute fine-grained performance statistics for every micro-operator. The results are strong: it outperforms AlphaEvolve and FunSearch styles on NAS-Bench-Zero by explicitly preserving diversity and enabling surgical, data-driven pruning rather than blind mutation. The critical takeaway is the 'alternative-level statistic' mechanism: by aggregating performance across all paths a component participates in, they generate a high-fidelity signal that tells the LLM exactly which lines of code are working, effectively solving the sparse reward problem in code evolution. We should immediately discuss refactoring our AlgoEvo representation to support this multi-alternative graph structure, as it maximizes signal extraction per LLM call.

### [Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search](https://arxiv.org/abs/2601.19622)

**2026-01-27** |  | M=7 P=5 I=8 *discuss*

*Method:* Evolutionary Heuristic Design (EoH) framework with Algorithmic-Contextual Prompt Augmentation (A-CEoH) | *LLM role:* heuristic_generator

> This paper introduces 'Algorithmic-Contextual EoH' (A-CEoH), which injects the actual source code of the search algorithm (e.g., the A* driver loop, neighbor generation) into the LLM prompt alongside the problem description. Experiments on the Unit-Load Pre-Marshalling Problem and Sliding Puzzle Problem demonstrate that this algorithmic context allows a 32B parameter model (Qwen2.5-Coder) to generate heuristics superior to those from GPT-4o and human experts. The results are credible and backed by comparisons against optimal baselines. The key takeaway is a transferable 'prompt trick': explicitly showing the LLM the code that *calls* its generated function aligns the heuristic significantly better with the search dynamics than natural language descriptions alone. We should immediately test injecting our ALNS/search driver code into our evolutionary prompt templates.

### [Evolutionary Discovery of Reinforcement Learning Algorithms via Large Language Models](https://arxiv.org/abs/2603.28416)

**2026-03-30** | Machine Perception and Interaction Lab, Örebro University, Sweden | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Evolutionary search over executable learning update rules using LLM-guided macro mutation and diversity-aware crossover | *LLM role:* evolutionary_search

> This paper evolves executable reinforcement learning update rules using LLMs as macro-mutation and crossover operators, explicitly forbidding standard RL mechanisms to force the discovery of novel algorithms. The results are backed by solid empirical evaluations on Gymnasium benchmarks, showing the evolved algorithms match or beat standard baselines like PPO and SAC on several tasks, though they struggle on a few complex continuous control environments. The single most useful takeaway for us is their diversity-aware crossover, which uses normalized Levenshtein distance to penalize recombining near-duplicate parents, alongside a post-evolution step where the LLM proposes bounds for internal scalar parameters before a final sweep. We should immediately test the Levenshtein-penalized crossover in AlgoEvo to prevent diversity collapse, and adopt the LLM-bounded HPO step to ensure we aren't discarding good heuristics simply because of bad default scalar parameters.

### [Reinforced Generation of Combinatorial Structures: Hardness of Approximation](https://arxiv.org/abs/2509.18057)

**2025-12-19** | Google DeepMind, Google, University of California, Berkeley | M=9 P=5 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search for combinatorial structures and verification procedures | *LLM role:* evolutionary_search

> Nagda et al. utilize AlphaEvolve to discover combinatorial gadgets that improve hardness of approximation bounds for MAX-CUT and TSP, validating findings with formal proofs. The standout contribution is not the hardness results themselves, but the methodology: they tasked AlphaEvolve with optimizing the *verification code* (checking correctness against a slow ground truth), achieving a 10,000x speedup that enabled searching gadgets of size 19 (vs. 11 previously). We should immediately adopt this 'evolve the verifier' loop for our computationally expensive fitness functions in AlgoEvo to break current scalability limits.

### [Game-Theoretic Co-Evolution for LLM-Based Heuristic Discovery](https://arxiv.org/abs/2601.22896)

**2026-02-09** | Tsinghua University, Chinese Academy of Sciences, University of Chinese Academy of Sciences, AiRiA | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Algorithm Space Response Oracles (ASRO), a game-theoretic framework for program-level co-evolution between solver and instance generator, extending PSRO to discrete program space with LLM-based best-response oracles | *LLM role:* program_synthesis, evolutionary_search

> ASRO adapts Policy Space Response Oracles (PSRO) to code generation, treating heuristic discovery as a zero-sum game where a 'Solver' evolves to minimize gaps and a 'Generator' evolves to create adversarial instances. The results are compelling: it consistently beats the static EoH baseline on TSPLIB and CVRPLIB, proving that adversarial training yields better generalization than training on fixed distributions. The critical takeaway is the architecture: explicitly co-evolving an 'Instance Generator' program alongside the solver prevents overfitting and exposes edge cases (like specific geometric traps in TSP) that static benchmarks miss. This is a direct upgrade to our AlgoEvo/AlphaEvolve pipelines, though it incurs higher computational costs due to the evaluation matrix required for the meta-game.

### [Structuring Collective Action with LLM-Guided Evolution: From Ill-Structured Problems to Executable Heuristics](https://arxiv.org/abs/2509.20412)

**2025-12-03** | University of Waterloo, Royal Bank of Canada | M=7 P=5 I=7 *discuss*

*Method:* LLM-driven evolutionary search for Python code heuristics and natural language messages (ECHO-MIMIC framework) | *LLM role:* variation engine, generator, modifier, fixer, agent (simulation) LLM

> ECHO-MIMIC presents a framework that first uses LLM-guided evolution to generate Python heuristics for agents (ECHO), and subsequently evolves natural language 'nudges' (MIMIC) to persuade simulated agents to adopt these global-optimal policies. While the experiments rely on synthetic data for agriculture and EV charging, the approach outperforms DSPy and AutoGen baselines in driving collective action. The most valuable takeaway is the architectural separation of 'policy discovery' (code evolution) and 'adoption mechanism' (message evolution)—a pattern we could adapt to evolve incentive structures or negotiation protocols in our multi-agent optimization systems (MASPRM/HERMES). The analysis of code complexity (Halstead metrics) versus fitness also provides a useful empirical reference for our observability work.

### [Autonomous Code Evolution Meets NP-Completeness](https://arxiv.org/abs/2509.07367)

**2025-09-09** | NVIDIA Research, University of Maryland | M=9 P=9 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* Autonomous agent-based code evolution system with Planning and Coding LLM agents | *LLM role:* evolutionary_search

> SATLUTION extends LLM evolutionary search to full-scale C++ repositories, autonomously evolving SAT solvers that outperform 2025 human competition winners using only 2024 training data. The results are highly rigorous, backed by 90k CPU hours of distributed evaluation and strict correctness proofs (DRAT), showing a clear monotonic improvement trajectory. The single most stealable insight is the **self-evolving rule system**: the agent autonomously updates a persistent set of markdown constraints (e.g., forbidden patterns, testing protocols) based on post-cycle failure analysis, effectively creating 'institutional memory' that prevents regression in long-horizon search. We must implement this meta-learning loop in AlgoEvo immediately to move beyond single-file optimization.

### [Reinforced Generation of Combinatorial Structures: Ramsey Numbers](https://arxiv.org/abs/2603.09172)

**2026-03-11** | Google DeepMind, Google, University of California, Berkeley | M=8 P=3 I=9 **MUST-READ** *discuss*

*Method:* AlphaEvolve, an LLM-based code mutation agent | *LLM role:* evolutionary_search

> Nagda et al. (DeepMind) apply the AlphaEvolve framework to discover novel stochastic search algorithms that improve lower bounds for five classical Ramsey numbers and match SoTA on 23 others. The results are mathematically verified and represent genuine SoTA advances in extremal combinatorics, proving the framework's capability to generate highly specialized, non-trivial heuristics. The single most useful takeaway for us is their meta-algorithm's scoring function: instead of only rewarding valid states, they evaluate a larger, infeasible 'prospect' state and provide a dense, continuous reward based on its violation count relative to a random baseline. We should immediately steal this 'prospect evaluation' trick for AlgoEvo to smooth the reward landscape when evolving heuristics for highly constrained OR problems like VRP, where finding strictly feasible intermediate solutions is a bottleneck.

### [WirelessAgent++: Automated Agentic Workflow Design and Benchmarking for Wireless Networks](https://arxiv.org/abs/2603.00501)

**2026-02-28** | The Hong Kong University of Science and Technology, Shenzhen University | M=7 P=4 I=8 *discuss*

*Method:* Domain-adapted Monte Carlo Tree Search (MCTS) for program search with LLM-based code mutation | *LLM role:* evolutionary_search, executor

> Tong et al. automate the design of LLM agent workflows for wireless tasks using a domain-adapted MCTS over a code-structured search space. The results are rigorously backed by numbers, showing significant improvements over baselines like AFlow and ADAS with a total search cost under $5. The single most useful takeaway for us is their method for handling search efficiency and noise: they use a zero-cost rule-based 'heuristic critic' to pre-screen code mutations via AST analysis before expensive LLM evaluation, and a 3-class experience replay (using an epsilon-threshold) to prevent the optimizer from chasing evaluation noise. While we do not care about wireless networks, we should absolutely steal these two tricks to reduce API costs and stabilize fitness signals in our AlgoEvo pipeline.

### [Online Operator Design in Evolutionary Optimization for Flexible Job Shop Scheduling via Large Language Models](https://arxiv.org/abs/2511.16485)

**2026-01-22** | City University of Hong Kong, Guangdong University of Technology | M=7 P=8 I=7 *discuss*

*Method:* Genetic Algorithm with LLM-driven online operator design and adaptive operator evolution | *LLM role:* evolutionary_search

> LLM4EO embeds an LLM directly into the Genetic Algorithm loop to dynamically generate and replace gene-selection operators whenever the population stagnates, rather than training them offline. Results on FJSP benchmarks (Brandimarte, Fattahi) show a 3-4% improvement over static GA and GP, with convergence plots demonstrating that LLM interventions successfully break local optima. The most stealable insight is the 'Perception and Analysis' prompt structure: it forces the LLM to explicitly diagnose *why* the current population is stuck (based on fitness stats) before generating new code, a mechanism we should port to AlgoEvo to handle search stagnation. This validates the viability of online, state-aware LLM intervention in OR scheduling problems.

### [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)

**2025-10-10** | UC Berkeley | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary search (MAP-Elites and island models) with automated code generation and empirical evaluation | *LLM role:* code_writer, reasoning_agent, feedback_generator

> The authors apply OpenEvolve (an AlphaEvolve-style framework) to 11 computer systems problems, achieving significant gains over human baselines, such as a 5.0x speedup in MoE expert placement and 26% cost reduction in cloud scheduling. The results are empirically rigorous, relying on high-fidelity simulators rather than toy problems. For us, the key takeaway is the engineering recipe: using an ensemble of reasoning models (o3) for exploration and fast models (Gemini) for diversity, combined with a specific 'failure taxonomy' to debug search stagnation. This is immediate proof-of-concept for your 'GPUSched' and 'AlgoEvo' projects; we should adopt their ensemble strategy and simulator-first evaluation pipeline.

### [Procedural Generation of Algorithm Discovery Tasks in Machine Learning](https://arxiv.org/abs/2603.17863)

**2026-03-18** | University of Oxford, University College London, University of California, Santa Barbara, University of Wisconsin–Madison, Delft University of Technology | M=6 P=8 I=8 **MUST-READ** *discuss*

*Method:* Procedural generation of algorithm discovery tasks using configurable parameters for domains, modules, and datasets | *LLM role:* research_agent, prompt_optimizer

> This paper introduces DiscoGen, a procedural generator that combinatorially creates millions of algorithm discovery tasks (varying domains, editable modules, and datasets) with strict meta-train/meta-test splits to evaluate and train Algorithm Discovery Agents (ADAs). The results are backed by extensive empirical evaluation of open-source LLMs on a fixed subset (DiscoBench), demonstrating that current ADAs struggle with multi-module discovery and that prompt-tuning over a diverse set of procedurally generated tasks significantly improves generalization. The single most useful takeaway is the combinatorial task generation approach (toggling which modules are editable vs. fixed), which provides a brilliant blueprint for creating an autocurriculum to train our 'evolver' agents. This matters immensely for us; we should immediately consider using DiscoGen to evaluate AlgoEvo, and adapt their procedural task generation strategy to create diverse OR/routing environments for our own RL-infused evolutionary search training.

### [GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms](https://arxiv.org/abs/2511.17592)

**2025-11-17** | Sber, Artificial Intelligence Research Institute (AIRI) | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* MAP-Elites quality-diversity algorithm with LLM-driven mutation operators (rewrite-based or diff-based) and bidirectional lineage tracking | *LLM role:* evolutionary_search

> GigaEvo is an open-source reproduction of the AlphaEvolve framework that implements MAP-Elites with an asynchronous DAG execution engine, successfully reproducing SOTA results on Heilbronn triangles and beating FunSearch on Weibull Bin Packing. The results are credible and backed by code, specifically highlighting that 'rewrite-based' mutation outperforms 'diff-based' approaches for open-weights models—a crucial engineering constraint for us. The most actionable takeaway is their 'bidirectional lineage tracking' mechanism, which enriches mutation prompts by analyzing both how a program improved over its ancestor and how its descendants further improved, a technique we should steal for AlgoEvo's mutation operator. Their negative result regarding multi-island MAP-Elites (added complexity, no gain) suggests we should deprioritize similar complex topologies.

### [Explainable AI-assisted Optimization for Feynman Integral Reduction](https://arxiv.org/abs/2502.09544)

**2025-02-13** | Peking University, Universit
Z
rich, Beijing Computational Science Research Center | M=7 P=3 I=8 *discuss*

*Method:* FunSearch algorithm for developing a priority function to optimize seeding integrals in Integration-by-Parts (IBP) reduction | *LLM role:* heuristic_generator

> Song et al. apply FunSearch to evolve priority functions for Feynman integral reduction, achieving up to 3058x reduction in seeding integrals compared to standard heuristics. The results are rigorous, enabling previously impossible multi-loop calculations. The critical insight for us is the successful transfer of heuristics evolved on trivial 1-loop instances (fast evaluation) to complex 5-loop problems without retraining. We should adopt this 'evolve-on-toy, deploy-on-giant' evaluation protocol to drastically reduce compute costs in our VRP and SAT solver evolutionary search pipelines.

### [LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI](https://arxiv.org/abs/2601.21511)

**2026-01-29** | Leiden University | M=8 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA-SAGE, an LLM-driven evolutionary algorithm that integrates structural feedback from Explainable AI (SHAP) analysis of Abstract Syntax Tree (AST) code features to guide mutations. | *LLM role:* evolutionary_search

> LLaMEA-SAGE augments LLM-based evolutionary search by extracting AST features (complexity, graph metrics) from generated code, training a surrogate model to predict fitness from these features, and using SHAP analysis to generate natural language prompts that guide the LLM to modify specific structural properties (e.g., 'increase cyclomatic complexity'). On the MA-BBOB benchmark, it outperforms state-of-the-art methods (MCTS-AHD, LHNS) and converges faster than vanilla LLaMEA, although the authors honestly report that statistical significance was limited (p=0.44) due to small sample sizes (5 runs). The critical takeaway for us is the pipeline of using static code analysis as a feedback signal—we can immediately steal this 'SAGE' loop to guide AlgoEvo or EvoCut by telling the LLM *how* to structurally mutate code based on surrogate correlations, rather than just hoping for random improvements.

### [Beyond Algorithm Evolution: An LLM-Driven Framework for the Co-Evolution of Swarm Intelligence Optimization Algorithms and Prompts](https://arxiv.org/abs/2512.09209)

**2025-12-10** | Peking University | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Collaborative evolution of Fireworks Algorithm operators and prompt templates, driven by a single LLM | *LLM role:* evolutionary_search

> The authors introduce a co-evolutionary framework where both the optimization algorithm (Fireworks Algorithm operators) and the prompt templates used to generate them are evolved simultaneously by the LLM. The results demonstrate a massive performance jump on constrained Aircraft Landing problems (from ~56% with FunSearch to 100% with their method), suggesting that static prompts are a primary failure mode for complex OR constraints. The critical takeaway is their prompt fitness function: evaluating a prompt template based on the *performance improvement* (`child - parent`) of the code it generates, rather than absolute performance. We should immediately implement this 'prompt-delta' fitness signal in AlgoEvo to automate our prompt engineering loop.

### [Landscape-aware Automated Algorithm Design: An Efficient Framework for Real-world Optimization](https://arxiv.org/abs/2602.04529)

**2026-02-04** |  | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining Genetic Programming (GP) for proxy function generation and an LLM-driven Evolutionary Algorithm (LLaMEA) for algorithm discovery, guided by Exploratory Landscape Analysis (ELA) features and Wasserstein distance. | *LLM role:* algorithm_designer

> Yin et al. introduce a framework that decouples algorithm discovery from expensive evaluations by using Genetic Programming to evolve symbolic proxy functions that statistically match the target problem's landscape (via ELA features). Empirical results on photonics problems confirm that algorithms evolved on these cheap proxies transfer successfully to the real tasks, outperforming standard baselines like LSHADE with only 50×D real evaluations. **Key Takeaway:** We can synthesize 'symbolic gyms' that statistically mimic our target problems to run thousands of LLM iterations at near-zero cost. This directly addresses the sample efficiency bottleneck in AlgoEvo and suggests we should move beyond standard neural surrogates to evolved symbolic proxies.

### [EvoGit: Decentralized Code Evolution via Git-Based Multi-Agent Collaboration](https://arxiv.org/abs/2506.02049)

**2025-06-01** | The Hong Kong Polytechnic University | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Decentralized multi-agent evolutionary process using a Git-based phylogenetic graph with mutation and three-way crossover operations | *LLM role:* code_writer, evaluator

> Huang et al. introduce EvoGit, a framework where LLM agents asynchronously evolve code by treating Git commits as the population and using 3-way merges (based on Lowest Common Ancestor) as crossover. While the experiments (web app, bin packing generator) are largely qualitative and lack rigorous statistical benchmarking against baselines like MetaGPT, the architectural contribution is significant. The key takeaway is using Git's native DAG structure to handle lineage, persistence, and asynchronous concurrency 'for free,' replacing complex custom population managers. This is directly actionable for our AlgoEvo infrastructure to enable massive parallelism and better memory/traceability without reinventing the wheel.

### [Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer](https://arxiv.org/abs/2601.05770)

**2026-01-09** |  | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Discrete Transformer with functional disentanglement, temperature-annealed sampling, hypothesis testing for attention, and symbolic regression for MLP | *LLM role:* none

> Zhang et al. introduce the 'Discrete Transformer,' a constrained architecture that learns algorithmic tasks via gradient descent and allows for the post-hoc extraction of exact, human-readable Python code. By enforcing functional disentanglement (using attention strictly for routing and MLPs for arithmetic) and employing temperature-annealed sampling, they recover symbolic laws for arithmetic and physics tasks with near-zero error. The critical takeaway is their 'continuous-to-discrete homotopy' strategy—annealing from soft to hard selection during training—which enables differentiable search to converge on discrete, symbolic solutions. This suggests a viable path to discover heuristics via continuous optimization rather than purely stochastic LLM evolution.

### [CDEoH: Category-Driven Automatic Algorithm Design With Large Language Models](https://arxiv.org/abs/2603.19284)

**2026-03-08** | City University of Hong Kong, Nanjing University, Hohai University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary algorithm with category-aware population management and reflection-based error repair | *LLM role:* heuristic_generator, code_writer, evaluator (for category induction), code_repairer

> This paper extends the EoH framework by using an LLM to classify generated heuristics into algorithmic categories (e.g., greedy, DP) and applying a two-stage selection strategy to explicitly maintain diversity, alongside a standard reflection loop for code repair. The authors demonstrate empirical improvements over EoH, FunSearch, and ReEvo on Online Bin Packing and TSP, though they lack statistical significance testing. The single most useful takeaway is the 'category pool' concept: prompting the LLM to label the paradigm of the generated code and ensuring the top-1 of each discovered paradigm survives to the next generation. This matters directly for our AlgoEvo project, as it offers a cheap, explicit way to maintain search space exploration without the massive computational overhead of FunSearch's multi-island model.

### [RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution](https://arxiv.org/abs/2602.16932)

**2026-02-18** | Walmart Global Tech, Santa Clara University, Independent Researcher | M=6 P=5 I=7 *discuss*

*Method:* LLM-guided program evolution based on AlphaEvolve | *LLM role:* evolutionary_search

> RankEvolve applies AlphaEvolve with MAP-Elites to evolve Python retrieval functions, achieving significant gains over BM25 on BEIR/BRIGHT by rediscovering concepts like soft stop-words and PMI-based scoring. The results are empirically rigorous, showing that 'Freeform' seeds (defining only I/O contracts) significantly outperform 'Composable' or 'Constrained' seeds, albeit at a 10x latency cost. For our AlgoEvo work, the key takeaway is the concrete evidence that constraining the search space to 'clean' components prematurely caps performance; we should adopt their 'Freeform' approach but add an explicit latency/cost constraint to the fitness function to manage the resulting complexity.



## Bridge Papers

Papers connecting multiple research fronts:

### [From Understanding to Excelling: Template-Free Algorithm Design through Structural-Functional Co-Evolution](https://arxiv.org/abs/2503.10721)

**TRUE SYNTHESIS** | score=0.64 | Front 0 → Front 1

> Zhao et al. propose CAE, a framework that co-evolves algorithm structure (workflow/call graphs) alongside function implementations, aiming to eliminate the fixed templates required by SOTA methods lik

### [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

**TRUE SYNTHESIS** | score=0.63 | Front 0 → Front 1

> AlphaEvolve extends FunSearch by evolving entire code files (rather than single functions) using a 'search/replace' diff format and Gemini 2.0, achieving SOTA results across matrix multiplication (bea


---

*Generated by Research Intelligence System*
