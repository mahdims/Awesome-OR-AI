# Living Review: LLMs for Algorithm Design

**Last Updated:** 2026-03-22

---

## Recent Papers

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

*5 fronts detected — snapshot 2026-03-10*

### Front 6 (29 papers) — STABLE

**Density:** 0.27 | **Methods:** llm_code_generation, program_synthesis, llm_evolutionary_search, llm_as_heuristic, evolution_of_heuristics | **Problems:** algorithm_discovery, heuristic_evolution, circle_packing, automated_algorithm_design, autocorrelation_inequalities

*Unique methods:* abductive_reflection, actor_critic, adaptive_boltzmann_selection, adaptive_optimization, adaptive_sampling, adaptive_sliding_window, agent_based_framework, automated_evaluation, automatic_differentiation, automatic_linearization, autotuning, bandit_tuned_uip_depth, bandit_tuned_vivification, basin_hopping, batch_sampling, binary_search, black_box_distillation, branch_and_bound, cma_es, code_generation, compressed_watch_architecture, conflict_driven_clause_learning, convex_programming, convexification, counterfactual_regret_minimization, dag_execution, darwin_godel_machine, data_diffusion, diversity_metrics, domain_specific_language, dynamic_resource_allocation, embedding_similarity, entropic_objective, eureka, execution_based_verification, execution_guided_program_synthesis, expectation_maximization, exponential_moving_average, generative_agent_based_modeling, gigaevo, global_optimization, gridcoder, hybrid_evolutionary_memory, hydra_configuration, importance_sampling, insight_generation, island_based_evolutionary_algorithm, island_evolution, iterative_debugging, jax_framework, kd_tree, knowledge_transfer, l_bfgs_b, langgraph, large_program_database, lineage_based_context_retrieval, lineage_tracking, linear_programming, llm_as_aggregator, llm_as_evolver, llm_as_executor, llm_as_mutation_operator, llm_as_planner, llm_as_policy, llm_as_summarizer, llm_ensemble, llm_proposal, llm_research_agent, local_verification_loop, lora_fine_tuning, majority_voting, meta_gradients, meta_guidance, meta_level_evolution, meta_prompting, mixed_integer_programming, monte_carlo_sampling, monte_carlo_simulation, multi_agent_llm, multi_agent_simulation, multi_domain_bandit_control, multi_island_evolution, multi_island_model, multi_level_database, multi_task_learning, multi_uip_clause_learning, multilayer_perceptron, neural_program_synthesis, nonlinear_programming, novelty_search, numpy_library, open_ended_evolution, openevolve, optimistic_regret_matching, plan_execute_summarize, polymorphic_execution_strategies, ppo, presolving, primal_heuristics, program_evolution, progressive_disclosure, prompt_optimization, puct_search, quasi_random_sampling, random_hill_climbing, retrieval_augmented_generation, reward_design, reward_model, reward_shaping, sat_solvers, self_improving_ai, semantic_delta, shapely_library, sigmoid_calibration, slsqp, spatial_branch_and_bound, spectral_graph_theory, symbolic_distillation, symmetry_breaking_preprocessing, sympy_library, test_time_fine_tuning, test_time_learning, tool_use, tree_search, truncated_svd, vivification_sensitivity, zeroth_order_optimization
*Shared methods:* alphaevolve, black_box_optimization, code_embedding, eoh, evolution_of_heuristics, evolution_strategies, evolutionary_algorithm, evolutionary_algorithms, evolutionary_search, genetic_algorithm, genetic_programming, gradient_descent, grpo, in_context_learning, island_model, island_model_ea, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, local_search, map_elites, meta_learning, multi_agent_system, multi_armed_bandit, policy_space_response_oracles, population_based_optimization, program_synthesis, quality_diversity, reevo, reinforcement_learning, self_improving_search, simulated_annealing, transformer, upper_confidence_bound

This research front is defined by the rapid advancement and meta-level optimization of the AlphaEvolve framework, an LLM-guided evolutionary algorithm for autonomous code and algorithm discovery. It encompasses the evolution of entire program files, not just parameters or small functions, and is applied across a wide spectrum of challenging domains. Key applications include Boolean Satisfiability (SATLUTION), Multi-Agent Learning (VAD-CFR, SHOR-PSRO), Compiler Optimization (Magellan), Cache Replacement Policies (ArchAgent), Mathematical Discovery (ThetaEvolve), Lexical Retrieval (RankEvolve), and even Synthetic Persona Generation (Persona Generators).

Significant contributions focus on enhancing AlphaEvolve's efficiency, robustness, and generalization. Innovations like ShinkaEvolve's novelty rejection and adaptive sampling, DeltaEvolve's semantic deltas, and LoongFlow's Plan-Execute-Summarize paradigm drastically improve sample efficiency and reduce token costs, achieving SOTA with fewer evaluations. Generalization is boosted by DeepEvolve's integrated debugging, ImprovEvolve's modular operator evolution, and SS-Logic's agentic meta-synthesis with 'Blind Review' for problem validity. Performance gains are evident in SATLUTION outperforming 2025 SAT competition winners, ArchAgent achieving 1-5% IPC speedups, and ThetaEvolve setting new SOTA bounds on mathematical problems. Furthermore, meta-learning approaches like DGM's self-modifying agents and READY's multi-task reward discovery are evolving the evolutionary process itself, while papers like Nagda et al. demonstrate evolving verifier code for 10,000x speedups in combinatorial structure discovery.

This front is in a rapid maturation phase, with the core AlphaEvolve paradigm firmly established and research now aggressively focused on addressing its limitations and scaling its capabilities. The trajectory suggests a move towards increasingly autonomous and sample-efficient discovery pipelines. Future work will likely concentrate on more sophisticated adaptive control and meta-learning mechanisms, deeper integration of external domain knowledge (e.g., via RAG or formal methods) to guide evolution, and hybrid architectures that combine LLM-driven code generation with classical optimization solvers. Expect to see continued efforts in automating the entire algorithm discovery lifecycle, from problem definition to rigorous, adversarial-resistant evaluation.

**Papers:**

### [Autonomous Code Evolution Meets NP-Completeness](https://arxiv.org/abs/2509.07367)

**2025-09-09** | NVIDIA Research, University of Maryland | M=9 P=9 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* Autonomous agent-based code evolution system with Planning and Coding LLM agents | *LLM role:* evolutionary_search

> SATLUTION extends LLM evolutionary search to full-scale C++ repositories, autonomously evolving SAT solvers that outperform 2025 human competition winners using only 2024 training data. The results are highly rigorous, backed by 90k CPU hours of distributed evaluation and strict correctness proofs (DRAT), showing a clear monotonic improvement trajectory. The single most stealable insight is the **self-evolving rule system**: the agent autonomously updates a persistent set of markdown constraints (e.g., forbidden patterns, testing protocols) based on post-cycle failure analysis, effectively creating 'institutional memory' that prevents regression in long-horizon search. We must implement this meta-learning loop in AlgoEvo immediately to move beyond single-file optimization.

### [Scientific Algorithm Discovery by Augmenting AlphaEvolve with Deep Research](https://arxiv.org/abs/2510.06056)

**2025-10-07** | MIT-IBM Watson AI Lab, IBM Research, University of Notre Dame | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Agent-based framework integrating deep research (planning, searching, writing) with algorithm evolution (coding, evaluation, evolutionary selection) and iterative debugging. | *LLM role:* heuristic_generator, code_writer, evaluator, decomposition_guide, debugger

> DeepEvolve augments the standard evolutionary coding loop (AlphaEvolve) with two critical components: a 'Deep Research' module that searches the web/literature to generate grounded mutation proposals, and an iterative debugging agent that fixes execution errors. While the '666%' improvement on Circle Packing is likely due to a weak baseline (fixed-size vs. generalized), the engineering results are compelling: the debugging agent raises execution success rates from ~13% to ~99% in complex tasks. The key takeaway for our AlgoEvo work is the architecture of generating a text-based 'research proposal' via RAG before attempting code generation, rather than mutating code directly. We should immediately adopt their debugging loop and consider injecting external literature search into our mutation operators to prevent search stagnation.

### [C-Evolve: Consensus-based Evolution for Prompt Groups](https://arxiv.org/abs/2509.23331)

**2025-09-27** | Westlake University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Island-based evolutionary algorithm with Exponential Moving Average (EMA) voting score as fitness, optimizing groups of prompts for consensus via majority voting or LLM-based aggregation. | *LLM role:* evolver, consensus_aggregator

> C-Evolve modifies island-based evolution to optimize a group of prompts that maximize consensus accuracy (majority vote) rather than individual performance. The authors introduce a 'voting score' fitness function—calculated via Exponential Moving Average (EMA) of an individual's contribution to sampled groups—which successfully drives the population toward diverse, complementary strategies that outperform ensembles of individually optimized prompts (beating AlphaEvolve by ~4% on Qwen3-8B). The single most actionable takeaway is the **EMA voting score mechanism**: we can steal this exact fitness formulation to evolve portfolios of complementary VRP heuristics in AlgoEvo, replacing our current focus on converging to a single 'best' solver. While the benchmarks are standard (MATH, HotpotQA), the method offers a robust solution to the 'single heuristic limitation' we face in OR.

### [Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning](https://arxiv.org/abs/2602.13218)

**2026-01-23** | Tencent, The Hong Kong University of Science and Technology (Guangzhou) | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Agentic Meta-Synthesis framework using a Generate–Validate–Repair closed loop for Generator–Validator program pairs | *LLM role:* program_synthesis, code_writer, decomposition_guide, evaluator, evolutionary_search

> Liu et al. introduce SS-Logic, an agentic framework that evolves Python 'Generator-Validator' pairs to scale logic task families, using a rigorous 'Code-Augmented Blind Review' where independent agents must write code to solve generated tasks to verify their validity. They expand 400 seed families to over 21k instances, achieving consistent gains on AIME (+3.0) and SynLogic (+5.2) via RLVR. **Crucial Takeaway:** We should steal the 'Blind Review' mechanism for AlgoEvo—using the solvability of a generated problem (by an independent code agent) as a strict fitness filter for the generator itself. This directly addresses our bottleneck in filtering invalid or hallucinated heuristics during evolutionary search.

### [Persona Generators: Generating Diverse Synthetic Personas at Scale](https://arxiv.org/abs/2602.03545)

**2026-02-03** | Google DeepMind | M=8 P=3 I=8 **MUST-READ** *discuss*

*Method:* AlphaEvolve-driven evolutionary search for Persona Generator code optimization with a two-stage LLM-based generation architecture | *LLM role:* evolutionary_search, code_writer, evaluator, research_agent

> Paglieri et al. (DeepMind) apply AlphaEvolve to optimize Python code that generates synthetic personas, explicitly maximizing diversity metrics (convex hull, coverage) in embedding space rather than just fidelity. They achieve >80% coverage of the behavioral space compared to <50% for baselines, proving that evolving the *generator function* is more effective than prompting for diversity. The key takeaway is their two-stage architecture (autoregressive high-level trait generation $\to$ parallel detail expansion), which we should steal to evolve 'Solution Generators' for VRP/OR that inherently resist mode collapse. This validates our direction with AlgoEvo but offers a concrete architectural pattern for maintaining population diversity.

### [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)

**2026-01-22** | Stanford University, NVIDIA, UC San Diego, Together AI, Astera Institute | M=9 P=10 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning with Entropic Objective and PUCT-based Reuse at Test Time | *LLM role:* Policy being optimized to generate solutions/code

> TTT-Discover introduces a method to fine-tune an LLM (gpt-oss-120b) *during* inference on a single test problem using RL, replacing the frozen-model evolutionary search of AlphaEvolve. They employ a novel 'entropic objective' that optimizes for the single best solution (discovery) rather than expected return, combined with PUCT-based state reuse. The results are empirically rigorous, setting new SOTA on Erdős’ problem, GPU kernel optimization, and AtCoder contests, directly beating AlphaEvolve and ShinkaEvolve. The critical takeaway is that for hard discovery tasks, shifting the model's distribution via online updates is superior to context-based search; we should immediately test their entropic objective in our AlgoEvo pipeline.

### [AlphaResearch: Accelerating New Algorithm Discovery with Language Models](https://arxiv.org/abs/2511.08522)

**2025-11-11** | Yale, NYU, Tsinghua, ByteDance | M=7 P=6 I=7 *discuss*

*Method:* Autonomous research agent with dual research environment combining execution-based verification and simulated real-world peer review | *LLM role:* research_agent

> AlphaResearch introduces a 'dual environment' for algorithm discovery: it generates natural language research ideas, filters them using a reward model fine-tuned on ICLR peer reviews, and then executes the surviving ideas. While it claims to beat human baselines on Packing Circles, the improvement is marginal (<0.1%) and it fails to improve upon baselines in 6/8 benchmark problems. The key takeaway for us is the mechanism of an 'Idea Critic'—using a learned reward model to filter the search space at the prompt level before wasting compute on execution—which directly addresses our sample efficiency goals in evolutionary search.

### [Discovering Multiagent Learning Algorithms with Large Language Models](https://arxiv.org/abs/2602.16928)

**2026-02-18** | Google DeepMind | M=10 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* AlphaEvolve, an LLM-powered evolutionary coding agent | *LLM role:* code_writer

> DeepMind applies AlphaEvolve to discover new variants of CFR and PSRO by evolving Python code for regret accumulation and meta-strategy solving. They identify VAD-CFR and SHOR-PSRO, which outperform human-designed SOTA (DCFR, PCFR+) on benchmarks like Leduc Poker and Liar's Dice; results are rigorous, using exact exploitability. The critical takeaway is the shift from evolving static functions to evolving **stateful classes** (e.g., tracking volatility via EWMA inside the accumulator), allowing the LLM to discover dynamic, adaptive schedules—a technique we should immediately port to AlgoEvo.

### [ThetaEvolve: Test-time Learning on Open Problems](https://arxiv.org/abs/2511.23473)

**2025-11-28** | Microsoft, University of Washington, Carnegie Mellon University, University of Wisconsin-Madison, University of California, San Diego | M=10 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Program evolution with test-time Reinforcement Learning (RL) using GRPO algorithm | *LLM role:* code_writer

> ThetaEvolve integrates test-time reinforcement learning (GRPO) directly into an AlphaEvolve-style loop, allowing a single 8B model to learn from its own successful mutations and achieve new SOTA bounds on Circle Packing and Autocorrelation inequalities. The results are rigorous, showing that RL applied to the *dynamic* environment (sampling from the evolving database) vastly outperforms RL on static prompts or pure inference search. The most stealable insight is the 'lazy penalty' mechanism—penalizing semantically equivalent code or stagnation—which forces the RL policy to learn genuine exploration strategies rather than memorization. This is a blueprint for the 'RL-infused evolution' milestone in our AlgoEvo roadmap.

### [RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution](https://arxiv.org/abs/2602.16932)

**2026-02-18** | Walmart Global Tech, Santa Clara University, Independent Researcher | M=6 P=5 I=7 *discuss*

*Method:* LLM-guided program evolution based on AlphaEvolve | *LLM role:* evolutionary_search

> RankEvolve applies AlphaEvolve with MAP-Elites to evolve Python retrieval functions, achieving significant gains over BM25 on BEIR/BRIGHT by rediscovering concepts like soft stop-words and PMI-based scoring. The results are empirically rigorous, showing that 'Freeform' seeds (defining only I/O contracts) significantly outperform 'Composable' or 'Constrained' seeds, albeit at a 10x latency cost. For our AlgoEvo work, the key takeaway is the concrete evidence that constraining the search space to 'clean' components prematurely caps performance; we should adopt their 'Freeform' approach but add an explicit latency/cost constraint to the fitness function to manage the resulting complexity.

### [ImprovEvolve: Ask AlphaEvolve to Improve the Input Solution and Then Improvise](https://arxiv.org/abs/2602.10233)

**2026-02-10** | MIRIAI, FusionBrain Lab, Institute of Numerical Mathematics | M=8 P=5 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search (MAP-Elites) for local search operators (improve, perturb, generate_config) within a basin-hopping framework | *LLM role:* evolutionary_search, initial_program_generator

> Kravatskiy et al. introduce ImprovEvolve, a framework that restricts the LLM to evolving `improve()` (local search) and `perturb()` (mutation) operators, which are then executed by a fixed basin-hopping algorithm. They achieve new state-of-the-art results on Hexagon Packing and the Second Autocorrelation Inequality, demonstrating that this modular approach generalizes to unseen problem sizes where monolithic AlphaEvolve solutions fail. The critical insight is that LLMs are poor at designing global search logic and tuning hyperparameters (LLM edits actively harmed performance), so we should isolate the LLM to generating local moves while keeping the meta-heuristic framework deterministic. We should immediately apply this 'operator-only' evolution strategy to our ALNS research for VRP.

### [Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve](https://arxiv.org/abs/2601.21096)

**2026-01-28** | Google DeepMind, Google, Cornell University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-powered coding agent (AlphaEvolve) with evolutionary search and black-box autotuning (Vizier) | *LLM role:* code_writer

> Magellan couples AlphaEvolve with a black-box autotuner (Vizier) to evolve C++ compiler heuristics, achieving >5% binary size reduction in LLVM and beating both human experts and prior neural policies. The results are rigorous, validated on production workloads and showing temporal generalization. **The critical takeaway is the 'Hierarchical Search' strategy:** rather than asking the LLM to write fully specified code, they prompt it to generate *templates* with exposed parameters (flags), delegating numerical tuning to a cheap external optimizer. This directly addresses the sample efficiency issues we face in AlgoEvo; we should immediately steal this architecture to separate structural evolution from parameter tuning.

### [Mathematical exploration and discovery at scale](https://arxiv.org/abs/2511.02864)

**2025-12-22** | Google DeepMind, UCLA, Brown University, Institute for Advanced Study | M=9 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search for programs (search heuristics) that find mathematical constructions | *LLM role:* Generates and mutates code for search heuristics; guides meta-level evolution of search strategies

> DeepMind applies AlphaEvolve to 67 math problems, formalizing the distinction between 'Search Mode' (evolving heuristics for fixed instances) and 'Generalizer Mode' (evolving algorithms that extrapolate from small to large n). Results are rigorous, establishing new bounds on Kakeya sets and 10+ other problems by exploiting verifier loopholes and heuristic specialization. The most critical takeaway for AlgoEvo is Section 44: evolving code that *calls* other LLMs leads to emergent prompt optimization and injection strategies, suggesting a path for our multi-agent optimization work. We must adopt their 'Generalizer' training curriculum (train on small n, test on large n) to fix our scalability bottlenecks.

### [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://arxiv.org/abs/2507.17668)

**2025-09-10** | University of Oxford | M=8 P=7 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Empirical comparison of meta-learning algorithms for reinforcement learning algorithm discovery | *LLM role:* code_writer

> Goldie et al. perform a rigorous empirical benchmark comparing LLM-based algorithm proposal against Black-box Evolution Strategies (ES) and various distillation methods. They find that while LLMs are sample-efficient for simple functions, they catastrophically fail to incorporate high-dimensional input features (e.g., the 20+ inputs in OPEN), where Black-box ES remains superior. The most actionable takeaway is 'Same-Size Distillation': distilling a learned black-box algorithm into a fresh network of identical size using synthetic data consistently improves out-of-distribution generalization with zero additional environment samples. We should implement this distillation step immediately and reconsider using LLMs for feature-heavy heuristic components.

### [Reinforced Generation of Combinatorial Structures: Hardness of Approximation](https://arxiv.org/abs/2509.18057)

**2025-12-19** | Google DeepMind, Google, University of California, Berkeley | M=9 P=5 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search for combinatorial structures and verification procedures | *LLM role:* evolutionary_search

> Nagda et al. utilize AlphaEvolve to discover combinatorial gadgets that improve hardness of approximation bounds for MAX-CUT and TSP, validating findings with formal proofs. The standout contribution is not the hardness results themselves, but the methodology: they tasked AlphaEvolve with optimizing the *verification code* (checking correctness against a slow ground truth), achieving a 10,000x speedup that enabled searching gadgets of size 19 (vs. 11 previously). We should immediately adopt this 'evolve the verifier' loop for our computationally expensive fitness functions in AlgoEvo to break current scalability limits.

### [READY: Reward Discovery for Meta-Black-Box Optimization](https://arxiv.org/abs/2601.21847)

**2026-01-29** |  | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* LLM-based program evolution with a multi-task niche-based architecture, fine-grained evolutionary operators, and explicit knowledge transfer | *LLM role:* evolutionary_search

> READY introduces a multi-task evolutionary framework where LLMs evolve reward functions for multiple MetaBBO algorithms simultaneously, utilizing explicit 'Knowledge Transfer' operators to translate successful logic between distinct tasks. The results are robust, demonstrating superior performance over Eureka and EoH on BBOB benchmarks with a 2-4x reduction in search time due to parallelization and shared heuristics. The most stealable insights are the 'History-Reflection' operator—which prompts the LLM to extrapolate trends from the evolutionary trajectory rather than just mutating the current state—and the cross-niche transfer mechanism, both of which should be implemented in our multi-agent optimization stack immediately.

### [LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm](https://arxiv.org/abs/2512.24077)

**2025-12-30** |  | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Plan-Execute-Summarize (PES) paradigm integrated with Hybrid Evolutionary Memory (Multi-Island, MAP-Elites, Adaptive Boltzmann Selection) | *LLM role:* planner, executor, summarizer

> LoongFlow replaces the standard stochastic mutation operator in LLM evolutionary search with a 'Plan-Execute-Summarize' (PES) cognitive loop. Instead of random code changes, a Planner retrieves the 'intent' and 'summary' of the parent solution's lineage to generate a directed hypothesis, which is then executed and summarized for the next generation. The authors demonstrate a 60% reduction in evaluations and a 100% success rate on AlphaEvolve tasks where standard methods fail or stagnate. The critical takeaway is the 'Lineage-Based Context Retrieval' mechanism: explicitly passing the parent's plan and retrospective summary to the child allows for directed rather than random walks in the search space. We must implement this PES loop in AlgoEvo immediately to fix our sample efficiency issues.

### [Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning](https://arxiv.org/abs/2507.15877)

**2025-09-21** |  | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Execution-guided multi-step neural program synthesis (EG-NPS) with Encoder-Decoder Transformer and tree search | *LLM role:* none

> Ouellette implements an Execution-Guided Neural Program Synthesis (EG-NPS) system for ARC-AGI that conditions the search on the intermediate execution state of every instruction, achieving 80% success on out-of-distribution tasks where TTFT (10%) and standard AlphaEvolve (0-14%) fail. The results are rigorous, using controlled OOD tasks to prove that TTFT relies on in-distribution priors rather than reasoning. The critical takeaway for our AlgoEvo work is the architecture of the 'state-conditioned decoder': instead of blind code generation, we should inject the tokenized execution result of step $t$ into the context for step $t+1$. This is effectively a dense process reward model that solves the sample efficiency bottleneck we face in evolutionary search.

### [DeltaEvolve: Accelerating Scientific Discovery through Momentum-Driven Evolution](https://arxiv.org/abs/2602.02919)

**2026-02-02** | Microsoft, The Ohio State University | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Momentum-driven evolutionary framework with semantic delta and progressive disclosure within an Expectation-Maximization (EM) process | *LLM role:* program_synthesizer

> DeltaEvolve replaces the standard full-code history in evolutionary search with 'semantic deltas'—structured text summaries capturing the 'from/to' logic of modifications and their hypotheses. Across 5 domains (including BBOB and Symbolic Regression), they demonstrate superior objective scores over AlphaEvolve while reducing token consumption by ~37%. The critical takeaway is the 'Progressive Disclosure' mechanism: treating history as a momentum vector (deltas) rather than a state archive (snapshots) allows us to fit a deeper evolutionary trajectory into the context window. We should immediately test their 'Delta Plan' prompt structure in AlgoEvo to improve sample efficiency and reduce costs.

### [Global Optimization for Combinatorial Geometry Problems Revisited in the Era of LLMs](https://arxiv.org/abs/2601.05943)

**2026-01-15** |  | M=4 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Global Nonlinear Programming (NLP) using spatial branch-and-bound | *LLM role:* none

> Berthold et al. demonstrate that standard global NLP solvers (SCIP, Xpress) outperform DeepMind's AlphaEvolve on its own benchmarks (circle/hexagon packing, min-max distance) without any learning or evolution. The results are rigorous, improving on 'newly discovered' solutions within minutes using default solver settings. **CRITICAL TAKEAWAY:** We must validate our AlgoEvo results against classical global solvers to ensure we aren't claiming 'discovery' on problems that are trivial for SCIP; furthermore, it suggests a hybrid path where LLMs generate NLP models for solvers rather than evolving raw heuristic code. This is a necessary reality check for our benchmarking strategy.

### [Mining Generalizable Activation Functions](https://arxiv.org/abs/2602.05688)

**2026-02-05** | Google DeepMind | M=8 P=5 I=8 **MUST-READ** *discuss*

*Method:* Evolutionary search powered by AlphaEvolve framework | *LLM role:* code_writer

> Vitvitskyi et al. (DeepMind) utilize AlphaEvolve to discover novel activation functions by evolving Python code on small, synthetic datasets explicitly designed to test OOD generalization (e.g., polynomials, Feynman equations). The results are credible and backed by downstream transfer: discovered functions like `GELU * (1 + 0.5 sinc(x))` outperform baselines on algorithmic reasoning tasks (CLRS-30) while matching standard vision benchmarks. **Key Takeaway:** The 'Small-Scale Lab' methodology—optimizing on cheap, synthetic proxy tasks to find generalizable logic—is a validated strategy to bypass the computational bottleneck of evaluating evolved candidates on large-scale instances. We should steal this 'proxy evolution' setup for AlgoEvo to drastically reduce evaluation costs while targeting generalization in VRP heuristics.

### [GigaEvo: An Open Source Optimization Framework Powered By LLMs And Evolution Algorithms](https://arxiv.org/abs/2511.17592)

**2025-11-17** | Sber, Artificial Intelligence Research Institute (AIRI) | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* MAP-Elites quality-diversity algorithm with LLM-driven mutation operators (rewrite-based or diff-based) and bidirectional lineage tracking | *LLM role:* evolutionary_search

> GigaEvo is an open-source reproduction of the AlphaEvolve framework that implements MAP-Elites with an asynchronous DAG execution engine, successfully reproducing SOTA results on Heilbronn triangles and beating FunSearch on Weibull Bin Packing. The results are credible and backed by code, specifically highlighting that 'rewrite-based' mutation outperforms 'diff-based' approaches for open-weights models—a crucial engineering constraint for us. The most actionable takeaway is their 'bidirectional lineage tracking' mechanism, which enriches mutation prompts by analyzing both how a program improved over its ancestor and how its descendants further improved, a technique we should steal for AlgoEvo's mutation operator. Their negative result regarding multi-island MAP-Elites (added complexity, no gain) suggests we should deprioritize similar complex topologies.

### [ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution](https://arxiv.org/abs/2509.19349)

**2025-09-17** | Sakana AI | M=9 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary framework with adaptive parent sampling, code novelty rejection-sampling, and bandit-based LLM ensemble selection | *LLM role:* mutation_operator, evaluator, decomposition_guide

> ShinkaEvolve presents an open-source evolutionary framework that drastically improves sample efficiency (e.g., beating AlphaEvolve on Circle Packing with only 150 evaluations vs. thousands) by integrating embedding-based novelty rejection, adaptive parent sampling, and bandit-based LLM selection. The results are credible, backed by code from Sakana AI, and directly target our primary pain point of high API costs/sample inefficiency in evolutionary search. **Key Takeaway:** We must implement their 'novelty rejection sampling' immediately—using a cheap embedding model to filter out semantically similar code mutations (threshold 0.95) before execution is a trivial but high-impact optimization for our AlgoEvo pipeline. This paper proves that smart filtering is superior to the brute-force compute strategies we have been relying on.

### [GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models](https://arxiv.org/abs/2509.21593)

**2025-09-25** | Massachusetts Institute of Technology, Stanford University, Technical University of Munich | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Multi-agent LLM framework combining OpenEvolve-based evolutionary search with GeoKnowRAG for geospatial domain knowledge injection | *LLM role:* code_writer, evaluator, prompt_optimizer, evolutionary_search, decomposition_guide

> GeoEvolve augments standard LLM-based evolutionary search (OpenEvolve) with an outer 'researcher' loop that queries a domain-specific RAG (textbooks/papers) to inject theoretical constraints into mutation prompts. On geospatial interpolation tasks, they report 13-21% error reduction over standard evolution, with ablations confirming that retrieved domain knowledge—not just iterative feedback—drives the performance gain. The critical takeaway is the architectural pattern of 'Knowledge-Guided Evolution': instead of relying on the LLM's internal weights for domain theory, they explicitly retrieve and inject theoretical priors (e.g., valid variogram definitions) to steer the search. We should adapt this 'Theory-RAG' outer loop for our AlgoEvo pipeline to force evolved VRP heuristics to respect OR theoretical bounds.

### [ArchAgent: Agentic AI-driven Computer Architecture Discovery](https://arxiv.org/abs/2602.22425)

**2026-02-25** | Google DeepMind, Google, University of California, Berkeley | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based evolutionary search using AlphaEvolve | *LLM role:* code_writer

> Gupta et al. (Google/DeepMind) apply AlphaEvolve to generate C++ cache replacement policies, achieving ~1-5% IPC gains over SOTA (Mockingjay/SHiP) on SPEC and Google traces. The results are rigorous and demonstrate that LLM-driven evolution can outperform human experts in mature hardware domains 3-5x faster. WE LEARNED: The documentation of 'simulator escapes'—where the agent exploited a compiler optimization bug to delete memory writes and artificially boost scores—is a critical warning for our own automated benchmarking; we must harden our OR evaluators against adversarial optimization. Additionally, their 'hyperspecialization' approach (tuning policies per workload) validates our interest in instance-specific heuristic evolution.

### [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/abs/2602.20133)

**2026-02-23** | University of California, Berkeley, Bespoke Labs | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven evolutionary algorithm with hierarchical adaptive optimization using an accumulated improvement signal to dynamically modulate local exploration intensity, global resource allocation via multi-armed bandit, and meta-level solution tactics generation | *LLM role:* semantic_mutation_operator

> AdaEvolve replaces static evolutionary schedules with a three-tier adaptive controller: local exploration intensity based on an 'accumulated improvement signal' (pseudo-gradient), global compute allocation via a normalized bandit, and meta-level 'tactic' generation when stagnation occurs. Results are highly convincing, showing SOTA on Circle Packing (beating AlphaEvolve) and 185 other tasks while using the same LLM backbone as baselines, proving the gains are algorithmic. The most stealable insight is the $G_t$ signal metric—an exponential moving average of squared normalized improvements—which allows auto-tuning exploration rates without manual intervention. This is a direct upgrade for our AlgoEvo architecture, specifically addressing our sample efficiency and stagnation bottlenecks.

### [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954)

**2025-09-26** | Sakana AI, Vector Institute, University of British Columbia, Canada CIFAR AI Chair | M=10 P=8 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* Darwin Gödel Machine (DGM) with iterative self-modification, empirical validation, and population-based open-ended exploration | *LLM role:* coding_agent, self_modifier, problem_solver, diagnosis_agent

> DGM implements a population-based evolutionary loop where agents modify their own Python source code (tools, memory, flow) to improve performance on coding benchmarks, rather than just optimizing prompts or parameters. Results are strong and verified: it boosts a base agent from 20% to 50% on SWE-bench Verified, matching handcrafted SoTA, with ablations proving the necessity of the population archive (open-endedness) over single-lineage hill climbing. **Key Takeaway:** The 'self-diagnosis' mechanism—feeding execution logs to a model to propose specific *architectural* code changes (e.g., implementing a 'str_replace' tool to fix granular editing errors)—is the exact mechanism we need to implement for evolving our heuristic searchers. This validates that LLM-driven code evolution is viable for complex logic improvement, not just toy tasks.

### [CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization](https://arxiv.org/abs/2510.14150)

**2026-01-06** | Inter&Co., Worcester Polytechnic Institute, Universidade Federal de Minas Gerais | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Islands-based genetic algorithm with modular LLM orchestration, context-aware recombination, adaptive meta-prompting, and depth-based exploitation | *LLM role:* code_writer

> CodeEvolve couples islands-based genetic algorithms with LLMs, utilizing CVT-MAP-Elites for diversity and a specific 'inspiration-based' crossover operator where the LLM integrates logic from high-ranking peer solutions. The results are strong and backed by numbers: they beat AlphaEvolve on 5/9 benchmarks and demonstrate that Qwen3-Coder-30B matches Gemini-2.5 performance at ~10% of the cost. The single most useful takeaway is the implementation of the 'inspiration' operator and the necessity of MAP-Elites over simple elitism to escape local optima in code space. We should immediately benchmark their open-source framework against our internal AlgoEvo builds.

### [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

**2025-06-16** | Google DeepMind | M=10 P=10 I=10 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary algorithm for code superoptimization, orchestrating an autonomous pipeline of LLMs for code generation, critique, and evolution, grounded by code execution and automatic evaluation. | *LLM role:* evolutionary_search

> AlphaEvolve extends FunSearch by evolving entire code files (rather than single functions) using a 'search/replace' diff format and Gemini 2.0, achieving SOTA results across matrix multiplication (beating Strassen), 50+ open math problems, and Google's production scheduling. The results are exceptionally strong and verified, including deployed improvements to Google's Borg scheduler (0.7% resource recovery) and TPU circuits. The critical takeaway is the move to **diff-based full-file evolution** and **meta-prompt evolution** (evolving the prompt instructions alongside the code), which allows the system to modify architecture and logic rather than just heuristics. This is a mandatory blueprint for the next iteration of our AlgoEvo and EvoCut projects.


### Front 8 (17 papers) — STABLE

**Density:** 0.43 | **Methods:** llm_code_generation, program_synthesis, llm_as_heuristic, funsearch, evolution_of_heuristics | **Problems:** heuristic_evolution, operator_discovery, algorithm_discovery, combinatorial_optimization, hyperparameter_optimization

*Unique methods:* acquisition_functions, adaptive_large_neighborhood_search, bayesian_optimization, best_fit_heuristic, binary_search_algorithm, combinatorial_reasoning, concept_learning, contrastive_learning, cross_entropy_method, decision_trees, deduplication, dynamic_programming, ensemble_heuristics, evoph, exploitation_exploration_tradeoff, few_shot_learning, few_shot_prompting, gaussian_processes, greedy_algorithm, heuristic_analysis, hgs, hill_climbing, human_llm_collaboration, hybrid_search, id3_algorithm, improved_seeding, input_space_partitioning, integration_by_parts_reduction, island_models, iterative_rounding, laporta_algorithm, llm_as_variation_operator, llm_guided_evolutionary_search, llm_in_context_learning, llm_rl_trained, manual_refinement, mcts, mdp_formulation, multi_agent_systems, multi_population_evolutionary_algorithm, optimization_based_analyzers, priority_function_design, program_search, prompt_engineering, proximal_policy_optimization, qube, random_forest, representation_learning, rl_dapo, smt_solvers, sparse_solver, tree_structured_parzen_estimator
*Shared methods:* co_evolution, eoh, evolution_of_heuristics, evolutionary_algorithm, evolutionary_algorithms, evolutionary_search, funsearch, genetic_algorithm, heuristic_evolution, hyper_heuristics, large_language_models, large_neighborhood_search, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, local_search, map_elites, meta_learning, program_synthesis, reevo, reinforcement_learning, self_improving_search, simulated_annealing, upper_confidence_bound

This research front centers on the application and significant enhancement of the FunSearch framework for automated algorithm and heuristic design. The core methodology involves leveraging Large Language Models (LLMs) within an evolutionary loop to generate, evaluate, and iteratively refine code for solving complex problems. Papers explore a diverse array of domains, ranging from traditional combinatorial optimization tasks like routing, bin packing, and scheduling to more specialized algorithm design challenges such as discovering Bayesian Optimization acquisition functions, adapting Vision-Language Models, and optimizing tensor network structures. The unifying theme is the LLM's pivotal role in program synthesis and variation, guided by an external evaluator.

Key contributions include substantial methodological advancements to the FunSearch paradigm and its successful application across various challenging domains. QUBE introduced an uncertainty-aware parent selection mechanism (UCB) to improve exploration, while CCTS leveraged LLM-extracted semantic concepts to guide the search process via a Tree-structured Parzen Estimator. G-LNS extended the approach to Large Neighborhood Search by co-evolving destroy and repair operators for routing problems, demonstrating superior performance over OR-Tools on large CVRP instances. Robusta integrated a Heuristic Analyzer and Suggester LLM to achieve a 28x improvement in worst-case performance for traffic engineering. Other works showcased FunBO for discovering Bayesian Optimization acquisition functions, EvoVLMA for Vision-Language Model adaptation, and LLM4DRD for dynamic flexible assembly flow shop scheduling, frequently achieving state-of-the-art results. Notably, Zhu et al. (2510.11121) provided empirical evidence that RL-finetuning a smaller LLM (Qwen-14B) for generating C++ crossover operators within the HGS solver can outperform GPT-4o and expert-designed components, introducing an innovative AST-based anti-plagiarism reward.

This front is rapidly maturing, with a clear trajectory towards more sophisticated and robust LLM-driven evolutionary search. The focus is shifting from basic program generation to enhancing the entire evolutionary loop, including advanced parent selection, effective diversity management (e.g., CCTS, QUBE, EvoPH island models), and the co-evolution of interacting algorithmic components (e.g., G-LNS destroy/repair, EvoPH prompts/heuristics, ECHO-MIMIC policies/nudges). The increasing emphasis on 'local evolution' (e.g., D-ID3 generating features at each decision tree split) and 'explainability-guided' search (Robusta) suggests a move towards more interpretable and context-aware algorithm design. Future research will likely explore multi-objective optimization, more complex constraint structures, and robust validation mechanisms for generated code, potentially integrating human-in-the-loop refinement more deeply.

**Papers:**

### [LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling](https://arxiv.org/abs/2601.15738)

**2026-01-22** |  | M=7 P=6 I=7 *discuss*

*Method:* LLM-assisted evolutionary algorithm for automatic dispatching rule design with dual-expert mechanism and feature-fitting rule evolution | *LLM role:* heuristic_generator, evaluator

> LLM4DRD employs a dual-agent framework (Generator & Evaluator) to evolve priority dispatching rules for dynamic flexible assembly flow shops. The core contribution is the **Hybrid Evaluation** mechanism, where the Evaluator generates qualitative critiques (strengths/weaknesses) that are injected into the Generator's prompts to guide specific operators like 'Dominance-Fusion Crossover' and 'Directed Optimization.' Empirical results show it outperforms FunSearch and EOH, avoiding the premature convergence seen in other methods. The most stealable insight is the prompt structure for crossover: rather than blindly combining code, it uses the Evaluator's analysis of parent strengths to direct the merger, a technique we should implement to improve sample efficiency in our evolutionary search.

### [FunBO: Discovering Acquisition Functions for Bayesian Optimization with FunSearch](https://arxiv.org/abs/2406.04824)

**2024-07-01** | Google DeepMind | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* FunSearch-based evolutionary algorithm for discovering acquisition functions in Python code | *LLM role:* evolutionary_search

> FunBO applies FunSearch to evolve Python code for Bayesian Optimization acquisition functions, evaluating fitness by running full BO loops on synthetic functions. The results are empirically strong, showing that evolved AFs generalize well to out-of-distribution functions and outperform standard baselines like EI and UCB. The most stealable insight is their 'few-shot' adaptation strategy, where a general-purpose heuristic is rapidly fine-tuned on a small set of target instances—a technique we should immediately test for our VRP heuristics. While the method is computationally expensive (brute-forcing the inner loop), the interpretable code outputs provide concrete ideas for dynamic exploration-exploitation trade-offs.

### [EvoVLMA: Evolutionary Vision-Language Model Adaptation](https://arxiv.org/abs/2508.01558)

**2025-08-03** | Chinese Academy of Sciences | M=7 P=4 I=7 *discuss*

*Method:* LLM-assisted two-stage evolutionary algorithm with crossover and mutation operators for optimizing feature selection and logits computation functions in code space | *LLM role:* code_writer

> This paper proposes EvoVLMA, an LLM-based evolutionary framework that searches for Python code to adapt Vision-Language Models (feature selection and logits computation). They demonstrate that **jointly** evolving two coupled algorithmic components fails (worse than random), whereas a **sequential** two-stage evolution strategy yields SOTA results (beating manual baselines by ~1-2%). For our AlgoEvo work, the key takeaway is the infrastructure design: they wrap code execution in restartable web services with a process monitor to handle the high rate of CUDA errors/timeouts in generated code—a practical 'trick' we should adopt to improve our search stability.

### [Amplifying human performance in combinatorial competitive programming](https://arxiv.org/abs/2411.19744)

**2024-11-29** | Google DeepMind | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* FunSearch, an evolutionary algorithm for program search using LLMs and a systematic evaluator | *LLM role:* evolutionary_search

> DeepMind applies FunSearch (using Gemini 1.5 Flash) to evolve scoring functions within human-written greedy backbones for Hash Code and AtCoder problems, achieving top-1% or rank-1 performance against humans. The results are robust, beating top human teams on 5/8 historical contests using a generic evolutionary setup. The critical takeaway is the 'switching variable' technique: using a single evolved function to handle multiple distinct decision points (e.g., selecting a vehicle vs. selecting a route) by passing a state flag, rather than evolving multiple interacting functions. This validates that generalist models (Flash) are sufficient for high-end OR evolution without code-specific fine-tuning. We should adopt their 'Backbone + Scorer' architecture for our VRP/Scheduling work immediately.

### [QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution](https://arxiv.org/abs/2412.20694)

**2025-02-21** | Westlake University, Zhejiang University, University of Electronic Science and Technology of China | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Algorithm with LLM as variation operator, guided by Quality-Uncertainty Trade-off Criterion (QUTC) using Uncertainty-Inclusive Quality (UIQ) metric | *LLM role:* variation_operator

> QUBE replaces FunSearch's naive score-based parent selection with a UCB algorithm that selects parents based on the *average quality of their offspring* (exploitation) plus an uncertainty term (exploration). The authors demonstrate that a parent's own score is a poor predictor of its ability to evolve further; treating parents as 'bandit arms' based on their lineage statistics yields significantly better results on Bin Packing and TSP with fewer samples. While they fail to beat DeepMind's massive-scale Cap Set record, the methodological insight regarding 'offspring-aware' selection is statistically validated and immediately transferable to our evolutionary search frameworks.

### [tnGPS: Discovering Unknown Tensor Network Structure Search Algorithms via Large Language Models (LLMs](https://arxiv.org/abs/2402.02456)

**2024-06-01** | RIKEN Center for Advanced Intelligence Project, Tencent Inc., Guangdong University of Technology | M=8 P=3 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven automation framework for algorithm discovery via iterative refinement and enhancement using a prompting pipeline | *LLM role:* evolutionary_search

> The authors propose tnGPS, a FunSearch-style framework that evolves Python code for Tensor Network Structure Search by mimicking human innovation stages (categorization, recombination, diversity injection). While the application (Tensor Networks) is niche, the results outperform standard heuristics like TNGA and TNLS. The critical takeaway for us is the 'Knowledge Categorization' phase: they use the LLM to semantically cluster the population of generated algorithms to manage diversity and guide the 'Diversity Injection' step. We should immediately implement this LLM-based population clustering in AlgoEvo to prevent convergence on similar code patterns.

### [The Art of Being Difficult: Combining Human and AI Strengths to Find Adversarial Instances for Heuristics](https://arxiv.org/abs/2601.16849)

**2026-01-23** | Google DeepMind, University of Bonn, University of Manitoba | M=5 P=8 I=7 *discuss*

*Method:* Human-LLM collaborative program search (Co-FunSearch) | *LLM role:* code_writer

> This paper applies FunSearch to generate adversarial instances for classical OR heuristics (Knapsack, Bin Packing, k-median), successfully breaking long-standing theoretical lower bounds. The results are rigorous: they disprove the output-polynomial time of the Nemhauser-Ullmann algorithm and improve the Best-Fit bin packing bound to 1.5. The key takeaway for our AlgoEvo work is the workflow: the LLM finds 'messy' structural patterns (e.g., repeated floats) which humans then manually generalize into asymptotic proofs. This validates Program Search over vector search but exposes the 'generalization gap'—we should implement a post-processing agent to automate this manual refinement step.

### [Programmatic Representation Learning with Language Models](https://arxiv.org/abs/2510.14825)

**2025-10-16** | Harvard University, Stanford University | M=9 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-synthesized programmatic feature functions combined with decision tree predictors, using Features FunSearch (F2) and Dynamic ID3 (D-ID3) algorithms | *LLM role:* code_writer

> The authors propose two algorithms, F2 (Features FunSearch) and D-ID3 (Dynamic ID3), to learn programmatic features for decision trees. D-ID3 is particularly novel: instead of evolving a global heuristic, it calls the LLM at *each split node* to generate a feature that discriminates the specific data subset at that leaf. Results are strong on Chess (matching Transformers trained on 250x more data) and Text, though the Image results (MNIST) are trivial. **Key Takeaway:** The D-ID3 architecture—using the solver's current state (leaf node data) to prompt the LLM for *local* code generation—is a powerful pattern we should steal for our VRP solvers (e.g., evolving local repair operators for specific route bottlenecks) and EvoCut work.

### [LLM-Guided Search for Deletion-Correcting Codes](https://arxiv.org/abs/2504.00613)

**2025-04-01** | Technical University of Munich, Munich Center for Machine Learning | M=7 P=4 I=8 **MUST-READ** *discuss*

*Method:* LLM-guided evolutionary search (FunSearch adaptation) for priority functions | *LLM role:* evolutionary_search

> Weindel and Heckel adapt FunSearch to discover priority functions for the Maximum Independent Set problem (applied to deletion-correcting codes), achieving new SOTA lower bounds for specific lengths (n=12, 13, 16). The critical takeaway for us is their **functional deduplication** step: they hash function outputs on a small subset of data to discard syntactically unique but logically identical programs, which significantly improves sample efficiency by preventing the evaluator from wasting cycles on 'comment changes' or variable renames. Additionally, they demonstrate that optimizing for the single hardest instance generalizes better than averaging performance across a curriculum—a counter-intuitive finding we should test in our reward modeling.

### [Explainable AI-assisted Optimization for Feynman Integral Reduction](https://arxiv.org/abs/2502.09544)

**2025-02-13** | Peking University, Universit
Z
rich, Beijing Computational Science Research Center | M=7 P=3 I=8 *discuss*

*Method:* FunSearch algorithm for developing a priority function to optimize seeding integrals in Integration-by-Parts (IBP) reduction | *LLM role:* heuristic_generator

> Song et al. apply FunSearch to evolve priority functions for Feynman integral reduction, achieving up to 3058x reduction in seeding integrals compared to standard heuristics. The results are rigorous, enabling previously impossible multi-loop calculations. The critical insight for us is the successful transfer of heuristics evolved on trivial 1-loop instances (fast evaluation) to complex 5-loop problems without retraining. We should adopt this 'evolve-on-toy, deploy-on-giant' evaluation protocol to drastically reduce compute costs in our VRP and SAT solver evolutionary search pipelines.

### [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://arxiv.org/abs/2509.24509)

**2025-09-30** | Tencent, Renmin University of China, City University of Hong Kong | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Experience-Guided Reflective Co-Evolution of Prompts and Heuristics (EvoPH) with island-based elites selection | *LLM role:* heuristic_generator

> EvoPH introduces a co-evolutionary framework where both the heuristic code and the LLM prompts are evolved, utilizing an island model for diversity and a 'strategy sampling' mechanism that dynamically selects mutation types (e.g., parameter tuning vs. rewrite) based on feedback. They report dominating performance over FunSearch and ReEvo on TSP and BPP (e.g., reducing Christofides gap from ~20% to ~5%), though the static performance of baselines suggests the gain comes largely from automating prompt engineering. The most stealable insight is the **Strategy Sampling** module: explicitly defining a pool of mutation operators and using an 'experience' buffer to select them is a practical implementation of the 'planner' concept we need for AlgoEvo. We should also adopt their island migration topology to improve diversity in our parallelized search.

### [G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design](https://arxiv.org/abs/2602.08253)

**2026-02-09** | Tsinghua University, University of Chinese Academy of Sciences, Northeastern University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Generative Large Neighborhood Search (G-LNS) with co-evolution of destroy and repair operators | *LLM role:* code_writer

> G-LNS extends LLM-based evolutionary search to ALNS by co-evolving Python code for Destroy and Repair operators rather than constructive priority rules. The authors introduce a 'Synergy Matrix' that tracks the performance of specific operator pairs during evaluation, using this data to guide a 'Synergistic Joint Crossover' where the LLM optimizes the coupling between destroy and repair logic. Results are strong: it significantly outperforms FunSearch and EoH on TSP/CVRP and beats OR-Tools on large-scale instances (N=200) under time constraints. The key takeaway for AlgoEvo is the synergy-aware co-evolution mechanism—explicitly tracking and prompting for component interaction is a concrete technique we can apply to multi-agent optimization systems.

### [Structuring Collective Action with LLM-Guided Evolution: From Ill-Structured Problems to Executable Heuristics](https://arxiv.org/abs/2509.20412)

**2025-12-03** | University of Waterloo, Royal Bank of Canada | M=7 P=5 I=7 *discuss*

*Method:* LLM-driven evolutionary search for Python code heuristics and natural language messages (ECHO-MIMIC framework) | *LLM role:* variation engine, generator, modifier, fixer, agent (simulation) LLM

> ECHO-MIMIC presents a framework that first uses LLM-guided evolution to generate Python heuristics for agents (ECHO), and subsequently evolves natural language 'nudges' (MIMIC) to persuade simulated agents to adopt these global-optimal policies. While the experiments rely on synthetic data for agriculture and EV charging, the approach outperforms DSPy and AutoGen baselines in driving collective action. The most valuable takeaway is the architectural separation of 'policy discovery' (code evolution) and 'adoption mechanism' (message evolution)—a pattern we could adapt to evolve incentive structures or negotiation protocols in our multi-agent optimization systems (MASPRM/HERMES). The analysis of code complexity (Halstead metrics) versus fitness also provides a useful empirical reference for our observability work.

### [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)

**2025-10-10** | UC Berkeley | M=6 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary search (MAP-Elites and island models) with automated code generation and empirical evaluation | *LLM role:* code_writer, reasoning_agent, feedback_generator

> The authors apply OpenEvolve (an AlphaEvolve-style framework) to 11 computer systems problems, achieving significant gains over human baselines, such as a 5.0x speedup in MoE expert placement and 26% cost reduction in cloud scheduling. The results are empirically rigorous, relying on high-fidelity simulators rather than toy problems. For us, the key takeaway is the engineering recipe: using an ensemble of reasoning models (o3) for exploration and fast models (Gemini) for diversity, combined with a specific 'failure taxonomy' to debug search stagnation. This is immediate proof-of-concept for your 'GPUSched' and 'AlgoEvo' projects; we should adopt their ensemble strategy and simulator-first evaluation pipeline.

### [Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM](https://arxiv.org/abs/2510.11121)

**2025-10-13** | Nanyang Technological University, Singapore, Singapore Management University, Singapore, Nanjing University of Information Science and Technology, China | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement Learning (DAPO) fine-tuning of LLM for crossover operator generation within Hybrid Genetic Search (HGS) | *LLM role:* heuristic_generator

> Zhu et al. fine-tune a Qwen-14B model using Reinforcement Learning (DAPO) to generate C++ crossover operators for the state-of-the-art HGS solver. Unlike typical prompting papers, they demonstrate that a small, specialized model can improve upon expert-designed components in a highly optimized solver, achieving superior results on CVRPLIB (up to 1000 nodes) where GPT-4o fails. The most stealable insight is their **AST-based anti-plagiarism reward**, which penalizes the model for generating code structurally identical to the prompt examples, effectively forcing exploration and preventing mode collapse—a technique we should immediately adopt for our evolutionary search agents. This confirms we should pivot from pure prompting to RL-finetuning for our code-generation agents.

### [Contrastive Concept-Tree Search for LLM-Assisted Algorithm Discovery](https://arxiv.org/abs/2602.03132)

**2026-02-03** |  | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Contrastive Concept-Tree Search (CCTS) using a hierarchical Bernoulli model and Tree-structured Parzen Estimator (TPE) for likelihood-ratio based parent reweighting, combined with cross-entropy updates for concept utility estimation. | *LLM role:* heuristic_generator

> The authors introduce Contrastive Concept-Tree Search (CCTS), which modifies the standard evolutionary loop by prompting the LLM to extract semantic 'concepts' from every generated program, building a dynamic hierarchy. They then apply a Tree-structured Parzen Estimator (TPE) to these concepts to learn a contrastive utility model (p(concept|good)/p(concept|bad)), using this to bias parent selection towards promising algorithmic strategies. Results are rigorous, showing consistent improvements over k-elite baselines on combinatorial tasks like Circle Packing, with a synthetic ablation confirming the model learns ground-truth concept utilities. **Key Takeaway:** We should immediately implement the 'Concept TPE' loop in AlgoEvo—asking the LLM to tag generated heuristics with concepts and maintaining a weight vector over these concepts provides a cheap, interpretable 'process reward model' to guide search.

### [Robust Heuristic Algorithm Design with LLMs](https://arxiv.org/abs/2510.08755)

**2025-10-09** | Microsoft, MIT, Microsoft Research, University of Southern California, The University of Texas at Austin | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Explanation-guided genetic search for heuristic design | *LLM role:* evolutionary_search, decomposition_guide, code_writer, prompt_optimizer

> Karimi et al. introduce 'Robusta', an enhancement to FunSearch that uses a Heuristic Analyzer (solver-based) to identify adversarial inputs and a Suggester LLM to explain *why* the current heuristic fails before generating new code. They demonstrate a 28x improvement in worst-case performance over FunSearch on traffic engineering tasks, with results backed by rigorous comparison against optimal solvers. The critical takeaway is the 'Suggester' intermediate step: converting raw failure instances into natural language coding strategies significantly improves the LLM's ability to fix logic bugs compared to raw samples alone. We should immediately attempt to replicate this 'Analyzer -> Explainer -> Coder' loop for our VRP work, using small-scale solvers to generate counter-examples for our evolved ALNS operators.


### Front 1 (15 papers) — STABLE

**Density:** 0.56 | **Methods:** program_synthesis, llm_evolutionary_search, llm_as_heuristic, llm_code_generation, evolution_of_heuristics | **Problems:** heuristic_evolution, tsp, operator_discovery, bin_packing, traveling_salesman_problem

*Unique methods:* abstract_syntax_tree, adaptive_scaling, automated_heuristic_design, bilevel_optimization, chain_of_thought, character_role_play, cumulative_diversity_index, direct_preference_optimization, dpo, evo_mcts, evolution_of_thought, evolutionary_computation, fixed_point_iteration, gnn, gradient_correction, harmony_search, lago, language_hyper_heuristics, learning_to_optimize, llm_as_heuristic_generator, mamba, monte_carlo_tree_search, multi_head_self_attention, multi_objective_optimization, neural_architecture_search, neural_combinatorial_optimization, neural_operator, neural_unrolling, nsga2, numerical_stability, one_plus_one_es, pareto_dominance, reflective_code_synthesis, reflective_evolution, rl_trained, self_reflection, shannon_wiener_diversity_index, spea2, spectral_normalization, supervised_fine_tuning
*Shared methods:* ant_colony_optimization, black_box_optimization, code_embedding, evolution_of_heuristics, evolutionary_algorithm, evolutionary_algorithms, evolutionary_search, genetic_algorithm, genetic_programming, gradient_descent, grpo, guided_local_search, heuristic_evolution, hyper_heuristics, in_context_learning, island_model, large_language_models, large_neighborhood_search, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, local_search, meta_learning, metaheuristics, population_based_optimization, program_synthesis, reflection, reinforcement_learning, self_improving_search

This research front advances LLM-based automated algorithm design by integrating sophisticated search mechanisms with adaptive LLM behavior. A primary focus is the online fine-tuning of LLMs during the evolutionary process, as seen in frameworks like CALM and EvoTune, which employ reinforcement learning (GRPO, DPO) to dynamically adjust LLM weights based on generated heuristic quality. Complementing this, significant effort is directed towards structural co-evolution, where systems like CAE and E2OC evolve not only code implementations but also higher-level algorithm structures or 'design thoughts,' thereby overcoming the limitations of fixed templates and isolated operator optimization.

Key contributions include the development of RL-infused evolutionary search, with CALM fine-tuning a Qwen-7B model to outperform GPT-4o baselines on Bin Packing and VRP, and EvoTune using DPO with Forward KL regularization to achieve up to 15% better optimality gaps on BP/FP and TSP. The front also features MCTS-guided evolution (Evo-MCTS, E2OC), demonstrating significant improvements (e.g., ~150% over ReEvo on signal processing) by structuring evolutionary lineage as a tree. Innovations in reflective evolution (ReEvo, SeEvo) generate 'verbal gradients' for mutations, while multi-objective heuristic evolution (MEoH) leverages AST-based dissimilarity for population diversity, yielding heuristics 16x faster on TSP. Furthermore, the LaGO framework introduces an 'Analyst' module for statistical characterization of solutions, and STRCMP integrates GNN-derived structural priors with LLMs for MILP/SAT, significantly reducing convergence times. Despite these advances, Zhang et al.'s work underscores the surprising effectiveness of a simple (1+1)-EPS baseline, prompting a re-evaluation of search complexity.

This front is rapidly maturing, transitioning from initial feasibility demonstrations to a focus on enhancing efficiency, robustness, and generalizability. The emphasis on online LLM adaptation, structural search, and advanced feedback mechanisms (e.g., statistical analysis, AST-based diversity) signals a trajectory towards more autonomous and less human-dependent algorithm discovery. Future work will likely converge on integrating these disparate advanced techniques into unified, scalable frameworks capable of addressing highly complex, real-world problems with dynamic environments, potentially exploring hybrid LLM architectures (e.g., Mamba-based encoders) and more sophisticated reward mechanisms for non-heuristic LLM trajectories.

**Papers:**

### [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://arxiv.org/abs/2505.12285)

**2025-05-18** | City University of Hong Kong, Southeast University, University of Victoria, Hon Hai Research Institute | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining verbal and numerical guidance for heuristic evolution, achieved by fine-tuning an LLM via reinforcement learning (GRPO) based on heuristic quality, co-evolving the LLM with the search process. | *LLM role:* heuristic_generator_and_fine_tuned_agent

> CALM introduces a hybrid evolutionary framework that fine-tunes the LLM generator *during* the search process using Group Relative Policy Optimization (GRPO), rather than relying solely on prompt evolution. Using a quantized Qwen-7B model on a single consumer GPU, it outperforms GPT-4o-based baselines (FunSearch, EoH) on Bin Packing and VRP benchmarks. The critical takeaway is their reward function design: instead of absolute performance, they reward the *relative improvement* of the generated code over the specific 'parent' heuristics in the prompt, stabilizing the RL signal. We should immediately test this 'online fine-tuning' approach to reduce our API costs and improve sample efficiency in AlgoEvo.

### [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://arxiv.org/abs/2402.01145)

**2024-10-14** | Peking University, KAIST, Singapore Management University, Southeast University, PKU-Wuhan Institute for AI | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* Genetic Programming with LLM-based Reflective Evolutionary Search | *LLM role:* heuristic_generator, decomposition_guide

> ReEvo integrates a 'Reflector LLM' into genetic programming that analyzes pairs of heuristics (better vs. worse) to generate textual 'verbal gradients' for crossover and mutation, maintaining a long-term memory of these insights. The results are strong and relevant: they outperform EoH (Evolution of Heuristics) and NCO baselines on TSP, CVRP, and Bin Packing with significantly higher sample efficiency (only ~100 evaluations). The single most useful takeaway is the 'Short-term Reflection' prompting strategy—explicitly asking the LLM to derive a mutation direction by comparing the logic of high-fitness vs. low-fitness parents—which we should immediately test in our AlgoEvo framework to reduce sample costs. This is a direct methodological upgrade for our current evolutionary search pipelines.

### [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://arxiv.org/abs/2407.10873)

**2024-07-15** | City University of Hong Kong, Southern University of Science and Technology | M=4 P=10 I=6 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-based Evolutionary Program Search (EPS) | *LLM role:* evolutionary_search

> Zhang et al. perform a rigorous benchmarking of major LLM-based evolutionary program search (EPS) methods (FunSearch, EoH, ReEvo) against a simple (1+1)-EPS baseline across four problems and nine LLMs. The results are empirically solid and sobering: the simple (1+1)-EPS baseline—iterative improvement via one-shot prompting—frequently matches or outperforms the complex population-based methods, particularly on bin packing, though EoH remains superior on TSP. **Crucial Takeaway:** We are likely over-engineering our search mechanisms; we must implement a (1+1)-EPS baseline in all future experiments (AlgoEvo, EvoCut) because if our multi-agent systems cannot beat this simple hill-climber, our papers will be rejected for unnecessary complexity. Additionally, they find that larger models (GPT-4) do not strictly guarantee better heuristic search performance compared to smaller, code-specialized models like CodeLlama-7B.

### [Automatic programming via large language models with population self-evolution for dynamic job shop scheduling problem](https://arxiv.org/abs/2410.22657)

**2024-10-30** | Huazhong University of Science and Technology | M=6 P=8 I=6 *discuss*

*Method:* LLM-based population self-evolutionary (SeEvo) method for automatic heuristic dispatching rules (HDRs) design | *LLM role:* heuristic_generator

> This paper introduces SeEvo, an LLM-based evolutionary search for Dynamic Job Shop Scheduling heuristics that adds an 'individual self-reflection' loop—prompting the LLM to analyze performance differences of a specific rule before and after mutation—alongside standard population-level reflection. While they claim significant improvements over GP/GEP and DRL, the ablation study reveals only a marginal <1% improvement over the existing ReEvo framework on benchmark instances. The primary takeaway for us is the specific prompt engineering technique of injecting an individual's mutation history (previous code vs. current code performance) into the context to guide the next mutation, which could potentially improve sample efficiency in our own evolutionary loops despite their weak empirical validation.

### [From Understanding to Excelling: Template-Free Algorithm Design through Structural-Functional Co-Evolution](https://arxiv.org/abs/2503.10721)

**2025-03-13** | Princeton University, Nanyang Technological University, City University of Hong Kong, University of Science and Technology of China, The Hong Kong University of Science and Technology (Guangzhou) | M=9 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-driven bi-dimensional structural-functional co-evolutionary algorithm | *LLM role:* code_writer, heuristic_generator, decomposition_guide, prompt_optimizer

> Zhao et al. propose CAE, a framework that co-evolves algorithm structure (workflow/call graphs) alongside function implementations, aiming to eliminate the fixed templates required by SOTA methods like FunSearch and EoH. On TSP benchmarks, they report reducing optimality gaps by ~2-5% compared to ReEvo, and in quadratic optimization, the system autonomously discovered numerical stability fixes (e.g., replacing matrix inversion with solvers) that human baselines missed. The critical takeaway is the 'bi-dimensional co-evolution' strategy: explicitly maintaining and mutating a population of control flow graphs separate from the function bodies, which allows the system to escape the local optima imposed by a fixed human-designed harness. We must evaluate if this structural search approach can be integrated into AlgoEvo to automate our harness design.

### [Learning to Evolve for Optimization via Stability-Inducing Neural Unrolling](https://arxiv.org/abs/2512.11453)

**2026-03-03** | The Hong Kong Polytechnic University, The University of Hong Kong | M=5 P=7 I=7 *discuss*

*Method:* Bilevel meta-optimization with stability-inducing neural unrolling, using a structured Mamba-based neural operator and a gradient-derived composite solver. | *LLM role:* none

> Gao et al. propose L2E, a meta-learned neural optimizer that uses Mamba blocks to parameterize evolutionary operators within a stability-enforcing unrolled loop (Krasnosel'skii-Mann iteration). Results on BBOB and LSGO-1000D are strong, showing it outperforms Transformer-based L2O methods (GLHF) and classical heuristics (DE) in sample efficiency and zero-shot generalization. **Key Takeaway:** We should investigate replacing Transformer-based population encoders in AlgoEvo with Mamba blocks to reduce complexity from quadratic to linear ($O(N)$), enabling larger population sizes in our meta-heuristic search. The theoretical framing of evolution as a fixed-point iteration also offers a rigorous stability constraint we could inject into our RL-guided search policies.

### [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/abs/2409.16867)

**2025-02-04** | City University of Hong Kong, Southern University of Science and Technology | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* LLM-based Multi-objective Evolutionary Algorithm with Dominance-Dissimilarity Mechanism | *LLM role:* heuristic_generator

> MEoH extends LLM-based heuristic evolution (like FunSearch/EoH) to multi-objective scenarios (e.g., Gap vs. Runtime) by introducing a 'Dominance-Dissimilarity' mechanism that selects parents based on both Pareto dominance and Abstract Syntax Tree (AST) code distance. The results are credible and strong: on TSP, they find heuristics matching EoH's quality but running 16x faster (1.37s vs 22.4s) by effectively navigating the complexity-performance trade-off. The single most useful takeaway is the **AST-based dissimilarity metric** for population management; we should immediately steal this to prune semantically identical code in our evolutionary loops, thereby forcing exploration and improving sample efficiency. This is a direct upgrade to our current single-objective evolutionary search methods.

### [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://arxiv.org/abs/2401.02051)

**2024-06-01** | Huawei Noah’s Ark Lab, City University of Hong Kong, Southern University of Science and Technology | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-assisted evolutionary algorithm for co-evolving natural language heuristic descriptions ('thoughts') and executable code implementations | *LLM role:* heuristic_generator

> EoH introduces a dual-track evolutionary framework that evolves both natural language 'thoughts' (heuristic logic) and their corresponding Python code, rather than code alone. On Online Bin Packing, it claims to outperform DeepMind's FunSearch while using only ~2,000 LLM queries (vs FunSearch's millions), and achieves SOTA gaps on TSP and FSSP via Guided Local Search. The critical takeaway is the 'E2' prompt strategy: explicitly asking the LLM to extract common ideas from parent heuristics into a natural language 'thought' before generating code, which acts as a genetic Chain-of-Thought to stabilize mutation. We should immediately implement this 'Thought-then-Code' mutation operator in our AlgoEvo pipeline to address our sample efficiency bottlenecks.

### [LLM Guided Evolution -- The Automation of Models Advancing Models](https://arxiv.org/abs/2403.11446)

**2024-03-18** | Georgia Tech Research Institute | M=5 P=6 I=7 *discuss*

*Method:* LLM-guided Genetic Algorithm for Neural Architecture Search (NAS) with multi-objective optimization (accuracy and parameter count), incorporating Evolution of Thought (EoT) and Character Role Play (CRP) for mutation and mating. | *LLM role:* evolutionary_search, code_writer, evaluator

> Morris et al. propose 'Guided Evolution,' an LLM-based NAS framework that introduces 'Evolution of Thought' (EoT) and 'Character Role Play' to guide code mutations. While the results are statistically negligible (single trials, ~0.8% gain on CIFAR-10), the EoT mechanism offers a specific, actionable prompt engineering technique: explicitly prompting the LLM to compare a successful elite individual against its original seed to extract 'reasoning' before applying mutations to new individuals. This serves as a lightweight, prompt-based memory/feedback mechanism that could immediately improve sample efficiency in our evolutionary search agents. The 'Character Role Play' (e.g., asking the LLM to act as 'Dr. MaGoo' for unorthodox ideas) is a gimmicky but potentially useful heuristic for maintaining population diversity.

### [Automated Algorithmic Discovery for Scientific Computing through LLM-Guided Evolutionary Search: A Case Study in Gravitational-Wave Detection](https://arxiv.org/abs/2508.03661)

**2025-11-16** | Tsinghua University, University of Chinese Academy of Sciences | M=9 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided Evolutionary Monte Carlo Tree Search (Evo-MCTS) with reflective code synthesis and multi-scale evolutionary operations | *LLM role:* code_writer, heuristic_generator, evaluator, evolutionary_search

> Evo-MCTS introduces a hybrid search architecture where MCTS manages the exploration-exploitation balance of an evolutionary process, using LLMs for node expansion via novel operators like 'Path-wise Crossover' (synthesizing code from full root-to-leaf trajectories). The results are empirically strong, outperforming standard LLM-evolution baselines (ReEvo) by ~150% on a complex signal processing task. We learned that structuring the evolutionary lineage as a tree and using MCTS Q-values to select parents—rather than standard population selection—drastically improves sample efficiency and solution quality. This is a blueprint for the 'RL-infused evolution' and 'persistent memory' features we have been planning for our own framework.

### [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://arxiv.org/abs/2412.14995)

**2024-12-19** | George Mason University, Hanoi University of Science and Technology | M=7 P=8 I=7 *discuss*

*Method:* Adaptive LLM-based Evolutionary Program Search (LLM-EPS) framework combining Harmony Search and Genetic Algorithm with diversity-driven mechanisms | *LLM role:* evolutionary_search

> HSEvo extends LLM-based evolutionary search (LLM-EPS) by integrating a numerical parameter tuning step (Harmony Search) and a token-efficient 'Flash Reflection' mechanism that batches analysis of parent pairs. They report superior results over ReEvo and FunSearch on Bin Packing and TSP, validated by proposed diversity metrics based on code embeddings. **Key Takeaway:** We should implement the hybrid tuning pattern: explicitly parsing LLM-generated code to extract constants and tuning them with a cheap numerical optimizer (rather than asking the LLM to tune parameters), and adopt batched reflections to reduce inference costs.

### [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/abs/2602.16038)

**2026-02-17** | Massachusetts Institute of Technology | M=9 P=10 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Language-Guided Optimization (LaGO) framework decomposing heuristic discovery into forward, backward, and update stages, utilizing LLMs for reasoned evolution, code-writing analysis, co-evolution of constructive and refinement heuristics, and diversity-aware population management. | *LLM role:* evolutionary_search

> LaGO decomposes automated heuristic design into three explicit modules: evaluation, a code-writing 'Analyst' (backward pass), and a diversity-aware 'Generator' (update), while co-evolving constructive and refinement heuristics. The authors demonstrate significant gains (+0.17 QYI) on PDPTW and Crew Pairing against ReEvo and EoH, showing that joint optimization of initialization and improvement prevents local optima. The critical takeaway is the 'Analyst' module: instead of asking the LLM for text critiques, they ask it to write Python feature extraction functions to statistically characterize solution quality—a technique we should immediately adopt to upgrade our fitness signals in AlgoEvo.

### [Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning](https://arxiv.org/abs/2504.05108)

**2025-08-04** | EPFL, Apple | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Search with Reinforcement Learning (DPO) fine-tuning | *LLM role:* heuristic_generator

> EvoTune augments LLM-based evolutionary search (FunSearch) by iteratively fine-tuning the LLM weights using Direct Preference Optimization (DPO) on the generated programs. The results are robust, consistently outperforming static FunSearch on Bin Packing, TSP, and Hash Code benchmarks by discovering better heuristics faster. The critical takeaway is the use of **Forward KL regularization** in DPO instead of the standard Reverse KL; this prevents the mode collapse that usually kills evolutionary diversity, allowing the model to learn from high-fitness samples while maintaining exploration. This is a direct blueprint for implementing the 'RL-infused evolution' component of our AlgoEvo project.

### [Evolving Interdependent Operators with Large Language Models for Multi-Objective Combinatorial Optimization](https://arxiv.org/abs/2601.17899)

**2026-02-01** |  | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Monte Carlo Tree Search for progressive design strategy search with operator rotation evolution | *LLM role:* heuristic_generator

> E2OC introduces a hierarchical search framework where MCTS optimizes 'design thoughts' (textual strategies) rather than raw code, subsequently using these strategies to guide a coordinate-descent-style evolution of interdependent operators. While the computational cost is high due to the inner-loop operator rotation, the results on FJSP/TSP (+20% HV vs expert) and comparisons against FunSearch/EoH demonstrate that explicitly modeling operator coupling is superior to isolated evolution. The critical takeaway for us is the **'strategy-first' search layer**: evolving a semantic blueprint for component interaction *before* code generation prevents the local optima trap of independent component optimization, a technique we should immediately test in AlgoEvo.

### [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/abs/2506.11057)

**2025-05-22** | Shanghai Key Laboratory of Scalable Computing and Systems, School of Computer Science, Shanghai Jiao Tong University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Structure-aware LLM-based algorithm discovery framework combining Graph Neural Network (GNN) for structural embeddings and LLM for solver-specific code generation, refined by an evolutionary algorithm. | *LLM role:* code_writer

> STRCMP introduces a composite architecture where a GNN encodes CO problem instances (MILP/SAT) into embeddings that condition an LLM (fine-tuned via SFT and DPO) to generate solver-specific heuristics within an evolutionary loop. The results are strong and empirically backed, showing significant reductions in convergence time and timeouts compared to text-only evolutionary methods like AutoSAT and LLM4Solver. The key takeaway is the architectural blueprint for fusing instance-specific structural embeddings (via soft prompting) with LLM code generation to drastically improve the sample efficiency of evolutionary search. This is immediately relevant to our EvoCut and AlgoEvo projects, suggesting we should move beyond pure text prompts for topology-heavy problems.


### Front 3 (9 papers) — STABLE

**Density:** 0.69 | **Methods:** llm_code_generation, evolution_of_heuristics, program_synthesis, llm_as_heuristic, llm_evolutionary_search | **Problems:** combinatorial_optimization, tsp, cvrp, bin_packing, heuristic_evolution

*Unique methods:* algorithm_space_response_oracles, ast_manipulation, automated_algorithm_design, best_response_oracle, differential_evolution, directed_acyclic_graph, fireworks_algorithm, game_theory, memetic_algorithm, memoization, meta_game, population_management, swarm_intelligence, ucb, zero_sum_game
*Shared methods:* ant_colony_optimization, co_evolution, eoh, evolution_of_heuristics, evolutionary_algorithm, exploratory_landscape_analysis, funsearch, genetic_algorithm, guided_local_search, island_model_ea, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_in_the_loop, llm_prompt_optimization, local_search, metaheuristics, multi_agent_system, multi_armed_bandit, policy_space_response_oracles, program_synthesis, quality_diversity, reevo, reflection, self_improving_search

This research front focuses on significant architectural and methodological advancements in LLM-driven evolutionary heuristic design, moving beyond basic Evolution of Heuristics (EoH) to incorporate sophisticated mechanisms. Key themes include game-theoretic co-evolution (ASRO), multi-agent collaboration (RoCo), novel program representations (EvoLattice), and nested evolutionary frameworks (TIDE). These approaches aim to enhance the robustness, generalization, and efficiency of LLM-generated algorithms for complex combinatorial optimization problems like TSP, CVRP, and Flexible Job Shop Scheduling, as well as automatic test function generation.

Papers introduce frameworks like ASRO, which uses a game-theoretic approach to co-evolve solvers and instance generators, outperforming standard EoH by 0.5-30% on OBP, TSP, and CVRP. RoCo employs a four-agent LLM collaboration loop with long-term reflection memory for heuristic refinement. EvoLattice introduces a multi-alternative Quality-Diversity DAG representation, yielding +158% spectral energy on NAS-Bench-Zero by providing fine-grained performance signals. TIDE decouples structural generation from numerical parameter tuning, achieving up to 7.35% gap reduction on TSP compared to EoH. Other contributions include Algorithmic-Contextual Prompt Augmentation (A-CEoH) for A* search, EoH-S for evolving complementary heuristic sets, and LLM4EO for online operator design in FJSP.

This front is rapidly emerging, characterized by a strong focus on overcoming the limitations of initial LLM-based evolutionary approaches. The trajectory indicates a shift towards more complex, integrated systems that leverage LLMs for higher-level reasoning (e.g., meta-game play, multi-agent collaboration, prompt evolution) while addressing challenges like numerical precision and generalization. The next papers will likely explore further integration of these architectural innovations, such as combining dynamic prompt co-evolution with multi-alternative program representations, or applying nested tuning to game-theoretic co-evolution for even more robust and transferable algorithm discovery across diverse and challenging problem domains.

**Papers:**

### [Game-Theoretic Co-Evolution for LLM-Based Heuristic Discovery](https://arxiv.org/abs/2601.22896)

**2026-02-09** | Tsinghua University, Chinese Academy of Sciences, University of Chinese Academy of Sciences, AiRiA | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Algorithm Space Response Oracles (ASRO), a game-theoretic framework for program-level co-evolution between solver and instance generator, extending PSRO to discrete program space with LLM-based best-response oracles | *LLM role:* program_synthesis, evolutionary_search

> ASRO adapts Policy Space Response Oracles (PSRO) to code generation, treating heuristic discovery as a zero-sum game where a 'Solver' evolves to minimize gaps and a 'Generator' evolves to create adversarial instances. The results are compelling: it consistently beats the static EoH baseline on TSPLIB and CVRPLIB, proving that adversarial training yields better generalization than training on fixed distributions. The critical takeaway is the architecture: explicitly co-evolving an 'Instance Generator' program alongside the solver prevents overfitting and exposes edge cases (like specific geometric traps in TSP) that static benchmarks miss. This is a direct upgrade to our AlgoEvo/AlphaEvolve pipelines, though it incurs higher computational costs due to the evaluation matrix required for the meta-game.

### [Automatic Design of Optimization Test Problems with Large Language Models](https://arxiv.org/abs/2602.02724)

**2026-02-02** | AGH University of Krakow, Warsaw University of Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* LLM-driven evolutionary search for Python function generation | *LLM role:* evolutionary_search

> Achtelik et al. adapt LLM-driven evolutionary search (EoH) to generate interpretable Python functions that match specific landscape features (ELA), effectively creating synthetic benchmarks on demand. Unlike prior neural network approaches that fail to scale, this method performs robustly in higher dimensions (3D-5D) and produces portable code. The key takeaway is the capability to procedurally generate 'hard' or specific-property instances; we should immediately adopt this to create a dynamic training curriculum for AlgoEvo, ensuring our evolved metaheuristics generalize beyond standard libraries like BBOB.

### [RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design](https://arxiv.org/abs/2512.03762)

**2025-12-04** | South China University of Technology | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-Agent Role-Based System (RoCo) for Automatic Heuristic Design (AHD) integrated into an Evolutionary Program Search (EoH) framework | *LLM role:* evolutionary_search

> RoCo replaces standard evolutionary mutation operators with a 4-agent collaboration loop (Explorer, Exploiter, Critic, Integrator) that iteratively refines heuristics and accumulates long-term reflection memory across generations. While the empirical gains over ReEvo are marginal (often <1%) and likely expensive in token cost, the architecture successfully demonstrates how to embed structured multi-agent reasoning into the evolutionary loop to stabilize black-box search. The key takeaway is their Long-term Reflection mechanism, which aggregates critic feedback into a persistent memory buffer to guide future mutations—a technique we should immediately test to improve sample efficiency in AlgoEvo.

### [EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery](https://arxiv.org/abs/2512.13857)

**2025-12-17** | aiXplain Inc | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLM-guided evolutionary search on a multi-alternative Quality-Diversity Directed Acyclic Graph (DAG) representation with alternative-level performance statistics and deterministic self-repair | *LLM role:* evolutionary_search

> EvoLattice replaces the standard 'overwrite-based' evolution of monolithic programs with a persistent DAG where each node holds multiple alternative implementations, evaluating all valid combinatorial paths to compute fine-grained performance statistics for every micro-operator. The results are strong: it outperforms AlphaEvolve and FunSearch styles on NAS-Bench-Zero by explicitly preserving diversity and enabling surgical, data-driven pruning rather than blind mutation. The critical takeaway is the 'alternative-level statistic' mechanism: by aggregating performance across all paths a component participates in, they generate a high-fidelity signal that tells the LLM exactly which lines of code are working, effectively solving the sparse reward problem in code evolution. We should immediately discuss refactoring our AlgoEvo representation to support this multi-alternative graph structure, as it maximizes signal extraction per LLM call.

### [Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search](https://arxiv.org/abs/2601.19622)

**2026-01-27** |  | M=7 P=5 I=8 *discuss*

*Method:* Evolutionary Heuristic Design (EoH) framework with Algorithmic-Contextual Prompt Augmentation (A-CEoH) | *LLM role:* heuristic_generator

> This paper introduces 'Algorithmic-Contextual EoH' (A-CEoH), which injects the actual source code of the search algorithm (e.g., the A* driver loop, neighbor generation) into the LLM prompt alongside the problem description. Experiments on the Unit-Load Pre-Marshalling Problem and Sliding Puzzle Problem demonstrate that this algorithmic context allows a 32B parameter model (Qwen2.5-Coder) to generate heuristics superior to those from GPT-4o and human experts. The results are credible and backed by comparisons against optimal baselines. The key takeaway is a transferable 'prompt trick': explicitly showing the LLM the code that *calls* its generated function aligns the heuristic significantly better with the search dynamics than natural language descriptions alone. We should immediately test injecting our ALNS/search driver code into our evolutionary prompt templates.

### [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://arxiv.org/abs/2508.03082)

**2025-08-20** | Huawei Noah

































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































Ark Lab, City University of Hong Kong | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary search framework with complementary population management and diversity-aware memetic search | *LLM role:* heuristic_generator

> EoH-S reformulates Automated Heuristic Design (AHD) to evolve a complementary *set* of heuristics rather than a single robust one, proving the objective is submodular and solvable via a greedy strategy. Results are strong and credible: on TSPLib and CVRPLib, their set of 10 heuristics reduces the optimality gap by ~40-60% compared to the top 10 heuristics from FunSearch or ReEvo. **KEY TAKEAWAY:** We should replace standard elitist selection in AlgoEvo with their 'Complementary Population Management' (CPM). By greedily selecting individuals based on marginal contribution to instance coverage (using instance-wise performance vectors), we can automatically generate diverse operator pools for ALNS instead of relying on hand-crafted diversity metrics.

### [Online Operator Design in Evolutionary Optimization for Flexible Job Shop Scheduling via Large Language Models](https://arxiv.org/abs/2511.16485)

**2026-01-22** | City University of Hong Kong, Guangdong University of Technology | M=7 P=8 I=7 *discuss*

*Method:* Genetic Algorithm with LLM-driven online operator design and adaptive operator evolution | *LLM role:* evolutionary_search

> LLM4EO embeds an LLM directly into the Genetic Algorithm loop to dynamically generate and replace gene-selection operators whenever the population stagnates, rather than training them offline. Results on FJSP benchmarks (Brandimarte, Fattahi) show a 3-4% improvement over static GA and GP, with convergence plots demonstrating that LLM interventions successfully break local optima. The most stealable insight is the 'Perception and Analysis' prompt structure: it forces the LLM to explicitly diagnose *why* the current population is stuck (based on fitness stats) before generating new code, a mechanism we should port to AlgoEvo to handle search stagnation. This validates the viability of online, state-aware LLM intervention in OR scheduling problems.

### [TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design](https://arxiv.org/abs/2601.21239)

**2026-01-29** |  | M=9 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Nested evolutionary framework with TSED-guided island model and co-evolutionary inner loop (UCB-based LLM logic generation + differential mutation for parameter tuning) | *LLM role:* heuristic_generator

> TIDE introduces a nested evolutionary framework that strictly decouples algorithmic structure generation (via LLM) from numerical parameter tuning (via Differential Evolution), managed by a Tree Similarity Edit Distance (TSED) guided island model. Results on 9 COPs (TSP, BPP, etc.) show it consistently outperforms ReEvo and EoH, primarily because the DE layer optimizes constants at zero token cost, preventing the discard of structurally sound but poorly tuned heuristics. The critical takeaway is the necessity of a gradient-free tuning layer for LLM-generated code; relying on LLMs for numerical constants is inefficient and imprecise. We should immediately implement a similar parameter-tuning inner loop in our AlgoEvo framework.

### [Beyond Algorithm Evolution: An LLM-Driven Framework for the Co-Evolution of Swarm Intelligence Optimization Algorithms and Prompts](https://arxiv.org/abs/2512.09209)

**2025-12-10** | Peking University | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Collaborative evolution of Fireworks Algorithm operators and prompt templates, driven by a single LLM | *LLM role:* evolutionary_search

> The authors introduce a co-evolutionary framework where both the optimization algorithm (Fireworks Algorithm operators) and the prompt templates used to generate them are evolved simultaneously by the LLM. The results demonstrate a massive performance jump on constrained Aircraft Landing problems (from ~56% with FunSearch to 100% with their method), suggesting that static prompts are a primary failure mode for complex OR constraints. The critical takeaway is their prompt fitness function: evaluating a prompt template based on the *performance improvement* (`child - parent`) of the code it generates, rather than absolute performance. We should immediately implement this 'prompt-delta' fitness signal in AlgoEvo to automate our prompt engineering loop.


### Front 5 (7 papers) — GROWING

**Density:** 0.43 | **Methods:** llm_code_generation, llm_evolutionary_search, evolutionary_algorithm, program_synthesis, evolution_of_heuristics | **Problems:** heuristic_evolution, black_box_optimization, automated_algorithm_design, expensive_continuous_optimization, operator_discovery

*Unique methods:* abstract_syntax_tree_analysis, adam_optimizer, algorithm_analysis, algorithm_distillation, anisotropic_diffusion, barzilai_borwein, butterworth_filter, code_validation, crossover, elitist_strategy, evolution_strategy, evolutionary_strategy, explainable_ai, git_based_coordination, guided_filter, huber_regularization, human_in_the_loop, image_quality_metrics, llamea, meta_optimization, mutation, pairwise_comparison, perceiver_attention, phase_unwrapping, phylogenetic_graph, proximal_operators, reward_free_evolution, search_space_analysis, self_attention, shap_analysis, static_code_analysis, structure_tensor_analysis, supervised_learning, surrogate_modeling, three_way_crossover, total_variation, version_control_system, wasserstein_distance, xgboost
*Shared methods:* alphaevolve, evolution_of_heuristics, evolution_strategies, evolutionary_algorithm, exploratory_landscape_analysis, funsearch, genetic_algorithm, genetic_programming, in_context_learning, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_in_the_loop, meta_learning, metaheuristics, multi_agent_system, program_synthesis, self_improving_search, transformer

This research front focuses on advanced LLM-driven evolutionary algorithm design, leveraging frameworks like LLaMEA, EvoGit, and Evolution Transformer. The core theme is enhancing the evolutionary loop with sophisticated LLM interactions, including autonomous discovery of regularization algorithms for ptychography, decentralized code evolution via Git, and meta-evolution for symbolic regression selection operators. The papers collectively push the boundaries of how LLMs can intelligently guide and accelerate the discovery of novel algorithms and heuristics.

Key contributions include Ptychi-Evolve (Paper 1) which achieved +0.26 SSIM and +8.3 dB PSNR in ptychographic reconstruction by introducing semantically-guided crossover and stateful operator discovery. EvoGit (Paper 2) presented a decentralized multi-agent code evolution framework utilizing Git for lineage and concurrency, simplifying population management. LLaMEA (Papers 3, 6, 7) was explored through behavioral analysis (Paper 3), demonstrating a 1+1 elitist strategy with a 'simplify' mutation prevents code bloat on BBOB 5D problems. LLaMEA-SAGE (Paper 6) integrated SHAP-based structural feedback from AST features, improving performance by 11.1 AUC units on SBOX-COST. A hybrid LLaMEA framework (Paper 7) used GP-evolved symbolic proxies to decouple discovery from expensive evaluations, transferring successfully to photonics problems with 50xD real evaluations. LLM-Meta-SR (Paper 4) introduced semantics-aware crossover for symbolic regression selection operators, outperforming ε-lexicase and achieving +2.3% R2 on SRBench. Finally, the Evolution Transformer (Paper 5) proposed a causal architecture for in-context evolutionary optimization, using Self-Referential Algorithm Distillation (SR-EAD) for self-improvement and generalizing to Brax tasks.

This front is rapidly emerging and growing, with a clear trajectory towards more sophisticated feedback mechanisms and architectural innovations. The focus is shifting from basic LLM-in-the-loop evolution to integrating deep semantic understanding, structural code analysis, and landscape-aware strategies. Future work will likely involve integrating more advanced Explainable AI (XAI) for fine-grained feedback, exploring multi-objective algorithm discovery, and applying these robust frameworks to higher-dimensional, real-world problems with enhanced transfer learning capabilities.

**Papers:**

### [Autonomous Algorithm Discovery for Ptychography via Evolutionary LLM Reasoning](https://arxiv.org/abs/2603.05696)

**2026-03-05** | Argonne National Laboratory, Rice University | M=7 P=3 I=8 *discuss*

*Method:* LLM-guided evolutionary search for regularization algorithms combining LLM-driven code generation with semantically-guided crossover and mutation | *LLM role:* evolutionary_search

> This paper applies LLM-guided evolutionary search (similar to FunSearch/AlphaEvolve) to discover novel regularization algorithms for ptychographic image reconstruction. The results are backed by solid empirical metrics, showing up to +0.26 SSIM improvements over unregularized baselines across multiple datasets. The most useful takeaway for us is twofold: first, their 'semantically-guided crossover' explicitly prompts the LLM to analyze two successful parent algorithms and intentionally merge their complementary mathematical strengths, rather than blindly recombining code. Second, the LLM autonomously discovered the benefit of embedding stateful optimizers (like Adam) and iterative sub-loops directly inside a single heuristic step. This matters for our AlgoEvo and VRP work because we should immediately steal this semantic crossover prompting strategy and ensure our evaluation API permits the LLM to generate stateful, multi-pass operators rather than just stateless, single-pass functions.

### [EvoGit: Decentralized Code Evolution via Git-Based Multi-Agent Collaboration](https://arxiv.org/abs/2506.02049)

**2025-06-01** | The Hong Kong Polytechnic University | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Decentralized multi-agent evolutionary process using a Git-based phylogenetic graph with mutation and three-way crossover operations | *LLM role:* code_writer, evaluator

> Huang et al. introduce EvoGit, a framework where LLM agents asynchronously evolve code by treating Git commits as the population and using 3-way merges (based on Lowest Common Ancestor) as crossover. While the experiments (web app, bin packing generator) are largely qualitative and lack rigorous statistical benchmarking against baselines like MetaGPT, the architectural contribution is significant. The key takeaway is using Git's native DAG structure to handle lineage, persistence, and asynchronous concurrency 'for free,' replacing complex custom population managers. This is directly actionable for our AlgoEvo infrastructure to enable massive parallelism and better memory/traceability without reinventing the wheel.

### [Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery](https://arxiv.org/abs/2507.03605)

**2025-07-04** | Leiden University, University of Stirling | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA framework with 1+1 elitist evolution strategy and dual mutation prompts (code simplification and random perturbation) | *LLM role:* evolutionary_search

> The authors introduce a behavioral analysis framework for LLM-driven algorithm discovery, mapping the 'behavior space' of generated heuristics using Search Trajectory Networks (STNs) and Code Evolution Graphs (CEGs). Results on BBOB (5D) show that a simple 1+1 elitist strategy alternating between 'simplify code' and 'random new' prompts significantly outperforms population-based approaches, effectively balancing exploitation and exploration while preventing code bloat. The primary takeaway is the critical role of a 'simplify' mutation operator—without it, LLM-generated code tends to drift into complexity without performance gains. We should immediately adopt their visualization metrics to debug our own evolutionary search trajectories and implement their 'simplify' prompt strategy in AlgoEvo.

### [LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression](https://arxiv.org/abs/2505.18602)

**2025-08-08** | Victoria University of Wellington, Michigan State University | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* LLM-driven meta-evolutionary framework for designing selection operators, incorporating semantics-aware selection, bloat control, and domain knowledge into prompts | *LLM role:* evolutionary_search

> Zhang et al. develop a meta-evolutionary framework to evolve selection operators for symbolic regression, achieving state-of-the-art results on SRBench by outperforming expert-designed methods like ε-lexicase. The standout contribution is **semantics-aware crossover**: rather than selecting parents based solely on scalar fitness, they compute complementarity scores using performance vectors across instances, explicitly retrieving parents that solve different subsets of the problem. This effectively treats parent selection as a retrieval task based on behavioral signatures, ensuring the LLM combines distinct functional capabilities. We should immediately implement this complementarity-based parent retrieval in AlgoEvo to improve how we merge heuristics.

### [Evolution Transformer: In-Context Evolutionary Optimization](https://arxiv.org/abs/2403.02985)

**2024-03-05** | Google DeepMind, TU Berlin | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolution Transformer, a causal Transformer architecture with self-attention and Perceiver cross-attention for search distribution updates | *LLM role:* evolutionary_search

> Lange et al. introduce the Evolution Transformer, a causal architecture that learns to perform evolutionary strategy updates by attending to optimization history, effectively 'distilling' algorithms like CMA-ES into a neural network. Crucially, they propose 'Self-Referential Algorithm Distillation' (SR-EAD), where the model improves itself by perturbing its own weights, generating trajectories, and filtering for the best ones to retrain on—eliminating the need for a teacher. The results are strong, showing generalization to unseen Brax control tasks and successful (though sometimes unstable) self-bootstrapping. The key takeaway for us is the SR-EAD loop as a mechanism for open-ended optimizer improvement, and their use of Perceiver cross-attention to handle variable population sizes—a technique we should immediately steal for our multi-agent memory architectures.

### [LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI](https://arxiv.org/abs/2601.21511)

**2026-01-29** |  | M=8 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* LLaMEA-SAGE, an LLM-driven evolutionary algorithm that integrates structural feedback from Explainable AI (SHAP) analysis of Abstract Syntax Tree (AST) code features to guide mutations. | *LLM role:* evolutionary_search

> LLaMEA-SAGE augments LLM-based evolutionary search by extracting AST features (complexity, graph metrics) from generated code, training a surrogate model to predict fitness from these features, and using SHAP analysis to generate natural language prompts that guide the LLM to modify specific structural properties (e.g., 'increase cyclomatic complexity'). On the MA-BBOB benchmark, it outperforms state-of-the-art methods (MCTS-AHD, LHNS) and converges faster than vanilla LLaMEA, although the authors honestly report that statistical significance was limited (p=0.44) due to small sample sizes (5 runs). The critical takeaway for us is the pipeline of using static code analysis as a feedback signal—we can immediately steal this 'SAGE' loop to guide AlgoEvo or EvoCut by telling the LLM *how* to structurally mutate code based on surrogate correlations, rather than just hoping for random improvements.

### [Landscape-aware Automated Algorithm Design: An Efficient Framework for Real-world Optimization](https://arxiv.org/abs/2602.04529)

**2026-02-04** |  | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hybrid framework combining Genetic Programming (GP) for proxy function generation and an LLM-driven Evolutionary Algorithm (LLaMEA) for algorithm discovery, guided by Exploratory Landscape Analysis (ELA) features and Wasserstein distance. | *LLM role:* algorithm_designer

> Yin et al. introduce a framework that decouples algorithm discovery from expensive evaluations by using Genetic Programming to evolve symbolic proxy functions that statistically match the target problem's landscape (via ELA features). Empirical results on photonics problems confirm that algorithms evolved on these cheap proxies transfer successfully to the real tasks, outperforming standard baselines like LSHADE with only 50×D real evaluations. **Key Takeaway:** We can synthesize 'symbolic gyms' that statistically mimic our target problems to run thousands of LLM iterations at near-zero cost. This directly addresses the sample efficiency bottleneck in AlgoEvo and suggests we should move beyond standard neural surrogates to evolved symbolic proxies.



## Bridge Papers

Papers connecting multiple research fronts:

### [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://arxiv.org/abs/2508.03082)

**TRUE SYNTHESIS** | score=0.65 | Front 3 → Front 1, Front 6, Front 8, Front 5

> EoH-S reformulates Automated Heuristic Design (AHD) to evolve a complementary *set* of heuristics rather than a single robust one, proving the objective is submodular and solvable via a greedy strateg

### [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

**TRUE SYNTHESIS** | score=0.56 | Front 6 → Front 1, Front 8, Front 3

> AlphaEvolve extends FunSearch by evolving entire code files (rather than single functions) using a 'search/replace' diff format and Gemini 2.0, achieving SOTA results across matrix multiplication (bea

### [Autonomous Algorithm Discovery for Ptychography via Evolutionary LLM Reasoning](https://arxiv.org/abs/2603.05696)

**TRUE SYNTHESIS** | score=0.54 | Front 5 → Front 1

> This paper applies LLM-guided evolutionary search (similar to FunSearch/AlphaEvolve) to discover novel regularization algorithms for ptychographic image reconstruction. The results are backed by solid


---

*Generated by Research Intelligence System*
