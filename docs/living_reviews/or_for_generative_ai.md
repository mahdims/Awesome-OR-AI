# Living Review: OR for Generative AI

**Last Updated:** 2026-03-01

---

## Recent Papers

#### 2026-03-01 (6 papers)

### [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548)

**2026-02-25** | DeepSeek-AI, Peking University, Tsinghua University | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* Dual-path KV-Cache loading architecture with CNIC-centric traffic management and adaptive request scheduling | *LLM role:* none

> Wu et al. introduce DualPath, a system that breaks the storage I/O bottleneck in agentic inference by utilizing idle decode-node bandwidth to load KV-cache and transferring it to prefill nodes via RDMA. The results are highly credible (1.87x throughput increase on DeepSeek-V3) and directly target the long-context, short-append patterns typical of our evolutionary search rollouts. The most valuable technical takeaway is the 'CNIC-centric' traffic management strategy, which isolates bulk KV transfer from latency-sensitive model collectives—a technique we should immediately steal for our own serving infrastructure. While their scheduling logic is a simple heuristic, the architectural change defines a new, more flexible state space for our OR-based resource allocation research.

### [Training Generalizable Collaborative Agents via Strategic Risk Aversion](https://arxiv.org/abs/2602.21515)

**2026-02-25** | Caltech | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Strategically Risk-Averse Policy Optimization (SRPO) based on Risk-Averse Quantal Response Equilibria (RQE) | *LLM role:* collaborative_agent

> Qu et al. introduce Strategically Risk-Averse Policy Optimization (SRPO), which trains agents against a 'constrained adversary' that minimizes their reward within a KL-divergence bound of the partner's current policy. Theoretical results prove this objective eliminates free-riding equilibria, and experiments on GSM8K multi-agent debate show it prevents 'lazy' agreement, improving joint accuracy by up to 19% when pairing heterogeneous LLMs (e.g., 0.6B with 4B). The key takeaway is that robustness to partner deviation—enforced via this specific adversarial objective—is a more principled way to fix lazy agent behavior than prompt engineering or simple dropout. We should immediately test this objective in our HERMES debate framework to improve the contribution quality of smaller models.

### [How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization](https://arxiv.org/abs/2602.19208)

**2026-02-22** | Meituan, Tsinghua University, Fudan University, Peking University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dual-pronged optimization framework (DynaMO) combining variance-minimizing dynamic rollout allocation and gradient-aware advantage modulation for GRPO-based policy optimization | *LLM role:* policy_model

> DynaMO introduces a dual-pronged optimization for RLVR: a dynamic rollout allocation strategy that prioritizes problems with high gradient variance (proxied by Bernoulli variance of success/failure), and a gradient modulation technique to stabilize updates. The results are strong (+11.8% Pass@1 over GRPO on Qwen-7B) and backed by clear ablations. **The critical takeaway for us is the allocation logic:** we should immediately replace uniform sampling in AlgoEvo with variance-based allocation ($n_i \propto \sqrt{p(1-p)}$). This ensures compute is spent on instances/components that are currently 'learnable' (high variance) rather than wasted on the trivial or impossible, directly optimizing our search budget.

### [ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning](https://arxiv.org/abs/2602.21534)

**2026-02-25** | University of California, Los Angeles, University of Wisconsin–Madison | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stable Agentic Multi-turn Policy Optimization (SAMPO) integrating sequence-level clipping, fine-grained advantage estimation, and dynamic filtering | *LLM role:* policy

> The authors dissect why standard RL (GRPO/PPO) fails in multi-turn agentic tasks, identifying that token-level importance sampling (IS) clipping allows negative-advantage outliers to destabilize training. They propose SAMPO, which enforces sequence-level clipping and integrates fine-grained step-level advantages (similar to process rewards) to stabilize learning. The results are rigorous, showing a jump from ~50% to 92% success on ALFWorld by fixing the gradient update mechanics rather than just prompt engineering. **Key Takeaway:** We must audit our RL implementations; if we are using token-level clipping for multi-step evolutionary agents, we are likely suffering from silent gradient instability—switching to sequence-level clipping and masking negative-advantage outliers is an immediate, code-level improvement we should adopt.

### [Reasoning-Driven Design of Single Atom Catalysts via a Multi-Agent Large Language Model Framework](https://arxiv.org/abs/2602.21533)

**2026-02-25** | Georgia Institute of Technology, Korea University, Sogang University, Ewha Womans University | M=7 P=3 I=8 *discuss*

*Method:* Multi-Agent-based Electrocatalyst Search Through Reasoning and Optimization (MAESTRO) framework using LLM agents and a Machine Learning Force Field (MLFF) surrogate model | *LLM role:* evolutionary_search

> Mok et al. propose MAESTRO, a multi-agent LLM framework for optimizing single-atom catalysts that explicitly separates search into exploration and exploitation phases, bridged by a textual 'Exploration Report.' Results are validated against high-fidelity DFT calculations, showing the system learns to break theoretical scaling relations via in-context learning, outperforming memory-less baselines. The key takeaway for us is the **Exploration Report Agent**: instead of just passing the best candidates to the exploitation phase, the system pauses to write a natural language strategy guide summarizing 'what worked and what didn't' from the exploration phase. We should steal this mechanism for AlgoEvo to let the search agent 'learn' from the initial population generation rather than just selecting from it.

### [GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization](https://arxiv.org/abs/2602.20427)

**2026-02-23** | Cornell University, University of Maryland, College Park | M=7 P=5 I=7 *discuss*

*Method:* Differentiable Scheduling Optimization via Gaussian Reparameterization with Augmented Lagrangian Method | *LLM role:* none

> GauS replaces the standard categorical (Gumbel-Softmax) relaxation in differentiable scheduling with Gaussian variables defined by mean and variance, reducing parameter space from O(N*D) to O(N). Results are strong: it scales to 57k nodes where previous differentiable methods OOM and exact solvers timeout, while maintaining near-100% GPU utilization. The key takeaway is a specific modeling technique: using Gaussian distributions to represent discrete ordinal values (like time steps) naturally captures temporal proximity and provides smoother gradients than categorical buckets. We should test this representation in our continuous latent-space optimization work to replace categorical relaxations for ordered parameters.


#### 2026-02-26 (8 papers)

### [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548)

**2026-02-25** | DeepSeek-AI, Peking University, Tsinghua University | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* Dual-path KV-Cache loading architecture with CNIC-centric traffic management and adaptive request scheduling | *LLM role:* none

> Wu et al. introduce DualPath, a system that breaks the storage I/O bottleneck in agentic inference by utilizing idle decode-node bandwidth to load KV-cache and transferring it to prefill nodes via RDMA. The results are highly credible (1.87x throughput increase on DeepSeek-V3) and directly target the long-context, short-append patterns typical of our evolutionary search rollouts. The most valuable technical takeaway is the 'CNIC-centric' traffic management strategy, which isolates bulk KV transfer from latency-sensitive model collectives—a technique we should immediately steal for our own serving infrastructure. While their scheduling logic is a simple heuristic, the architectural change defines a new, more flexible state space for our OR-based resource allocation research.

### [Training Generalizable Collaborative Agents via Strategic Risk Aversion](https://arxiv.org/abs/2602.21515)

**2026-02-25** | Caltech | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Strategically Risk-Averse Policy Optimization (SRPO) based on Risk-Averse Quantal Response Equilibria (RQE) | *LLM role:* collaborative_agent

> Qu et al. introduce Strategically Risk-Averse Policy Optimization (SRPO), which trains agents against a 'constrained adversary' that minimizes their reward within a KL-divergence bound of the partner's current policy. Theoretical results prove this objective eliminates free-riding equilibria, and experiments on GSM8K multi-agent debate show it prevents 'lazy' agreement, improving joint accuracy by up to 19% when pairing heterogeneous LLMs (e.g., 0.6B with 4B). The key takeaway is that robustness to partner deviation—enforced via this specific adversarial objective—is a more principled way to fix lazy agent behavior than prompt engineering or simple dropout. We should immediately test this objective in our HERMES debate framework to improve the contribution quality of smaller models.

### [How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization](https://arxiv.org/abs/2602.19208)

**2026-02-22** | Meituan, Tsinghua University, Fudan University, Peking University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dual-pronged optimization framework (DynaMO) combining variance-minimizing dynamic rollout allocation and gradient-aware advantage modulation for GRPO-based policy optimization | *LLM role:* policy_model

> DynaMO introduces a dual-pronged optimization for RLVR: a dynamic rollout allocation strategy that prioritizes problems with high gradient variance (proxied by Bernoulli variance of success/failure), and a gradient modulation technique to stabilize updates. The results are strong (+11.8% Pass@1 over GRPO on Qwen-7B) and backed by clear ablations. **The critical takeaway for us is the allocation logic:** we should immediately replace uniform sampling in AlgoEvo with variance-based allocation ($n_i \propto \sqrt{p(1-p)}$). This ensures compute is spent on instances/components that are currently 'learnable' (high variance) rather than wasted on the trivial or impossible, directly optimizing our search budget.

### [AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation](https://arxiv.org/abs/2602.17100)

**2026-02-19** | Shanghai Jiao Tong University, Meituan | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement learning (GRPO) optimized multi-agent system with LLM-based orchestrator agent for dynamic layered DAG topology generation | *LLM role:* orchestrator

> AgentConductor trains an LLM orchestrator via GRPO to dynamically generate and refine layered DAG interaction topologies (output as YAML) for code generation, optimizing for both correctness and token efficiency. The key innovation is a multi-objective reward that combines execution correctness with a 'difficulty-aware density penalty,' forcing the model to learn a policy that scales graph complexity with task hardness. Results are strong, showing ~14% gains on APPS while reducing token costs. We should immediately steal the **YAML-based topology action space** and the **density-aware reward formulation** to implement RL-driven structure optimization in AlgoEvo and MASPRM, replacing our static or hand-tuned interaction graphs.

### [ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning](https://arxiv.org/abs/2602.21534)

**2026-02-25** | University of California, Los Angeles, University of Wisconsin–Madison | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stable Agentic Multi-turn Policy Optimization (SAMPO) integrating sequence-level clipping, fine-grained advantage estimation, and dynamic filtering | *LLM role:* policy

> The authors dissect why standard RL (GRPO/PPO) fails in multi-turn agentic tasks, identifying that token-level importance sampling (IS) clipping allows negative-advantage outliers to destabilize training. They propose SAMPO, which enforces sequence-level clipping and integrates fine-grained step-level advantages (similar to process rewards) to stabilize learning. The results are rigorous, showing a jump from ~50% to 92% success on ALFWorld by fixing the gradient update mechanics rather than just prompt engineering. **Key Takeaway:** We must audit our RL implementations; if we are using token-level clipping for multi-step evolutionary agents, we are likely suffering from silent gradient instability—switching to sequence-level clipping and masking negative-advantage outliers is an immediate, code-level improvement we should adopt.

### [Reasoning-Driven Design of Single Atom Catalysts via a Multi-Agent Large Language Model Framework](https://arxiv.org/abs/2602.21533)

**2026-02-25** | Georgia Institute of Technology, Korea University, Sogang University, Ewha Womans University | M=7 P=3 I=8 *discuss*

*Method:* Multi-Agent-based Electrocatalyst Search Through Reasoning and Optimization (MAESTRO) framework using LLM agents and a Machine Learning Force Field (MLFF) surrogate model | *LLM role:* evolutionary_search

> Mok et al. propose MAESTRO, a multi-agent LLM framework for optimizing single-atom catalysts that explicitly separates search into exploration and exploitation phases, bridged by a textual 'Exploration Report.' Results are validated against high-fidelity DFT calculations, showing the system learns to break theoretical scaling relations via in-context learning, outperforming memory-less baselines. The key takeaway for us is the **Exploration Report Agent**: instead of just passing the best candidates to the exploitation phase, the system pauses to write a natural language strategy guide summarizing 'what worked and what didn't' from the exploration phase. We should steal this mechanism for AlgoEvo to let the search agent 'learn' from the initial population generation rather than just selecting from it.

### [GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization](https://arxiv.org/abs/2602.20427)

**2026-02-23** | Cornell University, University of Maryland, College Park | M=7 P=5 I=7 *discuss*

*Method:* Differentiable Scheduling Optimization via Gaussian Reparameterization with Augmented Lagrangian Method | *LLM role:* none

> GauS replaces the standard categorical (Gumbel-Softmax) relaxation in differentiable scheduling with Gaussian variables defined by mean and variance, reducing parameter space from O(N*D) to O(N). Results are strong: it scales to 57k nodes where previous differentiable methods OOM and exact solvers timeout, while maintaining near-100% GPU utilization. The key takeaway is a specific modeling technique: using Gaussian distributions to represent discrete ordinal values (like time steps) naturally captures temporal proximity and provides smoother gradients than categorical buckets. We should test this representation in our continuous latent-space optimization work to replace categorical relaxations for ordered parameters.

### [Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems](https://arxiv.org/abs/2602.17910)

**2026-02-20** | Lehigh University | M=7 P=5 I=7 *discuss*

*Method:* APEMO (Affect-aware Peak-End Modulation for Orchestration), a runtime scheduling layer that reallocates reasoning effort and repair across a trajectory under fixed computational budgets by operationalizing temporal-affective signals. | *LLM role:* agents_being_orchestrated

> Shi et al. introduce APEMO, a runtime orchestration layer that monitors agent trajectories for behavioral instability (e.g., repetition, drift) and dynamically reallocates a fixed compute budget to 'repair' these segments rather than spreading compute uniformly. The results are statistically rigorous, using bootstrap CIs to demonstrate significant improvements in trajectory robustness and completion rates without model retraining. **Key Takeaway:** We should steal the 'precision repair' logic: instead of uniform sampling in AlgoEvo, we can implement a 'stagnation detector' that triggers deeper inference or multi-agent debate only when the search gets stuck in local optima. This directly addresses our sample efficiency and resource allocation goals.


#### 2026-02-24 (4 papers)

### [How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization](https://arxiv.org/abs/2602.19208)

**2026-02-22** | Meituan, Tsinghua University, Fudan University, Peking University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dual-pronged optimization framework (DynaMO) combining variance-minimizing dynamic rollout allocation and gradient-aware advantage modulation for GRPO-based policy optimization | *LLM role:* policy_model

> DynaMO introduces a dual-pronged optimization for RLVR: a dynamic rollout allocation strategy that prioritizes problems with high gradient variance (proxied by Bernoulli variance of success/failure), and a gradient modulation technique to stabilize updates. The results are strong (+11.8% Pass@1 over GRPO on Qwen-7B) and backed by clear ablations. **The critical takeaway for us is the allocation logic:** we should immediately replace uniform sampling in AlgoEvo with variance-based allocation ($n_i \propto \sqrt{p(1-p)}$). This ensures compute is spent on instances/components that are currently 'learnable' (high variance) rather than wasted on the trivial or impossible, directly optimizing our search budget.

### [AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation](https://arxiv.org/abs/2602.17100)

**2026-02-19** | Shanghai Jiao Tong University, Meituan | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement learning (GRPO) optimized multi-agent system with LLM-based orchestrator agent for dynamic layered DAG topology generation | *LLM role:* orchestrator

> AgentConductor trains an LLM orchestrator via GRPO to dynamically generate and refine layered DAG interaction topologies (output as YAML) for code generation, optimizing for both correctness and token efficiency. The key innovation is a multi-objective reward that combines execution correctness with a 'difficulty-aware density penalty,' forcing the model to learn a policy that scales graph complexity with task hardness. Results are strong, showing ~14% gains on APPS while reducing token costs. We should immediately steal the **YAML-based topology action space** and the **density-aware reward formulation** to implement RL-driven structure optimization in AlgoEvo and MASPRM, replacing our static or hand-tuned interaction graphs.

### [AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence](https://arxiv.org/abs/2602.16873)

**2026-02-18** | Korea National Open University | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Task-adaptive topology routing algorithm based on DAG structural properties (parallelism width, critical path depth, coupling density) combined with an adaptive synthesis protocol | *LLM role:* decomposition_guide, executor, arbiter, synthesizer

> AdaptOrch introduces a control layer that dynamically routes tasks to one of four agent topologies (Parallel, Sequential, Hierarchical, Hybrid) by analyzing the task's dependency graph properties (parallelism width, coupling density). The results are strong and credible, showing a 9.8% improvement on SWE-bench over single-model baselines and significantly outperforming static multi-agent architectures like standard MoA. The most valuable takeaway is the **Topology Routing Algorithm**: a linear-time heuristic that maps DAG structure to optimal agent coordination patterns. We should adapt this for AlgoEvo to automatically parallelize search on loosely coupled code components while forcing sequential reasoning on critical paths, potentially improving our sample efficiency and cost scaling.

### [Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems](https://arxiv.org/abs/2602.17910)

**2026-02-20** | Lehigh University | M=7 P=5 I=7 *discuss*

*Method:* APEMO (Affect-aware Peak-End Modulation for Orchestration), a runtime scheduling layer that reallocates reasoning effort and repair across a trajectory under fixed computational budgets by operationalizing temporal-affective signals. | *LLM role:* agents_being_orchestrated

> Shi et al. introduce APEMO, a runtime orchestration layer that monitors agent trajectories for behavioral instability (e.g., repetition, drift) and dynamically reallocates a fixed compute budget to 'repair' these segments rather than spreading compute uniformly. The results are statistically rigorous, using bootstrap CIs to demonstrate significant improvements in trajectory robustness and completion rates without model retraining. **Key Takeaway:** We should steal the 'precision repair' logic: instead of uniform sampling in AlgoEvo, we can implement a 'stagnation detector' that triggers deeper inference or multi-agent debate only when the search gets stuck in local optima. This directly addresses our sample efficiency and resource allocation goals.


#### 2026-02-22 (3 papers)

### [AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation](https://arxiv.org/abs/2602.17100)

**2026-02-19** | Shanghai Jiao Tong University, Meituan | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement learning (GRPO) optimized multi-agent system with LLM-based orchestrator agent for dynamic layered DAG topology generation | *LLM role:* orchestrator

> AgentConductor trains an LLM orchestrator via GRPO to dynamically generate and refine layered DAG interaction topologies (output as YAML) for code generation, optimizing for both correctness and token efficiency. The key innovation is a multi-objective reward that combines execution correctness with a 'difficulty-aware density penalty,' forcing the model to learn a policy that scales graph complexity with task hardness. Results are strong, showing ~14% gains on APPS while reducing token costs. We should immediately steal the **YAML-based topology action space** and the **density-aware reward formulation** to implement RL-driven structure optimization in AlgoEvo and MASPRM, replacing our static or hand-tuned interaction graphs.

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence](https://arxiv.org/abs/2602.16873)

**2026-02-18** | Korea National Open University | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Task-adaptive topology routing algorithm based on DAG structural properties (parallelism width, critical path depth, coupling density) combined with an adaptive synthesis protocol | *LLM role:* decomposition_guide, executor, arbiter, synthesizer

> AdaptOrch introduces a control layer that dynamically routes tasks to one of four agent topologies (Parallel, Sequential, Hierarchical, Hybrid) by analyzing the task's dependency graph properties (parallelism width, coupling density). The results are strong and credible, showing a 9.8% improvement on SWE-bench over single-model baselines and significantly outperforming static multi-agent architectures like standard MoA. The most valuable takeaway is the **Topology Routing Algorithm**: a linear-time heuristic that maps DAG structure to optimal agent coordination patterns. We should adapt this for AlgoEvo to automatically parallelize search on loosely coupled code components while forcing sequential reasoning on critical paths, potentially improving our sample efficiency and cost scaling.


#### 2026-02-22 (1 papers)

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.


#### 2026-02-22 (1 papers)

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.


<!-- New papers are appended here by the daily updater -->

---

## Research Fronts

*9 fronts detected — snapshot 2026-02-24*

### Front 3 (10 papers) — STABLE

**Density:** 0.71 | **Methods:** pipeline_parallelism, integer_linear_programming, data_parallelism, tensor_parallelism, optimization | **Problems:** llm_serving_optimization, resource_allocation, gpu_scheduling, latency_minimization, edge_llm_inference

*Unique methods:* adaptive_scheduling, arima_time_series_forecasting, autoregressive_decoding, binary_search, cache_replacement_policy, chebyshev_guided_optimization, cvxpy, decomposition_algorithm, demand_forecasting, discrete_event_simulation, dynamic_batching, dynamic_scheduling, heuristic_initialization, heuristics, kv_cache, kv_cache_management, least_carbon_savings, llm_serving_optimization, max_flow_optimization, milp, milp_acceleration, milp_formulation, model_parallelism, network_communication_optimization, network_topology_modeling, neural_network, np_hardness_proof, optimal_transport, optimization, profiling, reactive_heuristics, resource_allocation_optimization, resource_management, sarima, shortest_path_algorithms, shortest_path_routing, simulation, system_algorithm_co_design, task_batching, threshold_based_routing, time_series_forecasting, weighted_round_robin, wireless_resource_allocation
*Shared methods:* bi_level_optimization, continuous_batching, data_parallelism, distributed_systems, dynamic_programming, greedy_algorithm, integer_linear_programming, llm_as_evaluator, load_balancing, mixed_integer_linear_programming, performance_modeling, pipeline_parallelism, proximal_policy_optimization, queueing_theory, reinforcement_learning, resource_allocation, robust_optimization, scheduling_algorithms, speculative_decoding, tensor_parallelism

This research front is unified by the application of Operations Research, specifically Integer Linear Programming (ILP) and Mixed-Integer Linear Programming (MILP), to optimize various aspects of Large Language Model (LLM) serving. The core theme revolves around enhancing efficiency, reducing latency, and managing resources for LLM inference in complex distributed and heterogeneous computing environments, including edge networks and cloud data centers. Key problems addressed include GPU scheduling, resource allocation, load balancing, and minimizing operational costs or carbon footprints.

Key contributions include frameworks like Cascadia, which achieves up to 3.3x higher throughput over CascadeServe using bi-level optimization, and Helix, demonstrating up to 3.3x decode throughput gains on mixed GPU clusters by formulating serving as a max-flow problem. SageServe reduces GPU hours by 25% and saves $2.5M/month through forecast-aware auto-scaling with ILP. AMPD's Dynamo framework significantly improves SLO attainment (67-340%) for multi-round LLM inference over disaggregated serving. Other papers introduce ILP for expert placement in MoE models, carbon-aware KV cache management (GreenCache), and joint optimization of speculative decoding parameters in edge networks.

This front is rapidly maturing, characterized by a strong emphasis on practical, deployable solutions for complex LLM serving challenges. The consistent use of rigorous ILP/MILP formulations, often validated with real-world benchmarks (e.g., Microsoft O365 production traces, Azure Conversation dataset), indicates a shift towards robust, production-ready systems. Future work will likely focus on integrating dynamic adaptation, multi-objective optimization (e.g., cost, carbon, latency), and scaling these OR approaches to even larger, more dynamic cloud and edge infrastructures, potentially leveraging hybrid OR-RL approaches.

**Papers:**

### [Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding](https://arxiv.org/abs/2510.11331)

**2025-10-13** | Queen Mary University of London, Kyung Hee University, Xidian University, Guangzhou Institute of Technology | M=5 P=7 I=6 

*Method:* Speculative Decoding (SD) with pipeline parallelism, combined with joint optimization of speculation length, task batching, and wireless communication resource allocation | *LLM role:* inference engine

> Zhu et al. propose a distributed Speculative Decoding framework for edge networks, formulating a Mixed-Integer Nonlinear Programming problem to jointly optimize task batching, speculation length, and wireless bandwidth. They solve the batching subproblem using a Dynamic Programming (DP) algorithm, achieving ~30-45% latency reduction over heuristics in simulations, though the approach relies on a rigid assumption of fixed maximum output lengths to remain tractable. The primary takeaway for our 'GPUSched' work is their DP formulation for optimizing batch boundaries in a pipelined draft-verify system, which offers a cleaner mathematical alternative to greedy heuristics for serving schedules. However, the heavy reliance on wireless channel modeling makes the full system less relevant to our datacenter-centric optimization problems.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [Cache Your Prompt When It's Green: Carbon-Aware Caching for Large Language Model Serving](https://arxiv.org/abs/2505.23970)

**2026-01-19** | University of Waterloo, Purdue University | M=5 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) based dynamic cache size reconfiguration with SARIMA load prediction and carbon-aware Least Carbon Savings (LCS) cache replacement policy | *LLM role:* none

> Tian et al. propose GreenCache, a framework using Integer Linear Programming (ILP) to dynamically resize KV caches for LLM serving, balancing operational carbon (compute) against embodied carbon (SSD storage). They demonstrate ~15% carbon reduction on Llama-3 70B using Azure traces, though the reliance on simulation rather than live deployment weakens the claims slightly. For our 'OR for AI systems' work, the key takeaway is their 'Least Carbon Savings' (LCS) eviction policy—a heuristic that weighs computation saved against storage cost and recency—which we could adapt for optimizing memory-constrained multi-agent systems (HERMES) or general serving resource allocation.

### [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs](https://arxiv.org/abs/2502.00722)

**2025-06-05** | University of Cambridge, ETH Zurich, Peking University, The Hong Kong University of Science and Technology, Purdue University | M=5 P=9 I=6 **MUST-READ** *discuss*

*Method:* Mixed-Integer Linear Programming (MILP) for scheduling | *LLM role:* none

> Jiang et al. formulate LLM serving on heterogeneous clouds as a Mixed-Integer Linear Programming (MILP) problem, co-optimizing GPU rental composition, parallelism strategies (TP/PP), and workload routing. They demonstrate ~25% throughput gains over SOTA systems (Helix, HexGen) using vLLM benchmarks, validating the approach with strong empirical ablations. For our **GPUSched** project, the key takeaway is their solver strategy: pre-generating valid configurations to linearize the problem and using a binary search wrapper on the makespan to avoid direct minimization overhead. We should adopt their heuristics for pruning the configuration space (e.g., restricting TP to intra-node) to improve our own solver times.

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**2025-11-12** | Microsoft, University of Illinois Urbana-Champaign, Georgia Institute of Technology, Indian Institute of Science | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Integer Linear Programming (ILP) for resource allocation, combined with ARIMA-based time-series forecasting and reactive heuristics for dynamic scaling and scheduling | *LLM role:* none

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interactive and non-interactive workloads. They validate this on real Microsoft O365 production traces (which they release), demonstrating a 25% reduction in GPU hours and $2.5M/month savings compared to reactive baselines. The primary value for us is the release of the production workload traces—allowing us to benchmark our 'GPUSched' formulations against real-world data rather than synthetic distributions—and their specific ILP formulation for unified capacity management, which directly competes with our internal OR models.

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**2025-03-05** | Carnegie Mellon University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Max-flow problem formulation on directed, weighted graphs with Mixed Integer Linear Programming (MILP) for joint model placement and per-request pipeline scheduling | *LLM role:* none

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow solution. Unlike standard static pipeline parallelism, it routes every request dynamically based on edge capacities, achieving up to 3.3x throughput gains over Swarm on mixed GPU clusters (L4/T4/A100). The results are rigorous, backed by both physical cluster experiments and high-fidelity simulations. The critical takeaway is the 'per-request pipeline' abstraction: decoupling request routing from static device assignment allows exact OR methods to maximize utilization of weaker hardware—a technique we should immediately evaluate for our GPUSched project.

### [Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via Reinforcement Learning](https://arxiv.org/abs/2507.10259)

**2025-09-16** | Shenzhen University of Advanced Technology, China Mobile Research Institute | M=6 P=9 I=6 **MUST-READ** *discuss*

*Method:* Proximal Policy Optimization (PPO) with Optimal Transport supervision | *LLM role:* none

> TORTA introduces a hierarchical scheduler for distributed LLM inference that uses a macro-level RL agent (PPO) supervised by an Optimal Transport (OT) baseline to manage inter-region allocation, followed by a micro-level greedy allocator. Results on simulated clusters (up to 50 servers) demonstrate a ~15% reduction in latency compared to reactive baselines (like SkyLB) specifically by optimizing for temporal smoothness and reducing switching costs. The key technical takeaway is the use of an exact OR solver (OT) as a dense supervision signal to train a faster RL policy, effectively combining the optimality of OR with the temporal foresight of RL. We should review our GPUSched formulations to ensure we aren't falling into the 'reactive' trap described here; if we are, this hybrid RL-OT architecture is a viable alternative.


### Front 0 (8 papers) — STABLE

**Density:** 0.46 | **Methods:** llm_fine_tuned, program_synthesis, llm_code_generation, reinforcement_learning, queueing_theory | **Problems:** llm_inference_scheduling, resource_allocation, gpu_scheduling, kv_cache_management, online_scheduling

*Unique methods:* adaptive_control, asymptotic_analysis, batch_scheduling, batching, binomial_thinning, bootstrapping, causal_inference, causal_intervention, competitive_ratio_analysis, data_augmentation, decode_prioritized_scheduling, demand_prediction, discrete_time_markov_chains, doobs_inequality, fair_queuing, fastertransformer, fcfs_scheduling, fluid_dynamics_approximation, fluid_limits, group_relative_policy_optimization, heuristic_filtering, hierarchical_clustering, instruction_tagging_system, instruction_tuning, kingmans_bound, lexicographical_optimization, lindley_recursion, llm_as_model_generator, llm_as_tagger, martingale_theory, mathematical_modeling, memory_centric_cost_modeling, memory_constrained_scheduling, mixed_batching, mlp, nested_wait_algorithm, non_preemptive_scheduling, online_algorithms, online_optimization, orca, outlier_detection, prefill_prioritized_scheduling, program_synthesis, sarathi_serve, self_improving_search, shortest_first, state_synchronization, statistical_testing, stochastic_processes, synthetic_data_generation, test_time_adaptation, test_time_reinforcement_learning, text_embedding, tf_idf, threshold_based_scheduling, union_bound, virtual_time_scheduling, vllm, wait_algorithm, work_conserving_scheduling
*Shared methods:* bin_packing, curriculum_learning, greedy_algorithm, integer_programming, linear_programming, llm_as_evaluator, llm_code_generation, llm_fine_tuned, llm_in_the_loop, load_balancing, lyapunov_function, online_scheduling, proximal_policy_optimization, queueing_theory, queuing_theory, reinforcement_learning, resource_allocation, scheduling, scheduling_algorithms, supervised_fine_tuning

This research front unifies efforts in applying Operations Research (OR) principles to two critical areas: optimizing Large Language Model (LLM) inference scheduling under stringent resource constraints, particularly KV cache memory, and enhancing LLMs' ability to perform automated optimization modeling. It features specific frameworks and algorithms such as Memory Constrained Shortest First (MC-SF), Staggered Batch Scheduling (SBS), and OR-Instruct for LLM fine-tuning, demonstrating a concerted push towards more efficient and intelligent AI systems.

Papers in this front introduce novel scheduling algorithms like MC-SF, achieving near-optimal latency (within 5% of hindsight optimal) by explicitly modeling KV cache growth [1]. Staggered Batch Scheduling (SBS) significantly reduces Time-to-First-Token (TTFT) by 30-40% and improves throughput by 15-20% on production workloads by optimizing batching for DP+EP architectures [4]. Throughput-optimal queueing-theoretic frameworks are established for LLM inference, proving the stability of work-conserving algorithms [3, 7]. For automated OR modeling, ORLM fine-tunes LLMs using OR-Instruct's synthetic data, outperforming GPT-4 by up to 38.4% on NL4OPT [2]. OR-R1 further refines this with Test-Time Group Relative Policy Optimization (TGRPO), achieving superior accuracy with 1/10th of the data [6]. Justitia introduces fair scheduling for LLM applications, reducing average job completion time by ~60% using memory-centric cost modeling [5].

This front is rapidly emerging, driven by the critical need for efficient LLM deployment and the growing interest in LLMs as tools for scientific discovery. The strong theoretical foundations in queueing theory and integer programming, coupled with practical, high-impact scheduling solutions, suggest a maturing trajectory for LLM inference optimization. Concurrently, the development of specialized LLMs for OR modeling, leveraging techniques like instruction tuning and reinforcement learning, indicates an emerging sub-field. The next papers will likely focus on integrating these scheduling policies into multi-GPU, distributed environments, and on developing more robust, generalizable LLM agents for complex, real-world OR problems.

**Papers:**

### [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115)

**2026-01-15** | Massachusetts Institute of Technology, Microsoft Research, HKUST | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Memory Constrained Shortest First (MC-SF) online batching and scheduling algorithm | *LLM role:* none

> This paper formulates LLM inference scheduling as an Integer Program (IP) that explicitly models the linear memory growth of KV caches, and proposes a 'Memory Constrained Shortest First' (MC-SF) algorithm. The results are rigorous, showing MC-SF achieves near-optimal performance (within 5% of hindsight optimal) on synthetic data and significantly outperforms standard FCFS/threshold heuristics on real traces. The critical takeaway is the 'future feasibility check' (Eq. 5), which validates that a batch will *remain* within memory limits throughout the generation process based on predicted output lengths—a necessary deviation from standard static-size scheduling. This is foundational reading for our GPUSched project, providing both the exact IP baseline we need and a strong heuristic to benchmark against.

### [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://arxiv.org/abs/2405.17743)

**2025-04-04** | Columbia University, Duke University, Shanghai Jiao Tong University, The Chinese University of Hong Kong, Shenzhen, Shenzhen Research Institute of Big Data, Shanghai University of Finance and Economics, Cardinal Operations | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Instruction tuning of open-source LLMs using semi-automated synthetic data generated by OR-Instruct framework | *LLM role:* data_synthesis, model_generator, code_writer

> The authors propose OR-Instruct, a framework that uses GPT-4 to synthesize over 32k optimization modeling pairs (natural language to COPT code) to fine-tune 7B-scale models (ORLM). They demonstrate that these fine-tuned models outperform GPT-4 on their new 'IndustryOR' benchmark, a result that appears robust given the specialized nature of the task. The most valuable takeaway is their specific data augmentation strategy—iteratively altering constraints and injecting specific modeling techniques (e.g., Big M)—which provides a concrete recipe we can steal to generate diverse instances for our OR-Bench project. While the methodology is standard instruction tuning, the resulting artifacts (benchmark and model) establish a new baseline for automated OR modeling that we cannot ignore.

### [Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents](https://arxiv.org/abs/2504.07347)

**2025-04-24** | Cornell University, Columbia University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Queueing-theoretic framework with discrete-time Markov chains and fluid limits for analyzing work-conserving scheduling algorithms | *LLM role:* none

> Li et al. formulate a batch queueing model for LLM inference, proving that 'work-conserving' algorithms (like Sarathi-Serve) which mix prefill and decode tokens are throughput-optimal, whereas separated strategies (vanilla vLLM, FasterTransformer) are theoretically unstable. The results are rigorous, combining fluid limit proofs with empirical validation on A100s showing queue blow-ups in non-optimal schedulers. The key takeaway is the precise definition of stability for token-level batching and the counter-intuitive finding that these locally optimal policies can fail in multi-agent networks due to cyclic resource dependencies. This is foundational reading for our GPUSched project and directly informs how we should model resource allocation for our multi-agent optimization systems.

### [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](https://arxiv.org/abs/2512.16134)

**2025-12-18** | Baidu Inc. | M=6 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Staggered Batch Scheduling (SBS) with Throughput-Adaptive Interval Control, Multi-tier State Synchronization, Prioritized Batch Allocation Algorithm (PBAA) for Prefill, and IQR-Aware Lexicographical Decode Scheduling for Decode | *LLM role:* none

> Tian et al. introduce Staggered Batch Scheduling (SBS) for DP+EP architectures, enforcing a buffering window to enable global bin-packing rather than immediate dispatch, which they prove causes Head-of-Line blocking in non-preemptive prefill phases. Tested on a production H800 cluster serving DeepSeek-V3, they demonstrate a 30-40% reduction in TTFT and a ~20% throughput increase backed by clear utilization metrics. The most valuable takeaway for our GPUSched project is their 'IQR-aware lexicographical' scheduling heuristic for the Decode phase, which robustly balances batch size against KV-cache memory variance—a constraint logic we should immediately adopt. This work validates that discrete batching is superior to continuous dispatch for MoE models, necessitating an update to our queuing theory models.

### [Justitia: Fair and Efficient Scheduling for LLM Applications](https://arxiv.org/abs/2510.17015)

**2025-10-19** | Shanghai Jiao Tong University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Virtual-time based fair queuing with memory-centric cost modeling and MLP-based demand prediction | *LLM role:* none

> Justitia introduces a scheduler for LLM agents that prioritizes applications based on their 'virtual finish time' (derived from a theoretical fair-sharing model) but executes them with full resource saturation to minimize completion time. The authors demonstrate a ~60% reduction in average job completion time compared to state-of-the-art fair schedulers (VTC) on vLLM, backed by rigorous experiments and theoretical delay bounds. The key takeaway is the 'KV token-time' cost metric (pd + d^2/2) which accurately captures memory bottlenecks in auto-regressive generation, and the insight that 'long-term fairness' allows for short-term resource saturation. This is immediately actionable for your GPUSched project and relevant for optimizing the serving infrastructure of AlgoEvo.

### [OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning](https://arxiv.org/abs/2511.09092)

**2025-11-12** | The Hong Kong University of Science and Technology, Arizona State University, University of North Carolina at Chapel Hill | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Supervised Fine-tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) with a composite reward function | *LLM role:* code_writer, heuristic_generator, evaluator

> OR-R1 introduces a data-efficient framework that fine-tunes Qwen3-8B using Supervised Fine-Tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) on unlabeled data. The results are empirically strong: it outperforms ORLM and LLMOPT while using only 1/10th of the synthetic training data, specifically narrowing the consistency gap between Pass@1 and Pass@8. The key takeaway for us is the effectiveness of GRPO (normalizing rewards within a sampled group to estimate baselines) combined with majority-voting rewards; this eliminates the need for a separate critic model while significantly improving code generation consistency. We should immediately evaluate GRPO as a lightweight alternative to PPO for the 'RL-infused' components of our evolutionary search methods.

### [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

**2026-01-05** | Massachusetts Institute of Technology, Peking University, Alibaba Group | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Fluid dynamics approximation and threshold-based online scheduling (WAIT and Nested WAIT algorithms) | *LLM role:* none

> This paper formulates LLM inference as a multi-stage stochastic scheduling problem, introducing 'Nested WAIT'—a threshold-based algorithm that handles unknown output lengths by letting prompts classify themselves as they survive into deeper decode segments. Unlike heuristic baselines (vLLM, Sarathi), they provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, validated on A100 simulations. The key takeaway is the 'nested segment' mechanism: instead of predicting job size, structure the queue so short jobs exit early and long jobs naturally migrate to lower-priority/protected tiers, effectively decoupling the memory risk. We should immediately evaluate this threshold logic for our GPUSched formulations, as it likely outperforms our current predictive or FCFS approaches for handling KV cache growth.

### [Beyond IID: Optimizing Instruction Learning from the Perspective of Instruction Interaction and Dependency](https://arxiv.org/abs/2409.07045)

**2024-09-11** | Beijing Academy of Artificial Intelligence | M=6 P=5 I=7 *discuss*

*Method:* Causal Intervention based Instruction Correlation Analysis and Ability Taxonomy Induction; Effect Equivalence-based Linear Programming for Category Proportion Optimization (EE-CPO); Dependency Taxonomy Guided Curriculum Supervised Fine-Tuning (DT-CSFT) | *LLM role:* tagger, base_model_for_analysis_and_finetuning

> The authors propose optimizing SFT data mixtures using Linear Programming (EE-CPO) by modeling the 'interaction' (synergy/antagonism) between instruction categories, rather than treating them as IID. They empirically derive a dependency taxonomy showing Math and Code are fundamental 'root' capabilities required before learning complex tasks, validating this via curriculum learning experiments that beat DEITA. The results are solid (+1.73 AlpacaEval over DEITA), though the cost of deriving the interaction matrix (training N models) is high. **Takeaway:** The 'Effect Equivalence Coefficient' matrix combined with an LP solver is a rigorous OR formulation for resource/data allocation that we should steal to optimize heuristic populations in our evolutionary search frameworks.


### Front 1 (8 papers) — GROWING

**Density:** 0.36 | **Methods:** llm_in_the_loop, resource_allocation, llm_as_heuristic, process_reward_model, rebase | **Problems:** mathematical_reasoning, llm_serving_optimization, llm_inference_optimization, resource_allocation, llm_test_time_scaling

*Unique methods:* adaptive_index_update, adaptive_sampling, advantage_modulation, all_to_all_collectives, analytical_modeling, approximate_nearest_neighbor_search, bayes_factor, bayesian_modeling, bayesian_optimization, bernoulli_variance_proxy, bert_embeddings, best_of_n, beta_distribution_modeling, clustering, cuda_graph, direction_oriented_resource_allocation, dirichlet_process_prior, distributed_inference, diverse_verifier_tree_search, dora, dynamic_dispatching, dynamic_rollout_allocation, embedding_model, entropy_dynamics_control, expert_parallelism, exponential_smoothing, fast_scanning, fluid_model_analysis, gradient_compensation, gradient_scheduling, gradient_variance_minimization, grouped_gemm, grpo, hierarchical_agglomerative_clustering, inference_optimization, inverted_file_index, kv_cache_optimization, latency_bounded_partitioning, llm_as_answer_generator, llm_ensemble, llm_inference_optimization, majority_voting, max_margin_optimization, memory_optimization, mixture_of_experts, monte_carlo_methods, nvshmem, online_distillation, performance_estimation, pipelining, policy_optimization, predictive_scheduling, process_reward_model, product_quantization, real_time_optimization, rebase, reinforcement_learning_with_verifiable_rewards, resource_partitioning, retrieval_augmented_generation_serving, reward_balanced_search, rl_ppo, semantic_similarity, soft_clustering, straggler_mitigation, supervised_learning, system_design, temperature_sampling, tree_search, triton, update_magnitude_stabilization, vector_similarity_search
*Shared methods:* beam_search, convex_optimization, curriculum_learning, greedy_algorithm, integer_linear_programming, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_in_the_loop, load_balancing, mixed_integer_linear_programming, queuing_theory, resource_allocation, scheduling, speculative_decoding

This research front focuses on applying advanced Operations Research (OR) principles, including adaptive resource allocation, intelligent search strategies, and policy optimization, to significantly enhance the efficiency and accuracy of Large Language Model (LLM) inference and reasoning. Key frameworks like DORA, ETS, GRPO, and DynaMO are central, addressing challenges in mathematical reasoning, RAG serving, and Mixture-of-Experts (MoE) inference. The unifying theme is the strategic use of OR to manage computational resources and guide LLM behavior for optimal performance.

Significant contributions include DORA's embedding-based soft clustering, which achieved state-of-the-art accuracy on MATH500 with 3.5x fewer FLOPs by optimizing compute budget. ETS formulated tree search pruning as an ILP, leading to a 1.8x KV cache reduction and 1.4x throughput increase on MATH500. Ao et al. demonstrated a 95.3% recovery rate for OR model debugging by using solver diagnostics as dense rewards for GRPO-trained LLMs. DynaMO further improved GRPO with variance-minimizing dynamic rollout allocation, yielding an 11.8% Pass@1 increase on Qwen-7B. VectorLiteRAG achieved 1.5x throughput gains for RAG serving, while PROBE delivered a 1.3x speedup for MoE inference via lookahead pipelining. The Best-of-Infinity paper introduced Bayesian adaptive stopping to reduce test-time compute by 2-5x.

This front is rapidly growing, showcasing a strong synergy between OR and AI. The trajectory indicates a shift towards more sophisticated, theoretically grounded OR techniques for fine-grained control and optimization of LLM systems. Future work will likely extend these adaptive allocation and search strategies to more complex, real-world LLM deployments, integrating multi-objective optimization and robust control for dynamic, stochastic environments.

**Papers:**

### [Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling](https://arxiv.org/abs/2506.15707)

**2025-10-20** | Beijing Institute of Technology, Xiaohongshu Inc | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Direction-Oriented Resource Allocation (DORA) | *LLM role:* reasoning_path_generator

> Wang et al. introduce Direction-Oriented Resource Allocation (DORA), which uses embedding-based soft clustering to group semantically similar reasoning paths and allocates compute budget to distinct 'directions' rather than individual solutions. They prove solution-level allocation (like REBASE) is suboptimal when paths are correlated and show DORA achieves state-of-the-art accuracy on MATH500 with 3.5x fewer FLOPs. **Key Takeaway:** We can immediately steal the 'semantic uniqueness reweighting' mechanism for AlgoEvo. By clustering generated heuristics via embeddings before expensive evaluation, we can drastically improve sample efficiency and stop wasting compute on minor variations of the same code.

### [VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG](https://arxiv.org/abs/2504.08930)

**2026-01-19** | Georgia Institute of Technology | M=5 P=7 I=6 *discuss*

*Method:* Analytical performance modeling and latency-bounded partitioning algorithm for hybrid CPU-GPU vector index, combined with a distributed runtime pipeline featuring query- and shard-aware routing and dynamic dispatcher. | *LLM role:* target_of_optimization

> VectorLiteRAG optimizes RAG serving throughput by dynamically partitioning vector indices between CPU and GPU memory based on access skew and latency SLOs. The results are credible, showing up to 1.5x throughput gains on H100/L40S setups by balancing retrieval speed against LLM KV-cache capacity. The most stealable insight is their use of a Beta distribution to analytically model the *minimum* hit rate within a batch (the bottleneck) to predict tail latency without full simulation—a technique we could adapt for stochastic constraints in our serving formulations. It solves a resource allocation problem we care about, though via systems engineering rather than the rigorous OR methods we prefer.

### [Solver-in-the-Loop: MDP-Based Benchmarks for Self-Correction and Behavioral Rationality in Operations Research](https://arxiv.org/abs/2601.21008)

**2026-02-08** | Massachusetts Institute of Technology, Alibaba Group | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Domain-specific Group Relative Policy Optimization (GRPO) with composite reward and three-stage curriculum learning | *LLM role:* agent_for_debugging_and_decision_making

> Ao et al. introduce a framework for iterative OR model debugging that trains an 8B model using Group Relative Policy Optimization (GRPO) and a Process Reward Model (PRM) to outperform GPT-4o-mini. They utilize Gurobi's Irreducible Infeasible Subsystem (IIS) not just as text feedback, but as a dense reward signal (IIS size reduction) for the PRM, achieving a 95.3% recovery rate versus 86.2% for frontier APIs. **Key Takeaway:** We should steal their PRM construction method—specifically using solver diagnostics (like IIS reduction or compiler error counts) as dense step-level rewards—and their 'faithfulness penalty' to prevent overfitting in our evolutionary search. This is a direct validation of RLVR (Reinforcement Learning with Verifiable Rewards) for OR, proving it superior to large-scale prompting.

### [ETS: Efficient Tree Search for Inference-Time Scaling](https://arxiv.org/abs/2502.13575)

**2025-06-11** | University of California, Berkeley, ICSI, LBNL | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Efficient Tree Search (ETS) using a linear programming cost model with KV cache sharing penalty and semantic coverage term | *LLM role:* candidate_generator, process_reward_model, search_guidance

> ETS formulates the tree search pruning step as a lightweight Integer Linear Program (ILP) that maximizes the reward of retained nodes while penalizing total KV cache size and enforcing semantic diversity via clustering. Unlike standard beam search or REBASE, it explicitly optimizes the trade-off between memory consumption (KV sharing) and exploration coverage. The authors demonstrate a 1.8x reduction in KV cache size and 1.4x throughput increase on MATH500 with minimal accuracy loss. We should steal the 'ILP-in-the-loop' mechanism for population management in our evolutionary search frameworks to optimize hardware utilization dynamically.

### [How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization](https://arxiv.org/abs/2602.19208)

**2026-02-22** | Meituan, Tsinghua University, Fudan University, Peking University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dual-pronged optimization framework (DynaMO) combining variance-minimizing dynamic rollout allocation and gradient-aware advantage modulation for GRPO-based policy optimization | *LLM role:* policy_model

> DynaMO introduces a dual-pronged optimization for RLVR: a dynamic rollout allocation strategy that prioritizes problems with high gradient variance (proxied by Bernoulli variance of success/failure), and a gradient modulation technique to stabilize updates. The results are strong (+11.8% Pass@1 over GRPO on Qwen-7B) and backed by clear ablations. **The critical takeaway for us is the allocation logic:** we should immediately replace uniform sampling in AlgoEvo with variance-based allocation ($n_i \propto \sqrt{p(1-p)}$). This ensures compute is spent on instances/components that are currently 'learnable' (high variance) rather than wasted on the trivial or impossible, directly optimizing our search budget.

### [GoodSpeed: Optimizing Fair Goodput with Adaptive Speculative Decoding in Distributed Edge Inference](https://arxiv.org/abs/2512.09963)

**2025-12-14** | The University of Sydney, Kyung Hee University | M=5 P=7 I=6 *discuss*

*Method:* Gradient-based scheduling algorithm maximizing logarithmic utility for proportional fairness with adaptive speculative decoding | *LLM role:* heuristic_generator, evaluator

> GoodSpeed uses gradient-based scheduling to dynamically allocate token generation budgets across distributed draft servers, maximizing a logarithmic utility function to balance throughput and fairness. The authors provide rigorous fluid sample path analysis to prove convergence, backed by experiments on H100/L4 clusters, although the baselines (fixed/random allocation) are relatively weak. The most useful takeaway is the mechanism of using exponentially smoothed acceptance rate estimates to drive real-time control in a stochastic system—a robust pattern we should adopt for our own stochastic resource allocation and RobustMAS projects.

### [Best-of-$\infty$ -- Asymptotic Performance of Test-Time Compute](https://arxiv.org/abs/2509.21091)

**2025-09-25** | Mohamed bin Zayed University of Artificial Intelligence, New York University, RIKEN AIP, Institute of Science Tokyo, NEC Corporation | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Adaptive sampling for majority voting using Bayesian modeling (Dirichlet process prior and Bayes factor) to determine stopping criteria, combined with optimally weighted LLM ensembles formulated as a Mixed-Integer Linear Program (MILP) with a max-margin solution. | *LLM role:* answer_generator

> This paper introduces a Bayesian adaptive stopping criterion (using Dirichlet process priors and Bayes factors) for majority voting, reducing test-time compute by 2-5x while maintaining asymptotic 'Best-of-Infinity' accuracy. They further demonstrate that optimizing weights for an ensemble of LLMs can be formulated as a Mixed-Integer Linear Program (MILP) by treating the decision boundaries as polytopes. **What we learned:** The Bayesian stopping logic is immediately transferable to AlgoEvo to reduce the cost of fitness evaluations—we can stop evaluating candidate solutions early if their performance distribution is statistically distinct. The MILP approach for ensembles also offers a concrete formulation we could adapt for our GPU scheduling and model serving optimization work.

### [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509)

**2026-02-03** | Kling Infra, Kuaishou Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Continuous Lookahead Pipelining with Gate-Initialized Lookahead Predictor, Hardware-Aware Balance Planning, and Phase-Locked Co-Scheduling | *LLM role:* none

> PROBE optimizes MoE inference by using a distilled MLP to predict next-layer expert activation, enabling proactive load balancing and weight prefetching hidden behind the current layer's computation. The results are strong (1.3x speedup on 235B models) and demonstrate that control plane overheads can be fully masked. The critical takeaway for our `GPUSched` project is the **Lookahead Pipelining** architecture: it carves out a deterministic execution window where we could inject our own specialized solvers (e.g., fast ALNS or IP formulations) to outperform their basic greedy resource allocator. This transforms the stochastic serving problem into a short-horizon deterministic routing problem we are well-equipped to solve.


### Front 8 (6 papers) — STABLE

**Density:** 0.53 | **Methods:** continuous_batching, mixed_integer_programming, heuristic_search, early_exit_llms, dynamic_rebatching | **Problems:** llm_serving_optimization, resource_allocation, llm_inference_scheduling, gpu_scheduling, scheduling

*Unique methods:* active_request_capping, adaptive_thresholding, attention_kernels, decode_limit, decode_router, disaggregated_expert_parallelism, dynamic_offset_adjustment, dynamic_rebatching, dynamic_resource_allocation, early_exit_llms, fine_grained_scheduling, first_come_first_serve, fluid_approximation, gate_and_route_policy, gemm, gemv, graph_partitioning, gurobi, heuristic_algorithm_design, heuristic_search, hybrid_optimization, kkt_conditions, kv_caching, lagrangian_heuristic, linear_performance_models, llm_inference_serving, llm_serving_systems, makespan_minimization, many_server_queueing, mathematical_analysis, matrix_multiplication, maximum_likelihood_estimation, optimal_gemm_tiling, optimization_problem_formulation, ordinary_least_squares, ping_pong_pipeline, preemptive_scheduling, prefill_admission_gate, queueing_network, real_time_tbt_deadline_tracking, resource_aware_dynamic_scheduler, scheduling_strategies, shortest_prefill_first_ordering, sla_aware_scheduling, slo_aware_llm_inference_scheduler, state_copying, stochastic_control, task_scheduling, token_budgeting, virtual_memory_management
*Shared methods:* bi_level_optimization, bin_packing, continuous_batching, convex_optimization, linear_programming, load_balancing, lyapunov_function, mixed_integer_programming, online_scheduling, performance_modeling, queueing_theory, scheduling

This research front focuses on applying advanced Operations Research (OR) and scheduling techniques to optimize large language model (LLM) inference serving. The core theme revolves around addressing critical challenges such as efficient handling of early-exit LLMs, managing prefill-decode contention in large-scale deployments, and optimizing distributed Mixture-of-Experts (MoE) inference. Key methodologies include dynamic rebatching, queueing-theoretic optimal scheduling, stochastic control with fluid approximations, Mixed-Integer Programming (MIP) for parallelism, and hybrid offline-online scheduling strategies.

Specific contributions include DREX's 2-12% throughput gain for Llama-EE-70B via dynamic rebatching, and Bari et al.'s RAD/SLAI schedulers achieving a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve. Lin et al. propose an asymptotically optimal 'Gate-and-Route' policy for prefill-decode contention, while She et al. demonstrate MIP's ability to reduce pipeline bubbles by 50% for DeepSeek V3's operator-level parallelism. Pang et al. report a ~9% utilization increase over vLLM with a hybrid MIP-based scheduler, and FinDEP achieves up to 1.61x throughput improvement for MoE inference by fine-grained task scheduling.

This front is rapidly maturing, moving from theoretical foundations to practical, system-level implementations. The trajectory indicates a strong push towards integrating diverse optimization techniques and addressing more complex real-world constraints. Future work will likely focus on combining early-exit and MoE optimizations, developing robust solutions for heterogeneous hardware and highly dynamic workloads, and providing rigorous tail-latency guarantees through advanced stochastic modeling.

**Papers:**

### [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](https://arxiv.org/abs/2512.15705)

**2025-12-17** | Microsoft Research, University of Pennsylvania | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Dynamic Rebatching with copy-free rebatching buffer and SLA-aware scheduler | *LLM role:* inference_target

> DREX introduces a system for 'Early-Exit' LLMs that dynamically splits and regroups batches at intermediate layers, using a cost-benefit heuristic (Adaptive Rebatching Threshold) to decide when rebatching is profitable versus forcing execution. Results are solid (2-12% throughput gain on A100s) and backed by real system measurements, not just simulations. The key takeaway for us is the analytical model for rebatching overhead (Eq. 6)—we can lift this constraint directly into our integer programming formulations for the GPUSched project to accurately model the trade-off between batch fragmentation and compute savings. Essential reading only for the serving optimization sub-team; irrelevant for the core evolutionary search group.

### [Optimal Scheduling Algorithms for LLM Inference: Theory and Practice](https://arxiv.org/abs/2508.01002)

**2025-12-01** | The University of Texas at Austin | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* Resource-Aware Dynamic (RAD) scheduler for throughput optimality based on optimal GeMM tiling and dynamic prefill/decode resource allocation; SLO-Aware LLM Inference (SLAI) scheduler for practical SLOs using real-time TBT deadline tracking, SPF prefill ordering, and dynamic offset adjustment based on GPU memory utilization. | *LLM role:* none

> Bari et al. develop a queueing-theoretic framework for LLM inference that proves throughput optimality requires satisfying two conditions: optimal GeMM tiling (batch sizes matching hardware tensor core dimensions) and dynamic resource allocation between prefill/decode phases. They propose RAD (theoretical) and SLAI (practical), where SLAI uses a 'last schedulable time' heuristic to delay decode iterations for non-critical requests, thereby freeing up compute for prefill to reduce TTFT. Results are strong, showing a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve on Mistral-7B. For our GPUSched project, the key takeaway is the explicit coupling of batch sizes to LCM(tile_dims) for theoretical optimality and the dynamic slack-based scheduling logic for heterogeneous SLOs.

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.

### [Automatic Operator-level Parallelism Planning for Distributed Deep Learning -- A Mixed-Integer Programming Approach](https://arxiv.org/abs/2503.09357)

**2025-03-12** | Huawei | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed-Integer Programming (MIP) formulation with a bi-level solution framework including a heuristic operation merging step | *LLM role:* none

> She et al. formulate distributed LLM training/inference as a Flexible Distributed Job Shop Scheduling Problem (FDJSSP) solved via Mixed-Integer Programming (MIP) combined with a heuristic graph coarsening step. They demonstrate that this automated approach not only reproduces DeepSeek V3's expert-designed "DualPipe" strategy but, when allowed to search longer, discovers a schedule with 50% fewer pipeline bubbles. The primary takeaway is the effectiveness of the bi-level optimization framework (greedy merging + MIP) to handle the scale of operator-level graphs, proving that formal OR methods can outperform manual system design for LLM infrastructure. This is a mandatory read for our GPUSched project, offering a concrete formulation for operator-level constraints we can directly adapt.

### [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)

**2025-02-14** | Noah’s Ark Lab, Huawei, Tsinghua University | M=6 P=10 I=7 **MUST-READ** *discuss*

*Method:* Hybrid offline-online method combining Minimizing Makespan Bin Packing (offline) with sorting, online preemption, and a Lagrangian-based heuristic (online) | *LLM role:* none

> Pang et al. formulate LLM inference scheduling as a Mixed-Integer Programming (MIP) model, solving it via a hybrid approach: offline bin-packing for request assignment and an online Lagrangian heuristic for prefill-decode preemption. They report a ~9% utilization increase (80.2% to 89.1%) over a vLLM-style baseline on LLaMA-65B, though the evaluation is limited to a single 8-GPU node and assumes deterministic output lengths for the offline component. The most actionable takeaway is their derivation of a simple cost-comparison threshold (prefill cost vs. decode wait cost) to dynamically inject prefill tasks into decoding streams. This provides a concrete, low-overhead heuristic baseline for our GPUSched work.

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**2025-12-25** | The Hong Kong University of Science and Technology, Harbin Institute of Technology, Hong Kong Baptist University | M=6 P=7 I=5 *discuss*

*Method:* Fine-grained task scheduling algorithm for disaggregated expert parallelism (DEP) with maximal task overlap, guided by linear performance models and analytical properties (monotonicity, convexity) | *LLM role:* none

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize overlap. The authors achieve 1.02x-1.61x speedups on H20/A6000 clusters compared to PPPipe, backed by solid empirical data. The key takeaway for our 'GPUSched' work is their methodology: deriving analytical properties (monotonicity and convexity) of the scheduling objective to reduce a complex search space into an $O(1)$ online solver, rather than relying on heavy solvers or RL. This confirms that simple linear performance models ($\alpha + \beta x$) are sufficient for accurate online resource allocation in LLM serving.


### Front 23 (3 papers) — EMERGING

**Density:** 1.00 | **Methods:** resource_allocation, distributed_training, sequence_parallelism, integer_linear_programming, load_balancing | **Problems:** communication_optimization, llm_training_efficiency, long_context_llm_training, llm_serving_optimization, distributed_system_optimization

*Unique methods:* activation_recomputation, activation_swapping, adaptive_parallelism, context_parallelism, cost_modeling, cpu_offloading, cuda_streams, distributed_training, expert_placement, flash_attention, gpu_memory_optimization, gradient_accumulation, memory_defragmentation, memory_management, sequence_packing, sequence_parallelism, system_level_optimization, system_optimization, tensor_management, token_routing_profiling
*Shared methods:* data_parallelism, distributed_systems, dynamic_programming, integer_linear_programming, linear_programming, load_balancing, mixed_integer_linear_programming, mixed_integer_programming, pipeline_parallelism, resource_allocation, tensor_parallelism

This research front unifies papers applying Integer Linear Programming (ILP) and Mixed Integer Programming (MIP) to optimize various aspects of large language model (LLM) distributed training and serving. The core theme is leveraging formal Operations Research (OR) methods to achieve significant performance gains in complex LLM systems, specifically within frameworks like Megatron-LM and DeepSpeed. These papers demonstrate that static and bi-level optimization strategies can effectively address challenges such as MoE expert placement, fine-grained memory management for long contexts, and adaptive sequence parallelism.

Key contributions include MoETuner, which uses ILP for balanced MoE expert placement and token routing, achieving up to 17.5% speedup on multi-node H200 clusters. MEMO introduces a bi-level MIP approach for fine-grained activation memory management, enabling 7B LLM training with 1M context on 8 GPUs and yielding a 1.97x MFU improvement. FlexSP employs MILP and dynamic programming for adaptive sequence parallelism, resulting in up to 1.98x speedup on A100 clusters by optimizing for varied-length sequences. All papers validate their approaches with concrete system measurements and benchmarks, proving the practical efficacy of OR solvers in real-world LLM infrastructure.

This front is clearly emerging, with all papers published in 2025, indicating a nascent but impactful trend of applying OR to LLM system optimization. The trajectory suggests a rapid expansion of these techniques to more complex, dynamic, and integrated optimization problems within distributed AI systems. The next papers will likely focus on integrating these specific OR formulations across different parallelism strategies and developing more scalable, potentially hybrid, optimization approaches.

**Papers:**

### [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643)

**2025-02-10** | Georgia Institute of Technology | M=8 P=9 I=7 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) for expert clustering and cluster-to-GPU assignment | *LLM role:* none

> Go et al. formulate the MoE expert placement problem as a two-stage Integer Linear Program (ILP) to balance token load and minimize communication tail latency, exploiting stable token routing dependencies across layers. They demonstrate real-world speedups of 17.5% on multi-node H200 clusters running Mixtral-8x7B, validating the approach with concrete systems measurements rather than just simulation. The key takeaway is the effectiveness of a min-max ILP objective for reducing tail latency in distributed inference, proving that static optimization based on profiling is sufficient for significant gains. This directly supports our 'OR for AI systems' track and provides a strong baseline formulation for our GPU scheduling work.

### [MEMO: Fine-grained Tensor Management For Ultra-long Context LLM Training](https://arxiv.org/abs/2407.12117)

**2025-01-15** | Peking University, Tencent Inc. | M=8 P=5 I=7 *discuss*

*Method:* Fine-grained activation memory management combining token-wise recomputation and swapping with bi-level Mixed Integer Programming (MIP) for memory planning | *LLM role:* none

> Memo enables training 7B LLMs with 1M context on 8 GPUs by combining token-wise activation swapping with a bi-level Mixed Integer Programming (MIP) approach to eliminate memory fragmentation. The results are strong (52% MFU vs ~30% for DeepSpeed) and demonstrate that static memory planning via OR solvers outperforms dynamic allocators for repetitive Transformer workloads. The key takeaway is the bi-level MIP strategy—solving the allocation for one layer and broadcasting it—which makes the NP-hard memory planning tractable. We should adapt this MIP formulation for our own GPU scheduling and inference resource allocation (GPUSched) projects.

### [FlexSP: Accelerating Large Language Model Training via Flexible Sequence Parallelism](https://arxiv.org/abs/2412.01523)

**2025-02-11** | Peking University, ByteDance Inc., Beihang University | M=8 P=6 I=7 *discuss*

*Method:* Heterogeneity-adaptive sequence parallelism using MILP and dynamic programming for optimal strategy selection | *LLM role:* none

> FlexSP optimizes distributed LLM training by dynamically assigning varied-length sequences to heterogeneous Sequence Parallelism (SP) groups using a Mixed-Integer Linear Programming (MILP) solver in the loop. The results are solid, showing up to 1.98x speedup on A100 clusters by mitigating communication bottlenecks for short sequences while preventing OOM for long ones. **Key Takeaway:** The authors use Dynamic Programming to 'bucket' similar sequences, drastically reducing the variable count for the MILP solver; this specific technique—reducing problem granularity to make exact solvers feasible in real-time systems—is directly applicable to our 'GPUSched' and inference resource allocation work. While we focus on evolution, this is a definitive reference for our 'OR for AI Systems' track, proving that formal optimization can beat heuristics in dynamic GPU scheduling.


### Front 36 (3 papers) — STABLE

**Density:** 1.00 | **Methods:** game_theory, llm_as_evaluator, multi_objective_optimization, convex_optimization, gradient_descent | **Problems:** llm_safety_alignment, safety_helpfulness_tradeoff, multiple_choice_qa, llm_alignment, instruction_following

*Unique methods:* adaptation_safety, best_of_k_sampling, black_box_optimization, blockwise_decoding, controlled_decoding, equilibrium_search, game_theory, gradient_aggregation, gradient_descent, inference_time_alignment, llm_alignment, llm_fine_tuning, maximin_optimization, multi_objective_optimization, noon_ppo, optimization_penalty_function, ppo, reward_modeling, rlhf, value_function_learning, zero_sum_game
*Shared methods:* convex_optimization, linear_programming, llm_as_evaluator, llm_in_the_loop, robust_optimization, supervised_fine_tuning

This research front unifies approaches leveraging game theory and convex optimization to address the complex challenge of multi-objective alignment in Large Language Models (LLMs). Key frameworks include the 'Safety Game' for black-box agentic LLMs, 'Robust Multi-Objective Decoding (RMOD)' for inference-time control, and 'Pareto Multi-Objective Alignment (PAMA)' for efficient Reinforcement Learning from Human Feedback (RLHF).

Key contributions include the 'Safety Game' which formulates LLM response selection as a zero-sum game solved by an LP solver, achieving up to two-fold accuracy improvement on SafetyBench by managing safety-helpfulness trade-offs. 'Robust Multi-Objective Decoding (RMOD)' employs a maximin game between adversarial reward weights and sampling policy, reducing to convex optimization, yielding +1.2% WCWR on LLM-as-Judge benchmarks. The 'Pareto Multi-Objective Alignment (PAMA)' algorithm transforms multi-objective RLHF into a convex optimization with a closed-form solution, outperforming baselines like MORLHF and MGDA-UB with stable convergence and significantly higher harmlessness scores (e.g., 0.4406 vs -0.1313).

This front is actively emerging, demonstrating novel applications of robust optimization and game theory to critical LLM alignment challenges. The trajectory suggests continued development in extending these principled optimization frameworks to handle more complex, dynamic, and sequential interaction settings, moving beyond static multiple-choice or block-based decoding to real-time, adaptive control of LLM behavior.

**Papers:**

### [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](https://arxiv.org/abs/2510.09330)

**2025-12-02** | University of Warwick | M=7 P=4 I=7 *discuss*

*Method:* Two-player zero-sum game formulation solved by a linear programming (LP) solver at inference time to compute minimax equilibrium strategies, using binary probes for helpfulness and safety scores, with a sigmoid penalty for risk. | *LLM role:* agent_response_selection, evaluator

> The authors formulate LLM response selection as a zero-sum game, solving a small Linear Program (LP) at inference time to mix candidate answers such that the expected risk never exceeds a 'safe fallback' baseline. Results are statistically significant, showing ~15% accuracy gains on SafetyBench by effectively managing the trade-off between helpfulness and safety probes. The key takeaway is the 'Adaptation Safety' constraint formulation: using an LP to guarantee that a stochastic policy is no worse than a heuristic baseline is a powerful, lightweight control mechanism we could adapt for selecting evolved algorithms or managing constraints in multi-agent optimization.

### [Robust Multi-Objective Controlled Decoding of Large Language Models](https://arxiv.org/abs/2503.08796)

**2025-03-11** | University College London, University of Basel, Ulsan National Institute of Science and Technology | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Maximin two-player game between adversarially computed reward weights and sampling policy, solvable through Nash equilibrium, reduced to convex optimization, with blockwise best-of-K sampling | *LLM role:* controlled_decoding_target

> RMOD formulates multi-objective decoding as a zero-sum game between a policy and adversarial weights, solving a convex optimization problem at each decoding step to maximize the worst-case value estimate (essentially a Process Reward Model). The results are empirically strong, outperforming MO-DPO and scalarized baselines on alignment benchmarks by dynamically preventing any single objective from collapsing. **Key Takeaway:** The efficient inference-time weight optimization algorithm (Eq. 10) is a 'stealable' mechanism for **AlgoEvo** and **RobustMAS**. We should implement this dynamic adversarial weighting to balance conflicting code metrics (e.g., runtime vs. solution quality) during evolutionary search, replacing our current static scalarization methods.

### [Pareto Multi-Objective Alignment for Language Models](https://arxiv.org/abs/2508.07768)

**2025-08-11** | Ruhr University Bochum | M=7 P=5 I=6 *discuss*

*Method:* PAMA (PAreto Multi-Objective Alignment) algorithm, which transforms multi-objective RLHF into a convex optimization problem with a closed-form solution, combined with Noon PPO. | *LLM role:* subject_of_optimization

> PAMA introduces a computationally efficient algorithm for multi-objective alignment by reformulating the expensive gradient-norm minimization of MGDA into a convex optimization problem with a closed-form solution, reducing complexity from O(n^2d) to O(n). Empirical results on LLaMA-2-7B are robust, showing stable convergence on conflicting objectives (e.g., harmlessness vs. length) where baselines like MGDA-UB oscillate or fail. The single most useful takeaway is the analytical derivation for optimal objective weighting (Theorem 1) and the 'Noon PPO' heuristic (clipping negative advantages); we could port this logic to our multi-objective process reward models in AlgoEvo to balance search signals efficiently. While the NLP experiments are trivial, the gradient balancing mechanism is directly applicable to our multi-objective RL controllers.


### Front 9 (2 papers) — EMERGING

**Density:** 1.00 | **Methods:** rate_distortion_theory, linear_programming, dual_linear_program, geometric_algorithm, token_classification | **Problems:** prompt_compression, llm_inference_efficiency, black_box_optimization, llm_inference_optimization, hardware_aware_optimization

*Unique methods:* adaptive_queryselect, blockwise_local_distillation, dual_linear_program, geometric_algorithm, global_knowledge_distillation, grouped_query_attention, knowledge_distillation, low_rank_approximation, multi_head_attention, neural_architecture_search, queryselect, rate_distortion_theory, reinforcement_learning_from_human_feedback, structured_sparsity, token_classification, transformer_architecture_optimization
*Shared methods:* beam_search, linear_programming, llm_as_heuristic, llm_fine_tuned, llm_in_the_loop, mixed_integer_programming, pruning

This front explores advanced Operations Research techniques, specifically rate-distortion theory formalized as linear programming and Mixed-Integer Programming (MIP), to optimize Large Language Models (LLMs) for efficiency. The core theme revolves around applying rigorous mathematical optimization to two distinct yet critical aspects of LLM deployment: prompt compression and neural architecture search for inference optimization.

Key contributions include Nagle et al.'s 'Adaptive QuerySelect,' a variable-rate prompt compression method derived from a rate-distortion framework. This method significantly outperforms fixed-rate baselines like LLMLingua-2 on synthetic and natural language datasets, demonstrating the importance of query-aware compression. Bercovich et al.'s 'Puzzle' framework utilizes decomposed Neural Architecture Search (NAS) with blockwise local knowledge distillation and MIP to optimize LLM architectures for specific hardware. Puzzle achieved a 2.17x inference throughput speedup for Llama-3.1-70B-Instruct (Nemotron-51B) while retaining 98.4% accuracy on benchmarks like Winogrande and MMLU, outperforming pruning methods.

This front is clearly emerging, showcasing novel applications of established OR paradigms to LLM challenges. The trajectory suggests a move towards more mathematically rigorous and hardware-aware optimization for LLMs. The next papers will likely focus on integrating these distinct optimization strategies, perhaps by developing unified frameworks that consider both prompt efficiency and model architecture concurrently, or by extending these methods to other LLM lifecycle stages like training or fine-tuning.

**Papers:**

### [Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models](https://arxiv.org/abs/2407.15504)

**2024-12-11** | UT Austin, EPFL | M=8 P=8 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Rate-distortion theory formalized as a linear program, solved via its dual using a geometric algorithm; Adaptive QuerySelect (query-aware, variable-rate token classification) | *LLM role:* token classifier

> Nagle et al. formalize prompt compression as a rate-distortion problem, deriving the fundamental theoretical limit via a dual linear program and proposing 'Adaptive QuerySelect,' a variable-rate compression technique. The results are rigorous: they calculate exact limits on synthetic data and use beam search approximations for NLP, demonstrating that existing fixed-rate methods leave significant performance on the table. The key takeaway is that **variable-rate compression**—keeping tokens based on a confidence threshold rather than a fixed percentage—is essential for approaching optimality; this allows 'hard' queries to retain more context while aggressively compressing 'easy' ones. This is immediately actionable for our AlgoEvo work: we should replace fixed-window history truncation with a query-aware, variable-rate compressor to maximize the useful information in our limited context window.

### [Puzzle: Distillation-Based NAS for Inference-Optimized LLMs](https://arxiv.org/abs/2411.19146)

**2025-06-03** | NVIDIA | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Decomposed Neural Architecture Search (NAS) using Blockwise Local Knowledge Distillation (BLD) for parallel architecture exploration and Mixed-Integer Programming (MIP) for precise constraint optimization, followed by Global Knowledge Distillation (GKD) | *LLM role:* none

> Bercovich et al. introduce Puzzle, a framework that optimizes LLM architectures for specific hardware by training a library of block variants (via local distillation) and using Mixed-Integer Programming (MIP) to select the optimal layer-wise configuration under strict latency and memory constraints. The results are robust: they compress Llama-70B to 51B, fitting on a single H100 with 2.17x throughput gain and 98.4% accuracy retention, significantly outperforming pruning baselines like Wanda. **Key takeaway:** The 'decomposed search' strategy—replacing expensive end-to-end evolutionary evaluation loops with local proxy scores (KL divergence) and a global MIP solver—is a highly efficient method for modular system configuration. This directly informs our 'GPUSched' and serving optimization work by demonstrating how to mathematically formulate hardware constraints (KV-cache, batch size, compute) into the model design process itself.


### Front 11 (2 papers) — STABLE

**Density:** 1.00 | **Methods:** post_training_quantization, mixed_precision_quantization, linear_programming, gptq, expert_quantization | **Problems:** llm_compression, memory_optimization, inference_efficiency, mixture_of_experts_compression, vlm_compression

*Unique methods:* binary_quantization, dynamic_expert_pruning, dynamic_pruning, expert_pruning, expert_quantization, gptq, gumbel_softmax, hqq, learnable_mask, mixed_precision_quantization, model_compression, moe_llm_compression, post_training_quantization, token_pruning
*Shared methods:* integer_programming, linear_programming, pruning

This research front centers on the Mixture Compressor (MC) and MC# frameworks, which leverage Linear Programming (LP) and Integer Linear Programming (ILP) for extreme compression of Mixture-of-Experts (MoE) Large Language Models. The unifying theme is the application of Operations Research techniques to optimally allocate mixed bit-widths and dynamically prune experts, enabling significant memory reduction while preserving model performance. This approach addresses the critical challenge of deploying large MoE models on resource-constrained hardware.

Key contributions include the development of hybrid post-training quantization and dynamic pruning strategies. Huang et al. (2025-02) introduced MC, using ILP to allocate 1-3 bit-widths based on activation frequency and routing weights, compressing Mixtral 8x7b to ~16GB (fitting on a single RTX 3090) with only a ~4% drop in zero-shot accuracy, significantly outperforming uniform quantization. Building on this, Huang et al. (2025-10) proposed MC#, combining Pre-Loading Mixed-Precision Quantization (PMQ) via LP with Online Top-any Pruning (OTP) via Gumbel-Softmax sampling, achieving a 6.2x weight reduction on DeepSeek-VL2 with less than 2% accuracy loss, and demonstrating 18.4% better performance than BSP 2.54-bit on Mixtral 8x7b LM-Eval.

This front is rapidly maturing, with MC# directly extending MC and demonstrating improved results and broader applicability (including VLMs). The trajectory indicates a strong focus on practical deployment. The next likely papers will focus on adapting these compression strategies for multimodal applications, optimizing them for specific hardware platforms, and further enhancing performance on challenging reasoning and long-context tasks where current methods still show some performance degradation.

**Papers:**

### [Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270)

**2025-02-22** | The University of Hong Kong, The Chinese University of Hong Kong, Beihang University, Centre for Perceptual and Interactive Intelligence, Hong Kong | M=5 P=7 I=6 *discuss*

*Method:* Hybrid Post-Training Quantization and Dynamic Pruning for MoE-LLMs using Linear Programming for bit-width allocation and significance-aware token protection | *LLM role:* none

> Huang et al. propose a compression framework for MoE-LLMs that uses Integer Programming to optimally allocate mixed bit-widths (1-3 bits) to experts based on activation frequency and routing weights. They achieve strong empirical results, compressing Mixtral 8x7b to ~16GB (fitting on a single RTX 3090) with only a ~4% drop in zero-shot accuracy, significantly outperforming uniform quantization. The key takeaway is the explicit IP formulation for minimizing quantization error under memory constraints—a clean 'OR for AI' pattern we can adapt for our GPU scheduling or memory allocation formulations. While not a methodological advance in evolution, this is highly relevant for our infrastructure: it enables deploying high-quality MoE models on cheaper hardware for our massive AlgoEvo loops.

### [MC#: Mixture Compressor for Mixture-of-Experts Large Models](https://arxiv.org/abs/2510.10962)

**2025-10-13** | NVIDIA Research, National University of Singapore, The University of Hong Kong, Beihang University, Hangzhou Innovation Institute | M=6 P=7 I=7 *discuss*

*Method:* Hybrid compression combining Pre-Loading Mixed-Precision Quantization (PMQ) via Linear Programming and Online Top-any Pruning (OTP) via Gumbel-Softmax sampling | *LLM role:* none

> Huang et al. propose MC#, a compression framework for MoE models that combines static mixed-precision quantization with dynamic expert pruning. They formulate bit-width allocation as an Integer Linear Programming (ILP) problem—optimizing expert importance vs. quantization error—and use a Gumbel-Softmax router for dynamic pruning. Results are strong, achieving 6.2x weight reduction on DeepSeek-VL2 with <2% accuracy loss. **Takeaway:** The ILP formulation (Eq. 7) is a clean, successful application of OR to AI infrastructure that we should replicate for our own resource allocation/scheduling problems; additionally, the differentiable router offers a template for dynamic agent selection in our multi-agent systems.


### Front 14 (2 papers) — EMERGING

**Density:** 1.00 | **Methods:** llm_code_generation, evolutionary_computation, llm_evolutionary_search, multi_agent_system, data_algorithm_co_evolution | **Problems:** MILP_general, heuristic_evolution, combinatorial_optimization, set_cover, combinatorial_auctions

*Unique methods:* branch_and_bound, constraint_programming_solver, data_algorithm_co_evolution, diving_heuristics, evolutionary_computation, in_context_learning, llm_evolutionary_search, multi_agent_system, neuro_symbolic_ai, parameter_efficient_fine_tuning, prompt_engineering, qlora, retrieval_augmented_generation, self_correction, tree_of_thoughts
*Shared methods:* llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_fine_tuned, llm_in_the_loop, supervised_fine_tuning

This research front explores advanced applications of Large Language Models (LLMs) in Operations Research, specifically focusing on two distinct yet complementary approaches: the DHEvo framework for data-algorithm co-evolution in heuristic search for Mixed-Integer Linear Programming (MILP), and the ConstraintLLM neuro-symbolic framework for automated Constraint Programming (CP) modeling.

DHEvo introduces a novel data-algorithm co-evolution framework that iteratively evolves heuristic code while dynamically filtering training instances to improve generalization and reduce performance variance. It significantly outperforms FunSearch and LLM4Solver on benchmarks like Setcover, achieving a primal gap of 9.74% compared to FunSearch's 77.99%. Complementarily, ConstraintLLM presents a neuro-symbolic approach for automated CP modeling, leveraging multi-instruction supervised fine-tuning and a unique Constraint-Aware Retrieval Module (CARM). This framework achieves approximately 51% accuracy on the challenging IndusCP benchmark, demonstrating its effectiveness in generating industrial-level CP models by focusing on structural constraint signatures rather than semantic similarity.

This front is clearly emerging, marked by two 'MUST-READ' papers that introduce paradigm-shifting methodologies. The trajectory suggests a strong emphasis on improving the robustness and generalization of LLM-generated OR solutions and models. Future work will likely expand on the DHEvo framework's data-algorithm co-evolution to other heuristic types and MILP solver components, while ConstraintLLM's structural retrieval and self-correction mechanisms will be refined and applied to broader CP domains and larger LLM models.

**Papers:**

### [DHEvo: Data-Algorithm Based Heuristic Evolution for Generalizable MILP Solving](https://arxiv.org/abs/2507.15615)

**2025-07-21** | Harbin Institute of Technology, Huawei Noah’s Ark Lab, Nanyang Technological University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Data-Algorithm Co-evolution Framework (DHEvo) with LLM-based Multi-Agent Evolution System (MA-Evolution System) | *LLM role:* code_writer

> DHEvo introduces a 'data-algorithm co-evolution' framework that iteratively evolves heuristic code while simultaneously filtering the training instance set to retain only 'representative' instances (those where current heuristics perform well/stably). Empirical results on SCIP diving heuristics show it outperforms FunSearch and EoH by ~60% on Setcover while significantly reducing performance variance, validating the claim that dynamic data curation prevents overfitting. The key takeaway is the counter-intuitive curriculum strategy: rather than training on the hardest instances, filtering for instances with 'regular' feasible regions (high fitness) stabilizes the evolutionary search for code. We should immediately test this dynamic instance filtering in AlgoEvo to improve sample efficiency and generalization.

### [ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming](https://arxiv.org/abs/2510.05774)

**2025-10-07** | University of Oxford, University of Chinese Academy of Sciences, Hangzhou Institute for Advanced Study, ISCAS, University of Science and Technology Beijing | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Neuro-Symbolic Framework integrating Multi-Instruction Supervised Fine-Tuning (SFT) of an open-source LLM, Constraint-Aware Retrieval Module (CARM), Tree-of-Thoughts (ToT) exploration, and Iterative Self-Correction with Guided Retrieval. | *LLM role:* code_writer

> ConstraintLLM fine-tunes a 32B model for Constraint Programming (CP) modeling, utilizing a "Constraint-Aware Retrieval Module" (CARM) that retrieves few-shot examples based on extracted constraint signatures (e.g., `AllDifferent`, `Cumulative`) rather than text embeddings. They also employ a Tree-of-Thoughts search pruned by test case execution and an iterative self-correction mechanism that retrieves "correction paths" (error-to-fix trajectories). Results are strong: on their new industrial benchmark (IndusCP), they achieve ~51% accuracy with a 32B model, matching or beating GPT-4o and DeepSeek-V3. **Key Takeaway:** The shift from semantic retrieval to *structural* retrieval (matching constraint profiles) is the "stealable" insight; we should implement this for our OR modeling tasks immediately, ignoring surface-level problem descriptions in favor of logical signatures. This directly impacts our OR-Bench and automated formulation work.



## Bridge Papers

<!-- Cross-front connectors, updated weekly -->

---

*Generated by Research Intelligence System*
