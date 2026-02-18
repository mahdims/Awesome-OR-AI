# Living Review: OR for Generative AI

**Last Updated:** 2026-02-18

---

## Recent Papers

#### 2026-02-17 (1 papers)

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.


#### 2026-02-17 (1 papers)

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.


#### 2026-02-17 (1 papers)

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.


<!-- New papers are appended here by the daily updater -->

---

## Research Fronts

*5 fronts detected — snapshot 2026-02-18*

### Front 14 (7 papers) — STABLE

**Density:** 0.81 | **Methods:** integer_linear_programming, performance_modeling, data_parallelism, tensor_parallelism, pipeline_parallelism | **Problems:** llm_serving_optimization, resource_allocation, gpu_scheduling, carbon_emission_reduction, sustainable_computing

*Unique methods:* adaptive_scheduling, arima_time_series_forecasting, bi_level_optimization, cache_replacement_policy, chebyshev_guided_optimization, continuous_batching, cvxpy, data_parallelism, decomposition_algorithm, discrete_event_simulation, distributed_systems, dynamic_batching, dynamic_scheduling, heuristic_initialization, integer_linear_programming, kv_cache, kv_cache_management, least_carbon_savings, llm_serving_optimization, max_flow_optimization, milp, milp_acceleration, milp_formulation, model_parallelism, network_communication_optimization, network_topology_modeling, np_hardness_proof, optimization, pipeline_parallelism, profiling, reactive_heuristics, resource_allocation_optimization, resource_management, sarima, shortest_path_algorithms, shortest_path_routing, simulation, system_algorithm_co_design, tensor_parallelism, threshold_based_routing, time_series_forecasting, weighted_round_robin
*Shared methods:* greedy_algorithm, llm_as_evaluator, load_balancing, mixed_integer_linear_programming, performance_modeling, queueing_theory, resource_allocation, robust_optimization, scheduling_algorithms

This research front focuses on leveraging Integer Linear Programming (ILP) and Mixed Integer Linear Programming (MILP) to optimize various aspects of Large Language Model (LLM) serving infrastructure. This includes carbon-aware cache management, efficient multi-round inference over disaggregated systems, forecast-aware auto-scaling in cloud data centers, resource allocation for geographically-distributed inference, topology-driven placement of Mixture-of-Expert (MoE) layers, bi-level optimization for cascade serving, and max-flow based serving over heterogeneous GPUs. The core theme is applying rigorous Operations Research methods to enhance the efficiency, sustainability, and performance of LLM deployment.

Key contributions include GreenCache's ILP for dynamic KV cache sizing, achieving up to 12.6% carbon reduction for Llama-3 70B. Dynamo's ILP-based offline planner and adaptive routing improved SLO attainment by 67-340% for multi-round inference. SageServe demonstrated 25% GPU-hours savings and $2.5M/month in cloud costs for Llama-2 using ILP and ARIMA forecasting. Other works optimized distributed inference, showing 60-80% latency reduction (Petals-derived), up to 39.1% network hop reduction for MoE placement (ILPLoad), 2.3x throughput gains for cascade serving (Cascadia), and 3.3x decode throughput on heterogeneous GPUs (Helix) via max-flow MILP.

This research front is stable, demonstrating a robust and expanding application of OR techniques to LLM serving. The trajectory indicates a move towards more complex, dynamic, and integrated optimization problems. Future work will likely focus on scaling these MILP solutions to larger clusters, incorporating more real-time and predictive elements, and adapting to evolving LLM architectures and serving paradigms. The next papers will likely explore dynamic, adaptive OR solutions that can respond to real-time changes in workload and infrastructure, potentially integrating machine learning for more accurate predictions within the optimization loop.

**Papers:**

### [Cache Your Prompt When It's Green: Carbon-Aware Caching for Large Language Model Serving](https://arxiv.org/abs/2505.23970)

**2026-01-19** | University of Waterloo, Purdue University | M=5 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) based dynamic cache size reconfiguration with SARIMA load prediction and carbon-aware Least Carbon Savings (LCS) cache replacement policy | *LLM role:* none

> Tian et al. propose GreenCache, a framework using Integer Linear Programming (ILP) to dynamically resize KV caches for LLM serving, balancing operational carbon (compute) against embodied carbon (SSD storage). They demonstrate ~15% carbon reduction on Llama-3 70B using Azure traces, though the reliance on simulation rather than live deployment weakens the claims slightly. For our 'OR for AI systems' work, the key takeaway is their 'Least Carbon Savings' (LCS) eviction policy—a heuristic that weighs computation saved against storage cost and recency—which we could adapt for optimizing memory-constrained multi-agent systems (HERMES) or general serving resource allocation.

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**2025-11-12** | Microsoft, University of Illinois Urbana-Champaign, Georgia Institute of Technology, Indian Institute of Science | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Integer Linear Programming (ILP) for resource allocation, combined with ARIMA-based time-series forecasting and reactive heuristics for dynamic scaling and scheduling | *LLM role:* none

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interactive and non-interactive workloads. They validate this on real Microsoft O365 production traces (which they release), demonstrating a 25% reduction in GPU hours and $2.5M/month savings compared to reactive baselines. The primary value for us is the release of the production workload traces—allowing us to benchmark our 'GPUSched' formulations against real-world data rather than synthetic distributions—and their specific ILP formulation for unified capacity management, which directly competes with our internal OR models.

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**2025-03-05** | Carnegie Mellon University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Max-flow problem formulation on directed, weighted graphs with Mixed Integer Linear Programming (MILP) for joint model placement and per-request pipeline scheduling | *LLM role:* none

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow solution. Unlike standard static pipeline parallelism, it routes every request dynamically based on edge capacities, achieving up to 3.3x throughput gains over Swarm on mixed GPU clusters (L4/T4/A100). The results are rigorous, backed by both physical cluster experiments and high-fidelity simulations. The critical takeaway is the 'per-request pipeline' abstraction: decoupling request routing from static device assignment allows exact OR methods to maximize utilization of weaker hardware—a technique we should immediately evaluate for our GPUSched project.


### Front 4 (3 papers) — STABLE

**Density:** 1.00 | **Methods:** online_optimization, scheduling, batching, integer_programming, competitive_ratio_analysis | **Problems:** llm_inference_scheduling, resource_allocation, kv_cache_management, online_scheduling, llm_serving_optimization

*Unique methods:* adaptive_control, asymptotic_analysis, batch_scheduling, batching, bin_packing, binomial_thinning, competitive_ratio_analysis, doobs_inequality, fluid_dynamics_approximation, integer_programming, kingmans_bound, lexicographical_optimization, lindley_recursion, martingale_theory, memory_constrained_scheduling, nested_wait_algorithm, online_algorithms, online_optimization, online_scheduling, outlier_detection, queuing_theory, scheduling, shortest_first, state_synchronization, threshold_based_scheduling, union_bound, wait_algorithm
*Shared methods:* greedy_algorithm, load_balancing, queueing_theory, resource_allocation, scheduling_algorithms

This research front unifies recent advancements in online scheduling for Large Language Model (LLM) inference, specifically addressing the critical challenges posed by KV cache memory constraints. It explores diverse operations research methodologies, including the Memory Constrained Shortest First (MC-SF) algorithm, Staggered Batch Scheduling (SBS), and fluid dynamics approximations with threshold-based online scheduling (Nested WAIT). The core theme is optimizing throughput and latency in dynamic LLM serving environments.

Key contributions include the MC-SF algorithm, which achieves near-optimal performance (within 5% of hindsight optimal) by incorporating a "future feasibility check" for KV cache management. Staggered Batch Scheduling (SBS) significantly reduces Time-to-First-Token (TTFT) by 30-40% and improves overall throughput by 15-20% on production workloads, leveraging an "IQR-aware lexicographical" decode scheduling. Furthermore, fluid-guided approaches like Nested WAIT provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, outperforming heuristic baselines like vLLM and Sarathi by employing a novel "nested segment" mechanism for handling unknown output lengths.

This front is rapidly emerging, establishing foundational mathematical and algorithmic frameworks for high-efficiency LLM inference. The trajectory suggests a move towards integrating these distinct approaches, refining parameter determination, and extending their applicability to more complex, heterogeneous, and distributed LLM serving architectures. The next papers will likely focus on combining predictive mechanisms with adaptive scheduling policies and scaling solutions for multi-GPU systems.

**Papers:**

### [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115)

**2026-01-15** | Massachusetts Institute of Technology, Microsoft Research, HKUST | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Memory Constrained Shortest First (MC-SF) online batching and scheduling algorithm | *LLM role:* none

> This paper formulates LLM inference scheduling as an Integer Program (IP) that explicitly models the linear memory growth of KV caches, and proposes a 'Memory Constrained Shortest First' (MC-SF) algorithm. The results are rigorous, showing MC-SF achieves near-optimal performance (within 5% of hindsight optimal) on synthetic data and significantly outperforms standard FCFS/threshold heuristics on real traces. The critical takeaway is the 'future feasibility check' (Eq. 5), which validates that a batch will *remain* within memory limits throughout the generation process based on predicted output lengths—a necessary deviation from standard static-size scheduling. This is foundational reading for our GPUSched project, providing both the exact IP baseline we need and a strong heuristic to benchmark against.

### [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](https://arxiv.org/abs/2512.16134)

**2025-12-18** | Baidu Inc. | M=6 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Staggered Batch Scheduling (SBS) with Throughput-Adaptive Interval Control, Multi-tier State Synchronization, Prioritized Batch Allocation Algorithm (PBAA) for Prefill, and IQR-Aware Lexicographical Decode Scheduling for Decode | *LLM role:* none

> Tian et al. introduce Staggered Batch Scheduling (SBS) for DP+EP architectures, enforcing a buffering window to enable global bin-packing rather than immediate dispatch, which they prove causes Head-of-Line blocking in non-preemptive prefill phases. Tested on a production H800 cluster serving DeepSeek-V3, they demonstrate a 30-40% reduction in TTFT and a ~20% throughput increase backed by clear utilization metrics. The most valuable takeaway for our GPUSched project is their 'IQR-aware lexicographical' scheduling heuristic for the Decode phase, which robustly balances batch size against KV-cache memory variance—a constraint logic we should immediately adopt. This work validates that discrete batching is superior to continuous dispatch for MoE models, necessitating an update to our queuing theory models.

### [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

**2026-01-05** | Massachusetts Institute of Technology, Peking University, Alibaba Group | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Fluid dynamics approximation and threshold-based online scheduling (WAIT and Nested WAIT algorithms) | *LLM role:* none

> This paper formulates LLM inference as a multi-stage stochastic scheduling problem, introducing 'Nested WAIT'—a threshold-based algorithm that handles unknown output lengths by letting prompts classify themselves as they survive into deeper decode segments. Unlike heuristic baselines (vLLM, Sarathi), they provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, validated on A100 simulations. The key takeaway is the 'nested segment' mechanism: instead of predicting job size, structure the queue so short jobs exit early and long jobs naturally migrate to lower-priority/protected tiers, effectively decoupling the memory risk. We should immediately evaluate this threshold logic for our GPUSched formulations, as it likely outperforms our current predictive or FCFS approaches for handling KV cache growth.


### Front 23 (3 papers) — GROWING

**Density:** 0.67 | **Methods:** multi_objective_optimization, convex_optimization, gradient_descent, game_theory, llm_as_evaluator | **Problems:** multi_objective_llm_alignment, sentiment_control, text_length_control, humor_generation, harmlessness_control

*Unique methods:* adaptation_safety, best_of_k_sampling, black_box_optimization, blockwise_decoding, controlled_decoding, equilibrium_search, game_theory, gradient_aggregation, gradient_descent, inference_time_alignment, llm_alignment, llm_fine_tuning, maximin_optimization, multi_objective_optimization, noon_ppo, optimization_penalty_function, ppo, reward_modeling, rlhf, supervised_fine_tuning, value_function_learning, zero_sum_game
*Shared methods:* convex_optimization, linear_programming, llm_as_evaluator, llm_in_the_loop, robust_optimization

This research front unifies recent advancements in applying Operations Research techniques, specifically convex optimization and game theory, to the challenging problem of multi-objective alignment for Large Language Models (LLMs). Papers introduce frameworks like PAMA, Safety Game, and Robust Multi-Objective Decoding (RMOD) to manage conflicting objectives such as harmlessness, helpfulness, sentiment, and length control, often at inference time.

Key contributions include the PAMA algorithm, which transforms multi-objective RLHF into an O(n) convex optimization problem with a closed-form solution, outperforming MORLHF and MGDA-UB on LLaMA-2 7B for harmlessness. The Safety Game formulates black-box LLM agent alignment as a zero-sum game solvable by an LP solver at inference, achieving up to two-fold accuracy improvement on SafetyBench. RMOD introduces a maximin two-player game for robust multi-objective decoding, solving a convex optimization problem at each step to maximize worst-case value, outperforming MO-DPO and scalarized baselines by +1.2% WCWR on Anthropic HH.

This front is rapidly growing, demonstrating the power of OR principles to bring robustness and efficiency to LLM alignment. The trajectory indicates a strong focus on mathematically grounded, inference-time control mechanisms. Future work will likely focus on extending these frameworks to more complex, dynamic, and multi-agent scenarios, improving their scalability to a greater number of objectives, and integrating these control mechanisms into broader agentic architectures.

**Papers:**

### [Pareto Multi-Objective Alignment for Language Models](https://arxiv.org/abs/2508.07768)

**2025-08-11** | Ruhr University Bochum | M=7 P=5 I=6 *discuss*

*Method:* PAMA (PAreto Multi-Objective Alignment) algorithm, which transforms multi-objective RLHF into a convex optimization problem with a closed-form solution, combined with Noon PPO. | *LLM role:* subject_of_optimization

> PAMA introduces a computationally efficient algorithm for multi-objective alignment by reformulating the expensive gradient-norm minimization of MGDA into a convex optimization problem with a closed-form solution, reducing complexity from O(n^2d) to O(n). Empirical results on LLaMA-2-7B are robust, showing stable convergence on conflicting objectives (e.g., harmlessness vs. length) where baselines like MGDA-UB oscillate or fail. The single most useful takeaway is the analytical derivation for optimal objective weighting (Theorem 1) and the 'Noon PPO' heuristic (clipping negative advantages); we could port this logic to our multi-objective process reward models in AlgoEvo to balance search signals efficiently. While the NLP experiments are trivial, the gradient balancing mechanism is directly applicable to our multi-objective RL controllers.

### [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](https://arxiv.org/abs/2510.09330)

**2025-12-02** | University of Warwick | M=7 P=4 I=7 *discuss*

*Method:* Two-player zero-sum game formulation solved by a linear programming (LP) solver at inference time to compute minimax equilibrium strategies, using binary probes for helpfulness and safety scores, with a sigmoid penalty for risk. | *LLM role:* agent_response_selection, evaluator

> The authors formulate LLM response selection as a zero-sum game, solving a small Linear Program (LP) at inference time to mix candidate answers such that the expected risk never exceeds a 'safe fallback' baseline. Results are statistically significant, showing ~15% accuracy gains on SafetyBench by effectively managing the trade-off between helpfulness and safety probes. The key takeaway is the 'Adaptation Safety' constraint formulation: using an LP to guarantee that a stochastic policy is no worse than a heuristic baseline is a powerful, lightweight control mechanism we could adapt for selecting evolved algorithms or managing constraints in multi-agent optimization.

### [Robust Multi-Objective Controlled Decoding of Large Language Models](https://arxiv.org/abs/2503.08796)

**2025-03-11** | University College London, University of Basel, Ulsan National Institute of Science and Technology | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Maximin two-player game between adversarially computed reward weights and sampling policy, solvable through Nash equilibrium, reduced to convex optimization, with blockwise best-of-K sampling | *LLM role:* controlled_decoding_target

> RMOD formulates multi-objective decoding as a zero-sum game between a policy and adversarial weights, solving a convex optimization problem at each decoding step to maximize the worst-case value estimate (essentially a Process Reward Model). The results are empirically strong, outperforming MO-DPO and scalarized baselines on alignment benchmarks by dynamically preventing any single objective from collapsing. **Key Takeaway:** The efficient inference-time weight optimization algorithm (Eq. 10) is a 'stealable' mechanism for **AlgoEvo** and **RobustMAS**. We should implement this dynamic adversarial weighting to balance conflicting code metrics (e.g., runtime vs. solution quality) during evolutionary search, replacing our current static scalarization methods.


### Front 28 (2 papers) — EMERGING

**Density:** 1.00 | **Methods:** task_scheduling, fine_grained_scheduling, disaggregated_expert_parallelism, ping_pong_pipeline, performance_modeling | **Problems:** llm_serving_optimization, moe_inference, gpu_resource_allocation, communication_optimization, task_scheduling

*Unique methods:* decode_router, disaggregated_expert_parallelism, fine_grained_scheduling, fluid_approximation, gate_and_route_policy, heuristic_algorithm_design, kkt_conditions, linear_performance_models, lyapunov_function, many_server_queueing, mathematical_analysis, maximum_likelihood_estimation, optimization_problem_formulation, ordinary_least_squares, ping_pong_pipeline, prefill_admission_gate, queueing_network, scheduling_strategies, stochastic_control, task_scheduling
*Shared methods:* convex_optimization, linear_programming, performance_modeling

This research front unifies approaches to optimize large-scale LLM inference, specifically addressing Mixture-of-Experts (MoE) scheduling and prefill-decode contention. One key direction involves fine-grained task scheduling, exemplified by the FinDEP algorithm, which maximizes task overlap for disaggregated expert parallelism. The other direction focuses on stochastic control, utilizing many-server queueing network models and fluid approximations to manage heterogeneous workloads and prefill-decode contention.

Key contributions include FinDEP's fine-grained task scheduling algorithm, which leverages linear performance models and analytical properties to achieve up to 1.61x throughput improvement over PPPipe on H20/A6000 clusters. Concurrently, Lin et al. propose a rigorous multiclass many-server queueing network model, deriving a 'Gate-and-Route' policy from a steady-state fluid LP. This policy effectively manages prefill-decode contention, demonstrating that separating prefill admission from decode routing maximizes revenue, with FI-WSP achieving approximately 30% lower revenue than OPT.

This front is clearly emerging, with both papers introducing novel, specific methodologies for critical LLM serving challenges. The trajectory suggests a move towards more robust and integrated solutions. Future work will likely focus on relaxing current model assumptions, extending applicability to more complex hardware and workload scenarios, and potentially combining the strengths of fine-grained scheduling with higher-level stochastic control policies.

**Papers:**

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**2025-12-25** | The Hong Kong University of Science and Technology, Harbin Institute of Technology, Hong Kong Baptist University | M=6 P=7 I=5 *discuss*

*Method:* Fine-grained task scheduling algorithm for disaggregated expert parallelism (DEP) with maximal task overlap, guided by linear performance models and analytical properties (monotonicity, convexity) | *LLM role:* none

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize overlap. The authors achieve 1.02x-1.61x speedups on H20/A6000 clusters compared to PPPipe, backed by solid empirical data. The key takeaway for our 'GPUSched' work is their methodology: deriving analytical properties (monotonicity and convexity) of the scheduling objective to reduce a complex search space into an $O(1)$ online solver, rather than relying on heavy solvers or RL. This confirms that simple linear performance models ($\alpha + \beta x$) are sufficient for accurate online resource allocation in LLM serving.

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.


### Front 29 (2 papers) — STABLE

**Density:** 1.00 | **Methods:** best_of_n, majority_voting, adaptive_sampling, bayesian_modeling, dirichlet_process_prior | **Problems:** llm_serving_optimization, llm_inference_optimization, mathematical_reasoning, scientific_reasoning, ensemble_optimization

*Unique methods:* adaptive_sampling, all_to_all_collectives, bayes_factor, bayesian_modeling, best_of_n, cuda_graph, dirichlet_process_prior, expert_parallelism, grouped_gemm, llm_as_answer_generator, llm_ensemble, llm_inference_optimization, majority_voting, max_margin_optimization, mixture_of_experts, monte_carlo_methods, nvshmem, online_distillation, pipelining, predictive_scheduling, real_time_optimization, straggler_mitigation, system_design, triton
*Shared methods:* greedy_algorithm, llm_in_the_loop, load_balancing, mixed_integer_linear_programming, resource_allocation

This research front unifies advanced optimization techniques to enhance the efficiency and performance of Large Language Model (LLM) inference and serving. Paper [1] focuses on adaptive sampling for majority voting using Bayesian modeling (Dirichlet process prior, Bayes factor) and optimally weighted LLM ensembles formulated as a Mixed-Integer Linear Program (MILP) to optimize LLM inference for reasoning tasks. Paper [2] introduces PROBE, a framework for Mixture-of-Experts (MoE) inference optimization, leveraging Continuous Lookahead Pipelining and predictive prefetching for dynamic load balancing.

Key contributions include Paper [1]'s adaptive generation scheme, which achieves the same accuracy with 2x-5x fewer samples and tokens compared to fixed-budget Best-of-N, and demonstrates LLM ensembles outperforming single LLMs (e.g., 93.3% vs 90.0% for GPT-OSS-20B on AIME2025). Paper [2]'s PROBE framework, with its hardware-aware balance planning and phase-locked co-scheduling, delivers significant speedups, such as 1.32x in prefill latency for SGLang and 1.26x higher decoding throughput over DeepSeek-EPLB on models like Qwen3-MoE-235B.

This front is rapidly emerging, showcasing novel OR/AI hybrid approaches to critical LLM efficiency challenges. The integration of Bayesian adaptive methods with predictive pipelining points towards a trajectory of more intelligent, dynamic, and resource-aware LLM serving and inference systems. The likely next steps involve integrating these adaptive and predictive control mechanisms with more sophisticated optimization solvers (e.g., fast ALNS or IP formulations) to address real-time, stochastic resource allocation in complex LLM serving environments.

**Papers:**

### [Best-of-$\infty$ -- Asymptotic Performance of Test-Time Compute](https://arxiv.org/abs/2509.21091)

**2025-09-25** | Mohamed bin Zayed University of Artificial Intelligence, New York University, RIKEN AIP, Institute of Science Tokyo, NEC Corporation | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Adaptive sampling for majority voting using Bayesian modeling (Dirichlet process prior and Bayes factor) to determine stopping criteria, combined with optimally weighted LLM ensembles formulated as a Mixed-Integer Linear Program (MILP) with a max-margin solution. | *LLM role:* answer_generator

> This paper introduces a Bayesian adaptive stopping criterion (using Dirichlet process priors and Bayes factors) for majority voting, reducing test-time compute by 2-5x while maintaining asymptotic 'Best-of-Infinity' accuracy. They further demonstrate that optimizing weights for an ensemble of LLMs can be formulated as a Mixed-Integer Linear Program (MILP) by treating the decision boundaries as polytopes. **What we learned:** The Bayesian stopping logic is immediately transferable to AlgoEvo to reduce the cost of fitness evaluations—we can stop evaluating candidate solutions early if their performance distribution is statistically distinct. The MILP approach for ensembles also offers a concrete formulation we could adapt for our GPU scheduling and model serving optimization work.

### [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509)

**2026-02-03** | Kling Infra, Kuaishou Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Continuous Lookahead Pipelining with Gate-Initialized Lookahead Predictor, Hardware-Aware Balance Planning, and Phase-Locked Co-Scheduling | *LLM role:* none

> PROBE optimizes MoE inference by using a distilled MLP to predict next-layer expert activation, enabling proactive load balancing and weight prefetching hidden behind the current layer's computation. The results are strong (1.3x speedup on 235B models) and demonstrate that control plane overheads can be fully masked. The critical takeaway for our `GPUSched` project is the **Lookahead Pipelining** architecture: it carves out a deterministic execution window where we could inject our own specialized solvers (e.g., fast ALNS or IP formulations) to outperform their basic greedy resource allocator. This transforms the stochastic serving problem into a short-horizon deterministic routing problem we are well-equipped to solve.



## Bridge Papers

Papers connecting multiple research fronts:

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**TRUE SYNTHESIS** | score=0.55 | Front 28 → Front 14

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize o


---

*Generated by Research Intelligence System*
