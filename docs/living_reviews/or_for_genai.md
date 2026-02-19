# Living Review: OR for Generative AI

**Last Updated:** 2026-02-19

---

## Recent Papers

#### 2026-02-19 (1 papers)

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.


#### 2026-02-18 (1 papers)

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

*6 fronts detected — snapshot 2026-02-18*

### Front 10 (10 papers) — GROWING

**Density:** 0.71 | **Methods:** integer_linear_programming, pipeline_parallelism, data_parallelism, tensor_parallelism, performance_modeling | **Problems:** llm_serving_optimization, resource_allocation, gpu_scheduling, llm_scheduling, service_level_objective_optimization

*Unique methods:* adaptive_scheduling, arima_time_series_forecasting, autoregressive_decoding, binary_search, cache_replacement_policy, chebyshev_guided_optimization, cvxpy, data_parallelism, decomposition_algorithm, demand_forecasting, discrete_event_simulation, distributed_systems, dynamic_batching, dynamic_programming, dynamic_scheduling, heuristic_initialization, heuristics, kv_cache, kv_cache_management, least_carbon_savings, llm_serving_optimization, max_flow_optimization, milp, milp_acceleration, milp_formulation, model_parallelism, network_communication_optimization, network_topology_modeling, neural_network, np_hardness_proof, optimal_transport, optimization, pipeline_parallelism, profiling, reactive_heuristics, resource_allocation_optimization, resource_management, sarima, shortest_path_algorithms, shortest_path_routing, simulation, system_algorithm_co_design, task_batching, tensor_parallelism, threshold_based_routing, time_series_forecasting, weighted_round_robin, wireless_resource_allocation
*Shared methods:* bi_level_optimization, continuous_batching, greedy_algorithm, integer_linear_programming, llm_as_evaluator, load_balancing, mixed_integer_linear_programming, performance_modeling, proximal_policy_optimization, queueing_theory, reinforcement_learning, resource_allocation, robust_optimization, scheduling_algorithms, speculative_decoding

This research front is unified by the application of Integer Linear Programming (ILP) and Mixed-Integer Linear Programming (MILP) to optimize various aspects of Large Language Model (LLM) serving. The core theme revolves around efficient resource allocation, scheduling, and deployment strategies for LLM inference, particularly addressing challenges posed by heterogeneous GPU clusters, disaggregated serving architectures, and geographically distributed systems. Specific problem domains include multi-round inference, Mixture-of-Expert (MoE) model placement, cascade serving, and carbon-aware caching.

Key contributions include Dynamo's ILP-based offline deployment for multi-round inference, achieving up to 340% SLO attainment improvement (Paper 1). Jiang et al. (Paper 3) demonstrated ~25% throughput gains over SOTA systems like Helix by co-optimizing GPU composition and parallelism using MILP for heterogeneous clouds. CASCADIA (Paper 5) introduced a bi-level optimization (MILP for deployment, Chebyshev for routing) for cascade serving, yielding 2.3x average throughput gains. SageServe (Paper 6) achieved 25% GPU-hours savings and $2.5M/month savings by coupling ILP with ARIMA forecasting for auto-scaling. Helix (Paper 10) formulated distributed LLM serving as a max-flow MILP, achieving up to 3.3x throughput gains on mixed GPU clusters by dynamic per-request routing. Other notable works include ILP for MoE expert placement (Paper 4), carbon-aware KV cache management (Paper 9), and hybrid RL-Optimal Transport for temporal-aware GPU allocation (Paper 7).

This front is rapidly maturing, driven by the increasing complexity and scale of LLM deployments. The consistent success of ILP/MILP in achieving significant performance, cost, and energy efficiency gains across diverse LLM serving scenarios indicates a strong and active research trajectory. Future work will likely focus on developing more scalable and dynamic OR solutions, integrating these with advanced LLM-specific optimizations, and expanding to multi-objective and real-time adaptive systems.

**Papers:**

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding](https://arxiv.org/abs/2510.11331)

**2025-10-13** | Queen Mary University of London, Kyung Hee University, Xidian University, Guangzhou Institute of Technology | M=5 P=7 I=6 

*Method:* Speculative Decoding (SD) with pipeline parallelism, combined with joint optimization of speculation length, task batching, and wireless communication resource allocation | *LLM role:* inference engine

> Zhu et al. propose a distributed Speculative Decoding framework for edge networks, formulating a Mixed-Integer Nonlinear Programming problem to jointly optimize task batching, speculation length, and wireless bandwidth. They solve the batching subproblem using a Dynamic Programming (DP) algorithm, achieving ~30-45% latency reduction over heuristics in simulations, though the approach relies on a rigid assumption of fixed maximum output lengths to remain tractable. The primary takeaway for our 'GPUSched' work is their DP formulation for optimizing batch boundaries in a pipelined draft-verify system, which offers a cleaner mathematical alternative to greedy heuristics for serving schedules. However, the heavy reliance on wireless channel modeling makes the full system less relevant to our datacenter-centric optimization problems.

### [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs](https://arxiv.org/abs/2502.00722)

**2025-06-05** | University of Cambridge, ETH Zurich, Peking University, The Hong Kong University of Science and Technology, Purdue University | M=5 P=9 I=6 **MUST-READ** *discuss*

*Method:* Mixed-Integer Linear Programming (MILP) for scheduling | *LLM role:* none

> Jiang et al. formulate LLM serving on heterogeneous clouds as a Mixed-Integer Linear Programming (MILP) problem, co-optimizing GPU rental composition, parallelism strategies (TP/PP), and workload routing. They demonstrate ~25% throughput gains over SOTA systems (Helix, HexGen) using vLLM benchmarks, validating the approach with strong empirical ablations. For our **GPUSched** project, the key takeaway is their solver strategy: pre-generating valid configurations to linearize the problem and using a binary search wrapper on the makespan to avoid direct minimization overhead. We should adopt their heuristics for pruning the configuration space (e.g., restricting TP to intra-node) to improve our own solver times.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**2025-11-12** | Microsoft, University of Illinois Urbana-Champaign, Georgia Institute of Technology, Indian Institute of Science | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Integer Linear Programming (ILP) for resource allocation, combined with ARIMA-based time-series forecasting and reactive heuristics for dynamic scaling and scheduling | *LLM role:* none

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interactive and non-interactive workloads. They validate this on real Microsoft O365 production traces (which they release), demonstrating a 25% reduction in GPU hours and $2.5M/month savings compared to reactive baselines. The primary value for us is the release of the production workload traces—allowing us to benchmark our 'GPUSched' formulations against real-world data rather than synthetic distributions—and their specific ILP formulation for unified capacity management, which directly competes with our internal OR models.

### [Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via Reinforcement Learning](https://arxiv.org/abs/2507.10259)

**2025-09-16** | Shenzhen University of Advanced Technology, China Mobile Research Institute | M=6 P=9 I=6 **MUST-READ** *discuss*

*Method:* Proximal Policy Optimization (PPO) with Optimal Transport supervision | *LLM role:* none

> TORTA introduces a hierarchical scheduler for distributed LLM inference that uses a macro-level RL agent (PPO) supervised by an Optimal Transport (OT) baseline to manage inter-region allocation, followed by a micro-level greedy allocator. Results on simulated clusters (up to 50 servers) demonstrate a ~15% reduction in latency compared to reactive baselines (like SkyLB) specifically by optimizing for temporal smoothness and reducing switching costs. The key technical takeaway is the use of an exact OR solver (OT) as a dense supervision signal to train a faster RL policy, effectively combining the optimality of OR with the temporal foresight of RL. We should review our GPUSched formulations to ensure we aren't falling into the 'reactive' trap described here; if we are, this hybrid RL-OT architecture is a viable alternative.

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [Cache Your Prompt When It's Green: Carbon-Aware Caching for Large Language Model Serving](https://arxiv.org/abs/2505.23970)

**2026-01-19** | University of Waterloo, Purdue University | M=5 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) based dynamic cache size reconfiguration with SARIMA load prediction and carbon-aware Least Carbon Savings (LCS) cache replacement policy | *LLM role:* none

> Tian et al. propose GreenCache, a framework using Integer Linear Programming (ILP) to dynamically resize KV caches for LLM serving, balancing operational carbon (compute) against embodied carbon (SSD storage). They demonstrate ~15% carbon reduction on Llama-3 70B using Azure traces, though the reliance on simulation rather than live deployment weakens the claims slightly. For our 'OR for AI systems' work, the key takeaway is their 'Least Carbon Savings' (LCS) eviction policy—a heuristic that weighs computation saved against storage cost and recency—which we could adapt for optimizing memory-constrained multi-agent systems (HERMES) or general serving resource allocation.

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**2025-03-05** | Carnegie Mellon University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Max-flow problem formulation on directed, weighted graphs with Mixed Integer Linear Programming (MILP) for joint model placement and per-request pipeline scheduling | *LLM role:* none

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow solution. Unlike standard static pipeline parallelism, it routes every request dynamically based on edge capacities, achieving up to 3.3x throughput gains over Swarm on mixed GPU clusters (L4/T4/A100). The results are rigorous, backed by both physical cluster experiments and high-fidelity simulations. The critical takeaway is the 'per-request pipeline' abstraction: decoupling request routing from static device assignment allows exact OR methods to maximize utilization of weaker hardware—a technique we should immediately evaluate for our GPUSched project.


### Front 6 (7 papers) — EMERGING

**Density:** 0.57 | **Methods:** queueing_theory, reinforcement_learning, llm_code_generation, llm_fine_tuned, program_synthesis | **Problems:** llm_inference_scheduling, resource_allocation, gpu_scheduling, online_scheduling, llm_serving_optimization

*Unique methods:* adaptive_control, asymptotic_analysis, batch_scheduling, batching, binomial_thinning, bootstrapping, competitive_ratio_analysis, data_augmentation, decode_prioritized_scheduling, demand_prediction, discrete_time_markov_chains, doobs_inequality, fair_queuing, fastertransformer, fcfs_scheduling, fluid_dynamics_approximation, fluid_limits, group_relative_policy_optimization, heuristic_filtering, instruction_tuning, kingmans_bound, lexicographical_optimization, lindley_recursion, llm_as_model_generator, llm_code_generation, llm_fine_tuned, martingale_theory, mathematical_modeling, memory_centric_cost_modeling, memory_constrained_scheduling, mixed_batching, mlp, nested_wait_algorithm, non_preemptive_scheduling, online_algorithms, online_optimization, orca, outlier_detection, prefill_prioritized_scheduling, program_synthesis, sarathi_serve, self_improving_search, shortest_first, state_synchronization, stochastic_processes, synthetic_data_generation, test_time_adaptation, test_time_reinforcement_learning, tf_idf, threshold_based_scheduling, union_bound, virtual_time_scheduling, vllm, wait_algorithm, work_conserving_scheduling
*Shared methods:* bin_packing, greedy_algorithm, integer_programming, llm_as_evaluator, llm_in_the_loop, load_balancing, lyapunov_function, online_scheduling, proximal_policy_optimization, queueing_theory, queuing_theory, reinforcement_learning, resource_allocation, scheduling, scheduling_algorithms, supervised_fine_tuning

This research front explores the application of advanced Operations Research (OR) principles and AI techniques to two critical challenges within the LLM ecosystem: optimizing LLM inference serving and automating the generation of OR models. For LLM inference, papers focus on sophisticated scheduling algorithms to manage GPU resources, KV cache memory, and request batching under various constraints. Concurrently, other works leverage large language models themselves to automatically formulate and solve complex OR problems, bridging the gap between natural language problem descriptions and executable optimization models.

Key contributions in LLM inference scheduling include Nested WAIT (Paper 1) for multi-stage online scheduling, achieving superior throughput and reduced latency on vLLM/Sarathi. Staggered Batch Scheduling (SBS) (Paper 2) for DP+EP architectures reduced Time-to-First-Token by 30-40% and increased throughput by 15-20% on DeepSeek-V3. Memory Constrained Shortest First (MC-SF) (Paper 4) achieved near-optimal latency (within 5% of hindsight optimal) for KV cache-aware online scheduling. Justitia (Paper 6) introduced a virtual-time based fair scheduler, reducing average job completion time by ~60%. In automated OR modeling, OR-R1 (Paper 3) integrated SFT and Test-Time Group Relative Policy Optimization (TGRPO) to improve modeling accuracy by +4.2% over ORLM. ORLM (Paper 7) and the OR-Instruct framework fine-tuned LLMs with synthetic data, outperforming GPT-4 by up to 38.4% on benchmarks like NL4OPT and IndustryOR.

This front is emerging, with significant activity in both LLM inference optimization and automated OR modeling. The trajectory suggests continued innovation in developing more robust and adaptive scheduling policies for increasingly complex LLM architectures and deployment scenarios. For automated OR, the next papers will likely focus on enhancing LLM capabilities for solution ranking, handling more diverse and complex problem types, and integrating multi-agent collaboration for sophisticated problem-solving.

**Papers:**

### [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

**2026-01-05** | Massachusetts Institute of Technology, Peking University, Alibaba Group | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Fluid dynamics approximation and threshold-based online scheduling (WAIT and Nested WAIT algorithms) | *LLM role:* none

> This paper formulates LLM inference as a multi-stage stochastic scheduling problem, introducing 'Nested WAIT'—a threshold-based algorithm that handles unknown output lengths by letting prompts classify themselves as they survive into deeper decode segments. Unlike heuristic baselines (vLLM, Sarathi), they provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, validated on A100 simulations. The key takeaway is the 'nested segment' mechanism: instead of predicting job size, structure the queue so short jobs exit early and long jobs naturally migrate to lower-priority/protected tiers, effectively decoupling the memory risk. We should immediately evaluate this threshold logic for our GPUSched formulations, as it likely outperforms our current predictive or FCFS approaches for handling KV cache growth.

### [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](https://arxiv.org/abs/2512.16134)

**2025-12-18** | Baidu Inc. | M=6 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Staggered Batch Scheduling (SBS) with Throughput-Adaptive Interval Control, Multi-tier State Synchronization, Prioritized Batch Allocation Algorithm (PBAA) for Prefill, and IQR-Aware Lexicographical Decode Scheduling for Decode | *LLM role:* none

> Tian et al. introduce Staggered Batch Scheduling (SBS) for DP+EP architectures, enforcing a buffering window to enable global bin-packing rather than immediate dispatch, which they prove causes Head-of-Line blocking in non-preemptive prefill phases. Tested on a production H800 cluster serving DeepSeek-V3, they demonstrate a 30-40% reduction in TTFT and a ~20% throughput increase backed by clear utilization metrics. The most valuable takeaway for our GPUSched project is their 'IQR-aware lexicographical' scheduling heuristic for the Decode phase, which robustly balances batch size against KV-cache memory variance—a constraint logic we should immediately adopt. This work validates that discrete batching is superior to continuous dispatch for MoE models, necessitating an update to our queuing theory models.

### [OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning](https://arxiv.org/abs/2511.09092)

**2025-11-12** | The Hong Kong University of Science and Technology, Arizona State University, University of North Carolina at Chapel Hill | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Supervised Fine-tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) with a composite reward function | *LLM role:* code_writer, heuristic_generator, evaluator

> OR-R1 introduces a data-efficient framework that fine-tunes Qwen3-8B using Supervised Fine-Tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) on unlabeled data. The results are empirically strong: it outperforms ORLM and LLMOPT while using only 1/10th of the synthetic training data, specifically narrowing the consistency gap between Pass@1 and Pass@8. The key takeaway for us is the effectiveness of GRPO (normalizing rewards within a sampled group to estimate baselines) combined with majority-voting rewards; this eliminates the need for a separate critic model while significantly improving code generation consistency. We should immediately evaluate GRPO as a lightweight alternative to PPO for the 'RL-infused' components of our evolutionary search methods.

### [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115)

**2026-01-15** | Massachusetts Institute of Technology, Microsoft Research, HKUST | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Memory Constrained Shortest First (MC-SF) online batching and scheduling algorithm | *LLM role:* none

> This paper formulates LLM inference scheduling as an Integer Program (IP) that explicitly models the linear memory growth of KV caches, and proposes a 'Memory Constrained Shortest First' (MC-SF) algorithm. The results are rigorous, showing MC-SF achieves near-optimal performance (within 5% of hindsight optimal) on synthetic data and significantly outperforms standard FCFS/threshold heuristics on real traces. The critical takeaway is the 'future feasibility check' (Eq. 5), which validates that a batch will *remain* within memory limits throughout the generation process based on predicted output lengths—a necessary deviation from standard static-size scheduling. This is foundational reading for our GPUSched project, providing both the exact IP baseline we need and a strong heuristic to benchmark against.

### [Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents](https://arxiv.org/abs/2504.07347)

**2025-04-24** | Cornell University, Columbia University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Queueing-theoretic framework with discrete-time Markov chains and fluid limits for analyzing work-conserving scheduling algorithms | *LLM role:* none

> Li et al. formulate a batch queueing model for LLM inference, proving that 'work-conserving' algorithms (like Sarathi-Serve) which mix prefill and decode tokens are throughput-optimal, whereas separated strategies (vanilla vLLM, FasterTransformer) are theoretically unstable. The results are rigorous, combining fluid limit proofs with empirical validation on A100s showing queue blow-ups in non-optimal schedulers. The key takeaway is the precise definition of stability for token-level batching and the counter-intuitive finding that these locally optimal policies can fail in multi-agent networks due to cyclic resource dependencies. This is foundational reading for our GPUSched project and directly informs how we should model resource allocation for our multi-agent optimization systems.

### [Justitia: Fair and Efficient Scheduling for LLM Applications](https://arxiv.org/abs/2510.17015)

**2025-10-19** | Shanghai Jiao Tong University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Virtual-time based fair queuing with memory-centric cost modeling and MLP-based demand prediction | *LLM role:* none

> Justitia introduces a scheduler for LLM agents that prioritizes applications based on their 'virtual finish time' (derived from a theoretical fair-sharing model) but executes them with full resource saturation to minimize completion time. The authors demonstrate a ~60% reduction in average job completion time compared to state-of-the-art fair schedulers (VTC) on vLLM, backed by rigorous experiments and theoretical delay bounds. The key takeaway is the 'KV token-time' cost metric (pd + d^2/2) which accurately captures memory bottlenecks in auto-regressive generation, and the insight that 'long-term fairness' allows for short-term resource saturation. This is immediately actionable for your GPUSched project and relevant for optimizing the serving infrastructure of AlgoEvo.

### [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://arxiv.org/abs/2405.17743)

**2025-04-04** | Columbia University, Duke University, Shanghai Jiao Tong University, The Chinese University of Hong Kong, Shenzhen, Shenzhen Research Institute of Big Data, Shanghai University of Finance and Economics, Cardinal Operations | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Instruction tuning of open-source LLMs using semi-automated synthetic data generated by OR-Instruct framework | *LLM role:* data_synthesis, model_generator, code_writer

> The authors propose OR-Instruct, a framework that uses GPT-4 to synthesize over 32k optimization modeling pairs (natural language to COPT code) to fine-tune 7B-scale models (ORLM). They demonstrate that these fine-tuned models outperform GPT-4 on their new 'IndustryOR' benchmark, a result that appears robust given the specialized nature of the task. The most valuable takeaway is their specific data augmentation strategy—iteratively altering constraints and injecting specific modeling techniques (e.g., Big M)—which provides a concrete recipe we can steal to generate diverse instances for our OR-Bench project. While the methodology is standard instruction tuning, the resulting artifacts (benchmark and model) establish a new baseline for automated OR modeling that we cannot ignore.


### Front 0 (6 papers) — EMERGING

**Density:** 0.53 | **Methods:** mixed_integer_programming, heuristic_search, continuous_batching, stochastic_control, queueing_network | **Problems:** llm_serving_optimization, resource_allocation, llm_inference_scheduling, scheduling, makespan_minimization

*Unique methods:* active_request_capping, adaptive_thresholding, attention_kernels, decode_limit, decode_router, disaggregated_expert_parallelism, dynamic_offset_adjustment, dynamic_rebatching, dynamic_resource_allocation, early_exit_llms, fine_grained_scheduling, first_come_first_serve, fluid_approximation, gate_and_route_policy, gemm, gemv, graph_partitioning, gurobi, heuristic_algorithm_design, heuristic_search, hybrid_optimization, kkt_conditions, kv_caching, lagrangian_heuristic, linear_performance_models, llm_inference_serving, llm_serving_systems, makespan_minimization, many_server_queueing, mathematical_analysis, matrix_multiplication, maximum_likelihood_estimation, mixed_integer_programming, optimal_gemm_tiling, optimization_problem_formulation, ordinary_least_squares, ping_pong_pipeline, preemptive_scheduling, prefill_admission_gate, queueing_network, real_time_tbt_deadline_tracking, resource_aware_dynamic_scheduler, scheduling_strategies, shortest_prefill_first_ordering, sla_aware_scheduling, slo_aware_llm_inference_scheduler, state_copying, stochastic_control, task_scheduling, token_budgeting, virtual_memory_management
*Shared methods:* bi_level_optimization, bin_packing, continuous_batching, convex_optimization, linear_programming, load_balancing, lyapunov_function, online_scheduling, performance_modeling, queueing_theory, scheduling

This research front unifies advanced Operations Research techniques, specifically many-server queueing network models, Mixed-Integer Programming (MIP), and stochastic control, to achieve optimal or near-optimal LLM inference scheduling and resource allocation. Key themes include managing prefill-decode contention, optimizing disaggregated expert parallelism in Mixture-of-Experts (MoE) models, and automatic operator-level parallelism planning for distributed deep learning.

Key contributions include Lin et al.'s (2026) 'Gate-and-Route' policy derived from a fluid LP, demonstrating ~30% lower revenue loss than OPT on Dolly-15k. She et al. (2025) used MIP for operator-level parallelism, reducing pipeline bubbles by 50% for DeepSeek V3. Bari et al. (2025) introduced RAD and SLAI schedulers, achieving a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve on Mistral-7B. Pang et al. (2025) combined offline bin-packing with an online Lagrangian heuristic, improving utilization by 8.86% over vLLM FCFS on LLaMA-65B. FinDEP (2025) optimized MoE inference with fine-grained scheduling, yielding up to 1.61x throughput improvement on Qwen3-235B, while DREX (2025) showed 2-12% throughput gains on Llama-EE-70B using dynamic rebatching for early-exit LLMs.

This front is emerging, characterized by a strong trend towards rigorous mathematical modeling to solve complex LLM serving challenges, moving beyond heuristic-driven approaches. The next papers will likely focus on relaxing simplifying assumptions (e.g., exponential service times), integrating stochastic programming for uncertainty, and extending these optimal control strategies to more heterogeneous and dynamic LLM architectures, such as those with speculative decoding or Mixture-of-Depths.

**Papers:**

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**2025-12-25** | The Hong Kong University of Science and Technology, Harbin Institute of Technology, Hong Kong Baptist University | M=6 P=7 I=5 *discuss*

*Method:* Fine-grained task scheduling algorithm for disaggregated expert parallelism (DEP) with maximal task overlap, guided by linear performance models and analytical properties (monotonicity, convexity) | *LLM role:* none

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize overlap. The authors achieve 1.02x-1.61x speedups on H20/A6000 clusters compared to PPPipe, backed by solid empirical data. The key takeaway for our 'GPUSched' work is their methodology: deriving analytical properties (monotonicity and convexity) of the scheduling objective to reduce a complex search space into an $O(1)$ online solver, rather than relying on heavy solvers or RL. This confirms that simple linear performance models ($\alpha + \beta x$) are sufficient for accurate online resource allocation in LLM serving.

### [Automatic Operator-level Parallelism Planning for Distributed Deep Learning -- A Mixed-Integer Programming Approach](https://arxiv.org/abs/2503.09357)

**2025-03-12** | Huawei | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed-Integer Programming (MIP) formulation with a bi-level solution framework including a heuristic operation merging step | *LLM role:* none

> She et al. formulate distributed LLM training/inference as a Flexible Distributed Job Shop Scheduling Problem (FDJSSP) solved via Mixed-Integer Programming (MIP) combined with a heuristic graph coarsening step. They demonstrate that this automated approach not only reproduces DeepSeek V3's expert-designed "DualPipe" strategy but, when allowed to search longer, discovers a schedule with 50% fewer pipeline bubbles. The primary takeaway is the effectiveness of the bi-level optimization framework (greedy merging + MIP) to handle the scale of operator-level graphs, proving that formal OR methods can outperform manual system design for LLM infrastructure. This is a mandatory read for our GPUSched project, offering a concrete formulation for operator-level constraints we can directly adapt.

### [Optimal Scheduling Algorithms for LLM Inference: Theory and Practice](https://arxiv.org/abs/2508.01002)

**2025-12-01** | The University of Texas at Austin | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* Resource-Aware Dynamic (RAD) scheduler for throughput optimality based on optimal GeMM tiling and dynamic prefill/decode resource allocation; SLO-Aware LLM Inference (SLAI) scheduler for practical SLOs using real-time TBT deadline tracking, SPF prefill ordering, and dynamic offset adjustment based on GPU memory utilization. | *LLM role:* none

> Bari et al. develop a queueing-theoretic framework for LLM inference that proves throughput optimality requires satisfying two conditions: optimal GeMM tiling (batch sizes matching hardware tensor core dimensions) and dynamic resource allocation between prefill/decode phases. They propose RAD (theoretical) and SLAI (practical), where SLAI uses a 'last schedulable time' heuristic to delay decode iterations for non-critical requests, thereby freeing up compute for prefill to reduce TTFT. Results are strong, showing a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve on Mistral-7B. For our GPUSched project, the key takeaway is the explicit coupling of batch sizes to LCM(tile_dims) for theoretical optimality and the dynamic slack-based scheduling logic for heterogeneous SLOs.

### [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)

**2025-02-14** | Noah’s Ark Lab, Huawei, Tsinghua University | M=6 P=10 I=7 **MUST-READ** *discuss*

*Method:* Hybrid offline-online method combining Minimizing Makespan Bin Packing (offline) with sorting, online preemption, and a Lagrangian-based heuristic (online) | *LLM role:* none

> Pang et al. formulate LLM inference scheduling as a Mixed-Integer Programming (MIP) model, solving it via a hybrid approach: offline bin-packing for request assignment and an online Lagrangian heuristic for prefill-decode preemption. They report a ~9% utilization increase (80.2% to 89.1%) over a vLLM-style baseline on LLaMA-65B, though the evaluation is limited to a single 8-GPU node and assumes deterministic output lengths for the offline component. The most actionable takeaway is their derivation of a simple cost-comparison threshold (prefill cost vs. decode wait cost) to dynamically inject prefill tasks into decoding streams. This provides a concrete, low-overhead heuristic baseline for our GPUSched work.

### [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](https://arxiv.org/abs/2512.15705)

**2025-12-17** | Microsoft Research, University of Pennsylvania | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Dynamic Rebatching with copy-free rebatching buffer and SLA-aware scheduler | *LLM role:* inference_target

> DREX introduces a system for 'Early-Exit' LLMs that dynamically splits and regroups batches at intermediate layers, using a cost-benefit heuristic (Adaptive Rebatching Threshold) to decide when rebatching is profitable versus forcing execution. Results are solid (2-12% throughput gain on A100s) and backed by real system measurements, not just simulations. The key takeaway for us is the analytical model for rebatching overhead (Eq. 6)—we can lift this constraint directly into our integer programming formulations for the GPUSched project to accurately model the trade-off between batch fragmentation and compute savings. Essential reading only for the serving optimization sub-team; irrelevant for the core evolutionary search group.


### Front 3 (6 papers) — GROWING

**Density:** 0.47 | **Methods:** llm_in_the_loop, resource_allocation, llm_as_heuristic, rebase, process_reward_model | **Problems:** llm_serving_optimization, llm_inference_optimization, mathematical_reasoning, scientific_reasoning, ensemble_optimization

*Unique methods:* adaptive_index_update, adaptive_sampling, all_to_all_collectives, analytical_modeling, approximate_nearest_neighbor_search, bayes_factor, bayesian_modeling, bayesian_optimization, beam_search, bert_embeddings, best_of_n, beta_distribution_modeling, clustering, cuda_graph, direction_oriented_resource_allocation, dirichlet_process_prior, distributed_inference, diverse_verifier_tree_search, dora, dynamic_dispatching, embedding_model, expert_parallelism, exponential_smoothing, fast_scanning, fluid_model_analysis, gradient_scheduling, grouped_gemm, hierarchical_agglomerative_clustering, inference_optimization, inverted_file_index, kv_cache_optimization, latency_bounded_partitioning, llm_as_answer_generator, llm_as_heuristic, llm_ensemble, llm_inference_optimization, majority_voting, max_margin_optimization, memory_optimization, mixture_of_experts, monte_carlo_methods, nvshmem, online_distillation, performance_estimation, pipelining, predictive_scheduling, process_reward_model, product_quantization, real_time_optimization, rebase, resource_partitioning, retrieval_augmented_generation_serving, reward_balanced_search, semantic_similarity, soft_clustering, straggler_mitigation, system_design, temperature_sampling, tree_search, triton, vector_similarity_search
*Shared methods:* convex_optimization, greedy_algorithm, integer_linear_programming, linear_programming, llm_as_evaluator, llm_in_the_loop, load_balancing, mixed_integer_linear_programming, queuing_theory, resource_allocation, scheduling, speculative_decoding

This research front focuses on applying advanced Operations Research techniques to optimize various aspects of Large Language Model (LLM) inference and serving. Key approaches include Bayesian adaptive stopping for efficient ensemble evaluation, gradient-based scheduling in GoodSpeed for distributed speculative decoding, PROBE's predictive Lookahead Pipelining for MoE inference, and ETS's Integer Linear Programming for KV cache optimization in tree search. Additionally, DORA employs embedding-based resource allocation for test-time search, and VectorLiteRAG uses analytical modeling for RAG serving.

Contributions include Bayesian adaptive sampling (2-5x compute reduction on AIME/GPQA) and MILP for LLM ensembles. GoodSpeed demonstrates gradient-based scheduling for distributed speculative decoding, achieving fair goodput on H100/L4 clusters. PROBE's Lookahead Pipelining yields a 1.3x speedup for Qwen3-MoE-235B inference, while ETS leverages ILP to achieve 1.8x KV cache reduction and 1.4x throughput increase on MATH500 compared to REBASE. VectorLiteRAG provides 1.5x throughput gains for RAG serving on H100/L40S, and DORA achieves state-of-the-art accuracy on MATH500 with 3.5x fewer FLOPs by optimizing test-time search.

This front is rapidly maturing, characterized by the increasing sophistication of OR methods integrated directly into LLM serving and reasoning pipelines. The trajectory points towards more tightly coupled, real-time optimization, where OR solvers dynamically adapt resource allocation and search strategies. Future work will likely focus on developing unified frameworks that combine predictive modeling, adaptive sampling, and advanced combinatorial optimization to handle the stochastic and dynamic nature of LLM workloads across diverse hardware architectures and reasoning tasks.

**Papers:**

### [Best-of-$\infty$ -- Asymptotic Performance of Test-Time Compute](https://arxiv.org/abs/2509.21091)

**2025-09-25** | Mohamed bin Zayed University of Artificial Intelligence, New York University, RIKEN AIP, Institute of Science Tokyo, NEC Corporation | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Adaptive sampling for majority voting using Bayesian modeling (Dirichlet process prior and Bayes factor) to determine stopping criteria, combined with optimally weighted LLM ensembles formulated as a Mixed-Integer Linear Program (MILP) with a max-margin solution. | *LLM role:* answer_generator

> This paper introduces a Bayesian adaptive stopping criterion (using Dirichlet process priors and Bayes factors) for majority voting, reducing test-time compute by 2-5x while maintaining asymptotic 'Best-of-Infinity' accuracy. They further demonstrate that optimizing weights for an ensemble of LLMs can be formulated as a Mixed-Integer Linear Program (MILP) by treating the decision boundaries as polytopes. **What we learned:** The Bayesian stopping logic is immediately transferable to AlgoEvo to reduce the cost of fitness evaluations—we can stop evaluating candidate solutions early if their performance distribution is statistically distinct. The MILP approach for ensembles also offers a concrete formulation we could adapt for our GPU scheduling and model serving optimization work.

### [GoodSpeed: Optimizing Fair Goodput with Adaptive Speculative Decoding in Distributed Edge Inference](https://arxiv.org/abs/2512.09963)

**2025-12-14** | The University of Sydney, Kyung Hee University | M=5 P=7 I=6 *discuss*

*Method:* Gradient-based scheduling algorithm maximizing logarithmic utility for proportional fairness with adaptive speculative decoding | *LLM role:* heuristic_generator, evaluator

> GoodSpeed uses gradient-based scheduling to dynamically allocate token generation budgets across distributed draft servers, maximizing a logarithmic utility function to balance throughput and fairness. The authors provide rigorous fluid sample path analysis to prove convergence, backed by experiments on H100/L4 clusters, although the baselines (fixed/random allocation) are relatively weak. The most useful takeaway is the mechanism of using exponentially smoothed acceptance rate estimates to drive real-time control in a stochastic system—a robust pattern we should adopt for our own stochastic resource allocation and RobustMAS projects.

### [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509)

**2026-02-03** | Kling Infra, Kuaishou Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Continuous Lookahead Pipelining with Gate-Initialized Lookahead Predictor, Hardware-Aware Balance Planning, and Phase-Locked Co-Scheduling | *LLM role:* none

> PROBE optimizes MoE inference by using a distilled MLP to predict next-layer expert activation, enabling proactive load balancing and weight prefetching hidden behind the current layer's computation. The results are strong (1.3x speedup on 235B models) and demonstrate that control plane overheads can be fully masked. The critical takeaway for our `GPUSched` project is the **Lookahead Pipelining** architecture: it carves out a deterministic execution window where we could inject our own specialized solvers (e.g., fast ALNS or IP formulations) to outperform their basic greedy resource allocator. This transforms the stochastic serving problem into a short-horizon deterministic routing problem we are well-equipped to solve.

### [ETS: Efficient Tree Search for Inference-Time Scaling](https://arxiv.org/abs/2502.13575)

**2025-06-11** | University of California, Berkeley, ICSI, LBNL | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Efficient Tree Search (ETS) using a linear programming cost model with KV cache sharing penalty and semantic coverage term | *LLM role:* candidate_generator, process_reward_model, search_guidance

> ETS formulates the tree search pruning step as a lightweight Integer Linear Program (ILP) that maximizes the reward of retained nodes while penalizing total KV cache size and enforcing semantic diversity via clustering. Unlike standard beam search or REBASE, it explicitly optimizes the trade-off between memory consumption (KV sharing) and exploration coverage. The authors demonstrate a 1.8x reduction in KV cache size and 1.4x throughput increase on MATH500 with minimal accuracy loss. We should steal the 'ILP-in-the-loop' mechanism for population management in our evolutionary search frameworks to optimize hardware utilization dynamically.

### [VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG](https://arxiv.org/abs/2504.08930)

**2026-01-19** | Georgia Institute of Technology | M=5 P=7 I=6 *discuss*

*Method:* Analytical performance modeling and latency-bounded partitioning algorithm for hybrid CPU-GPU vector index, combined with a distributed runtime pipeline featuring query- and shard-aware routing and dynamic dispatcher. | *LLM role:* target_of_optimization

> VectorLiteRAG optimizes RAG serving throughput by dynamically partitioning vector indices between CPU and GPU memory based on access skew and latency SLOs. The results are credible, showing up to 1.5x throughput gains on H100/L40S setups by balancing retrieval speed against LLM KV-cache capacity. The most stealable insight is their use of a Beta distribution to analytically model the *minimum* hit rate within a batch (the bottleneck) to predict tail latency without full simulation—a technique we could adapt for stochastic constraints in our serving formulations. It solves a resource allocation problem we care about, though via systems engineering rather than the rigorous OR methods we prefer.

### [Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling](https://arxiv.org/abs/2506.15707)

**2025-10-20** | Beijing Institute of Technology, Xiaohongshu Inc | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Direction-Oriented Resource Allocation (DORA) | *LLM role:* reasoning_path_generator

> Wang et al. introduce Direction-Oriented Resource Allocation (DORA), which uses embedding-based soft clustering to group semantically similar reasoning paths and allocates compute budget to distinct 'directions' rather than individual solutions. They prove solution-level allocation (like REBASE) is suboptimal when paths are correlated and show DORA achieves state-of-the-art accuracy on MATH500 with 3.5x fewer FLOPs. **Key Takeaway:** We can immediately steal the 'semantic uniqueness reweighting' mechanism for AlgoEvo. By clustering generated heuristics via embeddings before expensive evaluation, we can drastically improve sample efficiency and stop wasting compute on minor variations of the same code.


### Front 32 (3 papers) — GROWING

**Density:** 1.00 | **Methods:** multi_objective_optimization, convex_optimization, gradient_descent, game_theory, llm_as_evaluator | **Problems:** multi_objective_llm_alignment, sentiment_control, text_length_control, humor_generation, harmlessness_control

*Unique methods:* adaptation_safety, best_of_k_sampling, black_box_optimization, blockwise_decoding, controlled_decoding, equilibrium_search, game_theory, gradient_aggregation, gradient_descent, inference_time_alignment, llm_alignment, llm_fine_tuning, maximin_optimization, multi_objective_optimization, noon_ppo, optimization_penalty_function, ppo, reward_modeling, rlhf, value_function_learning, zero_sum_game
*Shared methods:* convex_optimization, linear_programming, llm_as_evaluator, llm_in_the_loop, robust_optimization, supervised_fine_tuning

This research front unifies recent advancements in applying Operations Research techniques, specifically convex optimization and game theory, to the challenging problem of multi-objective alignment for Large Language Models (LLMs). Papers introduce frameworks like PAMA, Safety Game, and Robust Multi-Objective Decoding (RMOD) to manage conflicting objectives such as harmlessness, helpfulness, sentiment, and length control, often at inference time.

Key contributions include the PAMA algorithm, which transforms multi-objective RLHF into an O(n) convex optimization problem with a closed-form solution, outperforming MORLHF and MGDA-UB on LLaMA-2 7B for harmlessness. The Safety Game formulates black-box LLM agent alignment as a zero-sum game solvable by an LP solver at inference, achieving up to two-fold accuracy improvement on SafetyBench. RMOD introduces a maximin two-player game for robust multi-objective decoding, solving a convex optimization problem at each step to maximize worst-case value, outperforming MO-DPO and scalarized baselines by +1.2% WCWR on Anthropic HH.

This front is rapidly growing, demonstrating the power of OR principles to bring robustness and efficiency to LLM alignment. The trajectory indicates a strong focus on mathematically grounded, inference-time control mechanisms. Future work will likely focus on extending these frameworks to more complex, dynamic, and multi-agent scenarios, improving their scalability to a greater number of objectives, and integrating these control mechanisms into broader agentic architectures.

**Papers:**

### [Pareto Multi-Objective Alignment for Language Models](https://arxiv.org/abs/2508.07768)

**2025-08-11** | Ruhr University Bochum | M=7 P=5 I=6 *discuss*

*Method:* PAMA (PAreto Multi-Objective Alignment) algorithm, which transforms multi-objective RLHF into a convex optimization problem with a closed-form solution, combined with Noon PPO. | *LLM role:* subject_of_optimization

> PAMA introduces a computationally efficient algorithm for multi-objective alignment by reformulating the expensive gradient-norm minimization of MGDA into a convex optimization problem with a closed-form solution, reducing complexity from O(n^2d) to O(n). Empirical results on LLaMA-2-7B are robust, showing stable convergence on conflicting objectives (e.g., harmlessness vs. length) where baselines like MGDA-UB oscillate or fail. The single most useful takeaway is the analytical derivation for optimal objective weighting (Theorem 1) and the 'Noon PPO' heuristic (clipping negative advantages); we could port this logic to our multi-objective process reward models in AlgoEvo to balance search signals efficiently. While the NLP experiments are trivial, the gradient balancing mechanism is directly applicable to our multi-objective RL controllers.

### [Robust Multi-Objective Controlled Decoding of Large Language Models](https://arxiv.org/abs/2503.08796)

**2025-03-11** | University College London, University of Basel, Ulsan National Institute of Science and Technology | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Maximin two-player game between adversarially computed reward weights and sampling policy, solvable through Nash equilibrium, reduced to convex optimization, with blockwise best-of-K sampling | *LLM role:* controlled_decoding_target

> RMOD formulates multi-objective decoding as a zero-sum game between a policy and adversarial weights, solving a convex optimization problem at each decoding step to maximize the worst-case value estimate (essentially a Process Reward Model). The results are empirically strong, outperforming MO-DPO and scalarized baselines on alignment benchmarks by dynamically preventing any single objective from collapsing. **Key Takeaway:** The efficient inference-time weight optimization algorithm (Eq. 10) is a 'stealable' mechanism for **AlgoEvo** and **RobustMAS**. We should implement this dynamic adversarial weighting to balance conflicting code metrics (e.g., runtime vs. solution quality) during evolutionary search, replacing our current static scalarization methods.

### [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](https://arxiv.org/abs/2510.09330)

**2025-12-02** | University of Warwick | M=7 P=4 I=7 *discuss*

*Method:* Two-player zero-sum game formulation solved by a linear programming (LP) solver at inference time to compute minimax equilibrium strategies, using binary probes for helpfulness and safety scores, with a sigmoid penalty for risk. | *LLM role:* agent_response_selection, evaluator

> The authors formulate LLM response selection as a zero-sum game, solving a small Linear Program (LP) at inference time to mix candidate answers such that the expected risk never exceeds a 'safe fallback' baseline. Results are statistically significant, showing ~15% accuracy gains on SafetyBench by effectively managing the trade-off between helpfulness and safety probes. The key takeaway is the 'Adaptation Safety' constraint formulation: using an LP to guarantee that a stochastic policy is no worse than a heuristic baseline is a powerful, lightweight control mechanism we could adapt for selecting evolved algorithms or managing constraints in multi-agent optimization.


### Front 34 (2 papers) — EMERGING

**Density:** 1.00 | **Methods:** post_training_quantization, mixed_precision_quantization, linear_programming, gptq, expert_quantization | **Problems:** llm_compression, memory_optimization, inference_efficiency, mixture_of_experts_compression, vlm_compression

*Unique methods:* binary_quantization, dynamic_expert_pruning, dynamic_pruning, expert_pruning, expert_quantization, gptq, gumbel_softmax, hqq, learnable_mask, mixed_precision_quantization, model_compression, moe_llm_compression, post_training_quantization, pruning, token_pruning
*Shared methods:* integer_programming, linear_programming

This research front centers on the Mixture Compressor (MC and MC#) frameworks, which leverage Operations Research techniques, specifically Linear Programming, for the extreme compression of Mixture-of-Experts (MoE) Large Language Models. The core theme involves optimizing mixed-precision quantization and dynamic expert pruning to significantly reduce model size while preserving performance, enabling deployment on resource-constrained hardware.

The initial paper, "Mixture Compressor for Mixture-of-Experts LLMs Gains More" (Huang et al., 2025-02), introduces a hybrid post-training quantization and dynamic pruning approach. It uses Linear Programming to optimally allocate mixed bit-widths (1-3 bits) to experts based on activation frequency, achieving strong empirical results such as compressing Mixtral 8x7b to ~16GB with only a ~4% drop in zero-shot accuracy, outperforming uniform GPTQ. Building on this, "MC#: Mixture Compressor for Mixture-of-Experts Large Models" (Huang et al., 2025-10) refines the framework by combining Pre-Loading Mixed-Precision Quantization (PMQ) via Linear Programming with Online Top-any Pruning (OTP) using Gumbel-Softmax sampling. This unified approach achieves a 6.2x weight reduction on DeepSeek-VL2 with less than 2% accuracy loss, further demonstrating the efficacy of OR-driven compression.

This front is currently emerging, with two closely related papers published in 2025, the second building directly on the first. The trajectory suggests a continued focus on refining these OR-driven compression techniques. The likely next paper would explore the adaptation of these methods to new model architectures or more challenging, complex reasoning tasks, while also addressing hardware-specific optimizations.

**Papers:**

### [Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270)

**2025-02-22** | The University of Hong Kong, The Chinese University of Hong Kong, Beihang University, Centre for Perceptual and Interactive Intelligence, Hong Kong | M=5 P=7 I=6 *discuss*

*Method:* Hybrid Post-Training Quantization and Dynamic Pruning for MoE-LLMs using Linear Programming for bit-width allocation and significance-aware token protection | *LLM role:* none

> Huang et al. propose a compression framework for MoE-LLMs that uses Integer Programming to optimally allocate mixed bit-widths (1-3 bits) to experts based on activation frequency and routing weights. They achieve strong empirical results, compressing Mixtral 8x7b to ~16GB (fitting on a single RTX 3090) with only a ~4% drop in zero-shot accuracy, significantly outperforming uniform quantization. The key takeaway is the explicit IP formulation for minimizing quantization error under memory constraints—a clean 'OR for AI' pattern we can adapt for our GPU scheduling or memory allocation formulations. While not a methodological advance in evolution, this is highly relevant for our infrastructure: it enables deploying high-quality MoE models on cheaper hardware for our massive AlgoEvo loops.

### [MC#: Mixture Compressor for Mixture-of-Experts Large Models](https://arxiv.org/abs/2510.10962)

**2025-10-13** | NVIDIA Research, National University of Singapore, The University of Hong Kong, Beihang University, Hangzhou Innovation Institute | M=6 P=7 I=7 *discuss*

*Method:* Hybrid compression combining Pre-Loading Mixed-Precision Quantization (PMQ) via Linear Programming and Online Top-any Pruning (OTP) via Gumbel-Softmax sampling | *LLM role:* none

> Huang et al. propose MC#, a compression framework for MoE models that combines static mixed-precision quantization with dynamic expert pruning. They formulate bit-width allocation as an Integer Linear Programming (ILP) problem—optimizing expert importance vs. quantization error—and use a Gumbel-Softmax router for dynamic pruning. Results are strong, achieving 6.2x weight reduction on DeepSeek-VL2 with <2% accuracy loss. **Takeaway:** The ILP formulation (Eq. 7) is a clean, successful application of OR to AI infrastructure that we should replicate for our own resource allocation/scheduling problems; additionally, the differentiable router offers a template for dynamic agent selection in our multi-agent systems.



## Bridge Papers

Papers connecting multiple research fronts:

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**TRUE SYNTHESIS** | score=0.55 | Front 28 → Front 14

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize o


---

*Generated by Research Intelligence System*
