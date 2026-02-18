# Living Review: OR for Generative AI

**Last Updated:** 2026-02-17

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

*4 fronts detected — snapshot 2026-02-17*

### Front 9 (6 papers) — NEW

**Density:** 0.87 | **Methods:** integer_linear_programming, performance_modeling, data_parallelism, adaptive_scheduling, resource_management | **Problems:** llm_serving_optimization, resource_allocation, gpu_scheduling, llm_scheduling, service_level_objective_optimization

*Unique methods:* adaptive_scheduling, arima_time_series_forecasting, bi_level_optimization, chebyshev_guided_optimization, continuous_batching, cvxpy, data_parallelism, decode_router, decomposition_algorithm, discrete_event_simulation, distributed_systems, dynamic_scheduling, fluid_approximation, gate_and_route_policy, integer_linear_programming, kkt_conditions, lyapunov_function, many_server_queueing, maximum_likelihood_estimation, milp_formulation, model_parallelism, network_communication_optimization, network_topology_modeling, np_hardness_proof, optimization, ordinary_least_squares, performance_modeling, prefill_admission_gate, queueing_network, reactive_heuristics, resource_allocation_optimization, resource_management, shortest_path_algorithms, shortest_path_routing, simulation, stochastic_control, system_algorithm_co_design, threshold_based_routing
*Shared methods:* greedy_algorithm, linear_programming, llm_as_evaluator, load_balancing, mixed_integer_linear_programming, pipeline_parallelism, queueing_theory, robust_optimization, tensor_parallelism

This research front unifies the application of Operations Research (OR) techniques, primarily Integer Linear Programming (ILP), Mixed Integer Linear Programming (MILP), and queueing theory with stochastic control, to optimize resource allocation and scheduling in complex Large Language Model (LLM) serving environments. These advanced environments include disaggregated serving frameworks like Dynamo, geographically distributed inference systems such as Petals, specialized placement for Mixture-of-Experts (MoE) models, cloud data centers with forecast-aware auto-scaling (SageServe), and bi-level optimized cascade serving (Cascadia). The core theme is leveraging formal OR models to systematically address the NP-hard optimization challenges inherent in these sophisticated LLM serving architectures, moving beyond heuristic approaches.

Key contributions demonstrate significant quantitative improvements across various facets of LLM serving. Dynamo achieved 67-340% average SLO attainment improvement through ILP-based offline deployment and adaptive routing. Lin et al. developed a multiclass many-server queueing network model with a fluid approximation and LP-based 'Gate-and-Route' policy, proving effective for prefill-decode contention. The Petals framework saw 60-80% latency reduction using a decomposed MILP heuristic for joint block placement and request routing. For MoE inference, ILP-based expert placement reduced network hops by up to 39.1% with a load-aware objective. SageServe, combining ILP with ARIMA forecasting, delivered 25% GPU-hours savings and $2.5M/month savings on real Microsoft O365 traces. Finally, Cascadia's bi-level optimization (MILP + Chebyshev) yielded 2.3x average throughput gains for cascade serving. These results underscore the efficacy of OR in enhancing LLM inference performance and cost-efficiency.

This front is rapidly emerging, driven by the escalating scale and complexity of LLM serving infrastructure. The consistent and successful application of ILP/MILP and queueing theory across diverse problem settings indicates a maturing methodology for applying OR to this domain. The trajectory suggests that future research will focus on integrating these OR solutions into more comprehensive, dynamic, and adaptive systems, addressing the limitations of current static or heuristic approaches. Expect to see more sophisticated modeling of real-world constraints, such as heterogeneous hardware, dynamic and multi-tenant workloads, and multi-objective optimization, as the field progresses towards fully autonomous and optimized LLM serving platforms.

**Papers:**

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**2025-11-12** | Microsoft, University of Illinois Urbana-Champaign, Georgia Institute of Technology, Indian Institute of Science | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Integer Linear Programming (ILP) for resource allocation, combined with ARIMA-based time-series forecasting and reactive heuristics for dynamic scaling and scheduling | *LLM role:* none

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interactive and non-interactive workloads. They validate this on real Microsoft O365 production traces (which they release), demonstrating a 25% reduction in GPU hours and $2.5M/month savings compared to reactive baselines. The primary value for us is the release of the production workload traces—allowing us to benchmark our 'GPUSched' formulations against real-world data rather than synthetic distributions—and their specific ILP formulation for unified capacity management, which directly competes with our internal OR models.

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.


### Front 6 (3 papers) — NEW

**Density:** 0.67 | **Methods:** scheduling_algorithms, batch_scheduling, adaptive_control, state_synchronization, greedy_algorithm | **Problems:** llm_serving_optimization, llm_inference_scheduling, distributed_scheduling, time_to_first_token_optimization, throughput_optimization

*Unique methods:* adaptive_control, asymptotic_analysis, batch_scheduling, bin_packing, binomial_thinning, doobs_inequality, dynamic_batching, fluid_dynamics_approximation, heuristic_initialization, kingmans_bound, kv_cache_management, lexicographical_optimization, lindley_recursion, martingale_theory, max_flow_optimization, milp, milp_acceleration, nested_wait_algorithm, online_scheduling, outlier_detection, queuing_theory, scheduling_algorithms, state_synchronization, threshold_based_scheduling, union_bound, wait_algorithm, weighted_round_robin
*Shared methods:* greedy_algorithm, load_balancing, pipeline_parallelism, queueing_theory, resource_allocation, tensor_parallelism

This research front unifies advanced operations research techniques for optimizing large language model (LLM) inference and serving. It specifically explores Staggered Batch Scheduling (SBS) for co-optimizing time-to-first-token (TTFT) and throughput in DP+EP architectures, max-flow problem formulations for distributed LLM serving over heterogeneous GPUs and networks, and fluid dynamics approximation with threshold-based online scheduling for memory-constrained environments. The common thread is the application of rigorous OR methods to address the complex resource allocation and scheduling challenges inherent in modern LLM serving.

Key contributions include Staggered Batch Scheduling (SBS) which reduces TTFT by 30-40% and improves overall throughput by 15-20% on DeepSeek-V3 workloads by enforcing buffering windows and using IQR-aware lexicographical decode scheduling. Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow MILP, achieving up to 3.3x decode throughput over Swarm on LLaMA 70B by dynamically routing requests based on edge capacities. Furthermore, the 'Nested WAIT' algorithm, based on fluid dynamics approximation, provides asymptotic optimality proofs and high-probability bounds against memory overflow for online LLM scheduling, outperforming heuristic baselines like vLLM and Sarathi.

This front is emerging, driven by the increasing scale and complexity of LLM serving infrastructure. The papers demonstrate a clear shift towards more rigorous, theoretically-backed OR formulations over purely heuristic approaches. The trajectory suggests that future work will focus on integrating these diverse advanced scheduling techniques, scaling MILP solutions for larger clusters, and adapting fluid-guided methods to multi-GPU and highly dynamic environments, while also developing analytical frameworks for up-to-date LLM inference models.

**Papers:**

### [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](https://arxiv.org/abs/2512.16134)

**2025-12-18** | Baidu Inc. | M=6 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Staggered Batch Scheduling (SBS) with Throughput-Adaptive Interval Control, Multi-tier State Synchronization, Prioritized Batch Allocation Algorithm (PBAA) for Prefill, and IQR-Aware Lexicographical Decode Scheduling for Decode | *LLM role:* none

> Tian et al. introduce Staggered Batch Scheduling (SBS) for DP+EP architectures, enforcing a buffering window to enable global bin-packing rather than immediate dispatch, which they prove causes Head-of-Line blocking in non-preemptive prefill phases. Tested on a production H800 cluster serving DeepSeek-V3, they demonstrate a 30-40% reduction in TTFT and a ~20% throughput increase backed by clear utilization metrics. The most valuable takeaway for our GPUSched project is their 'IQR-aware lexicographical' scheduling heuristic for the Decode phase, which robustly balances batch size against KV-cache memory variance—a constraint logic we should immediately adopt. This work validates that discrete batching is superior to continuous dispatch for MoE models, necessitating an update to our queuing theory models.

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**2025-03-05** | Carnegie Mellon University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Max-flow problem formulation on directed, weighted graphs with Mixed Integer Linear Programming (MILP) for joint model placement and per-request pipeline scheduling | *LLM role:* none

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow solution. Unlike standard static pipeline parallelism, it routes every request dynamically based on edge capacities, achieving up to 3.3x throughput gains over Swarm on mixed GPU clusters (L4/T4/A100). The results are rigorous, backed by both physical cluster experiments and high-fidelity simulations. The critical takeaway is the 'per-request pipeline' abstraction: decoupling request routing from static device assignment allows exact OR methods to maximize utilization of weaker hardware—a technique we should immediately evaluate for our GPUSched project.

### [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

**2026-01-05** | Massachusetts Institute of Technology, Peking University, Alibaba Group | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Fluid dynamics approximation and threshold-based online scheduling (WAIT and Nested WAIT algorithms) | *LLM role:* none

> This paper formulates LLM inference as a multi-stage stochastic scheduling problem, introducing 'Nested WAIT'—a threshold-based algorithm that handles unknown output lengths by letting prompts classify themselves as they survive into deeper decode segments. Unlike heuristic baselines (vLLM, Sarathi), they provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, validated on A100 simulations. The key takeaway is the 'nested segment' mechanism: instead of predicting job size, structure the queue so short jobs exit early and long jobs naturally migrate to lower-priority/protected tiers, effectively decoupling the memory risk. We should immediately evaluate this threshold logic for our GPUSched formulations, as it likely outperforms our current predictive or FCFS approaches for handling KV cache growth.


### Front 2 (2 papers) — NEW

**Density:** 1.00 | **Methods:** game_theory, llm_as_evaluator, controlled_decoding, multi_objective_optimization, robust_optimization | **Problems:** llm_alignment, instruction_following, helpfulness, safety, truthfulness

*Unique methods:* adaptation_safety, best_of_k_sampling, black_box_optimization, blockwise_decoding, controlled_decoding, convex_optimization, equilibrium_search, game_theory, gradient_descent, inference_time_alignment, maximin_optimization, multi_objective_optimization, optimization_penalty_function, reward_modeling, supervised_fine_tuning, value_function_learning, zero_sum_game
*Shared methods:* linear_programming, llm_as_evaluator, llm_in_the_loop, robust_optimization

This research front explores the application of game theory and robust optimization to achieve inference-time alignment and controlled decoding for Large Language Models (LLMs). Specifically, it encompasses frameworks like Robust Multi-Objective Decoding (RMOD) and the Safety Game, which formulate LLM decision-making as zero-sum games to balance conflicting objectives such as helpfulness, safety, and truthfulness.

Key contributions include the Robust Multi-Objective Decoding (RMOD) framework, which formulates multi-objective decoding as a zero-sum game solvable via convex optimization, demonstrating gains of +1.2% WCWR on LLM-as-Judge and +0.17 worst-case reward on HH benchmarks. Concurrently, the Safety Game introduces a novel game-theoretic formulation for black-box LLM agent alignment, employing an LP solver at inference time to balance safety and helpfulness. This approach achieved up to a two-fold improvement in accuracy on SafetyBench, outperforming baselines in 11 of 15 test cases by guaranteeing that stochastic policies are no worse than a safe fallback.

This front is emerging, showcasing novel applications of operations research techniques like game theory and robust optimization to critical LLM challenges at inference time. The trajectory suggests a move towards more sophisticated, dynamic control mechanisms for LLM behavior. The next likely paper will focus on extending these robust game-theoretic frameworks to handle sequential dialogues, a larger number of objectives, or dynamically tune trade-off parameters in real-time.

**Papers:**

### [Robust Multi-Objective Controlled Decoding of Large Language Models](https://arxiv.org/abs/2503.08796)

**2025-03-11** | University College London, University of Basel, Ulsan National Institute of Science and Technology | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Maximin two-player game between adversarially computed reward weights and sampling policy, solvable through Nash equilibrium, reduced to convex optimization, with blockwise best-of-K sampling | *LLM role:* controlled_decoding_target

> RMOD formulates multi-objective decoding as a zero-sum game between a policy and adversarial weights, solving a convex optimization problem at each decoding step to maximize the worst-case value estimate (essentially a Process Reward Model). The results are empirically strong, outperforming MO-DPO and scalarized baselines on alignment benchmarks by dynamically preventing any single objective from collapsing. **Key Takeaway:** The efficient inference-time weight optimization algorithm (Eq. 10) is a 'stealable' mechanism for **AlgoEvo** and **RobustMAS**. We should implement this dynamic adversarial weighting to balance conflicting code metrics (e.g., runtime vs. solution quality) during evolutionary search, replacing our current static scalarization methods.

### [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](https://arxiv.org/abs/2510.09330)

**2025-12-02** | University of Warwick | M=7 P=4 I=7 *discuss*

*Method:* Two-player zero-sum game formulation solved by a linear programming (LP) solver at inference time to compute minimax equilibrium strategies, using binary probes for helpfulness and safety scores, with a sigmoid penalty for risk. | *LLM role:* agent_response_selection, evaluator

> The authors formulate LLM response selection as a zero-sum game, solving a small Linear Program (LP) at inference time to mix candidate answers such that the expected risk never exceeds a 'safe fallback' baseline. Results are statistically significant, showing ~15% accuracy gains on SafetyBench by effectively managing the trade-off between helpfulness and safety probes. The key takeaway is the 'Adaptation Safety' constraint formulation: using an LP to guarantee that a stochastic policy is no worse than a heuristic baseline is a powerful, lightweight control mechanism we could adapt for selecting evolved algorithms or managing constraints in multi-agent optimization.


### Front 7 (2 papers) — NEW

**Density:** 1.00 | **Methods:** best_of_n, majority_voting, adaptive_sampling, bayesian_modeling, dirichlet_process_prior | **Problems:** llm_serving_optimization, llm_inference_optimization, mathematical_reasoning, scientific_reasoning, ensemble_optimization

*Unique methods:* adaptive_sampling, all_to_all_collectives, bayes_factor, bayesian_modeling, best_of_n, cuda_graph, dirichlet_process_prior, expert_parallelism, grouped_gemm, llm_as_answer_generator, llm_ensemble, llm_inference_optimization, majority_voting, max_margin_optimization, mixture_of_experts, monte_carlo_methods, nvshmem, online_distillation, pipelining, predictive_scheduling, real_time_optimization, straggler_mitigation, system_design, triton
*Shared methods:* greedy_algorithm, llm_in_the_loop, load_balancing, mixed_integer_linear_programming, resource_allocation

This research front explores advanced optimization techniques for Large Language Model (LLM) inference, focusing on enhancing efficiency and throughput. One key direction involves applying Bayesian adaptive sampling for majority voting in LLM ensembles, aiming to reduce test-time compute while maintaining accuracy. Concurrently, another significant contribution introduces predictive prefetching and dynamic load balancing for Mixture-of-Experts (MoE) models, specifically through the PROBE framework, to optimize inference performance.

Paper [1] introduces a Bayesian adaptive stopping criterion for majority voting, leveraging Dirichlet process priors and Bayes factors to reduce test-time compute by 2-5x for LLM reasoning tasks on benchmarks like AIME2025. It also formulates LLM ensemble optimization as a Mixed-Integer Linear Program (MILP), showing superior performance (e.g., 93.3% accuracy vs 90.0% for single LLMs). Paper [2] presents PROBE, a framework for MoE inference optimization that employs Continuous Lookahead Pipelining and predictive prefetching. PROBE demonstrates significant performance gains, achieving a 1.32x speedup in prefill latency with SGLang and 1.26x higher decoding throughput compared to DeepSeek-EPLB on large MoE models.

This front is clearly emerging, showcasing novel applications of Operations Research and AI techniques to critical LLM inference challenges. The trajectory suggests a focus on integrating sophisticated decision-making into LLM serving, moving beyond static configurations. Future work will likely involve refining adaptive sampling mechanisms, extending MILP formulations for dynamic ensemble management, and developing more advanced, specialized solvers within predictive pipelining architectures to further optimize resource utilization and throughput.

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

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**TRUE SYNTHESIS** | score=0.52 | Front 6 → Front 9

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow


---

*Generated by Research Intelligence System*
