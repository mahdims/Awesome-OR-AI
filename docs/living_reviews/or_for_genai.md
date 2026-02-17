# Living Review: OR for Generative AI

**Last Updated:** 2026-02-17

---

## Recent Papers

<!-- New papers are appended here by the daily updater -->

---

## Research Fronts

*1 fronts detected — snapshot 2026-02-17*

### Front 18 (4 papers) — NEW

**Density:** 0.67 | **Methods:** bi_level_optimization, mixed_integer_linear_programming, chebyshev_guided_optimization, data_parallelism, tensor_parallelism | **Problems:** llm_serving_optimization, resource_allocation, latency_quality_tradeoff, gpu_scheduling, multi_model_inference

*Unique methods:* bi_level_optimization, chebyshev_guided_optimization, cvxpy, data_parallelism, decode_router, decomposition_algorithm, dynamic_scheduling, fluid_approximation, gate_and_route_policy, greedy_algorithm, integer_linear_programming, kkt_conditions, linear_programming, llm_as_evaluator, load_balancing, lyapunov_function, many_server_queueing, maximum_likelihood_estimation, milp_formulation, mixed_integer_linear_programming, network_topology_modeling, np_hardness_proof, optimization, ordinary_least_squares, performance_modeling, pipeline_parallelism, prefill_admission_gate, queueing_network, resource_allocation_optimization, robust_optimization, shortest_path_algorithms, shortest_path_routing, simulation, stochastic_control, system_algorithm_co_design, tensor_parallelism, threshold_based_routing

This research front focuses on applying advanced Operations Research (OR) techniques, including bi-level optimization, Integer Linear Programming (ILP), and stochastic control, to optimize various aspects of large-scale Large Language Model (LLM) serving. The papers address critical challenges such as cascade serving, resource allocation for geographically-distributed inference, efficient placement of Mixture-of-Experts (MoE) layers, and managing prefill-decode contention under heterogeneous workloads.

Key contributions include Cascadia's bi-level optimization framework for LLM cascade serving, achieving up to 2.3x average throughput gains over existing systems on H100 clusters. For distributed LLM inference, a decomposed MILP heuristic demonstrated 60-80% latency reduction over PETALS' native methods on A100s by dynamically modeling attention cache. ILP-based placement of MoE experts, weighted by historical activation frequency, reduced network hops by up to 39.1%. Furthermore, a stochastic control approach with LP-based gate-and-route policies provided asymptotically optimal solutions for prefill-decode contention, maximizing revenue by effectively managing bottlenecks.

This is an emerging research front, rapidly applying established OR methodologies to the complex, dynamic environment of LLM inference. The trajectory indicates a strong push towards more rigorous mathematical modeling and optimization for real-world LLM deployment. Future work will likely involve integrating more sophisticated OR techniques with deeper understanding of LLM-specific architectural and workload characteristics, moving towards comprehensive, adaptive, and provably optimal serving systems.

**Papers:**

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.



## Bridge Papers

<!-- Cross-front connectors, updated weekly -->

---

*Generated by Research Intelligence System*
