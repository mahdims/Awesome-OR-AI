# Living Review: OR for Generative AI

**Last Updated:** 2026-02-22

---

## Recent Papers

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

<!-- Updated weekly by the revision agent -->

---

## Bridge Papers

<!-- Cross-front connectors, updated weekly -->

---

*Generated by Research Intelligence System*
