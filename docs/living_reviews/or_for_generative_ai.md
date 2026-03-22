# Living Review: OR for Generative AI

**Last Updated:** 2026-03-22

---

## Recent Papers

#### 2026-03-22 (5 papers)

### [MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild](https://arxiv.org/abs/2603.17187)

**2026-03-17** | Carnegie Mellon University, UC Berkeley, UNC-Chapel Hill, UC Santa Cruz | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Continual meta-learning with LLM-based gradient-free skill evolution and RL-based LoRA fine-tuning using a process reward model | *LLM role:* policy_executor, skill_generator, reward_model

> MetaClaw is a continual learning framework for LLM agents that combines gradient-free skill evolution (distilling failures into reusable prompt instructions) with asynchronous RL fine-tuning guided by a process reward model. The results are backed by solid empirical gains, showing an 8.25x improvement in end-to-end task completion on a 30-day simulated CLI benchmark. The single most useful takeaway for us is their 'Skill Generation Versioning' mechanism: when co-optimizing discrete prompts/heuristics and continuous model weights, you must strictly separate support data (used to evolve the heuristic) from query data (collected after the update) and flush the RL buffer upon evolution to prevent training on stale rewards. This directly addresses the continuous learning and RL-infused evolution bottlenecks in AlgoEvo, giving us a concrete trick to steal for managing our own RL buffers during evolutionary search.

### [Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective](https://arxiv.org/abs/2603.16104)

**2026-03-17** | National University of Singapore | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* Workflow-aware LLM serving framework integrating proactive KV cache management, global prompt caching, and cost-based cache-aware scheduling based on a templated radix tree | *LLM role:* none

> Helium optimizes LLM serving for batch agentic workflows by modeling them as query plans and using a Templated Radix Tree (TRT) to enable proactive KV caching and cache-aware scheduling. The results are rigorously backed by numbers, demonstrating up to 1.56x speedups over state-of-the-art systems (vLLM, Parrot, KVFlow) on complex multi-agent workflows, and the authors even validate their greedy scheduler's optimality gap against an MILP solver. The most valuable takeaway is the TRT abstraction, which captures global prefix hierarchies across a DAG of LLM calls to maximize KV cache reuse, rather than relying on reactive, per-call caching. This is highly actionable for us: we should directly examine their scheduling formulation and TRT implementation to improve resource allocation and memory management in our GPUSched and HERMES projects.

### [IEMAS: An Incentive-Efficiency Routing Framework for Open Agentic Web Ecosystems](https://arxiv.org/abs/2603.17302)

**2026-03-18** | Shanghai Jiao Tong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* VCG-based Min-Cost Max-Flow (MCMF) for bipartite matching, guided by Hoeffding Tree predictive QoS models | *LLM role:* none

> IEMAS routes client requests to distributed LLM agents by formulating the assignment as a Min-Cost Max-Flow bipartite matching problem, using VCG auctions to align economic incentives with KV-cache reuse. The results are backed by solid vLLM simulations, demonstrating an 80.2% KV-cache hit rate and a 35% cost reduction over baselines like GraphRouter. The most actionable takeaway for us is their method of quantifying KV-cache affinity (via Longest Common Prefix) and embedding it directly as a cost-reduction weight in a network flow optimization model. While the decentralized VCG auction mechanics might be overkill for centralized clusters, we should absolutely steal their cache-aware MCMF formulation for our OR-based LLM inference scheduling (GPUSched) work.

### [inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference](https://arxiv.org/abs/2603.16054)

**2026-03-17** | MBZUAI, McGill University, University of Chicago, Tensormesh Inc | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-phase optimization combining M/G/c Kimura approximation for analytical sweep and discrete-event simulation (DES) for verification | *LLM role:* none

> This paper introduces a two-phase capacity planner (M/G/c analytical sweep followed by discrete-event simulation) to find minimum-cost GPU fleet configurations for LLM inference under strict latency SLOs. The results are rigorously backed by simulation across multiple GPU profiles and workload traces, demonstrating that intuitive sizing rules often fail (e.g., slower GPUs can be cheaper due to KV-slot multipliers, and analytical models approve broken fleets for high-variance agent traffic). The most actionable takeaway for us is their lightweight, physics-informed (W, H, nmax) linear roofline model for GPU performance, which we can directly extract and embed into our integer programming formulations for GPUSched. This is highly relevant for our OR-for-AI infrastructure work, and we should use their open-source workload CDFs and performance constants to benchmark our own scheduling algorithms.

### [Guaranteeing Semantic and Performance Determinism in Flexible GPU Sharing](https://arxiv.org/abs/2603.15042)

**2026-03-17** | Shanghai Jiao Tong University, Chinese Academy of Sciences, University of Chinese Academy of Sciences | M=4 P=8 I=6 *discuss*

*Method:* GPU coroutines abstraction decoupling logical execution contexts (vCtx) from physical GPU resources (pCtx) via dynamic context binding and cooperative preemption | *LLM role:* none

> DETSHARE introduces 'GPU coroutines' to decouple logical execution contexts from physical GPU resources, enabling fine-grained spatial sharing without modifying kernels to preserve semantic determinism. The results are highly credible and backed by strong empirical numbers on A800/Hopper GPUs, demonstrating up to 79% higher training throughput and 69% lower inference latency compared to temporal sharing baselines. The most useful takeaway for us is the identification of 'semantic determinism'—specifically how dynamic spatial partitioning alters floating-point reduction trees and ruins training/RLHF stability. While we approach GPU scheduling via OR formulations (MIP/queueing) rather than OS-level CUDA hacking, this paper matters for our GPUSched project. We must incorporate their identified system realities—such as the 12% preemption overhead and the strict requirement for semantic determinism—into our mathematical models to ensure our theoretical schedules are actually deployable in production.


#### 2026-03-19 (6 papers)

### [MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild](https://arxiv.org/abs/2603.17187)

**2026-03-17** | Carnegie Mellon University, UC Berkeley, UNC-Chapel Hill, UC Santa Cruz | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Continual meta-learning with LLM-based gradient-free skill evolution and RL-based LoRA fine-tuning using a process reward model | *LLM role:* policy_executor, skill_generator, reward_model

> MetaClaw is a continual learning framework for LLM agents that combines gradient-free skill evolution (distilling failures into reusable prompt instructions) with asynchronous RL fine-tuning guided by a process reward model. The results are backed by solid empirical gains, showing an 8.25x improvement in end-to-end task completion on a 30-day simulated CLI benchmark. The single most useful takeaway for us is their 'Skill Generation Versioning' mechanism: when co-optimizing discrete prompts/heuristics and continuous model weights, you must strictly separate support data (used to evolve the heuristic) from query data (collected after the update) and flush the RL buffer upon evolution to prevent training on stale rewards. This directly addresses the continuous learning and RL-infused evolution bottlenecks in AlgoEvo, giving us a concrete trick to steal for managing our own RL buffers during evolutionary search.

### [Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective](https://arxiv.org/abs/2603.16104)

**2026-03-17** | National University of Singapore | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* Workflow-aware LLM serving framework integrating proactive KV cache management, global prompt caching, and cost-based cache-aware scheduling based on a templated radix tree | *LLM role:* none

> Helium optimizes LLM serving for batch agentic workflows by modeling them as query plans and using a Templated Radix Tree (TRT) to enable proactive KV caching and cache-aware scheduling. The results are rigorously backed by numbers, demonstrating up to 1.56x speedups over state-of-the-art systems (vLLM, Parrot, KVFlow) on complex multi-agent workflows, and the authors even validate their greedy scheduler's optimality gap against an MILP solver. The most valuable takeaway is the TRT abstraction, which captures global prefix hierarchies across a DAG of LLM calls to maximize KV cache reuse, rather than relying on reactive, per-call caching. This is highly actionable for us: we should directly examine their scheduling formulation and TRT implementation to improve resource allocation and memory management in our GPUSched and HERMES projects.

### [IEMAS: An Incentive-Efficiency Routing Framework for Open Agentic Web Ecosystems](https://arxiv.org/abs/2603.17302)

**2026-03-18** | Shanghai Jiao Tong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* VCG-based Min-Cost Max-Flow (MCMF) for bipartite matching, guided by Hoeffding Tree predictive QoS models | *LLM role:* none

> IEMAS routes client requests to distributed LLM agents by formulating the assignment as a Min-Cost Max-Flow bipartite matching problem, using VCG auctions to align economic incentives with KV-cache reuse. The results are backed by solid vLLM simulations, demonstrating an 80.2% KV-cache hit rate and a 35% cost reduction over baselines like GraphRouter. The most actionable takeaway for us is their method of quantifying KV-cache affinity (via Longest Common Prefix) and embedding it directly as a cost-reduction weight in a network flow optimization model. While the decentralized VCG auction mechanics might be overkill for centralized clusters, we should absolutely steal their cache-aware MCMF formulation for our OR-based LLM inference scheduling (GPUSched) work.

### [inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference](https://arxiv.org/abs/2603.16054)

**2026-03-17** | MBZUAI, McGill University, University of Chicago, Tensormesh Inc | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-phase optimization combining M/G/c Kimura approximation for analytical sweep and discrete-event simulation (DES) for verification | *LLM role:* none

> This paper introduces a two-phase capacity planner (M/G/c analytical sweep followed by discrete-event simulation) to find minimum-cost GPU fleet configurations for LLM inference under strict latency SLOs. The results are rigorously backed by simulation across multiple GPU profiles and workload traces, demonstrating that intuitive sizing rules often fail (e.g., slower GPUs can be cheaper due to KV-slot multipliers, and analytical models approve broken fleets for high-variance agent traffic). The most actionable takeaway for us is their lightweight, physics-informed (W, H, nmax) linear roofline model for GPU performance, which we can directly extract and embed into our integer programming formulations for GPUSched. This is highly relevant for our OR-for-AI infrastructure work, and we should use their open-source workload CDFs and performance constants to benchmark our own scheduling algorithms.

### [Guaranteeing Semantic and Performance Determinism in Flexible GPU Sharing](https://arxiv.org/abs/2603.15042)

**2026-03-17** | Shanghai Jiao Tong University, Chinese Academy of Sciences, University of Chinese Academy of Sciences | M=4 P=8 I=6 *discuss*

*Method:* GPU coroutines abstraction decoupling logical execution contexts (vCtx) from physical GPU resources (pCtx) via dynamic context binding and cooperative preemption | *LLM role:* none

> DETSHARE introduces 'GPU coroutines' to decouple logical execution contexts from physical GPU resources, enabling fine-grained spatial sharing without modifying kernels to preserve semantic determinism. The results are highly credible and backed by strong empirical numbers on A800/Hopper GPUs, demonstrating up to 79% higher training throughput and 69% lower inference latency compared to temporal sharing baselines. The most useful takeaway for us is the identification of 'semantic determinism'—specifically how dynamic spatial partitioning alters floating-point reduction trees and ruins training/RLHF stability. While we approach GPU scheduling via OR formulations (MIP/queueing) rather than OS-level CUDA hacking, this paper matters for our GPUSched project. We must incorporate their identified system realities—such as the 12% preemption overhead and the strict requirement for semantic determinism—into our mathematical models to ensure our theoretical schedules are actually deployable in production.

### [Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity](https://arxiv.org/abs/2603.12707)

**2026-03-13** | University of Illinois Urbana-Champaign | M=5 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Phase-aware runtime (HeteroServe) with modality-level partitioning, embedding-only transfer, and cross-type work stealing | *LLM role:* none

> Yu et al. demonstrate that partitioning multimodal LLM inference at the modality boundary (vision encoder vs. language decoder) reduces cross-device transfer costs by O(L), dropping requirements from GB-scale NVLink to MB-scale PCIe. This enables heterogeneous serving architectures where cheap compute-dense GPUs (RTX 4090) handle vision and expensive bandwidth-dense GPUs (A100) handle language. Results are strongly backed by real hardware deployments, showing a 37% improvement in cost-efficiency over homogeneous vLLM baselines. WHAT WE LEARNED: The phase-separable nature of MLLMs and the use of cross-tier work stealing (idle 4090s assisting with language decoding) are massive structural opportunities. We must immediately update our integer programming formulations in the GPUSched project to model this modality-level disaggregation, otherwise our OR models will be obsolete for multimodal workloads.


#### 2026-03-15 (1 papers)

### [Serving Compound Inference Systems on Datacenter GPUs](https://arxiv.org/abs/2603.08797)

**2026-03-09** | University of Illinois Urbana-Champaign | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed Integer Linear Programming (MILP) for joint optimization of model variants, GPU spatial partitions, and task-graph-informed budgeting | *LLM role:* none

> JIGSAWSERVE uses a Mixed Integer Linear Programming (MILP) formulation to jointly optimize model variant selection (accuracy scaling) and fine-grained GPU spatial partitioning (MIG/MPS) for serving compound inference DAGs. The results are strongly backed by empirical numbers on real hardware (H100s), demonstrating an 11.3x capacity improvement over the closest prior work (Loki) while maintaining under 0.6% SLO violations. The single most useful takeaway for us is their specific MILP formulation, which successfully linearizes the complexities of task-graph multiplicative factors and spatial GPU segments into a tractable objective for latency and accuracy SLOs. This is highly relevant for our GPUSched project; we should extract their MILP constraints for spatial partitioning and task-graph budgeting to adapt for our own LLM inference scheduling and multi-agent resource allocation models.


#### 2026-03-12 (4 papers)

### [SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference](https://arxiv.org/abs/2603.04716)

**2026-03-05** | Kingsoft Cloud | M=4 P=9 I=7 **MUST-READ** *discuss*

*Method:* Hybrid theoretical modeling (M/M/1 queuing theory) and empirical benchmarking for P/D resource calculation | *LLM role:* none

> This paper proposes a hybrid resource allocation method for disaggregated Prefill/Decode (P/D) inference, using M/M/1 queuing theory to model prefill throughput under TTFT constraints and empirical profiling for decode. The results are real and validated on NVIDIA H200 clusters running DeepSeek-V3.1. The key takeaway for us is the validated analytical relationship between TTFT, input length, and effective prefill throughput ($TP_{eff} = TP_{max} - \frac{L_{in}}{TTFT - T_{overhead}}$). We can steal this equation to serve as a cheap, differentiable constraint in our 'GPUSched' OR formulations or fitness functions, replacing expensive simulations.

### [Serving Compound Inference Systems on Datacenter GPUs](https://arxiv.org/abs/2603.08797)

**2026-03-09** | University of Illinois Urbana-Champaign | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed Integer Linear Programming (MILP) for joint optimization of model variants, GPU spatial partitions, and task-graph-informed budgeting | *LLM role:* none

> JIGSAWSERVE uses a Mixed Integer Linear Programming (MILP) formulation to jointly optimize model variant selection (accuracy scaling) and fine-grained GPU spatial partitioning (MIG/MPS) for serving compound inference DAGs. The results are strongly backed by empirical numbers on real hardware (H100s), demonstrating an 11.3x capacity improvement over the closest prior work (Loki) while maintaining under 0.6% SLO violations. The single most useful takeaway for us is their specific MILP formulation, which successfully linearizes the complexities of task-graph multiplicative factors and spatial GPU segments into a tractable objective for latency and accuracy SLOs. This is highly relevant for our GPUSched project; we should extract their MILP constraints for spatial partitioning and task-graph budgeting to adapt for our own LLM inference scheduling and multi-agent resource allocation models.

### [PromptTuner: SLO-Aware Elastic System for LLM Prompt Tuning](https://arxiv.org/abs/2603.05087)

**2026-03-05** | Nanyang Technological University, Unaffiliated | M=5 P=8 I=7 *discuss*

*Method:* SLO-aware elastic system combining a two-layer Prompt Bank for initial prompt selection and a Workload Scheduler for dynamic multi-GPU allocation | *LLM role:* feature_extractor

> PromptTuner is a cluster management system for LLM prompt tuning that combines a 'Prompt Bank' (retrieving similar past prompts to speed up convergence) with a hierarchical scheduler (warm/cold GPU pools) to meet latency SLOs. The authors demonstrate real-world efficacy on 32-96 GPU clusters, showing 4-8x reductions in SLO violations compared to INFless and ElasticFlow. The key takeaway for us is the 'Prompt Bank' mechanism: using K-medoids clustering on activation features to retrieve high-quality initial prompts. We should steal this initialization strategy for AlgoEvo to reduce the number of generations needed for convergence, and use the scheduling logic as a baseline for our GPUSched project.

### [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://arxiv.org/abs/2603.04918)

**2026-03-05** | Fudan University, Shanghai Innovation Institute | M=8 P=7 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Band-constrained Policy Optimization (BandPO) using a Band operator to project f-divergence trust regions into dynamic, probability-aware clipping intervals | *LLM role:* policy_agent

> BandPO replaces the standard static clipping in PPO/GRPO with dynamic bounds derived from projecting f-divergence trust regions, specifically addressing a bottleneck where allowable updates vanish for low-probability tokens. Empirical results are rigorous, showing consistent gains (2-10%) on math benchmarks and, crucially, maintaining policy entropy where baselines collapse. The key takeaway is that standard clipping scales update margins linearly with probability, effectively freezing rare tokens; BandPO decouples this, allowing the model to actually reinforce novel, high-advantage tail strategies. We should implement the closed-form TV or Chi-squared variants immediately in our RL optimizers to improve exploration efficiency.


#### 2026-03-08 (9 papers)

### [SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference](https://arxiv.org/abs/2603.04716)

**2026-03-05** | Kingsoft Cloud | M=4 P=9 I=7 **MUST-READ** *discuss*

*Method:* Hybrid theoretical modeling (M/M/1 queuing theory) and empirical benchmarking for P/D resource calculation | *LLM role:* none

> This paper proposes a hybrid resource allocation method for disaggregated Prefill/Decode (P/D) inference, using M/M/1 queuing theory to model prefill throughput under TTFT constraints and empirical profiling for decode. The results are real and validated on NVIDIA H200 clusters running DeepSeek-V3.1. The key takeaway for us is the validated analytical relationship between TTFT, input length, and effective prefill throughput ($TP_{eff} = TP_{max} - \frac{L_{in}}{TTFT - T_{overhead}}$). We can steal this equation to serve as a cheap, differentiable constraint in our 'GPUSched' OR formulations or fitness functions, replacing expensive simulations.

### [AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework](https://arxiv.org/abs/2603.03233)

**2026-03-03** | Fudan University, Shanghai Innovation Institute, Shanghai Academy of AI for Science | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Bayesian Adversarial Multi-agent Framework for AI4S (BAMF-AI4S) with recursive co-optimization of generated code, test cases, and prompts, guided by a non-LLM-based Bayesian updating rule and Bayesian Optimization for code performance estimation. | *LLM role:* code_writer, decomposition_guide, prompt_optimizer, test_case_generator, solution_generator

> The authors propose a multi-agent framework for scientific code generation that couples an adversarial 'Challenger' (generating difficult test cases) with a 'Solver', governed by a Bayesian update rule. Crucially, they employ Bayesian Optimization with a kernel based on code embeddings (AST + text) to estimate solution quality *before* running expensive tests, effectively acting as a learned surrogate model. Results on SciCode and ScienceAgentBench are strong, showing small models (Qwen-32B) outperforming GPT-4o when using this loop. **The killer feature for us is the surrogate modeling pipeline:** we should immediately steal the idea of using GP surrogates on code embeddings to filter candidates in our evolutionary search, potentially reducing our evaluation costs by orders of magnitude.

### [VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation](https://arxiv.org/abs/2603.02681)

**2026-03-03** | Tencent Hunyuan, Hong Kong University of Science and Technology | M=8 P=2 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Native visual-generation agentic model (VisionCreator) unifying Understanding, Thinking, Planning, and Creation (UTPC) capabilities, optimized via Progressive Specialization Training (PST) and Virtual Reinforcement Learning (VRL) with LtrReward in VisGenEnv. | *LLM role:* agentic_model

> This paper introduces VisionCreator, an agent trained via 'Virtual Reinforcement Learning' (VRL) where tool outputs and logic are simulated to train long-horizon planning policies without incurring expensive real-world execution costs. They employ a 'Plan-Driven Reward' model (combining LLM-based plan verification with rule-based execution checks) and prove theoretical bounds for the sim-to-real transfer, achieving performance superior to GPT-5 on visual tasks. **Key Takeaway:** We should steal the VRL architecture for AlgoEvo. By constructing a 'Virtual OR Environment' that simulates code validity and approximate heuristic performance, we can train our evolutionary search policies (RL-infused evolution) at a fraction of the current compute cost, bypassing the bottleneck of running full benchmarks during the search policy optimization phase.

### [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](https://arxiv.org/abs/2603.02637)

**2026-03-03** | University of Minnesota-Twin Cities | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent framework with rubric-based agentic reinforcement learning (GRPO) | *LLM role:* decomposition_guide, code_writer, evaluator

> StitchCUDA automates end-to-end GPU program generation using a multi-agent framework, but its core contribution is a training recipe that solves reward hacking in code optimization. They decompose expensive multi-turn agentic RL into single-turn 'atomic skills' (generation vs. refinement) and use GRPO with an LLM-evaluated 'Rubric Reward' (e.g., 'Did you use tiling?') rather than just sparse outcome metrics. This prevents the model from gaming the system (e.g., wrapping PyTorch code) and forces actual optimization behavior. We should steal the atomic skill decomposition to drastically reduce training costs for AlgoEvo and implement Rubric Rewards to fix our process reward models.

### [PromptTuner: SLO-Aware Elastic System for LLM Prompt Tuning](https://arxiv.org/abs/2603.05087)

**2026-03-05** | Nanyang Technological University, Unaffiliated | M=5 P=8 I=7 *discuss*

*Method:* SLO-aware elastic system combining a two-layer Prompt Bank for initial prompt selection and a Workload Scheduler for dynamic multi-GPU allocation | *LLM role:* feature_extractor

> PromptTuner is a cluster management system for LLM prompt tuning that combines a 'Prompt Bank' (retrieving similar past prompts to speed up convergence) with a hierarchical scheduler (warm/cold GPU pools) to meet latency SLOs. The authors demonstrate real-world efficacy on 32-96 GPU clusters, showing 4-8x reductions in SLO violations compared to INFless and ElasticFlow. The key takeaway for us is the 'Prompt Bank' mechanism: using K-medoids clustering on activation features to retrieve high-quality initial prompts. We should steal this initialization strategy for AlgoEvo to reduce the number of generations needed for convergence, and use the scheduling logic as a baseline for our GPUSched project.

### [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://arxiv.org/abs/2603.04918)

**2026-03-05** | Fudan University, Shanghai Innovation Institute | M=8 P=7 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Band-constrained Policy Optimization (BandPO) using a Band operator to project f-divergence trust regions into dynamic, probability-aware clipping intervals | *LLM role:* policy_agent

> BandPO replaces the standard static clipping in PPO/GRPO with dynamic bounds derived from projecting f-divergence trust regions, specifically addressing a bottleneck where allowable updates vanish for low-probability tokens. Empirical results are rigorous, showing consistent gains (2-10%) on math benchmarks and, crucially, maintaining policy entropy where baselines collapse. The key takeaway is that standard clipping scales update margins linearly with probability, effectively freezing rare tokens; BandPO decouples this, allowing the model to actually reinforce novel, high-advantage tail strategies. We should implement the closed-form TV or Chi-squared variants immediately in our RL optimizers to improve exploration efficiency.

### [AI4S-SDS: A Neuro-Symbolic Solvent Design System via Sparse MCTS and Differentiable Physics Alignment](https://arxiv.org/abs/2603.03686)

**2026-03-04** | Nanjing University, Suzhou Laboratory, Shanghai Artificial Intelligence Laboratory | M=8 P=4 I=8 **MUST-READ** *discuss*

*Method:* Neuro-symbolic framework integrating Sparse Monte Carlo Tree Search (MCTS) with Sibling-Aware Expansion, Memory-Driven Global Planning, and a Differentiable Physics Engine for continuous ratio optimization. | *LLM role:* semantic_generator

> Chen et al. introduce a neuro-symbolic MCTS framework for mixed discrete-continuous optimization, applying it to solvent design. They solve the LLM context bottleneck via 'Sparse State Storage' (storing only state abstractions and reconstructing paths on-demand) and fix mode collapse using 'Sibling-Aware Expansion' (conditioning the generator on sibling nodes to force orthogonality). While the chemical application is niche, the search architecture is highly relevant: we should steal the sibling-aware conditioning to improve diversity in our evolutionary code generation and adopt their sparse storage pattern to scale our search horizons.

### [Build, Judge, Optimize: A Blueprint for Continuous Improvement of Multi-Agent Consumer Assistants](https://arxiv.org/abs/2603.03565)

**2026-03-03** | DoorDash, WithMetis.ai | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Prompt-level optimization using GEPA and MAMUT GEPA | *LLM role:* evaluator, evolutionary_search, decomposition_guide, user_simulator

> This paper presents a production-grade framework for optimizing multi-agent systems by jointly evolving prompt bundles (MAMUT) rather than optimizing agents in isolation. They validate this on a grocery assistant, showing that system-level optimization outperforms local sub-agent optimization by ~7% because it captures coordination dynamics (e.g., context passing) that local metrics miss. The most stealable insight is their 'Judge Calibration' loop: they use evolutionary search (GEPA) to optimize the *evaluator's* prompt to match human labels (91.4% agreement) before using that judge to optimize the agents. This is a rigorous solution to the noisy fitness function problem we face in LLM evolutionary search.

### [MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing](https://arxiv.org/abs/2603.02885)

**2026-03-03** | Shanghai Jiao Tong University, National University of Singapore | M=6 P=7 I=7 *discuss*

*Method:* Hierarchical spatial-temporal backbone multiplexing with unified PEFT representations, dynamic programming for task fusion, priority-based subgraph scheduling, and chunk-based data alignment | *LLM role:* subject_of_optimization

> MuxTune introduces a hierarchical scheduler for multi-tenant PEFT that uses Dynamic Programming to optimally fuse tasks (spatial batching) or interleave them (temporal multiplexing) based on a pipeline cost model. Empirical results on H100s show up to 5x throughput gains over NeMo and S-LoRA, validated by ablation studies. The most stealable insight is their **chunk-based data alignment**: instead of standard padding or naive packing, they split packed sequences into fixed-size chunks to balance compute efficiency with memory waste—a trick we should immediately implement for batch evaluation in AlgoEvo and our serving optimization models.


#### 2026-03-05 (8 papers)

### [AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework](https://arxiv.org/abs/2603.03233)

**2026-03-03** | Fudan University, Shanghai Innovation Institute, Shanghai Academy of AI for Science | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Bayesian Adversarial Multi-agent Framework for AI4S (BAMF-AI4S) with recursive co-optimization of generated code, test cases, and prompts, guided by a non-LLM-based Bayesian updating rule and Bayesian Optimization for code performance estimation. | *LLM role:* code_writer, decomposition_guide, prompt_optimizer, test_case_generator, solution_generator

> The authors propose a multi-agent framework for scientific code generation that couples an adversarial 'Challenger' (generating difficult test cases) with a 'Solver', governed by a Bayesian update rule. Crucially, they employ Bayesian Optimization with a kernel based on code embeddings (AST + text) to estimate solution quality *before* running expensive tests, effectively acting as a learned surrogate model. Results on SciCode and ScienceAgentBench are strong, showing small models (Qwen-32B) outperforming GPT-4o when using this loop. **The killer feature for us is the surrogate modeling pipeline:** we should immediately steal the idea of using GP surrogates on code embeddings to filter candidates in our evolutionary search, potentially reducing our evaluation costs by orders of magnitude.

### [VisionCreator: A Native Visual-Generation Agentic Model with Understanding, Thinking, Planning and Creation](https://arxiv.org/abs/2603.02681)

**2026-03-03** | Tencent Hunyuan, Hong Kong University of Science and Technology | M=8 P=2 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Native visual-generation agentic model (VisionCreator) unifying Understanding, Thinking, Planning, and Creation (UTPC) capabilities, optimized via Progressive Specialization Training (PST) and Virtual Reinforcement Learning (VRL) with LtrReward in VisGenEnv. | *LLM role:* agentic_model

> This paper introduces VisionCreator, an agent trained via 'Virtual Reinforcement Learning' (VRL) where tool outputs and logic are simulated to train long-horizon planning policies without incurring expensive real-world execution costs. They employ a 'Plan-Driven Reward' model (combining LLM-based plan verification with rule-based execution checks) and prove theoretical bounds for the sim-to-real transfer, achieving performance superior to GPT-5 on visual tasks. **Key Takeaway:** We should steal the VRL architecture for AlgoEvo. By constructing a 'Virtual OR Environment' that simulates code validity and approximate heuristic performance, we can train our evolutionary search policies (RL-infused evolution) at a fraction of the current compute cost, bypassing the bottleneck of running full benchmarks during the search policy optimization phase.

### [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](https://arxiv.org/abs/2603.02637)

**2026-03-03** | University of Minnesota-Twin Cities | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent framework with rubric-based agentic reinforcement learning (GRPO) | *LLM role:* decomposition_guide, code_writer, evaluator

> StitchCUDA automates end-to-end GPU program generation using a multi-agent framework, but its core contribution is a training recipe that solves reward hacking in code optimization. They decompose expensive multi-turn agentic RL into single-turn 'atomic skills' (generation vs. refinement) and use GRPO with an LLM-evaluated 'Rubric Reward' (e.g., 'Did you use tiling?') rather than just sparse outcome metrics. This prevents the model from gaming the system (e.g., wrapping PyTorch code) and forces actual optimization behavior. We should steal the atomic skill decomposition to drastically reduce training costs for AlgoEvo and implement Rubric Rewards to fix our process reward models.

### [Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design](https://arxiv.org/abs/2602.23092)

**2026-02-26** | City University of Hong Kong, Southern University of Science and Technology | M=7 P=9 I=8 **MUST-READ** *discuss*

*Method:* Adaptive Iterated Local Search (AILS) with LLM-driven Evolutionary Computation for Automatic Heuristic Design (AHD) of ruin heuristics | *LLM role:* heuristic_generator

> This paper integrates LLM-driven evolutionary search into the AILS framework to evolve 'ruin' heuristics for CVRP, employing a Chain-of-Thought 'voting' mechanism to filter out poor heuristics before expensive evaluation. The results are empirically strong: they claim 8 new Best-Known Solutions on the CVRPLib large-scale benchmark, outperforming HGS and AILS-II. **Key Takeaway:** We should steal the 'acceleration mechanism'—using the LLM to predict heuristic quality via CoT prior to execution—to address the sample efficiency bottleneck in our own evolutionary search loops. This is a direct proof-of-concept that LLM-evolved components can beat hand-crafted SOTA on hard OR instances.

### [AI4S-SDS: A Neuro-Symbolic Solvent Design System via Sparse MCTS and Differentiable Physics Alignment](https://arxiv.org/abs/2603.03686)

**2026-03-04** | Nanjing University, Suzhou Laboratory, Shanghai Artificial Intelligence Laboratory | M=8 P=4 I=8 **MUST-READ** *discuss*

*Method:* Neuro-symbolic framework integrating Sparse Monte Carlo Tree Search (MCTS) with Sibling-Aware Expansion, Memory-Driven Global Planning, and a Differentiable Physics Engine for continuous ratio optimization. | *LLM role:* semantic_generator

> Chen et al. introduce a neuro-symbolic MCTS framework for mixed discrete-continuous optimization, applying it to solvent design. They solve the LLM context bottleneck via 'Sparse State Storage' (storing only state abstractions and reconstructing paths on-demand) and fix mode collapse using 'Sibling-Aware Expansion' (conditioning the generator on sibling nodes to force orthogonality). While the chemical application is niche, the search architecture is highly relevant: we should steal the sibling-aware conditioning to improve diversity in our evolutionary code generation and adopt their sparse storage pattern to scale our search horizons.

### [Build, Judge, Optimize: A Blueprint for Continuous Improvement of Multi-Agent Consumer Assistants](https://arxiv.org/abs/2603.03565)

**2026-03-03** | DoorDash, WithMetis.ai | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Prompt-level optimization using GEPA and MAMUT GEPA | *LLM role:* evaluator, evolutionary_search, decomposition_guide, user_simulator

> This paper presents a production-grade framework for optimizing multi-agent systems by jointly evolving prompt bundles (MAMUT) rather than optimizing agents in isolation. They validate this on a grocery assistant, showing that system-level optimization outperforms local sub-agent optimization by ~7% because it captures coordination dynamics (e.g., context passing) that local metrics miss. The most stealable insight is their 'Judge Calibration' loop: they use evolutionary search (GEPA) to optimize the *evaluator's* prompt to match human labels (91.4% agreement) before using that judge to optimize the agents. This is a rigorous solution to the noisy fitness function problem we face in LLM evolutionary search.

### [AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning](https://arxiv.org/abs/2602.23258)

**2026-02-26** | Alibaba Group, Harbin Institute of Technology, Shenzhen | M=6 P=7 I=8 *discuss*

*Method:* Test-time rectify-or-reject pruning framework with retrieval-augmented rectifier, failure-driven indicator pool, and dual-stage deduplication | *LLM role:* rectifier, teacher, deduplicator, reasoning_engine

> Wang et al. propose a test-time 'firewall' for multi-agent systems that intercepts messages and validates them against a retrieved set of error patterns (mined from offline failure trajectories). They achieve ~6% accuracy gains on math benchmarks by iteratively rectifying or pruning erroneous outputs before they propagate. The critical takeaway for our AlgoEvo work is the **Failure-Driven Indicator Pool**: we should implement a similar module that mines failed code generations to build a repository of 'forbidden patterns,' allowing a lightweight verifier to prune bad mutations before expensive execution. This effectively turns the 'graveyard' of failed runs into a persistent memory that improves sample efficiency.

### [MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing](https://arxiv.org/abs/2603.02885)

**2026-03-03** | Shanghai Jiao Tong University, National University of Singapore | M=6 P=7 I=7 *discuss*

*Method:* Hierarchical spatial-temporal backbone multiplexing with unified PEFT representations, dynamic programming for task fusion, priority-based subgraph scheduling, and chunk-based data alignment | *LLM role:* subject_of_optimization

> MuxTune introduces a hierarchical scheduler for multi-tenant PEFT that uses Dynamic Programming to optimally fuse tasks (spatial batching) or interleave them (temporal multiplexing) based on a pipeline cost model. Empirical results on H100s show up to 5x throughput gains over NeMo and S-LoRA, validated by ablation studies. The most stealable insight is their **chunk-based data alignment**: instead of standard padding or naive packing, they split packed sequences into fixed-size chunks to balance compute efficiency with memory waste—a trick we should immediately implement for batch evaluation in AlgoEvo and our serving optimization models.


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

*9 fronts detected — snapshot 2026-03-10*

### Front 17 (12 papers) — GROWING

**Density:** 0.29 | **Methods:** llm_in_the_loop, llm_as_heuristic, resource_allocation, process_reward_model, grpo | **Problems:** llm_inference_optimization, mathematical_reasoning, llm_serving_optimization, resource_allocation, throughput_optimization

*Unique methods:* adaptive_index_update, adaptive_queryselect, adaptive_sampling, advantage_modulation, all_to_all_collectives, analytical_modeling, approximate_nearest_neighbor_search, attention_sparsification, bayes_factor, bayesian_modeling, beam_search, bernoulli_variance_proxy, bert_embeddings, best_of_n, beta_distribution_modeling, bisection_method, blockwise_local_distillation, clipping_mechanism, clustering, control_theory, cuda_graph, direction_oriented_resource_allocation, dirichlet_process_prior, distributed_inference, diverse_verifier_tree_search, dora, dual_linear_program, dynamic_clipping_bounds, dynamic_dispatching, dynamic_rollout_allocation, embedding_model, entropy_control, entropy_dynamics_control, entropy_estimation, expert_parallelism, exponential_smoothing, f_divergences, fast_scanning, feedback_control, fluid_model_analysis, geometric_algorithm, global_knowledge_distillation, gradient_compensation, gradient_scheduling, gradient_variance_minimization, grouped_gemm, grouped_query_attention, hierarchical_agglomerative_clustering, inference_optimization, inverted_file_index, kl_divergence, knowledge_distillation, kv_cache_optimization, latency_bounded_partitioning, llm_as_answer_generator, llm_ensemble, llm_inference_optimization, llm_inference_scheduling, llm_reinforcement_learning, low_rank_approximation, majority_voting, max_margin_optimization, memory_optimization, mixture_of_experts, monte_carlo_methods, multi_head_attention, neural_architecture_search, nvshmem, online_distillation, paged_attention, pearson_chi_squared_divergence, performance_estimation, pipelining, predictive_scheduling, process_reward_model, product_quantization, queryselect, rate_distortion_theory, real_time_optimization, rebase, reinforcement_learning_from_human_feedback, reinforcement_learning_with_verifiable_rewards, resource_partitioning, retrieval_augmented_generation_serving, reward_balanced_search, rl_ppo, root_finding_algorithms, self_organizing_systems, semantic_similarity, soft_clustering, stochastic_thermodynamics, straggler_mitigation, structured_sparsity, temperature_sampling, token_classification, total_variation_divergence, transformer_architecture_optimization, tree_search, triton, trust_region_policy_optimization, update_magnitude_stabilization, vector_similarity_search
*Shared methods:* bayesian_optimization, convex_optimization, curriculum_learning, greedy_algorithm, group_relative_policy_optimization, grpo, integer_linear_programming, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_fine_tuned, llm_in_the_loop, load_balancing, mixed_integer_linear_programming, mixed_integer_programming, policy_optimization, proximal_policy_optimization, pruning, queuing_theory, resource_allocation, scheduling, speculative_decoding, supervised_learning, system_design, vllm

This research front explores advanced Operations Research (OR) and AI techniques to optimize various aspects of Large Language Model (LLM) systems, spanning inference, serving, and reasoning. A core theme is the application of mathematical optimization (e.g., Mixed-Integer Programming, Linear Programming, gradient-based scheduling) and reinforcement learning (e.g., GRPO, DynaMO, BandPO) to enhance LLM efficiency, performance, and decision-making. Key frameworks include GRPO for policy optimization, Puzzle (LANA) for NAS, DORA (REBASE) and ETS for test-time search and KV cache management, and Entropic-Time Inference for dynamic compute allocation.

Significant contributions include the Solver-in-the-Loop framework ([1]) using GRPO and Process Reward Models (PRM) for iterative OR model debugging, achieving +11.1% RR@5 over API models. Puzzle ([2]) leverages MIP and blockwise distillation for NAS, yielding 2.17x throughput speedup for Llama-70B with 98.4% accuracy retention. DORA ([3]) optimizes test-time search via embedding-based clustering and resource allocation, outperforming baselines with 3.5x fewer FLOPs on MATH500. Entropic-Time Inference ([4]) dynamically allocates compute based on entropy, showing 30-45% throughput increase over vLLM. BandPO ([12]) improves GRPO by 2-10% on math benchmarks by addressing policy update clipping. Other notable works include VectorLiteRAG ([6]) for RAG serving, GoodSpeed ([7]) for distributed edge inference, and PROBE ([11]) for MoE inference, all demonstrating substantial efficiency gains.

This research front is rapidly emerging and growing, characterized by a strong interdisciplinary fusion of OR and AI. The emphasis on verifiable rewards, solver-in-the-loop evaluation, and mathematically grounded resource allocation indicates a shift towards more robust and efficient LLM systems. The next wave of papers will likely focus on integrating these fine-grained optimization techniques across the entire LLM lifecycle, from training and architecture design to inference and complex reasoning. We can expect more sophisticated RL-infused evolutionary search frameworks that dynamically adapt compute based on uncertainty (e.g., Entropic Time, DynaMO's variance-based allocation) and MILP/ILP-driven control planes for real-time resource management in distributed LLM serving.

**Papers:**

### [Solver-in-the-Loop: MDP-Based Benchmarks for Self-Correction and Behavioral Rationality in Operations Research](https://arxiv.org/abs/2601.21008)

**2026-02-08** | Massachusetts Institute of Technology, Alibaba Group | M=9 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Domain-specific Group Relative Policy Optimization (GRPO) with composite reward and three-stage curriculum learning | *LLM role:* agent_for_debugging_and_decision_making

> Ao et al. introduce a framework for iterative OR model debugging that trains an 8B model using Group Relative Policy Optimization (GRPO) and a Process Reward Model (PRM) to outperform GPT-4o-mini. They utilize Gurobi's Irreducible Infeasible Subsystem (IIS) not just as text feedback, but as a dense reward signal (IIS size reduction) for the PRM, achieving a 95.3% recovery rate versus 86.2% for frontier APIs. **Key Takeaway:** We should steal their PRM construction method—specifically using solver diagnostics (like IIS reduction or compiler error counts) as dense step-level rewards—and their 'faithfulness penalty' to prevent overfitting in our evolutionary search. This is a direct validation of RLVR (Reinforcement Learning with Verifiable Rewards) for OR, proving it superior to large-scale prompting.

### [Puzzle: Distillation-Based NAS for Inference-Optimized LLMs](https://arxiv.org/abs/2411.19146)

**2025-06-03** | NVIDIA | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Decomposed Neural Architecture Search (NAS) using Blockwise Local Knowledge Distillation (BLD) for parallel architecture exploration and Mixed-Integer Programming (MIP) for precise constraint optimization, followed by Global Knowledge Distillation (GKD) | *LLM role:* none

> Bercovich et al. introduce Puzzle, a framework that optimizes LLM architectures for specific hardware by training a library of block variants (via local distillation) and using Mixed-Integer Programming (MIP) to select the optimal layer-wise configuration under strict latency and memory constraints. The results are robust: they compress Llama-70B to 51B, fitting on a single H100 with 2.17x throughput gain and 98.4% accuracy retention, significantly outperforming pruning baselines like Wanda. **Key takeaway:** The 'decomposed search' strategy—replacing expensive end-to-end evolutionary evaluation loops with local proxy scores (KL divergence) and a global MIP solver—is a highly efficient method for modular system configuration. This directly informs our 'GPUSched' and serving optimization work by demonstrating how to mathematically formulate hardware constraints (KV-cache, batch size, compute) into the model design process itself.

### [Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling](https://arxiv.org/abs/2506.15707)

**2025-10-20** | Beijing Institute of Technology, Xiaohongshu Inc | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Direction-Oriented Resource Allocation (DORA) | *LLM role:* reasoning_path_generator

> Wang et al. introduce Direction-Oriented Resource Allocation (DORA), which uses embedding-based soft clustering to group semantically similar reasoning paths and allocates compute budget to distinct 'directions' rather than individual solutions. They prove solution-level allocation (like REBASE) is suboptimal when paths are correlated and show DORA achieves state-of-the-art accuracy on MATH500 with 3.5x fewer FLOPs. **Key Takeaway:** We can immediately steal the 'semantic uniqueness reweighting' mechanism for AlgoEvo. By clustering generated heuristics via embeddings before expensive evaluation, we can drastically improve sample efficiency and stop wasting compute on minor variations of the same code.

### [Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention](https://arxiv.org/abs/2603.03310)

**2026-02-08** | UC Berkeley | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Self-organizing inference architecture based on entropy control, extending vLLM with entropy-aware scheduling, entropic pruning of paged attention blocks, and adaptive temperature control. | *LLM role:* none

> Kiruluta proposes 'Entropic-Time Inference,' a control framework that dynamically allocates compute (scheduling, attention, sampling) based on entropy reduction rather than token count, effectively skipping 'low-information' computation. While the 30-45% throughput gains over vLLM are self-reported and depend on efficient entropy estimation, the core mechanism is sound. The critical takeaway is the **'Entropic Time' metric**: we can steal this for AlgoEvo to allocate evolutionary compute only to solution branches with high uncertainty, rather than evolving all individuals uniformly. This directly addresses our sample efficiency bottleneck and provides a novel dynamic objective for our GPUSched project.

### [Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models](https://arxiv.org/abs/2407.15504)

**2024-12-11** | UT Austin, EPFL | M=8 P=8 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Rate-distortion theory formalized as a linear program, solved via its dual using a geometric algorithm; Adaptive QuerySelect (query-aware, variable-rate token classification) | *LLM role:* token classifier

> Nagle et al. formalize prompt compression as a rate-distortion problem, deriving the fundamental theoretical limit via a dual linear program and proposing 'Adaptive QuerySelect,' a variable-rate compression technique. The results are rigorous: they calculate exact limits on synthetic data and use beam search approximations for NLP, demonstrating that existing fixed-rate methods leave significant performance on the table. The key takeaway is that **variable-rate compression**—keeping tokens based on a confidence threshold rather than a fixed percentage—is essential for approaching optimality; this allows 'hard' queries to retain more context while aggressively compressing 'easy' ones. This is immediately actionable for our AlgoEvo work: we should replace fixed-window history truncation with a query-aware, variable-rate compressor to maximize the useful information in our limited context window.

### [VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG](https://arxiv.org/abs/2504.08930)

**2026-01-19** | Georgia Institute of Technology | M=5 P=7 I=6 *discuss*

*Method:* Analytical performance modeling and latency-bounded partitioning algorithm for hybrid CPU-GPU vector index, combined with a distributed runtime pipeline featuring query- and shard-aware routing and dynamic dispatcher. | *LLM role:* target_of_optimization

> VectorLiteRAG optimizes RAG serving throughput by dynamically partitioning vector indices between CPU and GPU memory based on access skew and latency SLOs. The results are credible, showing up to 1.5x throughput gains on H100/L40S setups by balancing retrieval speed against LLM KV-cache capacity. The most stealable insight is their use of a Beta distribution to analytically model the *minimum* hit rate within a batch (the bottleneck) to predict tail latency without full simulation—a technique we could adapt for stochastic constraints in our serving formulations. It solves a resource allocation problem we care about, though via systems engineering rather than the rigorous OR methods we prefer.

### [GoodSpeed: Optimizing Fair Goodput with Adaptive Speculative Decoding in Distributed Edge Inference](https://arxiv.org/abs/2512.09963)

**2025-12-14** | The University of Sydney, Kyung Hee University | M=5 P=7 I=6 *discuss*

*Method:* Gradient-based scheduling algorithm maximizing logarithmic utility for proportional fairness with adaptive speculative decoding | *LLM role:* heuristic_generator, evaluator

> GoodSpeed uses gradient-based scheduling to dynamically allocate token generation budgets across distributed draft servers, maximizing a logarithmic utility function to balance throughput and fairness. The authors provide rigorous fluid sample path analysis to prove convergence, backed by experiments on H100/L4 clusters, although the baselines (fixed/random allocation) are relatively weak. The most useful takeaway is the mechanism of using exponentially smoothed acceptance rate estimates to drive real-time control in a stochastic system—a robust pattern we should adopt for our own stochastic resource allocation and RobustMAS projects.

### [How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization](https://arxiv.org/abs/2602.19208)

**2026-02-22** | Meituan, Tsinghua University, Fudan University, Peking University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dual-pronged optimization framework (DynaMO) combining variance-minimizing dynamic rollout allocation and gradient-aware advantage modulation for GRPO-based policy optimization | *LLM role:* policy_model

> DynaMO introduces a dual-pronged optimization for RLVR: a dynamic rollout allocation strategy that prioritizes problems with high gradient variance (proxied by Bernoulli variance of success/failure), and a gradient modulation technique to stabilize updates. The results are strong (+11.8% Pass@1 over GRPO on Qwen-7B) and backed by clear ablations. **The critical takeaway for us is the allocation logic:** we should immediately replace uniform sampling in AlgoEvo with variance-based allocation ($n_i \propto \sqrt{p(1-p)}$). This ensures compute is spent on instances/components that are currently 'learnable' (high variance) rather than wasted on the trivial or impossible, directly optimizing our search budget.

### [Best-of-$\infty$ -- Asymptotic Performance of Test-Time Compute](https://arxiv.org/abs/2509.21091)

**2025-09-25** | Mohamed bin Zayed University of Artificial Intelligence, New York University, RIKEN AIP, Institute of Science Tokyo, NEC Corporation | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Adaptive sampling for majority voting using Bayesian modeling (Dirichlet process prior and Bayes factor) to determine stopping criteria, combined with optimally weighted LLM ensembles formulated as a Mixed-Integer Linear Program (MILP) with a max-margin solution. | *LLM role:* answer_generator

> This paper introduces a Bayesian adaptive stopping criterion (using Dirichlet process priors and Bayes factors) for majority voting, reducing test-time compute by 2-5x while maintaining asymptotic 'Best-of-Infinity' accuracy. They further demonstrate that optimizing weights for an ensemble of LLMs can be formulated as a Mixed-Integer Linear Program (MILP) by treating the decision boundaries as polytopes. **What we learned:** The Bayesian stopping logic is immediately transferable to AlgoEvo to reduce the cost of fitness evaluations—we can stop evaluating candidate solutions early if their performance distribution is statistically distinct. The MILP approach for ensembles also offers a concrete formulation we could adapt for our GPU scheduling and model serving optimization work.

### [ETS: Efficient Tree Search for Inference-Time Scaling](https://arxiv.org/abs/2502.13575)

**2025-06-11** | University of California, Berkeley, ICSI, LBNL | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Efficient Tree Search (ETS) using a linear programming cost model with KV cache sharing penalty and semantic coverage term | *LLM role:* candidate_generator, process_reward_model, search_guidance

> ETS formulates the tree search pruning step as a lightweight Integer Linear Program (ILP) that maximizes the reward of retained nodes while penalizing total KV cache size and enforcing semantic diversity via clustering. Unlike standard beam search or REBASE, it explicitly optimizes the trade-off between memory consumption (KV sharing) and exploration coverage. The authors demonstrate a 1.8x reduction in KV cache size and 1.4x throughput increase on MATH500 with minimal accuracy loss. We should steal the 'ILP-in-the-loop' mechanism for population management in our evolutionary search frameworks to optimize hardware utilization dynamically.

### [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509)

**2026-02-03** | Kling Infra, Kuaishou Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Continuous Lookahead Pipelining with Gate-Initialized Lookahead Predictor, Hardware-Aware Balance Planning, and Phase-Locked Co-Scheduling | *LLM role:* none

> PROBE optimizes MoE inference by using a distilled MLP to predict next-layer expert activation, enabling proactive load balancing and weight prefetching hidden behind the current layer's computation. The results are strong (1.3x speedup on 235B models) and demonstrate that control plane overheads can be fully masked. The critical takeaway for our `GPUSched` project is the **Lookahead Pipelining** architecture: it carves out a deterministic execution window where we could inject our own specialized solvers (e.g., fast ALNS or IP formulations) to outperform their basic greedy resource allocator. This transforms the stochastic serving problem into a short-horizon deterministic routing problem we are well-equipped to solve.

### [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://arxiv.org/abs/2603.04918)

**2026-03-05** | Fudan University, Shanghai Innovation Institute | M=8 P=7 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Band-constrained Policy Optimization (BandPO) using a Band operator to project f-divergence trust regions into dynamic, probability-aware clipping intervals | *LLM role:* policy_agent

> BandPO replaces the standard static clipping in PPO/GRPO with dynamic bounds derived from projecting f-divergence trust regions, specifically addressing a bottleneck where allowable updates vanish for low-probability tokens. Empirical results are rigorous, showing consistent gains (2-10%) on math benchmarks and, crucially, maintaining policy entropy where baselines collapse. The key takeaway is that standard clipping scales update margins linearly with probability, effectively freezing rare tokens; BandPO decouples this, allowing the model to actually reinforce novel, high-advantage tail strategies. We should implement the closed-form TV or Chi-squared variants immediately in our RL optimizers to improve exploration efficiency.


### Front 11 (12 papers) — STABLE

**Density:** 0.64 | **Methods:** integer_linear_programming, pipeline_parallelism, data_parallelism, tensor_parallelism, greedy_algorithm | **Problems:** llm_serving_optimization, resource_allocation, gpu_scheduling, cloud_scheduling, service_level_objective_optimization

*Unique methods:* adaptive_scheduling, arima_time_series_forecasting, autoregressive_decoding, binary_search, cache_replacement_policy, chebyshev_guided_optimization, cvxpy, decomposition_algorithm, demand_forecasting, discrete_event_simulation, dynamic_batching, dynamic_scheduling, empirical_benchmarking, gpu_resource_management, gurobi_solver, heuristic_initialization, heuristics, kv_cache, least_carbon_savings, llm_serving_optimization, m_m_1_model, max_flow_optimization, milp, milp_acceleration, milp_formulation, model_parallelism, multi_instance_gpu, network_communication_optimization, network_topology_modeling, neural_network, np_hardness_proof, optimal_transport, optimization, pre_initialization, profiling, reactive_heuristics, resource_allocation_modeling, resource_allocation_optimization, sarima, shortest_path_algorithms, shortest_path_routing, simulation, system_algorithm_co_design, task_batching, theoretical_modeling, threshold_based_routing, time_series_forecasting, transformer, weighted_round_robin, wireless_resource_allocation
*Shared methods:* bi_level_optimization, continuous_batching, data_parallelism, distributed_systems, dynamic_programming, dynamic_resource_allocation, greedy_algorithm, integer_linear_programming, kv_cache_management, llm_as_evaluator, load_balancing, mixed_integer_linear_programming, performance_modeling, pipeline_parallelism, proximal_policy_optimization, queueing_theory, queuing_theory, reinforcement_learning, resource_allocation, resource_management, robust_optimization, scheduling_algorithms, speculative_decoding, supervised_learning, tensor_parallelism

This research front centers on the application of Integer Linear Programming (ILP) and advanced heuristic algorithms to optimize resource allocation for Large Language Model (LLM) serving, particularly within disaggregated prefill/decode architectures and heterogeneous, geographically distributed GPU clusters. Key frameworks like Helix, SageServe, Cascadia, and Dynamo are developed to address challenges in dynamic resource provisioning, multi-tenancy, and carbon-aware caching, moving beyond reactive baselines to proactive, optimized solutions.

Papers in this front present diverse contributions, including SageServe's ILP and ARIMA-based forecasting achieving 25% GPU-hour savings on Microsoft O365 traces, and Helix's max-flow MILP formulation yielding up to 3.3x decode throughput on heterogeneous L4/T4/A100 clusters by enabling per-request pipeline scheduling. Cascadia employs bi-level optimization (MILP + Chebyshev) for LLM cascade serving, showing 2.3x throughput gains over SGLang. Other works like AMPD (Dynamo framework) demonstrate 67-340% SLO attainment improvements for multi-round inference via ILP-based offline planning and adaptive routing, while MIGRator uses ILP for dynamic NVIDIA MIG reconfiguration, improving 'Goodput' by ~20%. GreenCache introduces carbon-aware ILP for KV cache management, reducing carbon emissions by ~15%.

This front is maturing rapidly, driven by the critical need for efficient LLM serving infrastructure. The trajectory indicates a shift from foundational ILP formulations to more sophisticated hybrid approaches, integrating RL (TORTA) or advanced heuristics for dynamic, real-time adaptation. Future work will likely focus on scaling these OR solutions to extremely large clusters, accommodating more complex multi-tenant and multi-modal workloads, and integrating with emerging hardware capabilities like CXL memory, while also refining predictive models for greater robustness against real-world variability.

**Papers:**

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**2025-11-12** | Microsoft, University of Illinois Urbana-Champaign, Georgia Institute of Technology, Indian Institute of Science | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Integer Linear Programming (ILP) for resource allocation, combined with ARIMA-based time-series forecasting and reactive heuristics for dynamic scaling and scheduling | *LLM role:* none

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interactive and non-interactive workloads. They validate this on real Microsoft O365 production traces (which they release), demonstrating a 25% reduction in GPU hours and $2.5M/month savings compared to reactive baselines. The primary value for us is the release of the production workload traces—allowing us to benchmark our 'GPUSched' formulations against real-world data rather than synthetic distributions—and their specific ILP formulation for unified capacity management, which directly competes with our internal OR models.

### [SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference](https://arxiv.org/abs/2603.04716)

**2026-03-05** | Kingsoft Cloud | M=4 P=9 I=7 **MUST-READ** *discuss*

*Method:* Hybrid theoretical modeling (M/M/1 queuing theory) and empirical benchmarking for P/D resource calculation | *LLM role:* none

> This paper proposes a hybrid resource allocation method for disaggregated Prefill/Decode (P/D) inference, using M/M/1 queuing theory to model prefill throughput under TTFT constraints and empirical profiling for decode. The results are real and validated on NVIDIA H200 clusters running DeepSeek-V3.1. The key takeaway for us is the validated analytical relationship between TTFT, input length, and effective prefill throughput ($TP_{eff} = TP_{max} - \frac{L_{in}}{TTFT - T_{overhead}}$). We can steal this equation to serve as a cheap, differentiable constraint in our 'GPUSched' OR formulations or fitness functions, replacing expensive simulations.

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [Improving GPU Multi-Tenancy Through Dynamic Multi-Instance GPU Reconfiguration](https://arxiv.org/abs/2407.13126)

**2024-07-18** | UC San Diego, University of Pittsburgh, University of Arizona, University of Georgia | M=6 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) for dynamic Multi-Instance GPU (MIG) reconfiguration with Goodput objective | *LLM role:* none

> MIGRator formulates dynamic NVIDIA MIG partitioning as an Integer Linear Program (ILP) to optimize a compound 'Goodput' metric (SLO + accuracy) for continuous learning workloads. The results on A100s show ~20% gains over baselines like Ekya and PARIS, largely by mitigating the massive ~6s MIG reconfiguration overhead via a 'pre-initialization' lookahead strategy. For our GPUSched project, the key takeaway is the explicit modeling of reconfiguration penalties in the ILP and the technique of pre-assembling instances during idle time to hide latency. While the reliance on 200-second traffic prediction is a potential fragility, the rigorous handling of hardware constraints makes this a strong reference for our OR-based resource allocation work.

### [Cache Your Prompt When It's Green: Carbon-Aware Caching for Large Language Model Serving](https://arxiv.org/abs/2505.23970)

**2026-01-19** | University of Waterloo, Purdue University | M=5 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) based dynamic cache size reconfiguration with SARIMA load prediction and carbon-aware Least Carbon Savings (LCS) cache replacement policy | *LLM role:* none

> Tian et al. propose GreenCache, a framework using Integer Linear Programming (ILP) to dynamically resize KV caches for LLM serving, balancing operational carbon (compute) against embodied carbon (SSD storage). They demonstrate ~15% carbon reduction on Llama-3 70B using Azure traces, though the reliance on simulation rather than live deployment weakens the claims slightly. For our 'OR for AI systems' work, the key takeaway is their 'Least Carbon Savings' (LCS) eviction policy—a heuristic that weighs computation saved against storage cost and recency—which we could adapt for optimizing memory-constrained multi-agent systems (HERMES) or general serving resource allocation.

### [Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding](https://arxiv.org/abs/2510.11331)

**2025-10-13** | Queen Mary University of London, Kyung Hee University, Xidian University, Guangzhou Institute of Technology | M=5 P=7 I=6 

*Method:* Speculative Decoding (SD) with pipeline parallelism, combined with joint optimization of speculation length, task batching, and wireless communication resource allocation | *LLM role:* inference engine

> Zhu et al. propose a distributed Speculative Decoding framework for edge networks, formulating a Mixed-Integer Nonlinear Programming problem to jointly optimize task batching, speculation length, and wireless bandwidth. They solve the batching subproblem using a Dynamic Programming (DP) algorithm, achieving ~30-45% latency reduction over heuristics in simulations, though the approach relies on a rigid assumption of fixed maximum output lengths to remain tractable. The primary takeaway for our 'GPUSched' work is their DP formulation for optimizing batch boundaries in a pipelined draft-verify system, which offers a cleaner mathematical alternative to greedy heuristics for serving schedules. However, the heavy reliance on wireless channel modeling makes the full system less relevant to our datacenter-centric optimization problems.

### [Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via Reinforcement Learning](https://arxiv.org/abs/2507.10259)

**2025-09-16** | Shenzhen University of Advanced Technology, China Mobile Research Institute | M=6 P=9 I=6 **MUST-READ** *discuss*

*Method:* Proximal Policy Optimization (PPO) with Optimal Transport supervision | *LLM role:* none

> TORTA introduces a hierarchical scheduler for distributed LLM inference that uses a macro-level RL agent (PPO) supervised by an Optimal Transport (OT) baseline to manage inter-region allocation, followed by a micro-level greedy allocator. Results on simulated clusters (up to 50 servers) demonstrate a ~15% reduction in latency compared to reactive baselines (like SkyLB) specifically by optimizing for temporal smoothness and reducing switching costs. The key technical takeaway is the use of an exact OR solver (OT) as a dense supervision signal to train a faster RL policy, effectively combining the optimality of OR with the temporal foresight of RL. We should review our GPUSched formulations to ensure we aren't falling into the 'reactive' trap described here; if we are, this hybrid RL-OT architecture is a viable alternative.

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**2025-03-05** | Carnegie Mellon University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Max-flow problem formulation on directed, weighted graphs with Mixed Integer Linear Programming (MILP) for joint model placement and per-request pipeline scheduling | *LLM role:* none

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow solution. Unlike standard static pipeline parallelism, it routes every request dynamically based on edge capacities, achieving up to 3.3x throughput gains over Swarm on mixed GPU clusters (L4/T4/A100). The results are rigorous, backed by both physical cluster experiments and high-fidelity simulations. The critical takeaway is the 'per-request pipeline' abstraction: decoupling request routing from static device assignment allows exact OR methods to maximize utilization of weaker hardware—a technique we should immediately evaluate for our GPUSched project.

### [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs](https://arxiv.org/abs/2502.00722)

**2025-06-05** | University of Cambridge, ETH Zurich, Peking University, The Hong Kong University of Science and Technology, Purdue University | M=5 P=9 I=6 **MUST-READ** *discuss*

*Method:* Mixed-Integer Linear Programming (MILP) for scheduling | *LLM role:* none

> Jiang et al. formulate LLM serving on heterogeneous clouds as a Mixed-Integer Linear Programming (MILP) problem, co-optimizing GPU rental composition, parallelism strategies (TP/PP), and workload routing. They demonstrate ~25% throughput gains over SOTA systems (Helix, HexGen) using vLLM benchmarks, validating the approach with strong empirical ablations. For our **GPUSched** project, the key takeaway is their solver strategy: pre-generating valid configurations to linearize the problem and using a binary search wrapper on the makespan to avoid direct minimization overhead. We should adopt their heuristics for pruning the configuration space (e.g., restricting TP to intra-node) to improve our own solver times.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.


### Front 12 (9 papers) — GROWING

**Density:** 0.33 | **Methods:** llm_code_generation, llm_as_heuristic, llm_in_the_loop, llm_as_evaluator, program_synthesis | **Problems:** heuristic_evolution, resource_allocation, scientific_discovery, ai_for_science, code_generation

*Unique methods:* abstract_syntax_tree, adaptive_iterated_local_search, adaptive_routing, adversarial_learning, agentic_reinforcement_learning, binary_cross_entropy, branch_and_bound, chain_of_thought, code_embedding, constraint_programming_solver, continual_learning, cublas_epilogue, cuda_kernel_optimization, data_algorithm_co_evolution, data_layout_optimization, density_functional_theory, dice_loss, direct_preference_optimization, diving_heuristics, dynamic_analysis, dynamic_thresholding, evolution_of_heuristics, evolutionary_algorithm, evolutionary_computation, exploration_exploitation, funcdyn, gaussian_processes, gpt_4o, graph_neural_network, in_context_learning, iterated_local_search, k_means_clustering, llm_driven_automatic_heuristic_design, local_search, low_code_platform, machine_learning_force_field, map_elites, mixed_integer_nonlinear_programming, mixed_precision_computing, multi_agent_llm_framework, multi_agent_system, neuro_symbolic_ai, online_learning, openevolve, parameter_efficient_fine_tuning, performance_profiling, policynet, prompt_engineering, qlora, race_detection, rubric_reward, ruin_and_recreate, rule_based_routing, self_correction, supervised_finetuning, tensor_cores, tiling, tree_of_thoughts, u_net
*Shared methods:* bayesian_optimization, convex_optimization, grpo, kernel_fusion, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_prompt_optimization, mlp, ppo, program_synthesis, reinforcement_learning, retrieval_augmented_generation, self_improving_search, supervised_fine_tuning, supervised_learning

This research front focuses on advancing LLM-driven evolutionary computation and program synthesis for complex Operations Research (OR) and AI for Science (AI4S) problems. The core theme revolves around developing sophisticated frameworks that leverage large language models to automatically generate, optimize, and refine code, heuristics, and scientific models. Key approaches include multi-agent systems, advanced feedback mechanisms, and novel strategies to improve sample efficiency and generalization in LLM-guided search.

Key contributions include ParEVO, which achieves up to 106x speedup in parallel algorithm synthesis for irregular data using MAP-Elites and DPO fine-tuning. AILS-AHD demonstrates 8 new Best-Known Solutions on CVRPLib by evolving ruin heuristics with LLM-driven Chain-of-Thought voting. DHEvo introduces a data-algorithm co-evolution framework for MILP, outperforming FunSearch by ~60% on Setcover. ConstraintLLM fine-tunes LLMs for Constraint Programming modeling, achieving ~51% accuracy on industrial benchmarks via structural retrieval and iterative self-correction. BAMF-AI4S significantly boosts scientific code generation (+54.3% on HumanEval) using Bayesian surrogate models, while StitchCUDA automates GPU program generation with rubric-based reinforcement learning, yielding 1.72x speedup over baselines. MAESTRO employs an 'Exploration Report Agent' for materials discovery, and NetGPT uses SFT-anchored RL to prevent schema collapse in LLM routing.

This front is rapidly emerging and growing, marked by several

**Papers:**

### [AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework](https://arxiv.org/abs/2603.03233)

**2026-03-03** | Fudan University, Shanghai Innovation Institute, Shanghai Academy of AI for Science | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Bayesian Adversarial Multi-agent Framework for AI4S (BAMF-AI4S) with recursive co-optimization of generated code, test cases, and prompts, guided by a non-LLM-based Bayesian updating rule and Bayesian Optimization for code performance estimation. | *LLM role:* code_writer, decomposition_guide, prompt_optimizer, test_case_generator, solution_generator

> The authors propose a multi-agent framework for scientific code generation that couples an adversarial 'Challenger' (generating difficult test cases) with a 'Solver', governed by a Bayesian update rule. Crucially, they employ Bayesian Optimization with a kernel based on code embeddings (AST + text) to estimate solution quality *before* running expensive tests, effectively acting as a learned surrogate model. Results on SciCode and ScienceAgentBench are strong, showing small models (Qwen-32B) outperforming GPT-4o when using this loop. **The killer feature for us is the surrogate modeling pipeline:** we should immediately steal the idea of using GP surrogates on code embeddings to filter candidates in our evolutionary search, potentially reducing our evaluation costs by orders of magnitude.

### [Reasoning-Driven Design of Single Atom Catalysts via a Multi-Agent Large Language Model Framework](https://arxiv.org/abs/2602.21533)

**2026-02-25** | Georgia Institute of Technology, Korea University, Sogang University, Ewha Womans University | M=7 P=3 I=8 *discuss*

*Method:* Multi-Agent-based Electrocatalyst Search Through Reasoning and Optimization (MAESTRO) framework using LLM agents and a Machine Learning Force Field (MLFF) surrogate model | *LLM role:* evolutionary_search

> Mok et al. propose MAESTRO, a multi-agent LLM framework for optimizing single-atom catalysts that explicitly separates search into exploration and exploitation phases, bridged by a textual 'Exploration Report.' Results are validated against high-fidelity DFT calculations, showing the system learns to break theoretical scaling relations via in-context learning, outperforming memory-less baselines. The key takeaway for us is the **Exploration Report Agent**: instead of just passing the best candidates to the exploitation phase, the system pauses to write a natural language strategy guide summarizing 'what worked and what didn't' from the exploration phase. We should steal this mechanism for AlgoEvo to let the search agent 'learn' from the initial population generation rather than just selecting from it.

### [Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design](https://arxiv.org/abs/2602.23092)

**2026-02-26** | City University of Hong Kong, Southern University of Science and Technology | M=7 P=9 I=8 **MUST-READ** *discuss*

*Method:* Adaptive Iterated Local Search (AILS) with LLM-driven Evolutionary Computation for Automatic Heuristic Design (AHD) of ruin heuristics | *LLM role:* heuristic_generator

> This paper integrates LLM-driven evolutionary search into the AILS framework to evolve 'ruin' heuristics for CVRP, employing a Chain-of-Thought 'voting' mechanism to filter out poor heuristics before expensive evaluation. The results are empirically strong: they claim 8 new Best-Known Solutions on the CVRPLib large-scale benchmark, outperforming HGS and AILS-II. **Key Takeaway:** We should steal the 'acceleration mechanism'—using the LLM to predict heuristic quality via CoT prior to execution—to address the sample efficiency bottleneck in our own evolutionary search loops. This is a direct proof-of-concept that LLM-evolved components can beat hand-crafted SOTA on hard OR instances.

### [Learning to Incentivize: LLM-Empowered Contract for AIGC Offloading in Teleoperation](https://arxiv.org/abs/2508.03464)

**2025-08-05** | University of Houston, The Pennsylvania State University, University of Florida, Kyung Hee University, China University of Petroleum (East China), Prairie View A&M University | M=8 P=5 I=8 **MUST-READ** *discuss*

*Method:* LLM-evolved solver for ASP setting inference (P2) combined with convex optimization for contract derivation (P3) | *LLM role:* evolutionary_search

> Zhan et al. propose an LLM-based evolutionary framework to generate Python solvers for inferring hidden agent parameters in contract design (a bilevel OR problem). While the experiments are toy-scale (N=7 actions) and benchmarks are weak, the methodological architecture is highly relevant: they separate 'short-term reflectors' (analyzing parent pairs) from a 'long-term reflector' (aggregating insights across generations) to guide the Mutation LLM. This is a concrete, transferable implementation of evolutionary memory that we should test to improve sample efficiency in our own code-evolving agents.

### [ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution](https://arxiv.org/abs/2603.02510)

**2026-03-03** | Google DeepMind, Yale University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Evolutionary Coding Agent (ECA) with fine-tuned LLMs for C++ ParlayLib and Rust RPB, guided by compiler, race detector, and performance profiler feedback | *LLM role:* evolutionary_search

> ParEVO synthesizes high-performance parallel algorithms for irregular data by combining fine-tuned LLMs with an Evolutionary Coding Agent (ECA) that uses compilers, dynamic race detectors, and hardware profilers as absolute fitness functions. The results are highly rigorous and backed by extensive hardware benchmarks, showing up to a 106x speedup on ParEval and matching or beating expert human baselines on complex graph problems (e.g., 4.1x speedup on Maximal Independent Set). The single most useful takeaway for us is their evolutionary architecture: they use MAP-Elites (categorizing by code length, cyclomatic complexity, and sync primitives) to prevent diversity collapse, and they fine-tune their base models using Direct Preference Optimization (DPO) on past evolutionary trajectories to drastically improve sample efficiency. This is a must-read paper that directly advances our primary focus on LLM evolutionary search, providing concrete techniques we should immediately steal for AlgoEvo.

### [ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming](https://arxiv.org/abs/2510.05774)

**2025-10-07** | University of Oxford, University of Chinese Academy of Sciences, Hangzhou Institute for Advanced Study, ISCAS, University of Science and Technology Beijing | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Neuro-Symbolic Framework integrating Multi-Instruction Supervised Fine-Tuning (SFT) of an open-source LLM, Constraint-Aware Retrieval Module (CARM), Tree-of-Thoughts (ToT) exploration, and Iterative Self-Correction with Guided Retrieval. | *LLM role:* code_writer

> ConstraintLLM fine-tunes a 32B model for Constraint Programming (CP) modeling, utilizing a "Constraint-Aware Retrieval Module" (CARM) that retrieves few-shot examples based on extracted constraint signatures (e.g., `AllDifferent`, `Cumulative`) rather than text embeddings. They also employ a Tree-of-Thoughts search pruned by test case execution and an iterative self-correction mechanism that retrieves "correction paths" (error-to-fix trajectories). Results are strong: on their new industrial benchmark (IndusCP), they achieve ~51% accuracy with a 32B model, matching or beating GPT-4o and DeepSeek-V3. **Key Takeaway:** The shift from semantic retrieval to *structural* retrieval (matching constraint profiles) is the "stealable" insight; we should implement this for our OR modeling tasks immediately, ignoring surface-level problem descriptions in favor of logical signatures. This directly impacts our OR-Bench and automated formulation work.

### [DHEvo: Data-Algorithm Based Heuristic Evolution for Generalizable MILP Solving](https://arxiv.org/abs/2507.15615)

**2025-07-21** | Harbin Institute of Technology, Huawei Noah’s Ark Lab, Nanyang Technological University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Data-Algorithm Co-evolution Framework (DHEvo) with LLM-based Multi-Agent Evolution System (MA-Evolution System) | *LLM role:* code_writer

> DHEvo introduces a 'data-algorithm co-evolution' framework that iteratively evolves heuristic code while simultaneously filtering the training instance set to retain only 'representative' instances (those where current heuristics perform well/stably). Empirical results on SCIP diving heuristics show it outperforms FunSearch and EoH by ~60% on Setcover while significantly reducing performance variance, validating the claim that dynamic data curation prevents overfitting. The key takeaway is the counter-intuitive curriculum strategy: rather than training on the hardest instances, filtering for instances with 'regular' feasible regions (high fitness) stabilizes the evolutionary search for code. We should immediately test this dynamic instance filtering in AlgoEvo to improve sample efficiency and generalization.

### [StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning](https://arxiv.org/abs/2603.02637)

**2026-03-03** | University of Minnesota-Twin Cities | M=8 P=6 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-agent framework with rubric-based agentic reinforcement learning (GRPO) | *LLM role:* decomposition_guide, code_writer, evaluator

> StitchCUDA automates end-to-end GPU program generation using a multi-agent framework, but its core contribution is a training recipe that solves reward hacking in code optimization. They decompose expensive multi-turn agentic RL into single-turn 'atomic skills' (generation vs. refinement) and use GRPO with an LLM-evaluated 'Rubric Reward' (e.g., 'Did you use tiling?') rather than just sparse outcome metrics. This prevents the model from gaming the system (e.g., wrapping PyTorch code) and forces actual optimization behavior. We should steal the atomic skill decomposition to drastically reduce training costs for AlgoEvo and implement Rubric Rewards to fix our process reward models.

### [Optimizing NetGPT via Routing-Based Synergy and Reinforcement Learning](https://arxiv.org/abs/2511.22217)

**2025-11-27** | Zhejiang University, Huawei Technologies Co., Ltd., Zhejiang Lab, Macau University of Science and Technology, The University of Electro-Communications, Shenzhen CyberAray Network Technology Co., Ltd | M=5 P=6 I=7 *discuss*

*Method:* Unified router score with state-dependent fallback threshold and schema-preserving reinforcement learning (PPO with SFT anchor) for edge LLM policy update | *LLM role:* heuristic_generator

> Chen et al. propose a cloud-edge routing framework that dynamically offloads tool-calling tasks based on network conditions (RTT/Bandwidth) and a learned confidence score, while simultaneously updating the edge model via PPO. Results on 8,000 tasks show that dynamic thresholds outperform static baselines like FrugalGPT, and crucially, that interleaving SFT updates is required to prevent JSON schema collapse during RL. The primary takeaway for us is the 'SFT-anchored' update strategy: alternating between RL (for reward maximization) and SFT (on valid outputs) is a simple, effective stabilizer for maintaining structural constraints (like code syntax or JSON) during optimization. We should test this anchoring technique in AlgoEvo to keep evolved heuristics syntactically valid while maximizing fitness.


### Front 5 (8 papers) — STABLE

**Density:** 0.46 | **Methods:** llm_fine_tuned, program_synthesis, llm_code_generation, reinforcement_learning, supervised_fine_tuning | **Problems:** resource_allocation, llm_inference_scheduling, gpu_scheduling, linear_programming, kv_cache_management

*Unique methods:* adaptive_control, asymptotic_analysis, batch_scheduling, batching, binomial_thinning, bootstrapping, causal_inference, causal_intervention, competitive_ratio_analysis, data_augmentation, decode_prioritized_scheduling, demand_prediction, discrete_time_markov_chains, doobs_inequality, fair_queuing, fastertransformer, fcfs_scheduling, fluid_dynamics_approximation, fluid_limits, heuristic_filtering, hierarchical_clustering, instruction_tagging_system, instruction_tuning, kingmans_bound, lexicographical_optimization, lindley_recursion, llm_as_model_generator, llm_as_tagger, martingale_theory, mathematical_modeling, memory_centric_cost_modeling, memory_constrained_scheduling, mixed_batching, nested_wait_algorithm, non_preemptive_scheduling, online_algorithms, online_optimization, orca, outlier_detection, prefill_prioritized_scheduling, sarathi_serve, shortest_first, state_synchronization, statistical_testing, stochastic_processes, synthetic_data_generation, test_time_adaptation, test_time_reinforcement_learning, text_embedding, tf_idf, threshold_based_scheduling, union_bound, virtual_time_scheduling, wait_algorithm, work_conserving_scheduling
*Shared methods:* bin_packing, curriculum_learning, greedy_algorithm, group_relative_policy_optimization, integer_programming, linear_programming, llm_as_evaluator, llm_code_generation, llm_fine_tuned, llm_in_the_loop, load_balancing, lyapunov_function, mlp, online_scheduling, program_synthesis, proximal_policy_optimization, queueing_theory, queuing_theory, reinforcement_learning, resource_allocation, scheduling, scheduling_algorithms, self_improving_search, supervised_fine_tuning, vllm

This research front focuses on the intersection of Large Language Models (LLMs) and Operations Research (OR), primarily exploring how LLMs can be fine-tuned and reinforced to automate OR modeling and optimize instruction learning. Key frameworks like ORLM and OR-R1 leverage instruction tuning and test-time reinforcement learning to generate optimization models, while methods like EE-CPO apply linear programming to optimize instruction sets for LLM supervised fine-tuning. Concurrently, a significant portion of the front applies rigorous OR principles to optimize LLM inference scheduling, addressing challenges like KV cache management and throughput.

Specific contributions include ORLM-LLaMA-3, which, trained with the OR-Instruct framework, significantly outperforms GPT-4 on benchmarks like NL4OPT and IndustryOR for automated optimization modeling. OR-R1 further advances this by integrating Supervised Fine-tuning with Test-Time Group Relative Policy Optimization (TGRPO), achieving +7.6% average accuracy over LLMOPT with 1/10th of the training data and improving Pass@1 consistency. In instruction optimization, EE-CPO, utilizing causal intervention and linear programming, consistently outperforms DEITA by up to +1.73 on AlpacaEval2.0. For LLM inference, papers introduce algorithms like Memory Constrained Shortest First (MC-SF) for near-optimal latency, Nested WAIT for multi-stage scheduling with high-probability memory bounds, Staggered Batch Scheduling (SBS) reducing Time-to-First-Token by 30-40%, and Justitia, a fair scheduler reducing average job completion time by ~60% over VTC.

This front is stable, exhibiting a dual trajectory: the application of LLMs for automated OR modeling is rapidly emerging with novel RL-based approaches, while the use of OR for LLM inference scheduling is maturing, establishing rigorous theoretical foundations and practical, high-performance algorithms. Future work will likely see a convergence, with LLMs potentially generating or optimizing OR-based scheduling policies. The next generation of papers will focus on integrating advanced reinforcement learning techniques (e.g., RLHF, DPO) for solution ranking in automated OR modeling, developing multi-agent LLM systems for complex problem-solving, and extending OR scheduling frameworks to dynamic, multi-GPU, and multi-tenant LLM inference environments.

**Papers:**

### [ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling](https://arxiv.org/abs/2405.17743)

**2025-04-04** | Columbia University, Duke University, Shanghai Jiao Tong University, The Chinese University of Hong Kong, Shenzhen, Shenzhen Research Institute of Big Data, Shanghai University of Finance and Economics, Cardinal Operations | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Instruction tuning of open-source LLMs using semi-automated synthetic data generated by OR-Instruct framework | *LLM role:* data_synthesis, model_generator, code_writer

> The authors propose OR-Instruct, a framework that uses GPT-4 to synthesize over 32k optimization modeling pairs (natural language to COPT code) to fine-tune 7B-scale models (ORLM). They demonstrate that these fine-tuned models outperform GPT-4 on their new 'IndustryOR' benchmark, a result that appears robust given the specialized nature of the task. The most valuable takeaway is their specific data augmentation strategy—iteratively altering constraints and injecting specific modeling techniques (e.g., Big M)—which provides a concrete recipe we can steal to generate diverse instances for our OR-Bench project. While the methodology is standard instruction tuning, the resulting artifacts (benchmark and model) establish a new baseline for automated OR modeling that we cannot ignore.

### [Beyond IID: Optimizing Instruction Learning from the Perspective of Instruction Interaction and Dependency](https://arxiv.org/abs/2409.07045)

**2024-09-11** | Beijing Academy of Artificial Intelligence | M=6 P=5 I=7 *discuss*

*Method:* Causal Intervention based Instruction Correlation Analysis and Ability Taxonomy Induction; Effect Equivalence-based Linear Programming for Category Proportion Optimization (EE-CPO); Dependency Taxonomy Guided Curriculum Supervised Fine-Tuning (DT-CSFT) | *LLM role:* tagger, base_model_for_analysis_and_finetuning

> The authors propose optimizing SFT data mixtures using Linear Programming (EE-CPO) by modeling the 'interaction' (synergy/antagonism) between instruction categories, rather than treating them as IID. They empirically derive a dependency taxonomy showing Math and Code are fundamental 'root' capabilities required before learning complex tasks, validating this via curriculum learning experiments that beat DEITA. The results are solid (+1.73 AlpacaEval over DEITA), though the cost of deriving the interaction matrix (training N models) is high. **Takeaway:** The 'Effect Equivalence Coefficient' matrix combined with an LP solver is a rigorous OR formulation for resource/data allocation that we should steal to optimize heuristic populations in our evolutionary search frameworks.

### [OR-R1: Automating Modeling and Solving of Operations Research Optimization Problem via Test-Time Reinforcement Learning](https://arxiv.org/abs/2511.09092)

**2025-11-12** | The Hong Kong University of Science and Technology, Arizona State University, University of North Carolina at Chapel Hill | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Supervised Fine-tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) with a composite reward function | *LLM role:* code_writer, heuristic_generator, evaluator

> OR-R1 introduces a data-efficient framework that fine-tunes Qwen3-8B using Supervised Fine-Tuning (SFT) followed by Test-Time Group Relative Policy Optimization (TGRPO) on unlabeled data. The results are empirically strong: it outperforms ORLM and LLMOPT while using only 1/10th of the synthetic training data, specifically narrowing the consistency gap between Pass@1 and Pass@8. The key takeaway for us is the effectiveness of GRPO (normalizing rewards within a sampled group to estimate baselines) combined with majority-voting rewards; this eliminates the need for a separate critic model while significantly improving code generation consistency. We should immediately evaluate GRPO as a lightweight alternative to PPO for the 'RL-infused' components of our evolutionary search methods.

### [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115)

**2026-01-15** | Massachusetts Institute of Technology, Microsoft Research, HKUST | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Memory Constrained Shortest First (MC-SF) online batching and scheduling algorithm | *LLM role:* none

> This paper formulates LLM inference scheduling as an Integer Program (IP) that explicitly models the linear memory growth of KV caches, and proposes a 'Memory Constrained Shortest First' (MC-SF) algorithm. The results are rigorous, showing MC-SF achieves near-optimal performance (within 5% of hindsight optimal) on synthetic data and significantly outperforms standard FCFS/threshold heuristics on real traces. The critical takeaway is the 'future feasibility check' (Eq. 5), which validates that a batch will *remain* within memory limits throughout the generation process based on predicted output lengths—a necessary deviation from standard static-size scheduling. This is foundational reading for our GPUSched project, providing both the exact IP baseline we need and a strong heuristic to benchmark against.

### [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

**2026-01-05** | Massachusetts Institute of Technology, Peking University, Alibaba Group | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Fluid dynamics approximation and threshold-based online scheduling (WAIT and Nested WAIT algorithms) | *LLM role:* none

> This paper formulates LLM inference as a multi-stage stochastic scheduling problem, introducing 'Nested WAIT'—a threshold-based algorithm that handles unknown output lengths by letting prompts classify themselves as they survive into deeper decode segments. Unlike heuristic baselines (vLLM, Sarathi), they provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, validated on A100 simulations. The key takeaway is the 'nested segment' mechanism: instead of predicting job size, structure the queue so short jobs exit early and long jobs naturally migrate to lower-priority/protected tiers, effectively decoupling the memory risk. We should immediately evaluate this threshold logic for our GPUSched formulations, as it likely outperforms our current predictive or FCFS approaches for handling KV cache growth.

### [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](https://arxiv.org/abs/2512.16134)

**2025-12-18** | Baidu Inc. | M=6 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Staggered Batch Scheduling (SBS) with Throughput-Adaptive Interval Control, Multi-tier State Synchronization, Prioritized Batch Allocation Algorithm (PBAA) for Prefill, and IQR-Aware Lexicographical Decode Scheduling for Decode | *LLM role:* none

> Tian et al. introduce Staggered Batch Scheduling (SBS) for DP+EP architectures, enforcing a buffering window to enable global bin-packing rather than immediate dispatch, which they prove causes Head-of-Line blocking in non-preemptive prefill phases. Tested on a production H800 cluster serving DeepSeek-V3, they demonstrate a 30-40% reduction in TTFT and a ~20% throughput increase backed by clear utilization metrics. The most valuable takeaway for our GPUSched project is their 'IQR-aware lexicographical' scheduling heuristic for the Decode phase, which robustly balances batch size against KV-cache memory variance—a constraint logic we should immediately adopt. This work validates that discrete batching is superior to continuous dispatch for MoE models, necessitating an update to our queuing theory models.

### [Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents](https://arxiv.org/abs/2504.07347)

**2025-04-24** | Cornell University, Columbia University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Queueing-theoretic framework with discrete-time Markov chains and fluid limits for analyzing work-conserving scheduling algorithms | *LLM role:* none

> Li et al. formulate a batch queueing model for LLM inference, proving that 'work-conserving' algorithms (like Sarathi-Serve) which mix prefill and decode tokens are throughput-optimal, whereas separated strategies (vanilla vLLM, FasterTransformer) are theoretically unstable. The results are rigorous, combining fluid limit proofs with empirical validation on A100s showing queue blow-ups in non-optimal schedulers. The key takeaway is the precise definition of stability for token-level batching and the counter-intuitive finding that these locally optimal policies can fail in multi-agent networks due to cyclic resource dependencies. This is foundational reading for our GPUSched project and directly informs how we should model resource allocation for our multi-agent optimization systems.

### [Justitia: Fair and Efficient Scheduling for LLM Applications](https://arxiv.org/abs/2510.17015)

**2025-10-19** | Shanghai Jiao Tong University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Virtual-time based fair queuing with memory-centric cost modeling and MLP-based demand prediction | *LLM role:* none

> Justitia introduces a scheduler for LLM agents that prioritizes applications based on their 'virtual finish time' (derived from a theoretical fair-sharing model) but executes them with full resource saturation to minimize completion time. The authors demonstrate a ~60% reduction in average job completion time compared to state-of-the-art fair schedulers (VTC) on vLLM, backed by rigorous experiments and theoretical delay bounds. The key takeaway is the 'KV token-time' cost metric (pd + d^2/2) which accurately captures memory bottlenecks in auto-regressive generation, and the insight that 'long-term fairness' allows for short-term resource saturation. This is immediately actionable for your GPUSched project and relevant for optimizing the serving infrastructure of AlgoEvo.


### Front 8 (8 papers) — STABLE

**Density:** 0.68 | **Methods:** mixed_integer_programming, continuous_batching, heuristic_search, load_balancing, graph_partitioning | **Problems:** resource_allocation, llm_serving_optimization, gpu_scheduling, scheduling, makespan_minimization

*Unique methods:* active_request_capping, adaptive_thresholding, anomaly_detection, attention_kernels, autoscaling, constrained_optimization, decode_limit, decode_router, disaggregated_expert_parallelism, dynamic_offset_adjustment, dynamic_rebatching, early_exit_llms, fine_grained_scheduling, first_come_first_serve, fluid_approximation, gate_and_route_policy, gaussian_process_regression, gemm, gemv, genetic_algorithm, graph_partitioning, gurobi, heuristic_algorithm_design, heuristic_search, hybrid_optimization, joint_optimization, kkt_conditions, kv_caching, lagrangian_heuristic, linear_performance_models, llm_serving_systems, makespan_minimization, many_server_queueing, mathematical_analysis, matrix_multiplication, maximum_likelihood_estimation, memory_constrained_bayesian_optimization, multi_level_search, online_clustering, optimal_gemm_tiling, optimization_problem_formulation, ordinary_least_squares, ping_pong_pipeline, preemptive_scheduling, prefill_admission_gate, queueing_network, real_time_tbt_deadline_tracking, resource_aware_dynamic_scheduler, rolling_updates, scheduling_strategies, shortest_prefill_first_ordering, sla_aware_scheduling, slo_aware_llm_inference_scheduler, state_copying, stochastic_control, stream_processing, successive_halving, task_scheduling, token_budgeting, virtual_memory_management
*Shared methods:* bi_level_optimization, bin_packing, continuous_batching, convex_optimization, cost_modeling, data_parallelism, distributed_systems, dynamic_resource_allocation, linear_programming, llm_inference_serving, load_balancing, lyapunov_function, mixed_integer_linear_programming, mixed_integer_programming, online_scheduling, performance_modeling, pipeline_parallelism, queueing_theory, resource_scheduling, scheduling, tensor_parallelism

This research front unifies advanced Operations Research methodologies, including Mixed-Integer Programming, stochastic control, and fine-grained heuristic scheduling, to address the complex challenges of Large Language Model (LLM) inference and training optimization. The core theme revolves around efficiently managing computational resources, particularly GPUs, under diverse conditions such as prefill-decode contention, heterogeneous hardware, and specialized model architectures like Mixture-of-Experts (MoE) and Early-Exit LLMs.

Key contributions include Pang et al.'s hybrid offline-online MIP and Lagrangian heuristic, boosting vLLM utilization by 8.86% and reducing inference time by 10.42s. Lin et al. introduce asymptotically optimal 'Gate-and-Route' policies via stochastic control for prefill-decode contention, while Bari et al.'s RAD/SLAI schedulers achieve a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve. She et al. apply MIP to operator-level parallelism, reducing pipeline bubbles by 50% for DeepSeek V3. FinDEP optimizes MoE inference with 1.02x-1.61x throughput gains, and HetRL leverages multi-level search for 3-9x throughput increases in heterogeneous RLHF scheduling. DREX improves Early-Exit LLM throughput by 2-12% through dynamic rebatching, and Trident integrates MILP and Bayesian optimization for up to 2.01x throughput speedup in multimodal pipelines.

This front is rapidly maturing, characterized by a strong emphasis on rigorous mathematical modeling and empirical validation. The trajectory indicates a shift towards integrating diverse OR techniques to tackle increasingly complex, real-world LLM serving scenarios. Future work will likely focus on developing unified frameworks that combine stochastic programming, adaptive control, and advanced heuristics to manage dynamic, bursty workloads across highly heterogeneous and distributed GPU infrastructures, potentially incorporating LLM-guided meta-heuristics for online adaptation.

**Papers:**

### [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)

**2025-02-14** | Noah’s Ark Lab, Huawei, Tsinghua University | M=6 P=10 I=7 **MUST-READ** *discuss*

*Method:* Hybrid offline-online method combining Minimizing Makespan Bin Packing (offline) with sorting, online preemption, and a Lagrangian-based heuristic (online) | *LLM role:* none

> Pang et al. formulate LLM inference scheduling as a Mixed-Integer Programming (MIP) model, solving it via a hybrid approach: offline bin-packing for request assignment and an online Lagrangian heuristic for prefill-decode preemption. They report a ~9% utilization increase (80.2% to 89.1%) over a vLLM-style baseline on LLaMA-65B, though the evaluation is limited to a single 8-GPU node and assumes deterministic output lengths for the offline component. The most actionable takeaway is their derivation of a simple cost-comparison threshold (prefill cost vs. decode wait cost) to dynamically inject prefill tasks into decoding streams. This provides a concrete, low-overhead heuristic baseline for our GPUSched work.

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.

### [Optimal Scheduling Algorithms for LLM Inference: Theory and Practice](https://arxiv.org/abs/2508.01002)

**2025-12-01** | The University of Texas at Austin | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* Resource-Aware Dynamic (RAD) scheduler for throughput optimality based on optimal GeMM tiling and dynamic prefill/decode resource allocation; SLO-Aware LLM Inference (SLAI) scheduler for practical SLOs using real-time TBT deadline tracking, SPF prefill ordering, and dynamic offset adjustment based on GPU memory utilization. | *LLM role:* none

> Bari et al. develop a queueing-theoretic framework for LLM inference that proves throughput optimality requires satisfying two conditions: optimal GeMM tiling (batch sizes matching hardware tensor core dimensions) and dynamic resource allocation between prefill/decode phases. They propose RAD (theoretical) and SLAI (practical), where SLAI uses a 'last schedulable time' heuristic to delay decode iterations for non-critical requests, thereby freeing up compute for prefill to reduce TTFT. Results are strong, showing a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve on Mistral-7B. For our GPUSched project, the key takeaway is the explicit coupling of batch sizes to LCM(tile_dims) for theoretical optimality and the dynamic slack-based scheduling logic for heterogeneous SLOs.

### [Automatic Operator-level Parallelism Planning for Distributed Deep Learning -- A Mixed-Integer Programming Approach](https://arxiv.org/abs/2503.09357)

**2025-03-12** | Huawei | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed-Integer Programming (MIP) formulation with a bi-level solution framework including a heuristic operation merging step | *LLM role:* none

> She et al. formulate distributed LLM training/inference as a Flexible Distributed Job Shop Scheduling Problem (FDJSSP) solved via Mixed-Integer Programming (MIP) combined with a heuristic graph coarsening step. They demonstrate that this automated approach not only reproduces DeepSeek V3's expert-designed "DualPipe" strategy but, when allowed to search longer, discovers a schedule with 50% fewer pipeline bubbles. The primary takeaway is the effectiveness of the bi-level optimization framework (greedy merging + MIP) to handle the scale of operator-level graphs, proving that formal OR methods can outperform manual system design for LLM infrastructure. This is a mandatory read for our GPUSched project, offering a concrete formulation for operator-level constraints we can directly adapt.

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**2025-12-25** | The Hong Kong University of Science and Technology, Harbin Institute of Technology, Hong Kong Baptist University | M=6 P=7 I=5 *discuss*

*Method:* Fine-grained task scheduling algorithm for disaggregated expert parallelism (DEP) with maximal task overlap, guided by linear performance models and analytical properties (monotonicity, convexity) | *LLM role:* none

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize overlap. The authors achieve 1.02x-1.61x speedups on H20/A6000 clusters compared to PPPipe, backed by solid empirical data. The key takeaway for our 'GPUSched' work is their methodology: deriving analytical properties (monotonicity and convexity) of the scheduling objective to reduce a complex search space into an $O(1)$ online solver, rather than relying on heavy solvers or RL. This confirms that simple linear performance models ($\alpha + \beta x$) are sufficient for accurate online resource allocation in LLM serving.

### [HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments](https://arxiv.org/abs/2512.12476)

**2025-12-13** | Amazon Web Services, ETH Zurich | M=6 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-level search framework with nested successive halving and genetic algorithm with two-level swaps for constrained joint optimization of partitioning and assignment strategies | *LLM role:* none

> HetRL formulates the scheduling of RLHF workflows (PPO/GRPO) across heterogeneous GPUs and networks as a constrained joint optimization problem, solved via a multi-level search combining Successive Halving and Genetic Algorithms. The authors validate this with 20,000 GPU-hours of experiments, demonstrating 3-9x throughput gains over standard systems like 'verl' in heterogeneous settings. The key takeaway is the hierarchical decomposition of the search space (Task Grouping → Coarse Assignment → Fine-grained Assignment) and the use of SHA to efficiently allocate search budget among candidate configurations. This is directly actionable for your 'GPUSched' project and offers a concrete strategy to scale 'AlgoEvo' runs across cheaper, fragmented GPU resources.

### [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](https://arxiv.org/abs/2512.15705)

**2025-12-17** | Microsoft Research, University of Pennsylvania | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Dynamic Rebatching with copy-free rebatching buffer and SLA-aware scheduler | *LLM role:* inference_target

> DREX introduces a system for 'Early-Exit' LLMs that dynamically splits and regroups batches at intermediate layers, using a cost-benefit heuristic (Adaptive Rebatching Threshold) to decide when rebatching is profitable versus forcing execution. Results are solid (2-12% throughput gain on A100s) and backed by real system measurements, not just simulations. The key takeaway for us is the analytical model for rebatching overhead (Eq. 6)—we can lift this constraint directly into our integer programming formulations for the GPUSched project to accurately model the trade-off between batch fragmentation and compute savings. Essential reading only for the serving optimization sub-team; irrelevant for the core evolutionary search group.

### [Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines](https://arxiv.org/abs/2603.02075)

**2026-03-02** | HKUST, Huawei | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Closed-loop adaptive scheduling framework integrating Gaussian Process regression, memory-constrained Bayesian optimization, and Mixed-Integer Linear Programming (MILP) | *LLM role:* none

> Trident optimizes heterogeneous multimodal data pipelines on fixed clusters by integrating Gaussian Process capacity estimation, memory-constrained Bayesian Optimization for configuration tuning, and an MILP for joint parallelism and placement scheduling. The results are real and hardware-backed, demonstrating up to a 2.01x throughput speedup over static baselines and outperforming Ray Data on an 8-node Huawei NPU cluster. The most stealable insight is their MILP formulation for rolling updates (Eq 11-13), which elegantly linearizes cold-start overheads by precomputing discounted throughputs, alongside their use of a probabilistic memory constraint in BO to avoid OOMs during online tuning. This is highly relevant for our GPUSched project; we should evaluate their MILP constraints for cross-node data transfer and configuration transitions when building our own LLM inference scheduling models.


### Front 7 (4 papers) — GROWING

**Density:** 1.00 | **Methods:** llm_in_the_loop, llm_as_heuristic, multi_agent_systems, llm_as_evaluator, test_time_pruning | **Problems:** multi_agent_coordination, mathematical_reasoning, code_generation, error_propagation, resource_allocation

*Unique methods:* adversarial_training, autogen, deduplication, entropy_regularization, failure_driven_mining, independent_proximal_policy_optimization, iterative_refinement, llm_as_judge, multi_agent_reinforcement_learning, multi_agent_systems, offline_reinforcement_learning, policy_gradient, prompt_optimization, risk_averse_quantal_response_equilibria, srpo, strategic_risk_aversion, test_time_pruning, user_simulation
*Shared methods:* integer_linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_in_the_loop, llm_prompt_optimization, policy_optimization, ppo, reinforcement_learning, retrieval_augmented_generation

This research front focuses on enhancing the robustness, efficiency, and coordination of multi-agent LLM systems through a combination of formal optimization, test-time pruning, and risk-averse policy learning. Key frameworks include AgentDropoutV2 for error mitigation, BAMAS for budget-constrained structuring, SRPO for partner generalization, and GEPA for system-level prompt optimization.

AgentDropoutV2 demonstrates significant accuracy gains (+7.92% math) by rectifying erroneous outputs using a failure-driven indicator pool. BAMAS achieves substantial cost reductions (~80%) on benchmarks like GSM8K and MBPP by employing Integer Linear Programming for LLM selection and offline reinforcement learning for topology. SRPO eliminates free-riding equilibria in collaborative tasks, improving joint accuracy by up to 19% on GSM8K multi-agent debate. MAMUT GEPA optimizes multi-agent conversational assistants, yielding a +7.6% rubric pass rate by jointly evolving prompt bundles and calibrating the LLM judge.

This front is rapidly growing, demonstrating a clear trajectory towards more principled and robust multi-agent LLM design. Future work will likely focus on integrating these diverse optimization techniques, exploring dynamic adaptation during execution, and extending their application to more complex, real-world agentic AI settings and human-AI collaboration.

**Papers:**

### [AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning](https://arxiv.org/abs/2602.23258)

**2026-02-26** | Alibaba Group, Harbin Institute of Technology, Shenzhen | M=6 P=7 I=8 *discuss*

*Method:* Test-time rectify-or-reject pruning framework with retrieval-augmented rectifier, failure-driven indicator pool, and dual-stage deduplication | *LLM role:* rectifier, teacher, deduplicator, reasoning_engine

> Wang et al. propose a test-time 'firewall' for multi-agent systems that intercepts messages and validates them against a retrieved set of error patterns (mined from offline failure trajectories). They achieve ~6% accuracy gains on math benchmarks by iteratively rectifying or pruning erroneous outputs before they propagate. The critical takeaway for our AlgoEvo work is the **Failure-Driven Indicator Pool**: we should implement a similar module that mines failed code generations to build a repository of 'forbidden patterns,' allowing a lightweight verifier to prune bad mutations before expensive execution. This effectively turns the 'graveyard' of failed runs into a persistent memory that improves sample efficiency.

### [BAMAS: Structuring Budget-Aware Multi-Agent Systems](https://arxiv.org/abs/2511.21572)

**2025-11-26** | Tsinghua University, Peking University, University of Illinois Urbana-Champaign, Nanyang Technological University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Joint optimization of LLM selection via Integer Linear Programming (ILP) and agent collaboration topology selection via offline reinforcement learning (REINFORCE) | *LLM role:* agents

> BAMAS decouples agent resource provisioning from coordination strategy, using an Integer Linear Programming (ILP) solver to select the optimal set of LLMs under a strict budget and offline RL to select a fixed interaction topology. They demonstrate ~80% cost reduction on GSM8K and MBPP while matching SOTA accuracy, proving that formal optimization beats greedy heuristics for agent allocation. The key takeaway for us is the 'lexicographically optimal' ILP formulation for tier-based LLM selection, which we should steal immediately for our inference resource managers. While their topology search is limited to a fixed library (unlike our evolutionary approach), the hybrid ILP+RL architecture is a strong template for our 'OR for Generative AI' work.

### [Training Generalizable Collaborative Agents via Strategic Risk Aversion](https://arxiv.org/abs/2602.21515)

**2026-02-25** | Caltech | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Strategically Risk-Averse Policy Optimization (SRPO) based on Risk-Averse Quantal Response Equilibria (RQE) | *LLM role:* collaborative_agent

> Qu et al. introduce Strategically Risk-Averse Policy Optimization (SRPO), which trains agents against a 'constrained adversary' that minimizes their reward within a KL-divergence bound of the partner's current policy. Theoretical results prove this objective eliminates free-riding equilibria, and experiments on GSM8K multi-agent debate show it prevents 'lazy' agreement, improving joint accuracy by up to 19% when pairing heterogeneous LLMs (e.g., 0.6B with 4B). The key takeaway is that robustness to partner deviation—enforced via this specific adversarial objective—is a more principled way to fix lazy agent behavior than prompt engineering or simple dropout. We should immediately test this objective in our HERMES debate framework to improve the contribution quality of smaller models.

### [Build, Judge, Optimize: A Blueprint for Continuous Improvement of Multi-Agent Consumer Assistants](https://arxiv.org/abs/2603.03565)

**2026-03-03** | DoorDash, WithMetis.ai | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Prompt-level optimization using GEPA and MAMUT GEPA | *LLM role:* evaluator, evolutionary_search, decomposition_guide, user_simulator

> This paper presents a production-grade framework for optimizing multi-agent systems by jointly evolving prompt bundles (MAMUT) rather than optimizing agents in isolation. They validate this on a grocery assistant, showing that system-level optimization outperforms local sub-agent optimization by ~7% because it captures coordination dynamics (e.g., context passing) that local metrics miss. The most stealable insight is their 'Judge Calibration' loop: they use evolutionary search (GEPA) to optimize the *evaluator's* prompt to match human labels (91.4% agreement) before using that judge to optimize the agents. This is a rigorous solution to the noisy fitness function problem we face in LLM evolutionary search.


### Front 21 (4 papers) — STABLE

**Density:** 0.83 | **Methods:** load_balancing, resource_allocation, sequence_parallelism, distributed_training, distributed_system_design | **Problems:** llm_serving_optimization, communication_optimization, llm_training_efficiency, long_context_llm_training, agentic_llm_inference

*Unique methods:* activation_recomputation, activation_swapping, adaptive_parallelism, cnic_assisted_io, context_parallelism, cpu_offloading, cuda_streams, distributed_storage, distributed_system_design, distributed_training, expert_placement, flash_attention, gpu_memory_optimization, gradient_accumulation, layerwise_prefill, memory_defragmentation, memory_management, pd_disaggregation, qos, rdma, sequence_packing, sequence_parallelism, system_level_optimization, system_optimization, tensor_management, token_routing_profiling, traffic_management
*Shared methods:* cost_modeling, data_parallelism, distributed_systems, dynamic_programming, integer_linear_programming, kv_cache_management, linear_programming, llm_inference_serving, load_balancing, mixed_integer_linear_programming, mixed_integer_programming, pipeline_parallelism, resource_allocation, resource_scheduling, tensor_parallelism

This front unifies research on applying Operations Research (OR) techniques, specifically Integer Linear Programming (ILP), Mixed-Integer Linear Programming (MILP), and Mixed Integer Programming (MIP), to optimize distributed Large Language Model (LLM) serving and training infrastructure. Key systems like DualPath, MoETuner, FlexSP, and MEMO address critical bottlenecks in agentic inference, Mixture-of-Expert (MoE) serving, flexible sequence parallelism, and ultra-long context training by formulating resource allocation and scheduling as optimization problems.

DualPath ([1]) introduced a dual-path KV-Cache loading architecture with CNIC-centric traffic management, achieving up to 1.96x average online serving throughput for agentic LLM inference on datasets up to 64K MaxLen. MoETuner ([2]) leveraged ILP for balanced expert placement and token routing in MoE models, demonstrating a 17.5% end-to-end speedup on multi-node H200 clusters. FlexSP ([3]) utilized MILP and dynamic programming for adaptive sequence parallelism, accelerating LLM training by up to 1.98x on A100 clusters. MEMO ([4]) applied bi-level MIP for fine-grained tensor management, enabling 7B LLMs with 1M context on 8 GPUs and achieving up to 1.97x MFU improvement.

This front is maturing, demonstrating the significant impact of formal OR methods on complex LLM system design. The consistent use of ILP/MILP/MIP for problems like KV-cache management, expert placement, sequence parallelism, and tensor management indicates a robust and validated approach. Future work will likely focus on integrating these static optimization techniques with dynamic, online adaptation strategies, exploring approximate solvers for larger scales, and extending holistic optimization across different parallelism types. The next papers will likely combine these fine-grained OR solutions into more comprehensive, adaptive LLM serving and training frameworks.

**Papers:**

### [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548)

**2026-02-25** | DeepSeek-AI, Peking University, Tsinghua University | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* Dual-path KV-Cache loading architecture with CNIC-centric traffic management and adaptive request scheduling | *LLM role:* none

> Wu et al. introduce DualPath, a system that breaks the storage I/O bottleneck in agentic inference by utilizing idle decode-node bandwidth to load KV-cache and transferring it to prefill nodes via RDMA. The results are highly credible (1.87x throughput increase on DeepSeek-V3) and directly target the long-context, short-append patterns typical of our evolutionary search rollouts. The most valuable technical takeaway is the 'CNIC-centric' traffic management strategy, which isolates bulk KV transfer from latency-sensitive model collectives—a technique we should immediately steal for our own serving infrastructure. While their scheduling logic is a simple heuristic, the architectural change defines a new, more flexible state space for our OR-based resource allocation research.

### [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643)

**2025-02-10** | Georgia Institute of Technology | M=8 P=9 I=7 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) for expert clustering and cluster-to-GPU assignment | *LLM role:* none

> Go et al. formulate the MoE expert placement problem as a two-stage Integer Linear Program (ILP) to balance token load and minimize communication tail latency, exploiting stable token routing dependencies across layers. They demonstrate real-world speedups of 17.5% on multi-node H200 clusters running Mixtral-8x7B, validating the approach with concrete systems measurements rather than just simulation. The key takeaway is the effectiveness of a min-max ILP objective for reducing tail latency in distributed inference, proving that static optimization based on profiling is sufficient for significant gains. This directly supports our 'OR for AI systems' track and provides a strong baseline formulation for our GPU scheduling work.

### [FlexSP: Accelerating Large Language Model Training via Flexible Sequence Parallelism](https://arxiv.org/abs/2412.01523)

**2025-02-11** | Peking University, ByteDance Inc., Beihang University | M=8 P=6 I=7 *discuss*

*Method:* Heterogeneity-adaptive sequence parallelism using MILP and dynamic programming for optimal strategy selection | *LLM role:* none

> FlexSP optimizes distributed LLM training by dynamically assigning varied-length sequences to heterogeneous Sequence Parallelism (SP) groups using a Mixed-Integer Linear Programming (MILP) solver in the loop. The results are solid, showing up to 1.98x speedup on A100 clusters by mitigating communication bottlenecks for short sequences while preventing OOM for long ones. **Key Takeaway:** The authors use Dynamic Programming to 'bucket' similar sequences, drastically reducing the variable count for the MILP solver; this specific technique—reducing problem granularity to make exact solvers feasible in real-time systems—is directly applicable to our 'GPUSched' and inference resource allocation work. While we focus on evolution, this is a definitive reference for our 'OR for AI Systems' track, proving that formal optimization can beat heuristics in dynamic GPU scheduling.

### [MEMO: Fine-grained Tensor Management For Ultra-long Context LLM Training](https://arxiv.org/abs/2407.12117)

**2025-01-15** | Peking University, Tencent Inc. | M=8 P=5 I=7 *discuss*

*Method:* Fine-grained activation memory management combining token-wise recomputation and swapping with bi-level Mixed Integer Programming (MIP) for memory planning | *LLM role:* none

> Memo enables training 7B LLMs with 1M context on 8 GPUs by combining token-wise activation swapping with a bi-level Mixed Integer Programming (MIP) approach to eliminate memory fragmentation. The results are strong (52% MFU vs ~30% for DeepSpeed) and demonstrate that static memory planning via OR solvers outperforms dynamic allocators for repetitive Transformer workloads. The key takeaway is the bi-level MIP strategy—solving the allocation for one layer and broadcasting it—which makes the NP-hard memory planning tractable. We should adapt this MIP formulation for our own GPU scheduling and inference resource allocation (GPUSched) projects.


### Front 38 (4 papers) — STABLE

**Density:** 0.67 | **Methods:** multi_objective_optimization, game_theory, convex_optimization, gradient_descent, llm_as_evaluator | **Problems:** llm_alignment, instruction_following, helpfulness, safety, truthfulness

*Unique methods:* adaptation_safety, adapter_tuning, best_of_k_sampling, black_box_optimization, blockwise_decoding, communication_overlapping, controlled_decoding, data_alignment, diff_pruning, equilibrium_search, game_theory, gradient_aggregation, gradient_descent, inference_time_alignment, llm_alignment, llm_fine_tuning, lora, maximin_optimization, multi_objective_optimization, noon_ppo, optimization_penalty_function, reward_modeling, rlhf, subgraph_scheduling, value_function_learning, zero_sum_game
*Shared methods:* convex_optimization, data_parallelism, dynamic_programming, kernel_fusion, linear_programming, llm_as_evaluator, llm_in_the_loop, pipeline_parallelism, ppo, resource_management, robust_optimization, supervised_fine_tuning, system_design, tensor_parallelism

This research front explores the application of Operations Research techniques, including game theory, convex optimization, linear programming, and dynamic programming, to enhance Large Language Models. The core themes revolve around achieving robust multi-objective LLM alignment, ensuring safety and helpfulness, and optimizing the efficiency of multi-task LLM fine-tuning in datacenter environments.

Key contributions include the Robust Multi-Objective Decoding (RMOD) framework, which uses a maximin two-player game and convex optimization to improve worst-case reward by +1.2% on alignment benchmarks. The PAMA algorithm offers a computationally efficient convex optimization approach for multi-objective RLHF, demonstrating stable convergence where baselines like MGDA-UB oscillate. The Safety Game formulates LLM response selection as a zero-sum LP, achieving up to two-fold accuracy improvements on SafetyBench by dynamically balancing helpfulness and safety. Furthermore, MuxTune introduces a hierarchical scheduler leveraging dynamic programming for multi-tenant PEFT, yielding up to 5x throughput and 5.29x memory reduction compared to NeMo and S-LoRA.

This front is actively maturing, demonstrating a clear trajectory towards more robust and efficient LLM systems through principled OR methodologies. Future work will likely focus on extending these game-theoretic and optimization frameworks to handle more complex, dynamic interactions in LLM agents, such as sequential dialogues and multi-player scenarios. Additionally, there will be continued efforts to integrate these techniques with advanced cluster scheduling policies and adaptive resource management to guarantee performance SLOs and energy efficiency in large-scale LLM deployments.

**Papers:**

### [Robust Multi-Objective Controlled Decoding of Large Language Models](https://arxiv.org/abs/2503.08796)

**2025-03-11** | University College London, University of Basel, Ulsan National Institute of Science and Technology | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Maximin two-player game between adversarially computed reward weights and sampling policy, solvable through Nash equilibrium, reduced to convex optimization, with blockwise best-of-K sampling | *LLM role:* controlled_decoding_target

> RMOD formulates multi-objective decoding as a zero-sum game between a policy and adversarial weights, solving a convex optimization problem at each decoding step to maximize the worst-case value estimate (essentially a Process Reward Model). The results are empirically strong, outperforming MO-DPO and scalarized baselines on alignment benchmarks by dynamically preventing any single objective from collapsing. **Key Takeaway:** The efficient inference-time weight optimization algorithm (Eq. 10) is a 'stealable' mechanism for **AlgoEvo** and **RobustMAS**. We should implement this dynamic adversarial weighting to balance conflicting code metrics (e.g., runtime vs. solution quality) during evolutionary search, replacing our current static scalarization methods.

### [Pareto Multi-Objective Alignment for Language Models](https://arxiv.org/abs/2508.07768)

**2025-08-11** | Ruhr University Bochum | M=7 P=5 I=6 *discuss*

*Method:* PAMA (PAreto Multi-Objective Alignment) algorithm, which transforms multi-objective RLHF into a convex optimization problem with a closed-form solution, combined with Noon PPO. | *LLM role:* subject_of_optimization

> PAMA introduces a computationally efficient algorithm for multi-objective alignment by reformulating the expensive gradient-norm minimization of MGDA into a convex optimization problem with a closed-form solution, reducing complexity from O(n^2d) to O(n). Empirical results on LLaMA-2-7B are robust, showing stable convergence on conflicting objectives (e.g., harmlessness vs. length) where baselines like MGDA-UB oscillate or fail. The single most useful takeaway is the analytical derivation for optimal objective weighting (Theorem 1) and the 'Noon PPO' heuristic (clipping negative advantages); we could port this logic to our multi-objective process reward models in AlgoEvo to balance search signals efficiently. While the NLP experiments are trivial, the gradient balancing mechanism is directly applicable to our multi-objective RL controllers.

### [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](https://arxiv.org/abs/2510.09330)

**2025-12-02** | University of Warwick | M=7 P=4 I=7 *discuss*

*Method:* Two-player zero-sum game formulation solved by a linear programming (LP) solver at inference time to compute minimax equilibrium strategies, using binary probes for helpfulness and safety scores, with a sigmoid penalty for risk. | *LLM role:* agent_response_selection, evaluator

> The authors formulate LLM response selection as a zero-sum game, solving a small Linear Program (LP) at inference time to mix candidate answers such that the expected risk never exceeds a 'safe fallback' baseline. Results are statistically significant, showing ~15% accuracy gains on SafetyBench by effectively managing the trade-off between helpfulness and safety probes. The key takeaway is the 'Adaptation Safety' constraint formulation: using an LP to guarantee that a stochastic policy is no worse than a heuristic baseline is a powerful, lightweight control mechanism we could adapt for selecting evolved algorithms or managing constraints in multi-agent optimization.

### [MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing](https://arxiv.org/abs/2603.02885)

**2026-03-03** | Shanghai Jiao Tong University, National University of Singapore | M=6 P=7 I=7 *discuss*

*Method:* Hierarchical spatial-temporal backbone multiplexing with unified PEFT representations, dynamic programming for task fusion, priority-based subgraph scheduling, and chunk-based data alignment | *LLM role:* subject_of_optimization

> MuxTune introduces a hierarchical scheduler for multi-tenant PEFT that uses Dynamic Programming to optimally fuse tasks (spatial batching) or interleave them (temporal multiplexing) based on a pipeline cost model. Empirical results on H100s show up to 5x throughput gains over NeMo and S-LoRA, validated by ablation studies. The most stealable insight is their **chunk-based data alignment**: instead of standard padding or naive packing, they split packed sequences into fixed-size chunks to balance compute efficiency with memory waste—a trick we should immediately implement for batch evaluation in AlgoEvo and our serving optimization models.


### Front 3 (2 papers) — STABLE

**Density:** 1.00 | **Methods:** mixed_precision_quantization, linear_programming, post_training_quantization, gptq, dynamic_expert_pruning | **Problems:** llm_compression, mixture_of_experts_compression, vlm_compression, memory_optimization, inference_efficiency

*Unique methods:* binary_quantization, dynamic_expert_pruning, dynamic_pruning, expert_pruning, expert_quantization, gptq, gumbel_softmax, hqq, learnable_mask, mixed_precision_quantization, model_compression, moe_llm_compression, post_training_quantization, token_pruning
*Shared methods:* integer_programming, linear_programming, pruning

This research front centers on the MC and MC# frameworks, which leverage Integer Programming (IP) and Linear Programming (LP) to achieve extreme compression of Mixture-of-Experts (MoE) Large Language Models (LLMs) and Vision-Language Models (VLMs). The core unifying theme is the hybrid approach of combining static mixed-precision quantization with dynamic expert pruning, formulated as an optimization problem to minimize quantization error under memory constraints.

Key contributions include the development of Pre-Loading Mixed-Precision Quantization (PMQ) via LP and Online Top-any Pruning (OTP) via Gumbel-Softmax sampling. Paper [1] demonstrated MC# achieving 6.2x weight reduction on DeepSeek-VL2 with less than 2% accuracy loss, and PMQ showing 18.4% better performance than BSP 2.54-bit on Mixtral 8x7b LM-Eval. Paper [2] compressed Mixtral 8x7b to approximately 16GB (fitting on a single RTX 3090) with only a ~4% drop in zero-shot accuracy, significantly outperforming uniform GPTQ. Benchmarks used include C4, M4, WikiText2, and EleutherAI LM Harness.

This front is emerging, with both papers published in 2025, indicating active development in applying OR techniques to MoE compression. The trajectory suggests a focus on refining these hybrid compression strategies. Future work will likely address performance drops on challenging tasks like GSM8K and HumanEval, as well as adapting the frameworks for specific multimodal applications and optimizing them for diverse hardware platforms.

**Papers:**

### [MC#: Mixture Compressor for Mixture-of-Experts Large Models](https://arxiv.org/abs/2510.10962)

**2025-10-13** | NVIDIA Research, National University of Singapore, The University of Hong Kong, Beihang University, Hangzhou Innovation Institute | M=6 P=7 I=7 *discuss*

*Method:* Hybrid compression combining Pre-Loading Mixed-Precision Quantization (PMQ) via Linear Programming and Online Top-any Pruning (OTP) via Gumbel-Softmax sampling | *LLM role:* none

> Huang et al. propose MC#, a compression framework for MoE models that combines static mixed-precision quantization with dynamic expert pruning. They formulate bit-width allocation as an Integer Linear Programming (ILP) problem—optimizing expert importance vs. quantization error—and use a Gumbel-Softmax router for dynamic pruning. Results are strong, achieving 6.2x weight reduction on DeepSeek-VL2 with <2% accuracy loss. **Takeaway:** The ILP formulation (Eq. 7) is a clean, successful application of OR to AI infrastructure that we should replicate for our own resource allocation/scheduling problems; additionally, the differentiable router offers a template for dynamic agent selection in our multi-agent systems.

### [Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270)

**2025-02-22** | The University of Hong Kong, The Chinese University of Hong Kong, Beihang University, Centre for Perceptual and Interactive Intelligence, Hong Kong | M=5 P=7 I=6 *discuss*

*Method:* Hybrid Post-Training Quantization and Dynamic Pruning for MoE-LLMs using Linear Programming for bit-width allocation and significance-aware token protection | *LLM role:* none

> Huang et al. propose a compression framework for MoE-LLMs that uses Integer Programming to optimally allocate mixed bit-widths (1-3 bits) to experts based on activation frequency and routing weights. They achieve strong empirical results, compressing Mixtral 8x7b to ~16GB (fitting on a single RTX 3090) with only a ~4% drop in zero-shot accuracy, significantly outperforming uniform quantization. The key takeaway is the explicit IP formulation for minimizing quantization error under memory constraints—a clean 'OR for AI' pattern we can adapt for our GPU scheduling or memory allocation formulations. While not a methodological advance in evolution, this is highly relevant for our infrastructure: it enables deploying high-quality MoE models on cheaper hardware for our massive AlgoEvo loops.



## Bridge Papers

Papers connecting multiple research fronts:

### [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643)

**TRUE SYNTHESIS** | score=0.60 | Front 21 → Front 11, Front 8

> Go et al. formulate the MoE expert placement problem as a two-stage Integer Linear Program (ILP) to balance token load and minimize communication tail latency, exploiting stable token routing dependen

### [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)

**TRUE SYNTHESIS** | score=0.52 | Front 8 → Front 5, Front 11, Front 17

> Pang et al. formulate LLM inference scheduling as a Mixed-Integer Programming (MIP) model, solving it via a hybrid approach: offline bin-packing for request assignment and an online Lagrangian heurist

### [Optimal Scheduling Algorithms for LLM Inference: Theory and Practice](https://arxiv.org/abs/2508.01002)

**TRUE SYNTHESIS** | score=0.51 | Front 8 → Front 11, Front 5

> Bari et al. develop a queueing-theoretic framework for LLM inference that proves throughput optimality requires satisfying two conditions: optimal GeMM tiling (batch sizes matching hardware tensor cor


---

*Generated by Research Intelligence System*
