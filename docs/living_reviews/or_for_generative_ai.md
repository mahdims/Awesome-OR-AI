# Living Review: OR for Generative AI

**Last Updated:** 2026-04-21

---

## Recent Papers

#### 2026-04-12 (1 papers)

### [GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads](https://arxiv.org/abs/2604.04335)

**2026-04-06** | University of Washington, NVIDIA, Rice University, University of Waterloo, Cisco Research, Independent Researcher | M=6 P=7 I=6 *discuss*

*Method:* SLO-aware dynamic programming scheduler with intelligent video preemption, elastic sequence parallelism, and dynamic batching | *LLM role:* none

> GENSERVE optimizes the co-serving of text-to-image and text-to-video diffusion models on shared GPUs using a dynamic programming scheduler that jointly manages step-level preemption, dynamic batching, and elastic sequence parallelism. The results are empirically backed, demonstrating up to a 44% improvement in SLO attainment over baselines like SRTF on an 8-GPU cluster. The most useful takeaway is their two-stage DP formulation: they first generate a small set of anchored candidate actions (hold, resume, scale SP) per request, then run a knapsack DP to maximize global SLOs in under 2ms. While this targets diffusion models rather than LLMs, the OR formulation for elastic sequence parallelism and batching under strict latency SLOs is directly applicable to our GPUSched project.


#### 2026-04-02 (2 papers)

### [Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference](https://arxiv.org/abs/2603.26498)

**2026-03-27** | IMDEA Software Institute, Universidad Politécnica de Madrid | M=5 P=8 I=7 **MUST-READ** *discuss*

*Method:* Modality-aware dynamic priority scheduling with aging mechanism | *LLM role:* target_of_optimization

> RPS-Serve introduces a modality-aware scheduler for multimodal LLMs that classifies requests into 'rocks' (video), 'pebbles' (image), and 'sand' (text) based on predicted prefill latency and memory, using dynamic priorities and aging to prevent head-of-line blocking. The results are real and backed by solid systems experiments on vLLM, showing a 78.5% reduction in time-to-first-token for latency-critical text requests compared to FCFS and EDF baselines. The core takeaway is that multimodal workloads completely break standard text-only LLM serving assumptions because video/image prefill times and KV-cache footprints are orders of magnitude larger. This matters directly for our GPUSched project: any OR formulation we develop for LLM inference scheduling must now explicitly model this extreme multimodal variance rather than assuming homogeneous text workloads.

### [DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization](https://arxiv.org/abs/2603.25120)

**2026-03-26** | Microsoft Gray Systems Lab, SK Telecom, Yonsei University | M=6 P=5 I=7 *discuss*

*Method:* Data-driven co-optimization of 3D parallelism configuration and runtime microbatch scheduling using empirical profiling, an expected makespan minimization algorithm, and a hybrid ILP/LPT scheduler. | *LLM role:* none

> DFLOP optimizes distributed 3D parallelism for multimodal LLM training by combining offline profiling with an online ILP-based microbatch scheduler to minimize pipeline bubbles caused by heterogeneous data inputs. The results are real and backed by extensive hardware experiments, showing up to 3.6x throughput improvements over Megatron-LM. The single most useful takeaway for us is their hybrid online scheduling architecture: running an ILP solver asynchronously on the CPU to schedule the next batch while the GPU processes the current one, with a strict time limit that falls back to a fast Longest Processing Time (LPT) heuristic. Although this targets training, this exact asynchronous OR-based scheduling architecture and their adaptive throughput correction mechanism are directly stealable for our GPUSched project on LLM serving optimization.


#### 2026-03-26 (1 papers)

### [cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization](https://arxiv.org/abs/2603.19163)

**2026-03-19** | Independent Researcher, Shenzhen, China | M=5 P=6 I=7 *discuss*

*Method:* GPU-accelerated metaheuristic framework with 'one block evolves one solution' CUDA architecture, two-level adaptive operator selection (AOS), and hardware-aware resource management. | *LLM role:* modeling_assistant

> cuGenOpt is a GPU-accelerated metaheuristic framework that uses a 'one block evolves one solution' CUDA architecture and JIT compilation to solve combinatorial optimization problems. The results are rigorously backed by hardware benchmarks, showing it matches OR-Tools on small instances and vastly outperforms MIP solvers, though it struggles with large-scale VRP (>200 nodes) due to memory limits. WHAT WE LEARNED: The framework allows user-defined search operators (CUDA snippets) to be JIT-compiled and evaluated massively in parallel on the GPU. For us, this is highly actionable: we could use cuGenOpt as the high-throughput fitness evaluation backend for AlgoEvo, taking our LLM-generated ALNS operators, compiling them via their JIT pipeline, and evaluating them across thousands of instances in seconds to solve our scalability bottlenecks.


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

*3 fronts detected — snapshot 2026-04-21*

### Front 0 (56 papers) — STABLE

**Density:** 0.01 | **Methods:** llm_in_the_loop, integer_linear_programming, linear_programming, queueing_theory, resource_allocation | **Problems:** resource_allocation, llm_serving_optimization, gpu_scheduling, llm_inference_scheduling, throughput_optimization

*Unique methods:* 3d_parallelism, abstract_syntax_tree, accuracy_estimation, accuracy_scaling, active_request_capping, adapter_parallelism, adapter_tuning, adaptive_correction, adaptive_nano_batching, adaptive_operator_selection, adaptive_queryselect, adaptive_scheduling, adaptive_thresholding, advantage_estimation, adversarial_learning, agent_based_modeling, agentic_ai, aging_mechanism, aimd_controller, algorithm_selection, all_to_all_collectives, anomaly_detection, appo, asymptotic_analysis, attention_kernels, axiomatic_system, bandits_with_knapsacks, batching, bayes_factor, bayesian_modeling, bert_embeddings, best_of_n, bin_packing_algorithm, binary_quantization, binomial_thinning, black_box_evaluation, blockwise_local_distillation, budget_optimal_allocation, budget_partitioning, cache_replacement_policy, caching, capacity_planning_algorithms, causal_inference, causal_intervention, chance_constrained_programming, chunked_prefill, clustering, cnic_assisted_io, code_embedding, communication_overlapping, competitive_ratio_analysis, complementary_slackness, constrained_optimization, convex_minimization, cross_tier_serving, cuda_architecture, cuda_graph, curriculum_learning, data_alignment, data_migration, dataflow_programming, deadline_management, decision_focused_learning, decode_limit, decode_prioritized_scheduling, decode_router, decomposition_algorithm, delta_compression, demand_response, density_functional_theory, dice_loss, diff_pruning, differentiable_optimization, diffusion_model, direct_preference_optimization, dirichlet_process_prior, disaggregated_expert_parallelism, discrete_optimization, discrete_time_markov_chains, distributed_inference, distributed_storage, distributed_system_design, distributionally_robust_optimization, doobs_inequality, dual_learning, dual_linear_program, dynamic_expert_pruning, dynamic_offset_adjustment, dynamic_priority_scheduling, dynamic_pruning, dynamic_rebatching, early_dropping, early_exit_llms, embedding_based_prediction, embedding_only_transfer, empirical_profiling, encoding_abstraction, epoch_gluing, evolutionary_algorithm, evolutionary_algorithms, expected_improvement, expert_placement, expert_pruning, expert_quantization, exploration_exploitation, exponential_smoothing, fastertransformer, fcfs_scheduling, fine_grained_scheduling, first_come_first_serve, fixed_width_policy, flow_control, fluid_approximation, fluid_dynamics_approximation, fluid_limits, fluid_model_analysis, formal_verification, gate_and_route_policy, gaussian_process, gaussian_process_regression, gaussian_processes, gemm, gemv, genetic_algorithm, geometric_algorithm, global_knowledge_distillation, global_resolution, gptq, gpu_kernel_optimization, gpu_parallel_computing, gpu_performance_modeling, gpu_spatial_partitioning, gradient_descent_with_max_oracle, gradient_dynamics_analysis, gradient_scheduling, graph_neural_network, grouped_gemm, grouped_query_attention, gumbel_softmax, gurobi_solver, hardware_aware_optimization, heterogeneous_computing, heuristic_algorithm_design, hierarchical_agglomerative_clustering, hierarchical_clustering, hierarchical_incremental_grouping, hierarchical_scheduling, hqq, hybrid_inference, hybrid_optimization, impact_estimation, impala, inference_optimization, information_entropy_profiling, instruction_tagging_system, inter_model_communicator, island_model, iteration_level_scheduling, jit_compilation, job_scheduling, joint_optimization, kernel_fusion, kingmans_bound, kkt_conditions, knowledge_distillation, kv_cache, l_bfgs_b, lagrangian_heuristic, large_neighborhood_search, layerwise_prefill, learnable_mask, least_carbon_savings, lifo_eviction, limited_preemption_scheduling, lindley_recursion, linear_classifier, linear_performance_models, linear_regression, lipschitz_continuity, llm_as_answer_generator, llm_as_modeling_assistant, llm_as_planner, llm_as_tagger, llm_ensemble, llm_inference_optimization, llm_request_routing, llm_serving, log_cosh_loss, logical_inference, logits_convex_optimization, longest_processing_time_heuristic, low_code_platform, low_rank_adaptation, low_rank_approximation, lyapunov_function, m_g_1_queue, m_g_c_queue, machine_learning_force_field, majority_voting, makespan_minimization, many_server_queueing, martingale_theory, mathematical_analysis, mathematical_modeling, matrix_multiplication, max_flow, max_margin_optimization, maximum_likelihood_estimation, mean_variance_estimation, memory_constrained_bayesian_optimization, memory_constrained_scheduling, memory_optimization, metaheuristics, milp_formulation, milp_general, mixed_batching, mixed_precision_quantization, mixture_of_experts, modality_aware_scheduling, modality_level_partitioning, model_compression, model_parallelism, moe_llm_compression, monte_carlo_methods, mse_loss, multi_agent_llm_framework, multi_armed_bandit, multi_head_attention, multi_instance_gpu, multi_level_search, multi_process_service, multi_region_scheduling, nearest_neighbor_search, nested_wait_algorithm, network_communication_optimization, network_topology_modeling, neural_architecture_search, nonlinear_least_squares, np_hardness_proof, nsga_ii, nvshmem, offline_reinforcement_learning, online_algorithms, online_clustering, online_convex_optimization, online_distillation, online_linear_programming, online_optimization, online_probing, online_scheduling, operator_extensibility, optimal_gemm_tiling, optimization_framework, optimization_problem_formulation, optimization_steered_planning, orca, ordinary_least_squares, pd_disaggregation, peak_end_rule_application, penalty_convex_concave_procedure, performance_modeling, physics_informed_modeling, ping_pong_pipeline, polymatroid_theory, population_based_metaheuristics, post_training_quantization, power_modeling, pre_initialization, predictive_scheduling, prefill_admission_gate, prefill_prioritized_scheduling, primal_dual_methods, probing, problem_decomposition, proxy_tuning, pruning, pytorch, qos, quantile_regression, quantization_aware_fine_tuning, quantization_error_minimization, query_optimization, queryselect, queue_management, queueing_network, radix_tree, rate_distortion_theory, rdma, real_time_optimization, real_time_tbt_deadline_tracking, reconstruction_target_correction, regret_analysis, reliability_engineering, request_classification, resource_aware_dynamic_scheduler, resource_monitoring, resource_scheduling, rolling_updates, rule_based_controllers, runtime_orchestration, saddle_point_optimization, sarathi_serve, sarima, scheduling_algorithm, scheduling_strategies, self_improving_search, set_operations, shared_super_model_abstraction, shortest_first, shortest_path_algorithms, shortest_path_routing, shortest_prefill_first_ordering, shortest_remaining_processing_time, simulated_annealing, simulation, singular_value_decomposition, sla_aware_scheduling, slo_aware_llm_inference_scheduler, soap_framework, soft_actor_critic, speculative_sampling, state_copying, statistical_modeling, statistical_testing, stochastic_control, stochastic_modeling, stochastic_processes, straggler_mitigation, stream_processing, structured_sparsity, subgraph_scheduling, subset_selection, successive_halving, survival_analysis, system_design, system_level_optimization, task_scheduling, temporal_control, text_embedding, threshold_based_scheduling, threshold_rules, time_series_forecasting, time_series_prediction, token_budgeting, token_classification, token_pruning, token_routing_profiling, traffic_management, transformer, transformer_architecture_optimization, tree_search, triton, u_net, ucb, union_bound, unsupervised_evaluation, variance_regularization, virtual_memory_management, wait_algorithm, work_conserving_scheduling, work_stealing, workload_profiling
*Shared methods:* adaptive_sampling, autoscaling, bayesian_inference, bayesian_optimization, beam_search, bin_packing, binary_cross_entropy, continuous_batching, convex_optimization, cost_modeling, cvxpy, data_parallelism, discrete_event_simulation, distributed_systems, distributed_training, dynamic_programming, dynamic_resource_allocation, early_stopping, expert_parallelism, flash_attention, gpu_resource_management, gpu_scheduling, gradient_descent, graph_partitioning, greedy_algorithm, heuristic_initialization, heuristic_search, in_context_learning, integer_linear_programming, integer_programming, kl_divergence, kv_cache_management, kv_cache_optimization, kv_caching, linear_programming, llm_as_evaluator, llm_as_feature_extractor, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_inference_serving, llm_prompt_optimization, llm_serving_optimization, llm_serving_systems, load_balancing, lora, lstm, memory_management, mixed_integer_linear_programming, mixed_integer_nonlinear_programming, mixed_integer_programming, multi_agent_system, multi_agent_systems, multi_objective_optimization, optimal_transport, optimization, parameter_efficient_fine_tuning, pipeline_parallelism, pipelining, policy_gradient, policy_optimization, ppo, preemptive_scheduling, process_reward_model, profiling, program_synthesis, proximal_policy_optimization, queueing_theory, rebase, reinforcement_learning, reinforcement_learning_from_human_feedback, resource_allocation, resource_allocation_optimization, resource_management, retrieval_augmented_generation, robust_optimization, scheduling, speculative_decoding, supervised_fine_tuning, supervised_learning, tensor_parallelism, vllm

This research front unifies diverse applications of Operations Research (OR) techniques, including Integer Linear Programming (ILP), Linear Programming (LP), Mixed Integer Programming (MIP), Satisfiability Modulo Theories (SMT), Dynamic Programming (DP), Optimal Transport (OT), and convex optimization, to enhance the efficiency, reliability, and performance of Large Language Model (LLM) systems. Key themes include optimizing resource allocation for LLM serving and training (e.g., MoE expert placement, GPU scheduling, memory management), improving LLM reasoning and multi-agent coordination, and stabilizing policy optimization and alignment. Frameworks like OCCAM, TORTA, Twill, FlexSP, Helium, Continuum, and MEMO leverage exact OR solvers for system-level optimization, while ODAR, SRPO, ETS, LCO, Safety Game, and PAMA apply OR principles to agent behavior, reasoning, and alignment.

Significant contributions include OCCAM's ILP for optimal model selection, achieving 40% cost reduction on ImageNet-1K with minimal accuracy loss, and Twill's SMT/ILP formulation for GPU compiler optimization, matching hand-tuned Flash Attention performance within 1-2%. For LLM serving, TORTA's PPO with Optimal Transport supervision reduced distributed LLM inference latency by ~15%, while FlexSP's MILP/DP-based sequence parallelism achieved up to 1.98x training speedup. Helium's Templated Radix Tree and MILP-validated scheduler provided up to 39.5x speedup for agentic workflows, and Continuum's KV cache TTL reduced multi-turn agent delay by up to 3.66x. In policy optimization, LCO's convex optimization framework stabilized RL, yielding +7.8% Pass@1 on MATH500 over PPO, and SRPO's risk-averse policy optimization improved multi-agent debate accuracy by 19% on GSM8K.

This research front is rapidly emerging, characterized by a strong drive to integrate rigorous, provably optimal or near-optimal OR solutions into complex AI systems. The trajectory suggests a shift from heuristic-based approaches to mathematically grounded methods for resource management, agent coordination, and policy learning. Future work will likely focus on developing adaptive OR solvers that can operate in dynamic, uncertain environments, integrating real-time feedback loops for continuous optimization, and extending these principled approaches to handle the increasing complexity and scale of multi-modal, long-horizon agentic AI systems, potentially by combining exact OR with faster, learned policies.

**Papers:**

### [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548)

**2026-02-25** | DeepSeek-AI, Peking University, Tsinghua University | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* Dual-path KV-Cache loading architecture with CNIC-centric traffic management and adaptive request scheduling | *LLM role:* none

> Wu et al. introduce DualPath, a system that breaks the storage I/O bottleneck in agentic inference by utilizing idle decode-node bandwidth to load KV-cache and transferring it to prefill nodes via RDMA. The results are highly credible (1.87x throughput increase on DeepSeek-V3) and directly target the long-context, short-append patterns typical of our evolutionary search rollouts. The most valuable technical takeaway is the 'CNIC-centric' traffic management strategy, which isolates bulk KV transfer from latency-sensitive model collectives—a technique we should immediately steal for our own serving infrastructure. While their scheduling logic is a simple heuristic, the architectural change defines a new, more flexible state space for our OR-based resource allocation research.

### [cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization](https://arxiv.org/abs/2603.19163)

**2026-03-19** | Independent Researcher, Shenzhen, China | M=5 P=6 I=7 *discuss*

*Method:* GPU-accelerated metaheuristic framework with 'one block evolves one solution' CUDA architecture, two-level adaptive operator selection (AOS), and hardware-aware resource management. | *LLM role:* modeling_assistant

> cuGenOpt is a GPU-accelerated metaheuristic framework that uses a 'one block evolves one solution' CUDA architecture and JIT compilation to solve combinatorial optimization problems. The results are rigorously backed by hardware benchmarks, showing it matches OR-Tools on small instances and vastly outperforms MIP solvers, though it struggles with large-scale VRP (>200 nodes) due to memory limits. WHAT WE LEARNED: The framework allows user-defined search operators (CUDA snippets) to be JIT-compiled and evaluated massively in parallel on the GPU. For us, this is highly actionable: we could use cuGenOpt as the high-throughput fitness evaluation backend for AlgoEvo, taking our LLM-generated ALNS operators, compiling them via their JIT pipeline, and evaluating them across thousands of instances in seconds to solve our scalability bottlenecks.

### [Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints](https://arxiv.org/abs/2504.11320)

**2026-01-05** | Massachusetts Institute of Technology, Peking University, Alibaba Group | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Fluid dynamics approximation and threshold-based online scheduling (WAIT and Nested WAIT algorithms) | *LLM role:* none

> This paper formulates LLM inference as a multi-stage stochastic scheduling problem, introducing 'Nested WAIT'—a threshold-based algorithm that handles unknown output lengths by letting prompts classify themselves as they survive into deeper decode segments. Unlike heuristic baselines (vLLM, Sarathi), they provide rigorous asymptotic optimality proofs and high-probability bounds against memory overflow, validated on A100 simulations. The key takeaway is the 'nested segment' mechanism: instead of predicting job size, structure the queue so short jobs exit early and long jobs naturally migrate to lower-priority/protected tiers, effectively decoupling the memory risk. We should immediately evaluate this threshold logic for our GPUSched formulations, as it likely outperforms our current predictive or FCFS approaches for handling KV cache growth.

### [Don't Stop Me Now: Embedding Based Scheduling for LLMs](https://arxiv.org/abs/2410.01035)

**2024-10-01** | Harvard University, MIT | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* SRPT with limited preemption using iteration-level, embedding-based output length prediction refined by Bayesian inference | *LLM role:* feature_extractor

> TRAIL optimizes LLM inference scheduling by feeding the LLM's intermediate layer embeddings into a lightweight linear classifier to dynamically predict remaining output length, enabling a memory-aware Shortest Remaining Processing Time (SRPT) policy with limited preemption. The results are empirically strong and backed by real systems implementation; integrated into vLLM, it achieves up to 2x lower mean latency and 24x lower time-to-first-token on the Alpaca dataset compared to standard FCFS. The single most useful takeaway is the 'probing' technique: recycling intermediate layer embeddings (specifically layers 10-15 in Llama-3) to accurately predict generation length on the fly with near-zero overhead (0.03%). This matters significantly for our GPUSched project, as we can steal this embedding-based prediction mechanism to feed highly accurate, dynamic length estimates into our integer programming and stochastic scheduling formulations.

### [PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching](https://arxiv.org/abs/2602.00509)

**2026-02-03** | Kling Infra, Kuaishou Technology | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Continuous Lookahead Pipelining with Gate-Initialized Lookahead Predictor, Hardware-Aware Balance Planning, and Phase-Locked Co-Scheduling | *LLM role:* none

> PROBE optimizes MoE inference by using a distilled MLP to predict next-layer expert activation, enabling proactive load balancing and weight prefetching hidden behind the current layer's computation. The results are strong (1.3x speedup on 235B models) and demonstrate that control plane overheads can be fully masked. The critical takeaway for our `GPUSched` project is the **Lookahead Pipelining** architecture: it carves out a deterministic execution window where we could inject our own specialized solvers (e.g., fast ALNS or IP formulations) to outperform their basic greedy resource allocator. This transforms the stochastic serving problem into a short-horizon deterministic routing problem we are well-equipped to solve.

### [Alignment in Time: Peak-Aware Orchestration for Long-Horizon Agentic Systems](https://arxiv.org/abs/2602.17910)

**2026-02-20** | Lehigh University | M=7 P=5 I=7 *discuss*

*Method:* APEMO (Affect-aware Peak-End Modulation for Orchestration), a runtime scheduling layer that reallocates reasoning effort and repair across a trajectory under fixed computational budgets by operationalizing temporal-affective signals. | *LLM role:* agents_being_orchestrated

> Shi et al. introduce APEMO, a runtime orchestration layer that monitors agent trajectories for behavioral instability (e.g., repetition, drift) and dynamically reallocates a fixed compute budget to 'repair' these segments rather than spreading compute uniformly. The results are statistically rigorous, using bootstrap CIs to demonstrate significant improvements in trajectory robustness and completion rates without model retraining. **Key Takeaway:** We should steal the 'precision repair' logic: instead of uniform sampling in AlgoEvo, we can implement a 'stagnation detector' that triggers deeper inference or multi-agent debate only when the search gets stuck in local optima. This directly addresses our sample efficiency and resource allocation goals.

### [MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing](https://arxiv.org/abs/2603.02885)

**2026-03-03** | Shanghai Jiao Tong University, National University of Singapore | M=6 P=7 I=7 *discuss*

*Method:* Hierarchical spatial-temporal backbone multiplexing with unified PEFT representations, dynamic programming for task fusion, priority-based subgraph scheduling, and chunk-based data alignment | *LLM role:* subject_of_optimization

> MuxTune introduces a hierarchical scheduler for multi-tenant PEFT that uses Dynamic Programming to optimally fuse tasks (spatial batching) or interleave them (temporal multiplexing) based on a pipeline cost model. Empirical results on H100s show up to 5x throughput gains over NeMo and S-LoRA, validated by ablation studies. The most stealable insight is their **chunk-based data alignment**: instead of standard padding or naive packing, they split packed sequences into fixed-size chunks to balance compute efficiency with memory waste—a trick we should immediately implement for batch evaluation in AlgoEvo and our serving optimization models.

### [Optimal Scheduling Algorithms for LLM Inference: Theory and Practice](https://arxiv.org/abs/2508.01002)

**2025-12-01** | The University of Texas at Austin | M=8 P=8 I=7 **MUST-READ** *discuss*

*Method:* Resource-Aware Dynamic (RAD) scheduler for throughput optimality based on optimal GeMM tiling and dynamic prefill/decode resource allocation; SLO-Aware LLM Inference (SLAI) scheduler for practical SLOs using real-time TBT deadline tracking, SPF prefill ordering, and dynamic offset adjustment based on GPU memory utilization. | *LLM role:* none

> Bari et al. develop a queueing-theoretic framework for LLM inference that proves throughput optimality requires satisfying two conditions: optimal GeMM tiling (batch sizes matching hardware tensor core dimensions) and dynamic resource allocation between prefill/decode phases. They propose RAD (theoretical) and SLAI (practical), where SLAI uses a 'last schedulable time' heuristic to delay decode iterations for non-critical requests, thereby freeing up compute for prefill to reduce TTFT. Results are strong, showing a 53% reduction in median TTFT and 26% capacity increase over Sarathi-Serve on Mistral-7B. For our GPUSched project, the key takeaway is the explicit coupling of batch sizes to LCM(tile_dims) for theoretical optimality and the dynamic slack-based scheduling logic for heterogeneous SLOs.

### [inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference](https://arxiv.org/abs/2603.16054)

**2026-03-17** | MBZUAI, McGill University, University of Chicago, Tensormesh Inc | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-phase optimization combining M/G/c Kimura approximation for analytical sweep and discrete-event simulation (DES) for verification | *LLM role:* none

> This paper introduces a two-phase capacity planner (M/G/c analytical sweep followed by discrete-event simulation) to find minimum-cost GPU fleet configurations for LLM inference under strict latency SLOs. The results are rigorously backed by simulation across multiple GPU profiles and workload traces, demonstrating that intuitive sizing rules often fail (e.g., slower GPUs can be cheaper due to KV-slot multipliers, and analytical models approve broken fleets for high-variance agent traffic). The most actionable takeaway for us is their lightweight, physics-informed (W, H, nmax) linear roofline model for GPU performance, which we can directly extract and embed into our integer programming formulations for GPUSched. This is highly relevant for our OR-for-AI infrastructure work, and we should use their open-source workload CDFs and performance constants to benchmark our own scheduling algorithms.

### [BOA Constrictor: Squeezing Performance out of GPUs in the Cloud via Budget-Optimal Allocation](https://arxiv.org/abs/2602.01404)

**2026-02-01** | Carnegie Mellon University, University of Warwick, UNC Chapel Hill | M=8 P=7 I=7 *changes-thinking* *discuss*

*Method:* Budget-Optimal Allocation (BOA) policy derived from convex optimization | *LLM role:* none

> This paper derives a 'Budget-Optimal Allocation' (BOA) policy for ML training jobs, proving via queueing theory that a 'fixed-width' policy (no queueing, constant allocation per epoch) is optimal under general stochastic assumptions. They validate this on AWS, showing a 1.75x improvement in JCT and 2.2x budget reduction compared to Pollux, demonstrating that maximizing cluster utilization/efficiency is actually suboptimal for JCT. The key takeaway is the rigorous convex optimization formulation that replaces heuristic autoscaling, along with a practical 'Epoch Gluing' technique to handle rescaling overheads—both transferable to scheduling our malleable evolutionary search workloads.

### [Robust DNN Partitioning and Resource Allocation Under Uncertain Inference Time](https://arxiv.org/abs/2503.21476)

**2025-09-23** | Tsinghua University | M=6 P=7 I=6 *discuss*

*Method:* Problem decomposition into resource allocation and DNN partitioning subproblems, solved iteratively using Chance-Constrained Programming (CCP), convex optimization, and Penalty Convex-Concave Procedure (PCCP) | *LLM role:* none

> Nan et al. propose a robust optimization framework for DNN partitioning that handles uncertain inference times by converting probabilistic deadlines into deterministic constraints using mean/variance information (Chance-Constrained Programming). They decompose the resulting MINLP into a convex resource allocation problem and a partitioning problem solved via the Penalty Convex-Concave Procedure (PCCP). Experiments on real hardware (Jetson/RTX) demonstrate ~50% energy savings over worst-case baselines while maintaining violation probabilities below the risk threshold. For our 'GPUSched' and 'RobustMAS' work, the key takeaway is the specific analytic transformation of the chance constraint and the use of PCCP as a heuristic for the binary subproblem—a potential alternative to heavy evolutionary search for real-time scheduling components.

### [Improving GPU Multi-Tenancy Through Dynamic Multi-Instance GPU Reconfiguration](https://arxiv.org/abs/2407.13126)

**2024-07-18** | UC San Diego, University of Pittsburgh, University of Arizona, University of Georgia | M=6 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) for dynamic Multi-Instance GPU (MIG) reconfiguration with Goodput objective | *LLM role:* none

> MIGRator formulates dynamic NVIDIA MIG partitioning as an Integer Linear Program (ILP) to optimize a compound 'Goodput' metric (SLO + accuracy) for continuous learning workloads. The results on A100s show ~20% gains over baselines like Ekya and PARIS, largely by mitigating the massive ~6s MIG reconfiguration overhead via a 'pre-initialization' lookahead strategy. For our GPUSched project, the key takeaway is the explicit modeling of reconfiguration penalties in the ILP and the technique of pre-assembling instances during idle time to hide latency. While the reliance on 200-second traffic prediction is a potential fragility, the rigorous handling of hardware constraints makes this a strong reference for our OR-based resource allocation work.

### [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115)

**2026-01-15** | Massachusetts Institute of Technology, Microsoft Research, HKUST | M=8 P=10 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Memory Constrained Shortest First (MC-SF) online batching and scheduling algorithm | *LLM role:* none

> This paper formulates LLM inference scheduling as an Integer Program (IP) that explicitly models the linear memory growth of KV caches, and proposes a 'Memory Constrained Shortest First' (MC-SF) algorithm. The results are rigorous, showing MC-SF achieves near-optimal performance (within 5% of hindsight optimal) on synthetic data and significantly outperforms standard FCFS/threshold heuristics on real traces. The critical takeaway is the 'future feasibility check' (Eq. 5), which validates that a batch will *remain* within memory limits throughout the generation process based on predicted output lengths—a necessary deviation from standard static-size scheduling. This is foundational reading for our GPUSched project, providing both the exact IP baseline we need and a strong heuristic to benchmark against.

### [Balancing Fidelity and Plasticity: Aligning Mixed-Precision Fine-Tuning with Linguistic Hierarchies](https://arxiv.org/abs/2505.03802)

**2026-01-05** | Fudan University, Yale University, Zhejiang University | M=7 P=6 I=7 *discuss*

*Method:* Three-stage gradient-free search pipeline: Fidelity Sensitivity Profiling (information entropy based initialization), global exploration via Pareto-ranking genetic algorithm (NSGA-II), and local refinement using Bayesian Optimization (Gaussian Process with Matérn-5/2 Kernel and Expected Improvement acquisition function). | *LLM role:* none

> QR-Adaptor employs a three-stage gradient-free search (Entropy Profiling → NSGA-II → Bayesian Optimization) to jointly optimize per-layer quantization bits and LoRA ranks. The results are empirically strong, showing that strategic mixed-precision (avg ~3.5 bits) can rival 16-bit baselines by preserving fidelity in deep semantic layers. We should steal their **Fidelity Sensitivity Profiling** (using information entropy to bias the initial evolutionary population) and **Proxy Tuning** (using few-step training as a cheap fitness proxy); these are concrete mechanisms to improve sample efficiency in our own evolutionary search pipelines.

### [GoodSpeed: Optimizing Fair Goodput with Adaptive Speculative Decoding in Distributed Edge Inference](https://arxiv.org/abs/2512.09963)

**2025-12-14** | The University of Sydney, Kyung Hee University | M=5 P=7 I=6 *discuss*

*Method:* Gradient-based scheduling algorithm maximizing logarithmic utility for proportional fairness with adaptive speculative decoding | *LLM role:* heuristic_generator, evaluator

> GoodSpeed uses gradient-based scheduling to dynamically allocate token generation budgets across distributed draft servers, maximizing a logarithmic utility function to balance throughput and fairness. The authors provide rigorous fluid sample path analysis to prove convergence, backed by experiments on H100/L4 clusters, although the baselines (fixed/random allocation) are relatively weak. The most useful takeaway is the mechanism of using exponentially smoothed acceptance rate estimates to drive real-time control in a stochastic system—a robust pattern we should adopt for our own stochastic resource allocation and RobustMAS projects.

### [Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism](https://arxiv.org/abs/2512.21487)

**2025-12-25** | The Hong Kong University of Science and Technology, Harbin Institute of Technology, Hong Kong Baptist University | M=6 P=7 I=5 *discuss*

*Method:* Fine-grained task scheduling algorithm for disaggregated expert parallelism (DEP) with maximal task overlap, guided by linear performance models and analytical properties (monotonicity, convexity) | *LLM role:* none

> FinDEP optimizes distributed Mixture-of-Experts (MoE) inference by partitioning tasks (attention, experts, communication) into fine-grained micro-batches and solving a scheduling problem to maximize overlap. The authors achieve 1.02x-1.61x speedups on H20/A6000 clusters compared to PPPipe, backed by solid empirical data. The key takeaway for our 'GPUSched' work is their methodology: deriving analytical properties (monotonicity and convexity) of the scheduling objective to reduce a complex search space into an $O(1)$ online solver, rather than relying on heavy solvers or RL. This confirms that simple linear performance models ($\alpha + \beta x$) are sufficient for accurate online resource allocation in LLM serving.

### [DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads](https://arxiv.org/abs/2511.00117)

**2025-10-31** | Hewlett Packard Enterprise | M=5 P=7 I=6 *discuss*

*Method:* Reinforcement Learning (Soft Actor-Critic, PPO, APPO, IMPALA) and Rule-Based Controllers evaluated on a physics-informed simulation benchmark, with a multi-agent LLM framework for auditable control | *LLM role:* reasoning_and_planning_agent

> This paper introduces DCcluster-Opt, a high-fidelity simulation benchmark for geo-distributed AI workload scheduling that optimizes for cost, carbon footprint, and SLAs. The authors provide extensive simulation results demonstrating that while standard RL (SAC, PPO) can balance these multi-objective trade-offs, a multi-agent LLM framework distilled from the RL expert generalizes better to larger, unseen clusters (up to 10 DCs). The single most useful takeaway for us is their RL-to-LLM distillation pipeline: training an RL expert on small instances, converting state-action trajectories into text prompts, and fine-tuning an LLM to create an auditable heuristic that scales zero-shot to larger problem sizes. This matters for our GPUSched and multi-agent optimization projects, as we can leverage both the benchmark for testing our OR formulations and the distillation trick to improve the scalability of our own solvers.

### [Enhancing Delta Compression in LLMs via SVD-based Quantization Error Minimization](https://arxiv.org/abs/2506.11087)

**2025-09-27** | Tsinghua University, Fudan University, Southern University of Science and Technology, Shanghai University of Finance and Economics | M=5 P=7 I=6 *discuss*

*Method:* SVD-based mixed-precision delta compression formulated as a 0/1 Integer Linear Programming (ILP) problem for quantization error minimization, integrated with Reconstruction Target Correction (RTC) | *LLM role:* none

> PRINMIX replaces heuristic quantization of LLM delta-weights with a 0/1 Integer Linear Programming (ILP) formulation to minimize reconstruction error. The results are strong and backed by numbers, showing ~22% improvement on AIME2024 and 6x storage savings compared to Delta-CoMe. For us, the key takeaway is not the compression itself, but the formulation: it proves that exact OR modeling outperforms heuristics in LLM serving infrastructure. Additionally, the reported 30-minute solving time suggests this problem could serve as a valuable testbed for our own evolutionary solver acceleration (EvoCut/AlgoEvo).

### [tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models](https://arxiv.org/abs/2602.07263)

**2026-02-06** | University of Illinois Urbana-Champaign | M=5 P=8 I=6 *discuss*

*Method:* Shared Super-Model (SSM) abstraction, fused LoRA kernel with adaptive nano-batching, and online residual-capacity-aware scheduling algorithm | *LLM role:* none

> tLoRA optimizes multi-tenant LoRA training by fusing heterogeneous adapters into a 'Shared Super-Model' and employing an online scheduler that groups jobs based on residual GPU capacity and urgency. They report 1.2–1.8x throughput improvements and ~5x faster job completion times on A100 clusters compared to mLoRA, backed by realistic trace-driven experiments. For our GPUSched and resource allocation work, their hierarchical incremental grouping strategy serves as the state-of-the-art heuristic baseline we must outperform; additionally, their adaptive nano-batching (AIMD controller) is a transferable engineering trick for overlapping communication in distributed LLM workloads.

### [Beyond IID: Optimizing Instruction Learning from the Perspective of Instruction Interaction and Dependency](https://arxiv.org/abs/2409.07045)

**2024-09-11** | Beijing Academy of Artificial Intelligence | M=6 P=5 I=7 *discuss*

*Method:* Causal Intervention based Instruction Correlation Analysis and Ability Taxonomy Induction; Effect Equivalence-based Linear Programming for Category Proportion Optimization (EE-CPO); Dependency Taxonomy Guided Curriculum Supervised Fine-Tuning (DT-CSFT) | *LLM role:* tagger, base_model_for_analysis_and_finetuning

> The authors propose optimizing SFT data mixtures using Linear Programming (EE-CPO) by modeling the 'interaction' (synergy/antagonism) between instruction categories, rather than treating them as IID. They empirically derive a dependency taxonomy showing Math and Code are fundamental 'root' capabilities required before learning complex tasks, validating this via curriculum learning experiments that beat DEITA. The results are solid (+1.73 AlpacaEval over DEITA), though the cost of deriving the interaction matrix (training N models) is high. **Takeaway:** The 'Effect Equivalence Coefficient' matrix combined with an LP solver is a rigorous OR formulation for resource/data allocation that we should steal to optimize heuristic populations in our evolutionary search frameworks.

### [Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference](https://arxiv.org/abs/2508.09229)

**2025-08-12** | AIRI, Skoltech, Avito | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) with load-aware objective function | *LLM role:* none

> This paper formulates the placement of MoE experts (specifically DeepSeek-R1/V3) onto distributed GPU clusters as an Integer Linear Program (ILP) to minimize network hops. While the results are simulation-based (counting hops rather than measuring real latency), they demonstrate that ILP-based placement reduces traffic by ~14-30% compared to Round-Robin, but *only* when the objective function is weighted by historical expert activation frequency; unweighted ILP performs poorly. The key takeaway for our GPUSched project is the specific formulation of the load-aware objective function and the finding that topology-aware placement requires usage statistics to beat simple heuristics. We should adapt this ILP formulation for our resource allocation work.

### [Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents](https://arxiv.org/abs/2504.07347)

**2025-04-24** | Cornell University, Columbia University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Queueing-theoretic framework with discrete-time Markov chains and fluid limits for analyzing work-conserving scheduling algorithms | *LLM role:* none

> Li et al. formulate a batch queueing model for LLM inference, proving that 'work-conserving' algorithms (like Sarathi-Serve) which mix prefill and decode tokens are throughput-optimal, whereas separated strategies (vanilla vLLM, FasterTransformer) are theoretically unstable. The results are rigorous, combining fluid limit proofs with empirical validation on A100s showing queue blow-ups in non-optimal schedulers. The key takeaway is the precise definition of stability for token-level batching and the counter-intuitive finding that these locally optimal policies can fail in multi-agent networks due to cyclic resource dependencies. This is foundational reading for our GPUSched project and directly informs how we should model resource allocation for our multi-agent optimization systems.

### [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**2026-02-16** | University of Cambridge, Peking University, Shanghai Jiao Tong University, Ant Group, Southeast University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Adaptive routing and prefill reordering for online scheduling, combined with an Integer Linear Programming (ILP) based offline deployment planner | *LLM role:* none

> AMPD introduces a disaggregated serving framework tailored for multi-round LLM agents, utilizing an offline ILP solver to optimize resource allocation (TP/DP configurations) and an online adaptive routing mechanism to handle incremental prefill tasks. The results are strong, showing 67-340% improvements in SLO attainment over vLLM and NVIDIA Dynamo by dynamically routing incremental prefill to decode workers when slack exists. For our 'GPUSched' project, the key takeaway is the specific ILP formulation (Eq. 5) for partitioning prefill/decode resources under global GPU constraints, and the insight that multi-agent workflows create a unique 'incremental prefill' bottleneck that standard disaggregation handles poorly.

### [OSCAR: Optimization-Steered Agentic Planning for Composed Image Retrieval](https://arxiv.org/abs/2602.08603)

**2026-02-09** | Shanghai Jiao Tong University, OPPO | M=9 P=3 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Two-stage Mixed-Integer Programming (MIP) for optimal trajectory derivation, followed by VLM-steered in-context learning for inference | *LLM role:* planner_steered_by_optimization

> Wang et al. formulate agentic tool-use planning not as a heuristic search (ReAct), but as a two-stage Mixed-Integer Programming (MIP) problem that solves for the mathematically optimal trajectory (tool selection + set operations) on training data. These 'golden trajectories' are then used as retrieved in-context demonstrations to steer the VLM at inference time, achieving SOTA on CIR benchmarks with only 10% of training data. **Key Takeaway:** We can steal this 'Offline MIP $\to$ Online ICL' paradigm. Instead of relying on noisy online RL or expensive evolutionary loops to guide our AlgoEvo agents, we can solve MIPs on training instances to generate optimal reasoning traces, effectively 'solving' the prompt engineering problem via OR.

### [Dynamic Rebatching for Efficient Early-Exit Inference with DREX](https://arxiv.org/abs/2512.15705)

**2025-12-17** | Microsoft Research, University of Pennsylvania | M=5 P=8 I=6 **MUST-READ** *discuss*

*Method:* Dynamic Rebatching with copy-free rebatching buffer and SLA-aware scheduler | *LLM role:* inference_target

> DREX introduces a system for 'Early-Exit' LLMs that dynamically splits and regroups batches at intermediate layers, using a cost-benefit heuristic (Adaptive Rebatching Threshold) to decide when rebatching is profitable versus forcing execution. Results are solid (2-12% throughput gain on A100s) and backed by real system measurements, not just simulations. The key takeaway for us is the analytical model for rebatching overhead (Eq. 6)—we can lift this constraint directly into our integer programming formulations for the GPUSched project to accurately model the trade-off between batch fragmentation and compute savings. Essential reading only for the serving optimization sub-team; irrelevant for the core evolutionary search group.

### [DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization](https://arxiv.org/abs/2603.25120)

**2026-03-26** | Microsoft Gray Systems Lab, SK Telecom, Yonsei University | M=6 P=5 I=7 *discuss*

*Method:* Data-driven co-optimization of 3D parallelism configuration and runtime microbatch scheduling using empirical profiling, an expected makespan minimization algorithm, and a hybrid ILP/LPT scheduler. | *LLM role:* none

> DFLOP optimizes distributed 3D parallelism for multimodal LLM training by combining offline profiling with an online ILP-based microbatch scheduler to minimize pipeline bubbles caused by heterogeneous data inputs. The results are real and backed by extensive hardware experiments, showing up to 3.6x throughput improvements over Megatron-LM. The single most useful takeaway for us is their hybrid online scheduling architecture: running an ILP solver asynchronously on the CPU to schedule the next batch while the GPU processes the current one, with a strict time limit that falls back to a fast Longest Processing Time (LPT) heuristic. Although this targets training, this exact asynchronous OR-based scheduling architecture and their adaptive throughput correction mechanism are directly stealable for our GPUSched project on LLM serving optimization.

### [Stabilizing Policy Optimization via Logits Convexity](https://arxiv.org/abs/2603.00963)

**2026-03-01** | Tencent Inc, Sun Yat-sen University, Shanghai Innovation Institute, Shenzhen Loop Area Institute | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Logits Convex Optimization (LCO) framework, reformulating RL as an optimal target matching problem using regression-based (LCO-MSE, LCO-LCH) or distribution-based (LCO-KLD) alignment objectives. | *LLM role:* policy_model

> This paper identifies that RL instability in LLMs stems from the non-convexity of surrogate objectives (like PPO) in the logit space, and proposes Logits Convex Optimization (LCO) to match optimal targets derived from the RL objective using MSE, Log-cosh, or KL divergence. The results are rigorously backed by numbers, demonstrating that LCO-KLD and LCO-LCH outperform PPO and GRPO on MATH500 and AlpacaEval across multiple 3B-4B models (Qwen, Llama, Mistral) while requiring 2-3x fewer training samples. The key takeaway is that PPO's gradient spikes come from negative advantage samples in non-convex regions, which can be bypassed entirely by reformulating the RL update as a supervised alignment to an optimal logit target (z_old + A/beta). Because we are actively building RL-infused evolutionary search where sample efficiency and training stability are massive bottlenecks, we should immediately test replacing our PPO/GRPO loss with LCO-LCH or LCO-KLD to stabilize our generator updates.

### [Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective](https://arxiv.org/abs/2603.16104)

**2026-03-17** | National University of Singapore | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* Workflow-aware LLM serving framework integrating proactive KV cache management, global prompt caching, and cost-based cache-aware scheduling based on a templated radix tree | *LLM role:* none

> Helium optimizes LLM serving for batch agentic workflows by modeling them as query plans and using a Templated Radix Tree (TRT) to enable proactive KV caching and cache-aware scheduling. The results are rigorously backed by numbers, demonstrating up to 1.56x speedups over state-of-the-art systems (vLLM, Parrot, KVFlow) on complex multi-agent workflows, and the authors even validate their greedy scheduler's optimality gap against an MILP solver. The most valuable takeaway is the TRT abstraction, which captures global prefix hierarchies across a DAG of LLM calls to maximize KV cache reuse, rather than relying on reactive, per-call caching. This is highly actionable for us: we should directly examine their scheduling formulation and TRT implementation to improve resource allocation and memory management in our GPUSched and HERMES projects.

### [Cache Your Prompt When It's Green: Carbon-Aware Caching for Large Language Model Serving](https://arxiv.org/abs/2505.23970)

**2026-01-19** | University of Waterloo, Purdue University | M=5 P=7 I=6 *discuss*

*Method:* Integer Linear Programming (ILP) based dynamic cache size reconfiguration with SARIMA load prediction and carbon-aware Least Carbon Savings (LCS) cache replacement policy | *LLM role:* none

> Tian et al. propose GreenCache, a framework using Integer Linear Programming (ILP) to dynamically resize KV caches for LLM serving, balancing operational carbon (compute) against embodied carbon (SSD storage). They demonstrate ~15% carbon reduction on Llama-3 70B using Azure traces, though the reliance on simulation rather than live deployment weakens the claims slightly. For our 'OR for AI systems' work, the key takeaway is their 'Least Carbon Savings' (LCS) eviction policy—a heuristic that weighs computation saved against storage cost and recency—which we could adapt for optimizing memory-constrained multi-agent systems (HERMES) or general serving resource allocation.

### [Hybrid Offline-online Scheduling Method for Large Language Model Inference Optimization](https://arxiv.org/abs/2502.15763)

**2025-02-14** | Noah’s Ark Lab, Huawei, Tsinghua University | M=6 P=10 I=7 **MUST-READ** *discuss*

*Method:* Hybrid offline-online method combining Minimizing Makespan Bin Packing (offline) with sorting, online preemption, and a Lagrangian-based heuristic (online) | *LLM role:* none

> Pang et al. formulate LLM inference scheduling as a Mixed-Integer Programming (MIP) model, solving it via a hybrid approach: offline bin-packing for request assignment and an online Lagrangian heuristic for prefill-decode preemption. They report a ~9% utilization increase (80.2% to 89.1%) over a vLLM-style baseline on LLaMA-65B, though the evaluation is limited to a single 8-GPU node and assumes deterministic output lengths for the offline component. The most actionable takeaway is their derivation of a simple cost-comparison threshold (prefill cost vs. decode wait cost) to dynamically inject prefill tasks into decoding streams. This provides a concrete, low-overhead heuristic baseline for our GPUSched work.

### [Optimizing Resource Allocation for Geographically-Distributed Inference by Large Language Models](https://arxiv.org/abs/2512.21884)

**2025-12-26** | Pennsylvania State University, Virginia Tech, Indian Institute of Science | M=5 P=9 I=7 **MUST-READ** *discuss*

*Method:* Three-step heuristic algorithm decomposing MILP: Conservative Greedy Block Placement (CG-BP) for block allocation and Waiting-penalized Shortest-path Request Routing (WS-RR) for request routing. | *LLM role:* none

> This paper formulates geographically distributed LLM inference as a joint block placement and request routing problem, solved via a decomposed MILP heuristic (greedy placement + shortest path routing). The results are real and validated on A100 clusters, showing 60-80% latency reduction over PETALS' native heuristics. The key takeaway for us is their explicit modeling of 'attention cache' memory consumption as a function of concurrent requests—treating this as a dynamic constraint rather than a static buffer is the primary driver of their performance gains. This is a direct blueprint for the constraints we need in our 'GPUSched' formulations, though the algorithmic techniques themselves are standard OR fare.

### [AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework](https://arxiv.org/abs/2603.03233)

**2026-03-03** | Fudan University, Shanghai Innovation Institute, Shanghai Academy of AI for Science | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Bayesian Adversarial Multi-agent Framework for AI4S (BAMF-AI4S) with recursive co-optimization of generated code, test cases, and prompts, guided by a non-LLM-based Bayesian updating rule and Bayesian Optimization for code performance estimation. | *LLM role:* code_writer, decomposition_guide, prompt_optimizer, test_case_generator, solution_generator

> The authors propose a multi-agent framework for scientific code generation that couples an adversarial 'Challenger' (generating difficult test cases) with a 'Solver', governed by a Bayesian update rule. Crucially, they employ Bayesian Optimization with a kernel based on code embeddings (AST + text) to estimate solution quality *before* running expensive tests, effectively acting as a learned surrogate model. Results on SciCode and ScienceAgentBench are strong, showing small models (Qwen-32B) outperforming GPT-4o when using this loop. **The killer feature for us is the surrogate modeling pipeline:** we should immediately steal the idea of using GP surrogates on code embeddings to filter candidates in our evolutionary search, potentially reducing our evaluation costs by orders of magnitude.

### [HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments](https://arxiv.org/abs/2512.12476)

**2025-12-13** | Amazon Web Services, ETH Zurich | M=6 P=8 I=7 **MUST-READ** *discuss*

*Method:* Multi-level search framework with nested successive halving and genetic algorithm with two-level swaps for constrained joint optimization of partitioning and assignment strategies | *LLM role:* none

> HetRL formulates the scheduling of RLHF workflows (PPO/GRPO) across heterogeneous GPUs and networks as a constrained joint optimization problem, solved via a multi-level search combining Successive Halving and Genetic Algorithms. The authors validate this with 20,000 GPU-hours of experiments, demonstrating 3-9x throughput gains over standard systems like 'verl' in heterogeneous settings. The key takeaway is the hierarchical decomposition of the search space (Task Grouping → Coarse Assignment → Fine-grained Assignment) and the use of SHA to efficiently allocate search budget among candidate configurations. This is directly actionable for your 'GPUSched' project and offers a concrete strategy to scale 'AlgoEvo' runs across cheaper, fragmented GPU resources.

### [A Two-Layer Framework for Joint Online Configuration Selection and Admission Control](https://arxiv.org/abs/2602.07663)

**2026-02-07** | Massachusetts Institute of Technology, Stanford University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* SP-UCB-OLP algorithm, which solves an optimistic saddle point problem using UCB for exploration and a threshold-based admission rule derived from a switching-aware fluid oracle | *LLM role:* none

> The authors introduce a 'switching-aware' primal-dual framework for joint configuration selection (e.g., quantization, parallelism) and admission control, demonstrating that dynamically mixing configurations allows for higher resource utilization than any single fixed configuration. Results are rigorous, backed by $\tilde{O}(\sqrt{T})$ regret bounds and experiments on Alibaba cluster traces where the method achieves ~97% competitive ratio (vs. ~85% for greedy). The key takeaway is the 'switching-aware fluid oracle' concept: our resource allocation models for LLM serving must optimize over the convex hull of configurations (mixing CPU-heavy and Mem-heavy setups) rather than searching for a single static optimum. We should adapt their saddle-point formulation for the GPUSched project to handle heterogeneous resource constraints more effectively.

### [OCCAM: Towards Cost-Efficient and Accuracy-Aware Classification Inference](https://arxiv.org/abs/2406.04508)

**2025-02-25** | University of British Columbia | M=6 P=7 I=6 *discuss*

*Method:* Integer Linear Programming for optimal classifier assignment with nearest neighbor-based accuracy estimation and variance regularization | *LLM role:* none

> OCCAM formulates the inference model selection problem as an Integer Linear Program (ILP), using a nearest-neighbor estimator on validation embeddings to predict query-specific model accuracy. The authors provide theoretical guarantees for the estimator's bias and variance, demonstrating 40% cost reduction on ImageNet with <1% accuracy drop compared to heuristic baselines. The key takeaway is the **training-free, NNS-based accuracy estimator** combined with ILP; this avoids training complex routers and provides statistical guarantees. This is directly applicable to our **LLM serving optimization** (GPUSched) work for routing prompts between models of varying costs, and potentially for estimating fitness in **AlgoEvo** without full execution.

### [3D-Learning: Diffusion-Augmented Distributionally Robust Decision-Focused Learning](https://arxiv.org/abs/2602.02943)

**2026-02-03** | University of Houston | M=8 P=9 I=8 **MUST-READ** *discuss*

*Method:* Diffusion-Augmented Distributionally Robust Decision-Focused Learning (3D-Learning) using DDPM with U-Net backbone, integrating dual learning and diffusion policy optimization for min-max optimization | *LLM role:* none

> Wen et al. introduce '3D-Learning,' a framework that replaces analytic ambiguity sets (Wasserstein/KL) in Distributionally Robust Optimization (DRO) with a diffusion model trained via PPO to generate worst-case scenarios. Applied to LLM resource provisioning, they claim ~40-50% regret reduction on OOD Azure traces compared to standard DRO, though training computational cost is high (6.8GB memory vs 35MB). The critical takeaway is the methodology of parameterizing the ambiguity set with a generative model to find 'realistic' adversarial edge cases that respect the data manifold, solving the support shift issue of KL-DRO. We should steal this 'generative ambiguity set' concept for benchmarking our heuristics in RobustMAS and AlgoEvo.

### [CoEdge-RAG: Optimizing Hierarchical Scheduling for Retrieval-Augmented LLMs in Collaborative Edge Computing](https://arxiv.org/abs/2511.05915)

**2025-11-08** | Sun Yat-sen University | M=6 P=8 I=7 *discuss*

*Method:* Hierarchical scheduling combining PPO for online query identification, linear regression for inter-node capacity estimation, and online convex optimization with quadratic approximation for intra-node latency-quality trade-off | *LLM role:* generation, evaluation

> Hong et al. introduce CoEdge-RAG, a hierarchical scheduling framework for distributed edge RAG that combines PPO-based query routing with Online Convex Optimization (OCO) for local resource management. They empirically validate that a quadratic function best approximates LLM inference latency for OCO, allowing them to dynamically resize models and memory allocations under strict SLOs. The standout takeaway is the feedback loop: using PPO to learn a 'semantic routing policy' based on downstream generation quality (Rouge/BERTScore) rather than just load, effectively solving the 'black box' data distribution problem in privacy-preserving multi-agent systems. This hybrid RL/OR control stack is a transferable pattern for our distributed inference and multi-agent optimization work.

### [SkyNomad: On Using Multi-Region Spot Instances to Minimize AI Batch Job Cost](https://arxiv.org/abs/2601.06520)

**2026-01-10** | UC Berkeley, Shanghai Jiao Tong University, AMD, ICSI | M=6 P=9 I=7 **MUST-READ** *discuss*

*Method:* Multi-region scheduling policy guided by a unified monetary cost model, incorporating online availability probing, survival analysis for spot lifetime prediction, and deadline pressure estimation | *LLM role:* none

> SkyNomad presents a multi-region scheduler for AI batch jobs that minimizes cost by dynamically migrating spot instances based on real-time availability probing and survival-analysis-based lifetime prediction. The authors propose a 'Unified Cost Model' that quantifies the monetary value of deadline slack, allowing the system to mathematically trade off migration egress costs against cheaper spot prices. Empirical results on AWS and GCP are strong, demonstrating 1.25-3.96x cost savings over single-region baselines while guaranteeing deadlines. We should immediately adapt their 'Value of Progress' heuristic and lifetime prediction module to optimize our own large-scale parallel evolution infrastructure.

### [Logical Consistency Between Disagreeing Experts and Its Role in AI Safety](https://arxiv.org/abs/2510.00821)

**2025-10-01** | Data Engines | M=7 P=6 I=7 *discuss*

*Method:* Formulating and solving a Linear Programming problem in integer space to identify logically consistent group evaluations for classifiers based on observed agreement/disagreement statistics and universal linear equalities (axioms). | *LLM role:* subject_of_evaluation

> Corrada-Emmanuel formulates the unsupervised evaluation of classifiers as an Integer Linear Programming problem, defining the geometric space of possible ground truths consistent with observed agent disagreements. While the results are primarily theoretical demonstrations on MT-Bench (showing that certain disagreement patterns mathematically preclude accuracy >46%), the methodology is sound. The key takeaway is the concept of 'no-knowledge alarms': using LP constraints to flag when a multi-agent system or process reward model has become logically incoherent. We could implement this as a cheap, rigorous filter in our evolutionary search loops to prune branches where the evaluator agents are demonstrably unreliable.

### [MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/abs/2502.06643)

**2025-02-10** | Georgia Institute of Technology | M=8 P=9 I=7 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) for expert clustering and cluster-to-GPU assignment | *LLM role:* none

> Go et al. formulate the MoE expert placement problem as a two-stage Integer Linear Program (ILP) to balance token load and minimize communication tail latency, exploiting stable token routing dependencies across layers. They demonstrate real-world speedups of 17.5% on multi-node H200 clusters running Mixtral-8x7B, validating the approach with concrete systems measurements rather than just simulation. The key takeaway is the effectiveness of a min-max ILP objective for reducing tail latency in distributed inference, proving that static optimization based on profiling is sufficient for significant gains. This directly supports our 'OR for AI systems' track and provides a strong baseline formulation for our GPU scheduling work.

### [Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees](https://arxiv.org/abs/2604.11001)

**2026-04-13** | The University of Texas at Austin | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Flow-controlled scheduling policy based on queueing theory, limiting the rate of request activation | *LLM role:* subject_of_optimization

> The authors propose a flow-controlled scheduling framework for LLM inference that limits the rate at which requests enter the active set to prevent KV cache overflow and ensure system stability. Results are backed by empirical numbers, demonstrating improvements in token/request throughput and latency over standard greedy heuristics on both synthetic and real-world datasets. The key insight is that burst admissions cause synchronized memory peaks during decoding; strictly limiting the activation rate (flow control) smooths KV cache usage and provides provable stability guarantees, outperforming aggressive greedy packing under high load. This is highly relevant for our research in operations research formulations for LLM serving scheduling, as it provides both a theoretical queueing baseline and a practical heuristic to benchmark against our optimization models.

### [ALTO: Adaptive LoRA Tuning and Orchestration for Heterogeneous LoRA Training Workloads](https://arxiv.org/abs/2604.05426)

**2026-04-10** | Rice University, Independent Researcher | M=7 P=7 I=7 *discuss*

*Method:* Co-designed system with loss-aware early-exit, fused grouped GEMM, rank-local adapter parallelism, and hierarchical scheduling (MILP inter-task, greedy intra-task) | *LLM role:* none

> ALTO is a co-designed system that accelerates LoRA hyperparameter tuning by combining loss-aware early termination, fused grouped GEMM kernels, and a constraint programming scheduler to pack heterogeneous tasks onto GPU clusters. The results are backed by strong empirical evidence, demonstrating up to a 13.8x speedup over state-of-the-art multi-LoRA systems on 7B-70B models without degrading downstream accuracy. The key insight is that the predictability of LoRA training durations allows the inter-task scheduling problem to be formulated as a heterogeneous-resource strip-packing problem, solvable via MILP in under a second, which dynamically replans upon early-exit events to eliminate GPU fragmentation. This is highly relevant for our research in OR formulations for AI systems; the specific MILP formulation and dynamic replanning architecture provide a concrete blueprint for optimizing resource allocation and GPU scheduling in large-scale LLM infrastructure.

### [Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models](https://arxiv.org/abs/2407.15504)

**2024-12-11** | UT Austin, EPFL | M=8 P=8 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Rate-distortion theory formalized as a linear program, solved via its dual using a geometric algorithm; Adaptive QuerySelect (query-aware, variable-rate token classification) | *LLM role:* token classifier

> Nagle et al. formalize prompt compression as a rate-distortion problem, deriving the fundamental theoretical limit via a dual linear program and proposing 'Adaptive QuerySelect,' a variable-rate compression technique. The results are rigorous: they calculate exact limits on synthetic data and use beam search approximations for NLP, demonstrating that existing fixed-rate methods leave significant performance on the table. The key takeaway is that **variable-rate compression**—keeping tokens based on a confidence threshold rather than a fixed percentage—is essential for approaching optimality; this allows 'hard' queries to retain more context while aggressively compressing 'easy' ones. This is immediately actionable for our AlgoEvo work: we should replace fixed-window history truncation with a query-aware, variable-rate compressor to maximize the useful information in our limited context window.

### [MC#: Mixture Compressor for Mixture-of-Experts Large Models](https://arxiv.org/abs/2510.10962)

**2025-10-13** | NVIDIA Research, National University of Singapore, The University of Hong Kong, Beihang University, Hangzhou Innovation Institute | M=6 P=7 I=7 *discuss*

*Method:* Hybrid compression combining Pre-Loading Mixed-Precision Quantization (PMQ) via Linear Programming and Online Top-any Pruning (OTP) via Gumbel-Softmax sampling | *LLM role:* none

> Huang et al. propose MC#, a compression framework for MoE models that combines static mixed-precision quantization with dynamic expert pruning. They formulate bit-width allocation as an Integer Linear Programming (ILP) problem—optimizing expert importance vs. quantization error—and use a Gumbel-Softmax router for dynamic pruning. Results are strong, achieving 6.2x weight reduction on DeepSeek-VL2 with <2% accuracy loss. **Takeaway:** The ILP formulation (Eq. 7) is a clean, successful application of OR to AI infrastructure that we should replicate for our own resource allocation/scheduling problems; additionally, the differentiable router offers a template for dynamic agent selection in our multi-agent systems.

### [Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Minimization](https://arxiv.org/abs/2511.15898)

**2025-11-19** | Stanford University, Ritual | M=9 P=8 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Global Resolution algorithm via convex minimization and max-flow | *LLM role:* none

> The authors solve the Optimal Transport Linear Program (OTLP) for multi-draft speculative sampling by reducing it to a convex minimization problem using polymatroid theory and max-flow, rather than using slow general LP solvers. They prove this 'Global Resolution' algorithm is exact for i.i.d. drafts and achieves >90% acceptance with negligible overhead (<100ms), running 10,000x faster than baselines. **Key Takeaway:** The reduction of a discrete token selection problem to a convex optimization problem via polymatroids is a brilliant theoretical trick we could potentially adapt for selecting diverse solution subsets in AlgoEvo. This is a definitive 'OR for LLM infra' paper that obsoletes heuristic verification strategies.

### [Serving Compound Inference Systems on Datacenter GPUs](https://arxiv.org/abs/2603.08797)

**2026-03-09** | University of Illinois Urbana-Champaign | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed Integer Linear Programming (MILP) for joint optimization of model variants, GPU spatial partitions, and task-graph-informed budgeting | *LLM role:* none

> JIGSAWSERVE uses a Mixed Integer Linear Programming (MILP) formulation to jointly optimize model variant selection (accuracy scaling) and fine-grained GPU spatial partitioning (MIG/MPS) for serving compound inference DAGs. The results are strongly backed by empirical numbers on real hardware (H100s), demonstrating an 11.3x capacity improvement over the closest prior work (Loki) while maintaining under 0.6% SLO violations. The single most useful takeaway for us is their specific MILP formulation, which successfully linearizes the complexities of task-graph multiplicative factors and spatial GPU segments into a tractable objective for latency and accuracy SLOs. This is highly relevant for our GPUSched project; we should extract their MILP constraints for spatial partitioning and task-graph budgeting to adapt for our own LLM inference scheduling and multi-agent resource allocation models.

### [Puzzle: Distillation-Based NAS for Inference-Optimized LLMs](https://arxiv.org/abs/2411.19146)

**2025-06-03** | NVIDIA | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Decomposed Neural Architecture Search (NAS) using Blockwise Local Knowledge Distillation (BLD) for parallel architecture exploration and Mixed-Integer Programming (MIP) for precise constraint optimization, followed by Global Knowledge Distillation (GKD) | *LLM role:* none

> Bercovich et al. introduce Puzzle, a framework that optimizes LLM architectures for specific hardware by training a library of block variants (via local distillation) and using Mixed-Integer Programming (MIP) to select the optimal layer-wise configuration under strict latency and memory constraints. The results are robust: they compress Llama-70B to 51B, fitting on a single H100 with 2.17x throughput gain and 98.4% accuracy retention, significantly outperforming pruning baselines like Wanda. **Key takeaway:** The 'decomposed search' strategy—replacing expensive end-to-end evolutionary evaluation loops with local proxy scores (KL divergence) and a global MIP solver—is a highly efficient method for modular system configuration. This directly informs our 'GPUSched' and serving optimization work by demonstrating how to mathematically formulate hardware constraints (KV-cache, batch size, compute) into the model design process itself.

### [Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines](https://arxiv.org/abs/2603.02075)

**2026-03-02** | HKUST, Huawei | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Closed-loop adaptive scheduling framework integrating Gaussian Process regression, memory-constrained Bayesian optimization, and Mixed-Integer Linear Programming (MILP) | *LLM role:* none

> Trident optimizes heterogeneous multimodal data pipelines on fixed clusters by integrating Gaussian Process capacity estimation, memory-constrained Bayesian Optimization for configuration tuning, and an MILP for joint parallelism and placement scheduling. The results are real and hardware-backed, demonstrating up to a 2.01x throughput speedup over static baselines and outperforming Ray Data on an 8-node Huawei NPU cluster. The most stealable insight is their MILP formulation for rolling updates (Eq 11-13), which elegantly linearizes cold-start overheads by precomputing discounted throughputs, alongside their use of a probabilistic memory constraint in BO to avoid OOMs during online tuning. This is highly relevant for our GPUSched project; we should evaluate their MILP constraints for cross-node data transfer and configuration transitions when building our own LLM inference scheduling models.

### [Reasoning-Driven Design of Single Atom Catalysts via a Multi-Agent Large Language Model Framework](https://arxiv.org/abs/2602.21533)

**2026-02-25** | Georgia Institute of Technology, Korea University, Sogang University, Ewha Womans University | M=7 P=3 I=8 *discuss*

*Method:* Multi-Agent-based Electrocatalyst Search Through Reasoning and Optimization (MAESTRO) framework using LLM agents and a Machine Learning Force Field (MLFF) surrogate model | *LLM role:* evolutionary_search

> Mok et al. propose MAESTRO, a multi-agent LLM framework for optimizing single-atom catalysts that explicitly separates search into exploration and exploitation phases, bridged by a textual 'Exploration Report.' Results are validated against high-fidelity DFT calculations, showing the system learns to break theoretical scaling relations via in-context learning, outperforming memory-less baselines. The key takeaway for us is the **Exploration Report Agent**: instead of just passing the best candidates to the exploitation phase, the system pauses to write a natural language strategy guide summarizing 'what worked and what didn't' from the exploration phase. We should steal this mechanism for AlgoEvo to let the search agent 'learn' from the initial population generation rather than just selecting from it.

### [ETS: Efficient Tree Search for Inference-Time Scaling](https://arxiv.org/abs/2502.13575)

**2025-06-11** | University of California, Berkeley, ICSI, LBNL | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Efficient Tree Search (ETS) using a linear programming cost model with KV cache sharing penalty and semantic coverage term | *LLM role:* candidate_generator, process_reward_model, search_guidance

> ETS formulates the tree search pruning step as a lightweight Integer Linear Program (ILP) that maximizes the reward of retained nodes while penalizing total KV cache size and enforcing semantic diversity via clustering. Unlike standard beam search or REBASE, it explicitly optimizes the trade-off between memory consumption (KV sharing) and exploration coverage. The authors demonstrate a 1.8x reduction in KV cache size and 1.4x throughput increase on MATH500 with minimal accuracy loss. We should steal the 'ILP-in-the-loop' mechanism for population management in our evolutionary search frameworks to optimize hardware utilization dynamically.

### [Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270)

**2025-02-22** | The University of Hong Kong, The Chinese University of Hong Kong, Beihang University, Centre for Perceptual and Interactive Intelligence, Hong Kong | M=5 P=7 I=6 *discuss*

*Method:* Hybrid Post-Training Quantization and Dynamic Pruning for MoE-LLMs using Linear Programming for bit-width allocation and significance-aware token protection | *LLM role:* none

> Huang et al. propose a compression framework for MoE-LLMs that uses Integer Programming to optimally allocate mixed bit-widths (1-3 bits) to experts based on activation frequency and routing weights. They achieve strong empirical results, compressing Mixtral 8x7b to ~16GB (fitting on a single RTX 3090) with only a ~4% drop in zero-shot accuracy, significantly outperforming uniform quantization. The key takeaway is the explicit IP formulation for minimizing quantization error under memory constraints—a clean 'OR for AI' pattern we can adapt for our GPU scheduling or memory allocation formulations. While not a methodological advance in evolution, this is highly relevant for our infrastructure: it enables deploying high-quality MoE models on cheaper hardware for our massive AlgoEvo loops.

### [Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity](https://arxiv.org/abs/2603.12707)

**2026-03-13** | University of Illinois Urbana-Champaign | M=5 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Phase-aware runtime (HeteroServe) with modality-level partitioning, embedding-only transfer, and cross-type work stealing | *LLM role:* none

> Yu et al. demonstrate that partitioning multimodal LLM inference at the modality boundary (vision encoder vs. language decoder) reduces cross-device transfer costs by O(L), dropping requirements from GB-scale NVLink to MB-scale PCIe. This enables heterogeneous serving architectures where cheap compute-dense GPUs (RTX 4090) handle vision and expensive bandwidth-dense GPUs (A100) handle language. Results are strongly backed by real hardware deployments, showing a 37% improvement in cost-efficiency over homogeneous vLLM baselines. WHAT WE LEARNED: The phase-separable nature of MLLMs and the use of cross-tier work stealing (idle 4090s assisting with language decoding) are massive structural opportunities. We must immediately update our integer programming formulations in the GPUSched project to model this modality-level disaggregation, otherwise our OR models will be obsolete for multimodal workloads.

### [BAMAS: Structuring Budget-Aware Multi-Agent Systems](https://arxiv.org/abs/2511.21572)

**2025-11-26** | Tsinghua University, Peking University, University of Illinois Urbana-Champaign, Nanyang Technological University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Joint optimization of LLM selection via Integer Linear Programming (ILP) and agent collaboration topology selection via offline reinforcement learning (REINFORCE) | *LLM role:* agents

> BAMAS decouples agent resource provisioning from coordination strategy, using an Integer Linear Programming (ILP) solver to select the optimal set of LLMs under a strict budget and offline RL to select a fixed interaction topology. They demonstrate ~80% cost reduction on GSM8K and MBPP while matching SOTA accuracy, proving that formal optimization beats greedy heuristics for agent allocation. The key takeaway for us is the 'lexicographically optimal' ILP formulation for tier-based LLM selection, which we should steal immediately for our inference resource managers. While their topology search is limited to a fixed library (unlike our evolutionary approach), the hybrid ILP+RL architecture is a strong template for our 'OR for Generative AI' work.

### [Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control](https://arxiv.org/abs/2602.02987)

**2026-02-03** | The Hong Kong University of Science and Technology | M=8 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stochastic control with fluid approximation and LP-based gate-and-route policies | *LLM role:* none

> Lin et al. formulate LLM inference scheduling as a multiclass many-server queueing network, deriving a 'Gate-and-Route' policy from a steady-state fluid LP that explicitly manages prefill-decode contention. Calibrated on A100s, their approach proves that separating prefill admission (via occupancy tracking) from decode routing (work-conserving) eliminates decode backlogs and maximizes revenue. **Key Takeaway:** The decomposition of scheduling into 'static planning' (solving an LP for target occupancies) and 'dynamic control' (a simple gate tracking those targets) is a scalable alternative to online combinatorial optimization for your GPUSched work. It mathematically formalizes the intuition that prefill is the bottleneck and decode should be kept strictly critical but not backlogged.

### [Best-of-$\infty$ -- Asymptotic Performance of Test-Time Compute](https://arxiv.org/abs/2509.21091)

**2025-09-25** | Mohamed bin Zayed University of Artificial Intelligence, New York University, RIKEN AIP, Institute of Science Tokyo, NEC Corporation | M=8 P=7 I=8 **MUST-READ** *discuss*

*Method:* Adaptive sampling for majority voting using Bayesian modeling (Dirichlet process prior and Bayes factor) to determine stopping criteria, combined with optimally weighted LLM ensembles formulated as a Mixed-Integer Linear Program (MILP) with a max-margin solution. | *LLM role:* answer_generator

> This paper introduces a Bayesian adaptive stopping criterion (using Dirichlet process priors and Bayes factors) for majority voting, reducing test-time compute by 2-5x while maintaining asymptotic 'Best-of-Infinity' accuracy. They further demonstrate that optimizing weights for an ensemble of LLMs can be formulated as a Mixed-Integer Linear Program (MILP) by treating the decision boundaries as polytopes. **What we learned:** The Bayesian stopping logic is immediately transferable to AlgoEvo to reduce the cost of fitness evaluations—we can stop evaluating candidate solutions early if their performance distribution is statistically distinct. The MILP approach for ensembles also offers a concrete formulation we could adapt for our GPU scheduling and model serving optimization work.

### [Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference](https://arxiv.org/abs/2603.26498)

**2026-03-27** | IMDEA Software Institute, Universidad Politécnica de Madrid | M=5 P=8 I=7 **MUST-READ** *discuss*

*Method:* Modality-aware dynamic priority scheduling with aging mechanism | *LLM role:* target_of_optimization

> RPS-Serve introduces a modality-aware scheduler for multimodal LLMs that classifies requests into 'rocks' (video), 'pebbles' (image), and 'sand' (text) based on predicted prefill latency and memory, using dynamic priorities and aging to prevent head-of-line blocking. The results are real and backed by solid systems experiments on vLLM, showing a 78.5% reduction in time-to-first-token for latency-critical text requests compared to FCFS and EDF baselines. The core takeaway is that multimodal workloads completely break standard text-only LLM serving assumptions because video/image prefill times and KV-cache footprints are orders of magnitude larger. This matters directly for our GPUSched project: any OR formulation we develop for LLM inference scheduling must now explicitly model this extreme multimodal variance rather than assuming homogeneous text workloads.


### Front 2 (35 papers) — EMERGING

**Density:** 0.01 | **Methods:** llm_as_heuristic, llm_in_the_loop, llm_as_evaluator, pipeline_parallelism, resource_allocation | **Problems:** resource_allocation, llm_serving_optimization, gpu_scheduling, mathematical_reasoning, multi_agent_coordination

*Unique methods:* active_inference, actor_critic, adamw_optimizer, adaptive_greedy_heuristic, adaptive_precision, adaptive_replacement, adaptive_routing, advantage_modulation, adversarial_training, aflow, alphaevolve, amortized_inference, archive_based_search, arima_time_series_forecasting, asynchronous_reinforcement_learning, attention_sparsification, autoencoder, autogen, automated_system_design, autoregressive_decoding, batch_size_optimization, bayesian_regression, bernoulli_variance_proxy, best_of_k_sampling, beta_distribution, binary_search, bisection_method, blockwise_decoding, cayley_graphs, clipping_mechanism, cluster_scheduling, co_scheduling, code_generation, compiler_optimization, conditional_graph_generation, conflict_aware_meta_verification, consensus_set_partitioning, conservative_stochastic_planning, constrained_search, constraint_satisfaction_problem, continual_learning, continual_meta_learning, control_theory, controlled_decoding, cosine_similarity, cost_benefit_analysis, cost_normalization, cpu_gpu_aware_micro_batching, cpu_parallelism, deduplication, deepspeed_zero, demand_forecasting, difficulty_estimation, direction_oriented_resource_allocation, distributed_data_parallelism, divergence_analysis, diverse_verifier_tree_search, dora, dynamic_clipping_bounds, dynamic_parallelism, dynamic_rollout_allocation, dynamic_thresholding, elastic_system, embedding_model, empirical_benchmarking, empirical_feedback, energy_analysis, entropy_control, entropy_dynamics_control, entropy_estimation, entropy_regularization, environment_aware_optimization, eoh, exact_nearest_neighbor_search, expert_data_parallelism, f_divergences, failure_driven_mining, feature_engineering, feature_extraction, feature_sampling, feedback_control, fine_tuning, free_energy_principle, fully_sharded_data_parallelism, funcdyn, funsearch, glpk_mi, gpu_allocation, gpu_consolidation, gradient_compensation, gradient_free_optimization, gradient_variance_minimization, graph_neural_networks, graphsage, greedy_sequence_packing, heuristics, heuristics_analysis, highs_solver, hypothesis_testing, independent_proximal_policy_optimization, inter_group_scheduling, intra_group_scheduling, iterative_optimization, iterative_refinement, k_medoids_clustering, knowledge_gradient_policy, latency_analysis, latency_prediction, lexical_summarization, llm_adaptive_inference, llm_adversarial_search, llm_as_generator, llm_as_instruction_generator, llm_as_judge, llm_as_refiner, llm_inference_scheduling, llm_multi_agent_communication, llm_reinforcement_learning, llm_research_agent, local_search, locality_aware_routing, long_tail_migration, m_m_1_model, max_flow_optimization, maximin_optimization, maximum_a_posteriori, meta_learning, micro_batching, milp, milp_acceleration, mixed_integer_conic_optimization, mixed_precision_training, mixed_workload_scheduling, model_quantization, modular_system_design, modulo_scheduling, monte_carlo_sampling, multi_layer_perceptron, multi_processing, multi_start, multi_threading, neural_network, optimal_learning, overlapping_scheduling, paged_attention, pearson_chi_squared_divergence, performance_profiling, phase_centric_control, policynet, program_level_fcfs, prompt_engineering, prompt_optimization, prompt_retrieval, prompt_tuning, quantization, queueing_delay_reduction, random_forest_regression, reactive_heuristics, reinforcement_learning_with_verifiable_rewards, representation_learning, resource_allocation_modeling, reward_balanced_search, reward_modeling, risk_averse_quantal_response_equilibria, risk_sensitive_control, root_finding_algorithms, round_robin_scheduling, rule_based_routing, satisfiability_modulo_theories, scheduling_optimization, scheduling_policies, self_organizing_systems, semantic_similarity, sequential_decision_making, skill_evolution, soft_clustering, software_pipelining, srpo, stackelberg_security_game, statistics_collection, stochastic_thermodynamics, strategic_risk_aversion, structured_trajectory_representation, supervised_finetuning, system_characterization, task_batching, temperature_sampling, tensor_model_parallelism, test_time_pruning, theoretical_analysis, theoretical_modeling, throughput_analysis, time_multiplexing, time_to_live_caching, token_scheduling, topology_aware_model_synchronization, total_variation_divergence, triplet_loss, trust_region_policy_optimization, unsupervised_learning, update_magnitude_stabilization, user_simulation, value_function_learning, variational_graph_encoder, vector_quantization, warm_cold_pooling, warm_start_mechanism, warp_specialization, weighted_round_robin, wireless_resource_allocation, workload_scheduling
*Shared methods:* adaptive_sampling, autoscaling, bayesian_inference, bayesian_optimization, beam_search, binary_cross_entropy, black_box_optimization, continuous_batching, convex_optimization, cost_modeling, cvxpy, data_parallelism, discrete_event_simulation, distributed_systems, dynamic_batching, dynamic_programming, evolution_of_heuristics, expert_parallelism, game_theory, gpu_scheduling, gradient_descent, graph_partitioning, greedy_algorithm, group_relative_policy_optimization, grpo, heuristic_initialization, in_context_learning, integer_linear_programming, integer_programming, k_means_clustering, kl_divergence, kv_cache_management, linear_programming, llm_as_evaluator, llm_as_feature_extractor, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_in_the_loop, llm_inference_serving, llm_orchestration, llm_prompt_optimization, llm_serving_systems, lora, mixed_integer_linear_programming, mlp, multi_agent_reinforcement_learning, multi_agent_system, multi_agent_systems, multi_objective_optimization, online_learning, optimal_transport, optimization, parameter_efficient_fine_tuning, pipeline_parallelism, pipelining, policy_optimization, ppo, process_reward_model, profiling, program_synthesis, proximal_policy_optimization, queuing_theory, rebase, reinforcement_learning, reinforcement_learning_from_human_feedback, resource_allocation, resource_management, retrieval_augmented_generation, rl_ppo, robust_optimization, scheduling_algorithms, speculative_decoding, supervised_fine_tuning, supervised_learning, tensor_parallelism, vllm

This research front is unified by the application of rigorous Operations Research (OR) methodologies, predominantly Mixed-Integer Programming (MILP) and Integer Linear Programming (ILP), to optimize the design and operation of large language model (LLM) systems. It addresses two primary areas: efficient resource allocation and scheduling for LLM serving and training across heterogeneous and distributed hardware, exemplified by frameworks like AREAL-Hex, Helix, SageServe, and BAMAS; and the principled orchestration and design of multi-agent LLM systems, as seen in AgentConductor, AdaptOrch, and MaMa.

Key contributions include the formulation of LLM serving and training as MILP problems to co-optimize GPU composition, parallelism strategies, and workload routing, achieving significant throughput and cost efficiencies (e.g., Helix's 3.3x decode throughput, SageServe's 25% GPU-hour reduction). Papers like 'Predicting Future Utility' apply convex optimization for global KV cache eviction, yielding 80% cache reduction. In multi-agent systems, innovations range from RL-optimized dynamic topology generation (AgentConductor, 14% gains on APPS) and task-adaptive orchestration (AdaptOrch, 9.8% on SWE-bench) to adversarial co-evolution for safety (MaMa, reducing attack success rates to 15-25%). Other notable advancements include SAMPO's sequence-level clipping for stable agentic RL (ARLArena, 92% success on ALFWorld) and ILP-driven budget-aware agent structuring (BAMAS, 80% cost reduction).

This front is rapidly emerging, characterized by a shift from heuristic-driven solutions to mathematically rigorous optimization for LLM systems. The trajectory indicates a deepening integration of OR techniques with LLM-specific challenges, such as dynamic adaptation to multimodal workloads (HeteroServe), real-time predictive prefetching for MoE (PROBE), and uncertainty-aware multi-agent coordination (UMAD). Future work will likely focus on scaling these hybrid OR/ML approaches to increasingly complex, distributed, and heterogeneous environments, incorporating dynamic online adaptation, and robustly handling stochasticity and adversarial conditions to further enhance efficiency, safety, and performance.

**Papers:**

### [Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs](https://arxiv.org/abs/2502.00722)

**2025-06-05** | University of Cambridge, ETH Zurich, Peking University, The Hong Kong University of Science and Technology, Purdue University | M=5 P=9 I=6 **MUST-READ** *discuss*

*Method:* Mixed-Integer Linear Programming (MILP) for scheduling | *LLM role:* none

> Jiang et al. formulate LLM serving on heterogeneous clouds as a Mixed-Integer Linear Programming (MILP) problem, co-optimizing GPU rental composition, parallelism strategies (TP/PP), and workload routing. They demonstrate ~25% throughput gains over SOTA systems (Helix, HexGen) using vLLM benchmarks, validating the approach with strong empirical ablations. For our **GPUSched** project, the key takeaway is their solver strategy: pre-generating valid configurations to linearize the problem and using a binary search wrapper on the makespan to avoid direct minimization overhead. We should adopt their heuristics for pruning the configuration space (e.g., restricting TP to intra-node) to improve our own solver times.

### [Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230)

**2026-01-30** | UC Berkeley, Stanford University, Tensormesh, Tsinghua University | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* KV cache time-to-live (TTL) mechanism with a cost-benefit model and program-level FCFS scheduling | *LLM role:* none

> Continuum introduces a dynamic Time-to-Live (TTL) KV cache retention policy and program-level FCFS scheduling to minimize prefill overhead and queueing delays during multi-turn LLM agent tool calls. The results are highly credible, demonstrating up to 3.66x delay reduction and 3.22x throughput improvement on real agentic workloads (SWE-Bench, BFCL) using Llama-3.1 8B/70B on vLLM. The most valuable takeaway for us is their cost-benefit formulation for TTL, which balances the opportunity cost of GPU memory against the expected savings in prefill time and per-turn queueing delay using a 'memoryfulness' factor and empirical tool-call CDFs. This is highly actionable for our GPUSched project; we must ensure our OR formulations for LLM inference scheduling account for the asynchronous pauses and KV cache lifecycle of multi-turn agentic workloads, rather than treating them as independent requests.

### [HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference](https://arxiv.org/abs/2508.19373)

**2025-08-26** | Huawei Noah’s Ark Lab, Shandong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Integer Linear Programming (ILP) for dynamic selection of hybrid parallel strategies (DP, TP, EP) for Attention and Expert modules, guided by module-specific inference latency simulation models | *LLM role:* none

> HAP replaces static parallelization heuristics in MoE inference with an Integer Linear Programming (ILP) solver that dynamically selects optimal strategies (TP, EP, DP) for Attention and Expert modules. They achieve verified ~1.6x speedups on A100/A6000 GPUs by modeling the inference process as a two-stage problem (prefill vs. decoding) with explicit transition costs, allowing the system to switch parallelism strategies mid-inference. For our work on OR-based resource allocation (GPUSched), the key takeaway is their formulation of **transition overheads** within the ILP constraints—a technique we should steal to model dynamic reconfiguration in our scheduling solvers. This confirms that symbolic OR methods can outperform standard systems heuristics in the LLM serving stack.

### [BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning](https://arxiv.org/abs/2603.04918)

**2026-03-05** | Fudan University, Shanghai Innovation Institute | M=8 P=7 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Band-constrained Policy Optimization (BandPO) using a Band operator to project f-divergence trust regions into dynamic, probability-aware clipping intervals | *LLM role:* policy_agent

> BandPO replaces the standard static clipping in PPO/GRPO with dynamic bounds derived from projecting f-divergence trust regions, specifically addressing a bottleneck where allowable updates vanish for low-probability tokens. Empirical results are rigorous, showing consistent gains (2-10%) on math benchmarks and, crucially, maintaining policy entropy where baselines collapse. The key takeaway is that standard clipping scales update margins linearly with probability, effectively freezing rare tokens; BandPO decouples this, allowing the model to actually reinforce novel, high-advantage tail strategies. We should implement the closed-form TV or Chi-squared variants immediately in our RL optimizers to improve exploration efficiency.

### [Glia: A Human-Inspired AI for Automated Systems Design and Optimization](https://arxiv.org/abs/2510.27176)

**2025-11-17** | MIT CSAIL, Independent Researcher | M=9 P=9 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Human-inspired multi-agent LLM workflow with specialized Researcher and Supervisor agents collaborating through an evaluation framework for iterative hypothesis formation, experimentation, and analysis. | *LLM role:* research_agent

> Glia replaces black-box LLM evolutionary search (like AlphaEvolve) with a multi-agent workflow (Researcher + Supervisor) that instruments simulators, analyzes telemetry, forms hypotheses, and writes interpretable system algorithms for LLM serving. The results are backed by strong empirical numbers; it outperforms OpenEvolve, FunSearch, and EoH by 1.3-1.7x in mean response time using significantly fewer simulations (massive sample efficiency gains), and the results transfer to a real vLLM cluster. The single most useful takeaway is shifting from pure code-level mutation to 'idea-level' evolution: forcing the LLM to analyze simulation logs and write a chain-of-thought hypothesis before generating code, guided by a Supervisor agent that prevents stagnation. This is a direct threat and massive inspiration for AlgoEvo; it proves that for systems optimization, agentic reasoning with observability beats blind evolutionary search, and we should immediately test their log-analysis loop and Multi-Context parallel sampling strategy.

### [Training Generalizable Collaborative Agents via Strategic Risk Aversion](https://arxiv.org/abs/2602.21515)

**2026-02-25** | Caltech | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Strategically Risk-Averse Policy Optimization (SRPO) based on Risk-Averse Quantal Response Equilibria (RQE) | *LLM role:* collaborative_agent

> Qu et al. introduce Strategically Risk-Averse Policy Optimization (SRPO), which trains agents against a 'constrained adversary' that minimizes their reward within a KL-divergence bound of the partner's current policy. Theoretical results prove this objective eliminates free-riding equilibria, and experiments on GSM8K multi-agent debate show it prevents 'lazy' agreement, improving joint accuracy by up to 19% when pairing heterogeneous LLMs (e.g., 0.6B with 4B). The key takeaway is that robustness to partner deviation—enforced via this specific adversarial objective—is a more principled way to fix lazy agent behavior than prompt engineering or simple dropout. We should immediately test this objective in our HERMES debate framework to improve the contribution quality of smaller models.

### [CARD: Towards Conditional Design of Multi-agent Topological Structures](https://arxiv.org/abs/2603.01089)

**2026-03-01** | Monash University, Southeast University, Griffith University | M=6 P=8 I=7 *discuss*

*Method:* Conditional graph generation via encoder-decoder graph neural network with environment-aware optimization | *LLM role:* agentic reasoning and task execution

> CARD dynamically generates multi-agent communication topologies using a conditional graph encoder-decoder that adapts to runtime environmental signals like model capability, tool availability, and token cost. The results are backed by solid empirical numbers on HumanEval, MATH, and MMLU, showing a 1-3% improvement over static learned topologies (like G-designer and Aflow), particularly when generalizing to unseen LLMs or degraded search tools. The most actionable takeaway for us is their method of explicitly encoding environmental constraints (e.g., API costs, tool reliability) into a shared embedding space to dynamically threshold communication edges and balance task utility against inference cost. This is highly relevant for our RobustMAS and MAS resource allocation work; we should steal their condition-embedding approach to dynamically route or prune agent interactions when facing budget constraints or API rate limits.

### [A Sequential Optimal Learning Approach to Automated Prompt Engineering in Large Language Models](https://arxiv.org/abs/2501.03508)

**2025-01-07** | Northwestern University, Stevens Institute of Technology | M=8 P=7 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Sequential Optimal Learning with Knowledge-Gradient (KG) policy using Bayesian regression | *LLM role:* instruction_generator, evaluator

> Wang et al. treat prompt engineering as a Bayesian optimal experimental design problem, representing prompts as discrete feature vectors (template, tone, examples) and selecting the next candidate using a Knowledge-Gradient (KG) policy solved via mixed-integer second-order cone programming. Results are rigorous and show that this OR-based approach outperforms evolutionary (EvoPrompt) and bandit baselines on instruction induction tasks, specifically in low-sample regimes (N=30). The critical takeaway is the **replacement of random evolutionary mutation with a KG policy over a structured feature space** to maximize information gain per step. We should steal this formulation to optimize high-level meta-parameters or strategy selection in AlgoEvo, leveraging our team's OR background to solve our sample efficiency bottleneck.

### [A CPU-Centric Perspective on Agentic AI](https://arxiv.org/abs/2511.00739)

**2025-11-29** | Intel, Georgia Institute of Technology | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* System characterization, full-system profiling (latency, throughput, energy), CPU and GPU-Aware Micro-batching (CGAM), Mixed Agentic Workload Scheduling (MAWS) | *LLM role:* none

> This paper profiles the system-level bottlenecks of agentic AI frameworks, revealing that CPU-bound tool execution (e.g., Python interpreters, retrieval) dominates latency, and proposes micro-batching scheduling strategies (CGAM, MAWS) to mitigate CPU core over-subscription. The results are backed by rigorous hardware profiling on Intel/NVIDIA setups, demonstrating that CPU processing takes up to 90.6% of latency and that their scheduling heuristics yield a 2.1x P50 latency speedup. The critical takeaway for us is that scaling agentic AI throughput is bottlenecked by CPU context switching and dynamic energy at large batch sizes, not just GPU KV cache. This is highly actionable for our GPUSched and AlgoEvo projects; we must incorporate CPU-awareness and micro-batching constraints into our formal OR scheduling models, as ignoring the CPU overhead of code execution invalidates standard GPU-only serving models.

### [Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via Reinforcement Learning](https://arxiv.org/abs/2507.10259)

**2025-09-16** | Shenzhen University of Advanced Technology, China Mobile Research Institute | M=6 P=9 I=6 **MUST-READ** *discuss*

*Method:* Proximal Policy Optimization (PPO) with Optimal Transport supervision | *LLM role:* none

> TORTA introduces a hierarchical scheduler for distributed LLM inference that uses a macro-level RL agent (PPO) supervised by an Optimal Transport (OT) baseline to manage inter-region allocation, followed by a micro-level greedy allocator. Results on simulated clusters (up to 50 servers) demonstrate a ~15% reduction in latency compared to reactive baselines (like SkyLB) specifically by optimizing for temporal smoothness and reducing switching costs. The key technical takeaway is the use of an exact OR solver (OT) as a dense supervision signal to train a faster RL policy, effectively combining the optimality of OR with the temporal foresight of RL. We should review our GPUSched formulations to ensure we aren't falling into the 'reactive' trap described here; if we are, this hybrid RL-OT architecture is a viable alternative.

### [B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests](https://arxiv.org/abs/2409.08692)

**2024-09-13** | Zhejiang University, Singapore Management University | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* Bayesian Maximum A Posteriori (MAP) estimation approximated by Beta functions (B4) | *LLM role:* none

> Chen et al. derive a Bayesian posterior estimator (B4) for selecting correct code solutions using unreliable (LLM-generated) tests, explicitly modeling the probability of incorrect code passing incorrect tests. They demonstrate statistically significant improvements (up to 50% relative gain on hard problems) over state-of-the-art heuristics like CodeT and MaxPass on HumanEval and APPS. The key takeaway is the B4 scoring formula: a product of four Beta functions that weighs consensus sets based on priors about test reliability (e.g., incorrect code rarely passes incorrect tests). This is immediately actionable for AlgoEvo: we can replace our naive fitness aggregation with B4 to improve selection accuracy when using generated unit tests, directly boosting sample efficiency.

### [Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding](https://arxiv.org/abs/2510.11331)

**2025-10-13** | Queen Mary University of London, Kyung Hee University, Xidian University, Guangzhou Institute of Technology | M=5 P=7 I=6 

*Method:* Speculative Decoding (SD) with pipeline parallelism, combined with joint optimization of speculation length, task batching, and wireless communication resource allocation | *LLM role:* inference engine

> Zhu et al. propose a distributed Speculative Decoding framework for edge networks, formulating a Mixed-Integer Nonlinear Programming problem to jointly optimize task batching, speculation length, and wireless bandwidth. They solve the batching subproblem using a Dynamic Programming (DP) algorithm, achieving ~30-45% latency reduction over heuristics in simulations, though the approach relies on a rigid assumption of fixed maximum output lengths to remain tractable. The primary takeaway for our 'GPUSched' work is their DP formulation for optimizing batch boundaries in a pipelined draft-verify system, which offers a cleaner mathematical alternative to greedy heuristics for serving schedules. However, the heavy reliance on wireless channel modeling makes the full system less relevant to our datacenter-centric optimization problems.

### [PromptTuner: SLO-Aware Elastic System for LLM Prompt Tuning](https://arxiv.org/abs/2603.05087)

**2026-03-05** | Nanyang Technological University, Unaffiliated | M=5 P=8 I=7 *discuss*

*Method:* SLO-aware elastic system combining a two-layer Prompt Bank for initial prompt selection and a Workload Scheduler for dynamic multi-GPU allocation | *LLM role:* feature_extractor

> PromptTuner is a cluster management system for LLM prompt tuning that combines a 'Prompt Bank' (retrieving similar past prompts to speed up convergence) with a hierarchical scheduler (warm/cold GPU pools) to meet latency SLOs. The authors demonstrate real-world efficacy on 32-96 GPU clusters, showing 4-8x reductions in SLO violations compared to INFless and ElasticFlow. The key takeaway for us is the 'Prompt Bank' mechanism: using K-medoids clustering on activation features to retrieve high-quality initial prompts. We should steal this initialization strategy for AlgoEvo to reduce the number of generations needed for convergence, and use the scheduling logic as a baseline for our GPUSched project.

### [How to Allocate, How to Learn? Dynamic Rollout Allocation and Advantage Modulation for Policy Optimization](https://arxiv.org/abs/2602.19208)

**2026-02-22** | Meituan, Tsinghua University, Fudan University, Peking University | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Dual-pronged optimization framework (DynaMO) combining variance-minimizing dynamic rollout allocation and gradient-aware advantage modulation for GRPO-based policy optimization | *LLM role:* policy_model

> DynaMO introduces a dual-pronged optimization for RLVR: a dynamic rollout allocation strategy that prioritizes problems with high gradient variance (proxied by Bernoulli variance of success/failure), and a gradient modulation technique to stabilize updates. The results are strong (+11.8% Pass@1 over GRPO on Qwen-7B) and backed by clear ablations. **The critical takeaway for us is the allocation logic:** we should immediately replace uniform sampling in AlgoEvo with variance-based allocation ($n_i \propto \sqrt{p(1-p)}$). This ensures compute is spent on instances/components that are currently 'learnable' (high variance) rather than wasted on the trivial or impossible, directly optimizing our search budget.

### [AReaL-Hex: Accommodating Asynchronous RL Training over Heterogeneous GPUs](https://arxiv.org/abs/2511.00796)

**2025-11-02** | HKUST, Tsinghua University, Ant Group | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* Two-phase scheduling algorithm combining constrained search with Mixed-Integer Linear Programming (MILP) for parallelization strategies and workload assignments, and a cost-guided graph partitioning for resource allocation | *LLM role:* none

> AREAL-HEX optimizes asynchronous RL training for LLMs across heterogeneous GPUs by using a two-phase scheduler (MILP for workload assignment and graph partitioning for resource allocation) to map HBM-bound rollout generation to high-bandwidth GPUs (H20) and compute-bound training to high-FLOP GPUs (H800). The results are rigorously backed by hardware benchmarks, demonstrating up to 1.5x higher throughput and 1.46x cost reduction over homogeneous clusters on 1.5B-14B models. The single most useful takeaway for us is their specific MILP formulation that balances data staleness constraints with multi-stage pipeline throughput across disjoint hardware pools. This is highly actionable for our GPUSched project (OR for AI systems) and provides a concrete blueprint for scaling AlgoEvo's generation/evaluation loops across heterogeneous cloud resources to minimize costs.

### [SNIP: An Adaptive Mixed Precision Framework for Subbyte Large Language Model Training](https://arxiv.org/abs/2602.01410)

**2026-02-01** | University of Michigan, NTT Research, Inc., University of Massachusetts Amherst | M=7 P=5 I=7 *discuss*

*Method:* Integer Linear Programming (ILP) for layer-wise precision optimization guided by loss divergence and weight divergence metrics | *LLM role:* none

> Pan et al. introduce SNIP, a framework that periodically solves a Knapsack-style Integer Linear Program (ILP) to assign layer-wise precision (FP4/FP8) during training, minimizing a custom 'divergence' metric subject to FLOPs constraints. Results are simulated via fake quantization (proxy FLOPs) rather than wall-clock time on native hardware, but the method scales to 70B models and outperforms heuristic baselines. **Key Takeaway:** The design pattern of 'collect sensitivity stats -> solve lightweight ILP -> dynamic reconfiguration' is highly relevant for our work on optimizing LLM serving and agent compute budgets; it proves standard OR solvers are fast enough to operate within the runtime loop of high-performance AI systems.

### [Fine-grained MoE Load Balancing with Linear Programming](https://arxiv.org/abs/2511.16947)

**2026-01-15** | Peking University, Institute of Computing Technology Chinese Academy of Sciences | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* FineEP, a novel expert parallelism strategy leveraging token scheduling formulated as a linear programming problem, combined with tailored expert placement strategies (symmetric Cayley graphs, asymmetric Monte Carlo sampling) and an adaptive replacement mechanism. | *LLM role:* none

> FineMoE replaces heuristic load balancing in MoE training with a Linear Programming (LP) formulation solved per micro-batch to minimize maximum GPU load, achieving ~37-47% throughput gains over Megatron-LM. They utilize warm-started simplex solvers to keep optimization time under 1ms and employ Cayley graphs to optimize static expert placement. For our `GPUSched` work, this is a critical data point: it proves that formal OR solvers can replace heuristics in real-time LLM infrastructure without becoming a latency bottleneck.

### [SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference](https://arxiv.org/abs/2603.04716)

**2026-03-05** | Kingsoft Cloud | M=4 P=9 I=7 **MUST-READ** *discuss*

*Method:* Hybrid theoretical modeling (M/M/1 queuing theory) and empirical benchmarking for P/D resource calculation | *LLM role:* none

> This paper proposes a hybrid resource allocation method for disaggregated Prefill/Decode (P/D) inference, using M/M/1 queuing theory to model prefill throughput under TTFT constraints and empirical profiling for decode. The results are real and validated on NVIDIA H200 clusters running DeepSeek-V3.1. The key takeaway for us is the validated analytical relationship between TTFT, input length, and effective prefill throughput ($TP_{eff} = TP_{max} - \frac{L_{in}}{TTFT - T_{overhead}}$). We can steal this equation to serve as a cheap, differentiable constraint in our 'GPUSched' OR formulations or fitness functions, replacing expensive simulations.

### [RollMux: Phase-Level Multiplexing for Disaggregated RL Post-Training](https://arxiv.org/abs/2512.11306)

**2025-12-15** | Alibaba Group, Hong Kong University of Science and Technology, UIUC | M=7 P=8 I=7 *discuss*

*Method:* Two-tier cluster scheduling framework with inter-group conservative stochastic planning and intra-group round-robin orchestration | *LLM role:* none

> ROLLMUX proposes a cluster scheduler that interleaves the rollout and training phases of multiple RL jobs to eliminate the idle 'dependency bubbles' inherent in synchronous on-policy learning. Tested on a production-scale cluster (328 H800s + 328 H20s), they demonstrate a 1.84x cost reduction with real-world traces, validating the approach beyond simulation. The most stealable insight is 'long-tail migration': dynamically detecting straggler requests during generation and migrating them to a small subset of nodes, freeing the main cluster to proceed immediately. We should implement this logic in our AlgoEvo evaluation loops to mitigate stochastic evaluation times.

### [Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention](https://arxiv.org/abs/2603.03310)

**2026-02-08** | UC Berkeley | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Self-organizing inference architecture based on entropy control, extending vLLM with entropy-aware scheduling, entropic pruning of paged attention blocks, and adaptive temperature control. | *LLM role:* none

> Kiruluta proposes 'Entropic-Time Inference,' a control framework that dynamically allocates compute (scheduling, attention, sampling) based on entropy reduction rather than token count, effectively skipping 'low-information' computation. While the 30-45% throughput gains over vLLM are self-reported and depend on efficient entropy estimation, the core mechanism is sound. The critical takeaway is the **'Entropic Time' metric**: we can steal this for AlgoEvo to allocate evolutionary compute only to solution branches with high uncertainty, rather than evolving all individuals uniformly. This directly addresses our sample efficiency bottleneck and provides a novel dynamic objective for our GPUSched project.

### [MARINE: Theoretical Optimization and Design for Multi-Agent Recursive IN-context Enhancement](https://arxiv.org/abs/2512.07898)

**2025-12-05** | ZTE | M=8 P=6 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Multi-Agent Recursive IN-context Enhancement (MARINE) framework for iterative refinement of a persistent reference trajectory via a theoretically-grounded refinement operator | *LLM role:* trajectory_generator, evaluator, refiner

> MARINE proposes a multi-agent framework that iteratively refines a single 'reference trajectory' by generating small batches of candidates, verifying logical/factual conflicts, and merging superior segments rather than regenerating the whole chain. Results are impressive, with an 80B model matching 1000B baselines on retrieval tasks, backed by a theoretical derivation showing that batch size M=2 is optimal for fixed-budget refinement. The critical takeaway is the 'conflict-aware meta-verification' and segment merger, which functions effectively as a process-reward-guided mutation operator. We should immediately test the M=2 configuration in our evolutionary loops and consider adapting their merger logic to replace random crossover in our code generation agents.

### [ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference](https://arxiv.org/abs/2602.23681)

**2026-02-27** | Carnegie Mellon University, Nanyang Technological University, Beihang University, University of the Chinese Academy of Sciences, Renmin University of China, Sun Yat-sen University, Northeast University (Qinhuangdao Campus) | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Adaptive routing framework using a Difficulty Estimator trained via amortized active inference to route queries between a heuristic Fast Agent and a deliberative Slow Agent, with answer selection via Free-Energy-Principled (FEP) fusion. | *LLM role:* orchestrator and multi-agent reasoner

> ODAR dynamically routes LLM queries to fast or slow agents based on a lightweight difficulty estimator, and fuses multiple reasoning paths using a Free-Energy-Principled (FEP) objective that penalizes high-variance log-probabilities. The results are highly rigorous and backed by extensive numbers across 23 benchmarks, notably matching the performance of fully supervised PRMs on MATH (96.7% vs 97.1%) without any training, while cutting compute by 82%. WHAT WE LEARNED: Ignore the biological 'theta-gamma' framing; the killer trick here is the FEP fusion formula. By scoring candidates using their character-normalized log-probability minus a temporal variance penalty (varentropy), they effectively filter out epistemic uncertainty and hallucinations. This matters immensely for us: we should immediately test this varentropy-based scoring mechanism as a training-free fitness signal in AlgoEvo and as a baseline for MASPRM.

### [Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference](https://arxiv.org/abs/2604.07472)

**2026-04-08** | Arizona State University | M=8 P=8 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Adaptive Greedy Heuristic (AGH) with multi-start construction, relocate-based local search, and GPU consolidation, guided by three constraint-aware mechanisms | *LLM role:* none

> Cheng et al. propose an Adaptive Greedy Heuristic (AGH) to jointly optimize model selection, GPU provisioning, parallelism configuration, and workload routing for LLM inference under strict SLOs. Backed by Azure LLM trace simulations, AGH achieves a >260x speedup over exact MILP solvers on large instances and reduces costs by up to 48% under high demand volatility via 5-minute rolling re-optimization. The key insight is that exact cost-minimal MILP solutions are highly fragile under operational uncertainty, whereas constraint-aware heuristics implicitly provision headroom, making them significantly more robust to delay and error inflation. This is highly relevant for our research in OR formulations for LLM serving scheduling, offering a concrete, scalable alternative to exact solvers that improves real-world deployment robustness.

### [Build, Judge, Optimize: A Blueprint for Continuous Improvement of Multi-Agent Consumer Assistants](https://arxiv.org/abs/2603.03565)

**2026-03-03** | DoorDash, WithMetis.ai | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Prompt-level optimization using GEPA and MAMUT GEPA | *LLM role:* evaluator, evolutionary_search, decomposition_guide, user_simulator

> This paper presents a production-grade framework for optimizing multi-agent systems by jointly evolving prompt bundles (MAMUT) rather than optimizing agents in isolation. They validate this on a grocery assistant, showing that system-level optimization outperforms local sub-agent optimization by ~7% because it captures coordination dynamics (e.g., context passing) that local metrics miss. The most stealable insight is their 'Judge Calibration' loop: they use evolutionary search (GEPA) to optimize the *evaluator's* prompt to match human labels (91.4% agreement) before using that judge to optimize the agents. This is a rigorous solution to the noisy fitness function problem we face in LLM evolutionary search.

### [AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning](https://arxiv.org/abs/2602.23258)

**2026-02-26** | Alibaba Group, Harbin Institute of Technology, Shenzhen | M=6 P=7 I=8 *discuss*

*Method:* Test-time rectify-or-reject pruning framework with retrieval-augmented rectifier, failure-driven indicator pool, and dual-stage deduplication | *LLM role:* rectifier, teacher, deduplicator, reasoning_engine

> Wang et al. propose a test-time 'firewall' for multi-agent systems that intercepts messages and validates them against a retrieved set of error patterns (mined from offline failure trajectories). They achieve ~6% accuracy gains on math benchmarks by iteratively rectifying or pruning erroneous outputs before they propagate. The critical takeaway for our AlgoEvo work is the **Failure-Driven Indicator Pool**: we should implement a similar module that mines failed code generations to build a repository of 'forbidden patterns,' allowing a lightweight verifier to prune bad mutations before expensive execution. This effectively turns the 'graveyard' of failed runs into a persistent memory that improves sample efficiency.

### [Robust Multi-Objective Controlled Decoding of Large Language Models](https://arxiv.org/abs/2503.08796)

**2025-03-11** | University College London, University of Basel, Ulsan National Institute of Science and Technology | M=8 P=6 I=8 **MUST-READ** *discuss*

*Method:* Maximin two-player game between adversarially computed reward weights and sampling policy, solvable through Nash equilibrium, reduced to convex optimization, with blockwise best-of-K sampling | *LLM role:* controlled_decoding_target

> RMOD formulates multi-objective decoding as a zero-sum game between a policy and adversarial weights, solving a convex optimization problem at each decoding step to maximize the worst-case value estimate (essentially a Process Reward Model). The results are empirically strong, outperforming MO-DPO and scalarized baselines on alignment benchmarks by dynamically preventing any single objective from collapsing. **Key Takeaway:** The efficient inference-time weight optimization algorithm (Eq. 10) is a 'stealable' mechanism for **AlgoEvo** and **RobustMAS**. We should implement this dynamic adversarial weighting to balance conflicting code metrics (e.g., runtime vs. solution quality) during evolutionary search, replacing our current static scalarization methods.

### [Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs](https://arxiv.org/abs/2512.18134)

**2025-12-19** | Stanford University, NVIDIA | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Joint optimization of modulo scheduling and warp specialization formulated as a constraint satisfaction problem, solved by Integer Linear Programming (ZLP) for initial modulo schedule and Satisfiability Modulo Theories (SMT) solver for combined SWP and WS | *LLM role:* none

> Twill formulates the complex interplay of software pipelining and warp specialization on modern GPUs (Hopper/Blackwell) as a joint SMT/ILP optimization problem, automatically rediscovering expert-tuned Flash Attention schedules without heuristics. The results are rigorous, matching hand-tuned performance within 1-2% and handling new hardware constraints (Blackwell TMEM) automatically. The key takeaway is the 'cost normalization' technique via ILP to make the scheduling search space tractable, and the demonstration that exact constraint solvers can replace human intuition for complex kernel generation. This is essential reading for your work on OR formulations for GPU scheduling and LLM serving optimization, offering a deterministic baseline to compare against evolutionary approaches.

### [Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters](https://arxiv.org/abs/2512.10271)

**2025-12-11** | Virginia Tech, Kuwait University, Northwestern Polytechnical University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Proximal Policy Optimization (PPO) with Actor-Critic network for dynamic prioritization coupled with Mixed-Integer Linear Programming (MILP) for multi-resource allocation | *LLM role:* none

> RLTune introduces a hybrid scheduling architecture where an RL agent (PPO) handles dynamic job prioritization based on cluster state, while a MILP solver optimizes the specific job-to-node packing constraints for the top-K jobs. The results are robust, demonstrating a ~25% makespan reduction over Slurm on a physical cluster and significant gains over pure RL baselines on standard traces (Philly, Helios). The critical takeaway is the architectural separation of concerns: delegating 'fuzzy' long-term objectives to RL and 'hard' constraint satisfaction to a symbolic solver. We should evaluate this 'RL-guided Solver' pattern for our `GPUSched` and `EvoCut` projects to improve constraint handling without losing adaptivity.

### [MaMa: A Game-Theoretic Approach for Designing Safe Agentic Systems](https://arxiv.org/abs/2602.04431)

**2026-02-04** | Max Planck Institute for Software Systems | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Iterative Stackelberg Security Game solving via LLM-based adversarial search, building on AFlow for system design updates | *LLM role:* decomposition_guide, prompt_optimizer, heuristic_generator, evaluator

> MaMa automates the design of multi-agent systems by formulating the problem as a Stackelberg Security Game: a Meta-Agent evolves system architectures (tools, communication graphs) while a Meta-Adversary iteratively optimizes worst-case agent compromises to break them. Empirical results on the BAD-ACTS benchmark show this adversarial co-evolution reduces attack success rates from ~50% (static baselines) to ~15-25% without degrading task quality. The critical takeaway is the implementation of an **adversarial co-evolution loop** within the architecture search—optimizing the 'threat' alongside the 'solution'—which directly addresses the robustness objectives in our RobustMAS project. We should implement this 'Meta-Adversary' concept to stress-test our evolved algorithms during the search phase rather than post-hoc.

### [Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling](https://arxiv.org/abs/2506.15707)

**2025-10-20** | Beijing Institute of Technology, Xiaohongshu Inc | M=8 P=8 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Direction-Oriented Resource Allocation (DORA) | *LLM role:* reasoning_path_generator

> Wang et al. introduce Direction-Oriented Resource Allocation (DORA), which uses embedding-based soft clustering to group semantically similar reasoning paths and allocates compute budget to distinct 'directions' rather than individual solutions. They prove solution-level allocation (like REBASE) is suboptimal when paths are correlated and show DORA achieves state-of-the-art accuracy on MATH500 with 3.5x fewer FLOPs. **Key Takeaway:** We can immediately steal the 'semantic uniqueness reweighting' mechanism for AlgoEvo. By clustering generated heuristics via embeddings before expensive evaluation, we can drastically improve sample efficiency and stop wasting compute on minor variations of the same code.

### [FORGE: Foundational Optimization Representations from Graph Embeddings](https://arxiv.org/abs/2508.20330)

**2025-09-24** | Brown University, Northeastern University, Fidelity Investments | M=4 P=8 I=6 **MUST-READ** *discuss*

*Method:* Vector-quantized graph autoencoder with GraphSAGE layers | *LLM role:* none

> FORGE pre-trains a vector-quantized graph autoencoder on diverse MIP instances to learn unsupervised, discrete instance- and variable-level embeddings. The results are rigorously backed by numbers, demonstrating 30-85% primal gap improvements in Gurobi via pseudo-cuts and search guidance, notably outperforming the LLM-based Li et al. (2025) baseline. The key takeaway is their use of vector quantization to create a discrete 'vocabulary' of optimization codes, which successfully captures global MIP structure where standard GNNs over-smooth. While we will not pivot from LLM evolutionary search to GNNs, this paper is highly relevant for EvoCut as it establishes the new deep learning baseline for pseudo-cut generation that our evolutionary methods must beat.

### [MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild](https://arxiv.org/abs/2603.17187)

**2026-03-17** | Carnegie Mellon University, UC Berkeley, UNC-Chapel Hill, UC Santa Cruz | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Continual meta-learning with LLM-based gradient-free skill evolution and RL-based LoRA fine-tuning using a process reward model | *LLM role:* policy_executor, skill_generator, reward_model

> MetaClaw is a continual learning framework for LLM agents that combines gradient-free skill evolution (distilling failures into reusable prompt instructions) with asynchronous RL fine-tuning guided by a process reward model. The results are backed by solid empirical gains, showing an 8.25x improvement in end-to-end task completion on a 30-day simulated CLI benchmark. The single most useful takeaway for us is their 'Skill Generation Versioning' mechanism: when co-optimizing discrete prompts/heuristics and continuous model weights, you must strictly separate support data (used to evolve the heuristic) from query data (collected after the update) and flush the RL buffer upon evolution to prevent training on stale rewards. This directly addresses the continuous learning and RL-infused evolution bottlenecks in AlgoEvo, giving us a concrete trick to steal for managing our own RL buffers during evolutionary search.

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**2025-11-12** | Microsoft, University of Illinois Urbana-Champaign, Georgia Institute of Technology, Indian Institute of Science | M=7 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Integer Linear Programming (ILP) for resource allocation, combined with ARIMA-based time-series forecasting and reactive heuristics for dynamic scaling and scheduling | *LLM role:* none

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interactive and non-interactive workloads. They validate this on real Microsoft O365 production traces (which they release), demonstrating a 25% reduction in GPU hours and $2.5M/month savings compared to reactive baselines. The primary value for us is the release of the production workload traces—allowing us to benchmark our 'GPUSched' formulations against real-world data rather than synthetic distributions—and their specific ILP formulation for unified capacity management, which directly competes with our internal OR models.

### [Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow](https://arxiv.org/abs/2406.01566)

**2025-03-05** | Carnegie Mellon University | M=8 P=9 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Max-flow problem formulation on directed, weighted graphs with Mixed Integer Linear Programming (MILP) for joint model placement and per-request pipeline scheduling | *LLM role:* none

> Helix formulates distributed LLM serving on heterogeneous clusters as a max-flow problem, using MILP to optimize model placement and deriving a per-request weighted round-robin scheduler from the flow solution. Unlike standard static pipeline parallelism, it routes every request dynamically based on edge capacities, achieving up to 3.3x throughput gains over Swarm on mixed GPU clusters (L4/T4/A100). The results are rigorous, backed by both physical cluster experiments and high-fidelity simulations. The critical takeaway is the 'per-request pipeline' abstraction: decoupling request routing from static device assignment allows exact OR methods to maximize utilization of weaker hardware—a technique we should immediately evaluate for our GPUSched project.

### [Optimizing NetGPT via Routing-Based Synergy and Reinforcement Learning](https://arxiv.org/abs/2511.22217)

**2025-11-27** | Zhejiang University, Huawei Technologies Co., Ltd., Zhejiang Lab, Macau University of Science and Technology, The University of Electro-Communications, Shenzhen CyberAray Network Technology Co., Ltd | M=5 P=6 I=7 *discuss*

*Method:* Unified router score with state-dependent fallback threshold and schema-preserving reinforcement learning (PPO with SFT anchor) for edge LLM policy update | *LLM role:* heuristic_generator

> Chen et al. propose a cloud-edge routing framework that dynamically offloads tool-calling tasks based on network conditions (RTT/Bandwidth) and a learned confidence score, while simultaneously updating the edge model via PPO. Results on 8,000 tasks show that dynamic thresholds outperform static baselines like FrugalGPT, and crucially, that interleaving SFT updates is required to prevent JSON schema collapse during RL. The primary takeaway for us is the 'SFT-anchored' update strategy: alternating between RL (for reward maximization) and SFT (on valid outputs) is a simple, effective stabilizer for maintaining structural constraints (like code syntax or JSON) during optimization. We should test this anchoring technique in AlgoEvo to keep evolved heuristics syntactically valid while maximizing fitness.


### Front 1 (24 papers) — EMERGING

**Density:** 0.00 | **Methods:** llm_in_the_loop, greedy_algorithm, sequence_parallelism, mixed_integer_linear_programming, llm_fine_tuned | **Problems:** resource_allocation, llm_serving_optimization, gpu_scheduling, multi_agent_coordination, llm_inference_optimization

*Unique methods:* activation_recomputation, activation_swapping, adaptation_safety, adaptive_control, adaptive_index_update, adaptive_parallelism, adaptive_synthesis, advantage_design, aggregate_llm_pipeline, alternating_optimization, analytical_modeling, approximate_nearest_neighbor_search, batch_scheduling, batching_optimization, behavior_cloning, bert, beta_distribution_modeling, bi_level_optimization, bipartite_matching, budget_scheduling, chain_of_thought, chebyshev_guided_optimization, cispo, combinatorial_optimization, context_migration, context_parallelism, convex_hull_relaxation, cpu_offloading, cuda_streams, dapo, demand_prediction, difficulty_classification, diffusion_model_inference, distributed_routing, dynamic_dispatching, dynamic_filtering, dynamic_scheduling, dynamic_tree_search, dynamic_voltage_and_frequency_scaling, embedding_similarity, empg, equilibrium_search, fair_queuing, fast_scanning, fault_isolation, format_penalty, fractional_gpu_allocation, gated_recurrent_unit, gigpo, gpu_coroutines, gpu_memory_optimization, gradient_accumulation, gradient_aggregation, graph_algorithms, graph_convolutional_network, gspo, gurobi, hardware_software_co_design, hierarchical_heuristic, hoeffding_tree_classification, hoeffding_tree_regression, inference_time_alignment, inter_process_communication, inverted_file_index, isotonic_regression, kl_regularization, lagrangian_relaxation, latency_bounded_partitioning, least_laxity_first_scheduling, lexicographical_optimization, linear_approximation, llm_alignment, llm_as_agent, llm_as_orchestrator, llm_fine_tuning, llm_in_context_learning, llm_iterative_refinement, lookup_table, mechanism_design, memory_centric_cost_modeling, memory_defragmentation, min_cost_max_flow, multi_agent_llm, multiplicative_increase_additive_decrease, non_preemptive_scheduling, noon_ppo, nvidia_green_contexts, nvidia_mps, offline_profiling, optimization_penalty_function, outlier_detection, pava, performance_estimation, ppo_style_optimization, predictive_modeling, preemption, primal_dual_optimization, product_quantization, resource_partitioning, retrieval_augmented_generation_serving, rl_grpo, rlhf, sapo, sequence_level_clipping, sequence_packing, sequence_parallelism, slo_aware_scheduling, sparse_gating, spatial_sharing, state_synchronization, subgradient_method, submodular_maximization, system_algorithm_co_design, system_optimization, task_decomposition, tensor_management, tf_idf, threshold_based_routing, topology_aware_placement, topology_routing_algorithm, transformer_hidden_states_analysis, uncertainty_quantification, vcg_auction, vector_similarity_search, virtual_time_scheduling, zero_sum_game
*Shared methods:* bayesian_inference, bin_packing, black_box_optimization, convex_optimization, cost_modeling, data_parallelism, distributed_training, dynamic_batching, dynamic_programming, dynamic_resource_allocation, early_stopping, evolution_of_heuristics, flash_attention, game_theory, gpu_resource_management, gpu_scheduling, gradient_descent, graph_partitioning, greedy_algorithm, group_relative_policy_optimization, grpo, heuristic_search, k_means_clustering, kv_cache_optimization, kv_caching, linear_programming, llm_as_evaluator, llm_as_heuristic, llm_code_generation, llm_evolutionary_search, llm_fine_tuned, llm_in_the_loop, llm_orchestration, llm_serving_optimization, load_balancing, lora, lstm, memory_management, mixed_integer_linear_programming, mixed_integer_nonlinear_programming, mixed_integer_programming, mlp, multi_agent_reinforcement_learning, multi_agent_systems, multi_objective_optimization, online_learning, pipeline_parallelism, policy_gradient, ppo, preemptive_scheduling, profiling, program_synthesis, queueing_theory, queuing_theory, reinforcement_learning, resource_allocation, resource_allocation_optimization, resource_management, rl_ppo, scheduling, scheduling_algorithms, speculative_decoding, supervised_learning, tensor_parallelism

This research front focuses on applying advanced Operations Research (OR) techniques, including Mixed-Integer Linear Programming (MILP), Integer Linear Programming (ILP), Linear Programming (LP), Queueing Theory, Convex Optimization, and Stochastic Control, to optimize the complex challenges of Large Language Model (LLM) inference serving, GPU scheduling, and resource allocation. The core objective is to maximize throughput, minimize latency (e.g., Time-to-First-Token), and ensure efficient utilization of heterogeneous GPU and CPU resources, often under strict Service Level Objectives (SLOs). This front moves beyond heuristic-based system design by leveraging formal mathematical modeling to derive provably optimal or near-optimal solutions for critical LLM infrastructure problems.

Key contributions include MILP/ILP formulations for fine-grained GPU partitioning (JIGSAWSERVE achieving 11.3x capacity over Loki), MoE expert placement (MoETuner with 17.5% speedup on H200s, FineMoE with 37-47% throughput gains over Megatron-LM, HAP with ~1.6x speedups), dynamic precision assignment (SNIP), and prompt cache management (GreenCache reducing carbon by ~15%). Queueing theory and stochastic control are used for throughput-optimal scheduling (Li et al. proving work-conserving policies are optimal, Lin et al.'s Gate-and-Route policy), capacity planning (inference-fleet-sim), and prefill-decode disaggregation (SLO-Aware Compute). Convex optimization enables optimal multi-draft speculative sampling (Global Resolution, 10,000x faster than LP solvers) and robust resource allocation (Nan et al. achieving ~50% energy savings). Hybrid approaches like RLTune combine RL for prioritization with MILP for allocation, demonstrating ~25% makespan reduction over Slurm. TRAIL's embedding-based prediction in vLLM achieved 24x lower TTFT.

This is a maturing research front, rapidly transitioning from foundational OR formulations to integrated, real-time systems. The consistent "MUST-READ" and "CHANGES-THINKING" annotations across papers highlight its high impact and ongoing evolution. Future work is likely to focus on dynamic adaptation to varying workloads, seamless integration with heterogeneous hardware (including CPU-GPU co-optimization), and scaling solutions to multi-node and multi-cloud environments. The trajectory indicates a move towards more robust optimization under uncertainty, incorporating multimodal data pipelines, and developing hybrid AI/OR architectures that leverage LLMs for high-level reasoning while relying on OR for low-level, constrained decision-making.

**Papers:**

### [GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads](https://arxiv.org/abs/2604.04335)

**2026-04-06** | University of Washington, NVIDIA, Rice University, University of Waterloo, Cisco Research, Independent Researcher | M=6 P=7 I=6 *discuss*

*Method:* SLO-aware dynamic programming scheduler with intelligent video preemption, elastic sequence parallelism, and dynamic batching | *LLM role:* none

> GENSERVE optimizes the co-serving of text-to-image and text-to-video diffusion models on shared GPUs using a dynamic programming scheduler that jointly manages step-level preemption, dynamic batching, and elastic sequence parallelism. The results are empirically backed, demonstrating up to a 44% improvement in SLO attainment over baselines like SRTF on an 8-GPU cluster. The most useful takeaway is their two-stage DP formulation: they first generate a small set of anchored candidate actions (hold, resume, scale SP) per request, then run a knapsack DP to maximize global SLOs in under 2ms. While this targets diffusion models rather than LLMs, the OR formulation for elastic sequence parallelism and batching under strict latency SLOs is directly applicable to our GPUSched project.

### [Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines](https://arxiv.org/abs/2604.15186)

**2026-04-16** | Imperial College London, Independent Researcher | M=7 P=8 I=8 **MUST-READ** *discuss*

*Method:* Aggregate LLM Pipeline for performance prediction combined with a hierarchical heuristic search for joint throughput/latency optimization of GPU allocations and topology-aware fractional placement | *LLM role:* none

> Scepsy is a serving system that schedules multi-LLM agentic workflows onto GPU clusters by profiling relative LLM execution times to create an Aggregate LLM Pipeline and using a hierarchical heuristic for fractional GPU allocation. The results are strongly backed by empirical numbers on a 16-GPU cluster, showing up to 2.4x higher throughput and 27x lower latency compared to baselines like Kubernetes HPA and Ayo. The key insight is that instead of modeling the highly variable end-to-end latency of dynamic agentic workflows, systems can achieve stable steady-state performance predictions by modeling the aggregate fractional demand each LLM places on the system. This is highly relevant for our research in LLM serving scheduling and GPU resource allocation, providing a strong heuristic baseline and modeling abstraction that our formal OR formulations must benchmark against.

### [Cascadia: An Efficient Cascade Serving System for Large Language Models](https://arxiv.org/abs/2506.04203)

**2025-09-29** | Princeton University, University of Cambridge, Tsinghua University, HKUST, Shanghai Jiaotong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Bi-level optimization with MILP for deployment and Chebyshev-guided method for routing | *LLM role:* evaluator

> Jiang et al. propose CASCADIA, a bi-level optimization framework for LLM cascade serving that iterates between an MILP solver for hardware deployment (choosing DP/TP/PP strategies) and a Chebyshev-guided solver for routing thresholds. They demonstrate 2.3x average throughput gains over SGLang and CascadeServe on H100 clusters, backed by rigorous ablation studies. The key takeaway is the effective decomposition of the NP-hard joint optimization problem: freezing routing to solve deployment via MILP, then optimizing routing against that deployment. This is a direct reference point for our 'GPUSched' project, validating the efficacy of formal integer programming in LLM resource allocation.

### [Pareto Multi-Objective Alignment for Language Models](https://arxiv.org/abs/2508.07768)

**2025-08-11** | Ruhr University Bochum | M=7 P=5 I=6 *discuss*

*Method:* PAMA (PAreto Multi-Objective Alignment) algorithm, which transforms multi-objective RLHF into a convex optimization problem with a closed-form solution, combined with Noon PPO. | *LLM role:* subject_of_optimization

> PAMA introduces a computationally efficient algorithm for multi-objective alignment by reformulating the expensive gradient-norm minimization of MGDA into a convex optimization problem with a closed-form solution, reducing complexity from O(n^2d) to O(n). Empirical results on LLaMA-2-7B are robust, showing stable convergence on conflicting objectives (e.g., harmlessness vs. length) where baselines like MGDA-UB oscillate or fail. The single most useful takeaway is the analytical derivation for optimal objective weighting (Theorem 1) and the 'Noon PPO' heuristic (clipping negative advantages); we could port this logic to our multi-objective process reward models in AlgoEvo to balance search signals efficiently. While the NLP experiments are trivial, the gradient balancing mechanism is directly applicable to our multi-objective RL controllers.

### [ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning](https://arxiv.org/abs/2602.21534)

**2026-02-25** | University of California, Los Angeles, University of Wisconsin–Madison | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Stable Agentic Multi-turn Policy Optimization (SAMPO) integrating sequence-level clipping, fine-grained advantage estimation, and dynamic filtering | *LLM role:* policy

> The authors dissect why standard RL (GRPO/PPO) fails in multi-turn agentic tasks, identifying that token-level importance sampling (IS) clipping allows negative-advantage outliers to destabilize training. They propose SAMPO, which enforces sequence-level clipping and integrates fine-grained step-level advantages (similar to process rewards) to stabilize learning. The results are rigorous, showing a jump from ~50% to 92% success on ALFWorld by fixing the gradient update mechanics rather than just prompt engineering. **Key Takeaway:** We must audit our RL implementations; if we are using token-level clipping for multi-step evolutionary agents, we are likely suffering from silent gradient instability—switching to sequence-level clipping and masking negative-advantage outliers is an immediate, code-level improvement we should adopt.

### [Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference](https://arxiv.org/abs/2512.16134)

**2025-12-18** | Baidu Inc. | M=6 P=9 I=7 **MUST-READ** *changes-thinking* *discuss*

*Method:* Staggered Batch Scheduling (SBS) with Throughput-Adaptive Interval Control, Multi-tier State Synchronization, Prioritized Batch Allocation Algorithm (PBAA) for Prefill, and IQR-Aware Lexicographical Decode Scheduling for Decode | *LLM role:* none

> Tian et al. introduce Staggered Batch Scheduling (SBS) for DP+EP architectures, enforcing a buffering window to enable global bin-packing rather than immediate dispatch, which they prove causes Head-of-Line blocking in non-preemptive prefill phases. Tested on a production H800 cluster serving DeepSeek-V3, they demonstrate a 30-40% reduction in TTFT and a ~20% throughput increase backed by clear utilization metrics. The most valuable takeaway for our GPUSched project is their 'IQR-aware lexicographical' scheduling heuristic for the Decode phase, which robustly balances batch size against KV-cache memory variance—a constraint logic we should immediately adopt. This work validates that discrete batching is superior to continuous dispatch for MoE models, necessitating an update to our queuing theory models.

### [FREESH: Fair, Resource- and Energy-Efficient Scheduling for LLM Serving on Heterogeneous GPUs](https://arxiv.org/abs/2511.00807)

**2025-11-05** | University of Washington, University of Alberta, Hong Kong University of Science and Technology (Guangzhou), Huazhong University of Science and Technology, Renmin University of China | M=7 P=9 I=8 **MUST-READ** *discuss*

*Method:* Hierarchical optimization combining Mixed Integer Linear Programming (MILP) for pool-level routing and resource allocation, Multiplicative Increase Additive Decrease (MIAD) for GPU frequency scaling, and Least-Laxity-First (LLF) for request scheduling | *LLM role:* none

> FREESH optimizes LLM serving across geographically distributed, heterogeneous GPUs by combining a slow-timescale MILP for carbon-aware routing with fast-timescale MIAD frequency scaling and Least-Laxity-First (LLF) queueing. The results are backed by solid simulations using real PJM carbon intensity traces and Llama3-70B, demonstrating a 45% emission reduction and 28% energy reduction. The single most useful takeaway is the adaptation of TCP congestion control (MIAD) for dynamic GPU frequency scaling, which elegantly decouples real-time power management from the heavier MILP resource allocation. This is a must-read for the GPUSched team, as we should benchmark our OR formulations against their multi-timescale architecture and potentially steal their spatiotemporal carbon-aware objective function.

### [IEMAS: An Incentive-Efficiency Routing Framework for Open Agentic Web Ecosystems](https://arxiv.org/abs/2603.17302)

**2026-03-18** | Shanghai Jiao Tong University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* VCG-based Min-Cost Max-Flow (MCMF) for bipartite matching, guided by Hoeffding Tree predictive QoS models | *LLM role:* none

> IEMAS routes client requests to distributed LLM agents by formulating the assignment as a Min-Cost Max-Flow bipartite matching problem, using VCG auctions to align economic incentives with KV-cache reuse. The results are backed by solid vLLM simulations, demonstrating an 80.2% KV-cache hit rate and a 35% cost reduction over baselines like GraphRouter. The most actionable takeaway for us is their method of quantifying KV-cache affinity (via Longest Common Prefix) and embedding it directly as a cost-reduction weight in a network flow optimization model. While the decentralized VCG auction mechanics might be overkill for centralized clusters, we should absolutely steal their cache-aware MCMF formulation for our OR-based LLM inference scheduling (GPUSched) work.

### [Predicting Future Utility: Global Combinatorial Optimization for Task-Agnostic KV Cache Eviction](https://arxiv.org/abs/2602.08585)

**2026-02-09** | Baidu, Fudan University | M=8 P=9 I=7 **MUST-READ** *discuss*

*Method:* Convex-hull relaxation and marginal-utility-based greedy solver for global combinatorial optimization | *LLM role:* none

> Tang et al. formulate KV cache eviction not as a heuristic filtering task, but as a global combinatorial optimization problem maximizing 'Oracle Importance' (future utility) across all attention heads. They solve this NP-hard problem efficiently by applying Isotonic Regression (via PAVA) to create a convex surrogate of the eviction loss, enabling an optimal greedy allocation strategy that is deployed via an offline-computed lookup table. Results are strong: they achieve 80% cache reduction on LongBench and RULER with minimal degradation, significantly outperforming dynamic heuristics like AdaKV. **Key Takeaway:** The decomposition of error into 'ranking error' vs. 'allocation error'—and solving the latter via convex-hull relaxation—is a powerful OR pattern we should apply to our own resource allocation and scheduling problems.

### [Epistemic Gain, Aleatoric Cost: Uncertainty Decomposition in Multi-Agent Debate for Math Reasoning](https://arxiv.org/abs/2603.01221)

**2026-03-01** | ByteDance, CUHK Shenzhen, TU Darmstadt, Tongji University | M=8 P=8 I=8 **MUST-READ** *discuss*

*Method:* Uncertainty-Guided Multi-Agent Reinforcement Learning (UMAD) extending Group Relative Policy Optimization (GRPO) | *LLM role:* debate_agent

> This paper decomposes uncertainty in multi-agent debate (MAD) into epistemic and aleatoric components, and trains agents using GRPO with an intrinsic reward for improving peers' performance and an advantage penalty for high aleatoric noise. The results are backed by solid numbers on MATH and AIME benchmarks, showing up to 14% improvement over zero-shot MAD in heterogeneous setups. The single most useful takeaway is the 'epistemic influence intrinsic reward'—explicitly rewarding an agent based on the delta improvement in a peer's correctness in the subsequent turn. This is highly relevant for us; we should steal this peer-improvement reward formulation for MASPRM to train agents that generate genuinely useful feedback rather than sycophantic agreement.

### [Batching-Aware Joint Model Onloading and Offloading for Hierarchical Multi-Task Inference](https://arxiv.org/abs/2508.13380)

**2025-08-18** | The University of Texas at Austin, DEVCOM Army Research Laboratory | M=5 P=8 I=7 *discuss*

*Method:* Alternating optimization combining greedy submodular maximization with Lagrangian relaxation for onloading and constrained linear programming for offloading | *LLM role:* none

> Cha et al. propose an alternating optimization framework (J3O) for joint model placement and query routing in hierarchical inference systems, decomposing the MINLP into greedy Lagrangian submodular maximization and linear programming. They explicitly model batching latency at the edge using a linear surrogate to handle the non-convex batch setup costs, achieving ~97% of Gurobi's optimal accuracy with <15% of the runtime. **Takeaway:** We should steal their linear surrogate formulation for batching overhead (approximating the L0-norm of task arrival) for our 'GPUSched' integer programs; it offers a tractable way to model batching efficiency in serving systems without full non-linear solvers.

### [ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios](https://arxiv.org/abs/2604.09603)

**2026-03-10** | Qwen Applications Business Group of Alibaba, Zhejiang University | M=7 P=8 I=7 *discuss*

*Method:* Elastic Speculative Decoding with Sparse Confidence Gating and Unified Elastic Budget Scheduling | *LLM role:* none

> ECHO optimizes speculative decoding in high-concurrency LLM serving by formulating dynamic tree construction as a budget scheduling problem governed by sparse confidence gating. The results are rigorously backed by empirical data, demonstrating up to 5.35x walltime speedups and outperforming state-of-the-art static trees (EAGLE-3) on industrial-scale models (Qwen3-235B) at high batch sizes (BS=256). The key insight is treating speculative decoding not merely as a single-request tree search, but as a batch-level resource allocation problem where a fixed verification compute budget is dynamically shifted from low-confidence requests to extend the depth of high-confidence ones. This is highly relevant to our research in LLM serving scheduling, as it provides a concrete, kernel-compatible framework for applying operations research allocation principles directly to inference acceleration.

### [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](https://arxiv.org/abs/2510.09330)

**2025-12-02** | University of Warwick | M=7 P=4 I=7 *discuss*

*Method:* Two-player zero-sum game formulation solved by a linear programming (LP) solver at inference time to compute minimax equilibrium strategies, using binary probes for helpfulness and safety scores, with a sigmoid penalty for risk. | *LLM role:* agent_response_selection, evaluator

> The authors formulate LLM response selection as a zero-sum game, solving a small Linear Program (LP) at inference time to mix candidate answers such that the expected risk never exceeds a 'safe fallback' baseline. Results are statistically significant, showing ~15% accuracy gains on SafetyBench by effectively managing the trade-off between helpfulness and safety probes. The key takeaway is the 'Adaptation Safety' constraint formulation: using an LP to guarantee that a stochastic policy is no worse than a heuristic baseline is a powerful, lightweight control mechanism we could adapt for selecting evolved algorithms or managing constraints in multi-agent optimization.

### [VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG](https://arxiv.org/abs/2504.08930)

**2026-01-19** | Georgia Institute of Technology | M=5 P=7 I=6 *discuss*

*Method:* Analytical performance modeling and latency-bounded partitioning algorithm for hybrid CPU-GPU vector index, combined with a distributed runtime pipeline featuring query- and shard-aware routing and dynamic dispatcher. | *LLM role:* target_of_optimization

> VectorLiteRAG optimizes RAG serving throughput by dynamically partitioning vector indices between CPU and GPU memory based on access skew and latency SLOs. The results are credible, showing up to 1.5x throughput gains on H100/L40S setups by balancing retrieval speed against LLM KV-cache capacity. The most stealable insight is their use of a Beta distribution to analytically model the *minimum* hit rate within a batch (the bottleneck) to predict tail latency without full simulation—a technique we could adapt for stochastic constraints in our serving formulations. It solves a resource allocation problem we care about, though via systems engineering rather than the rigorous OR methods we prefer.

### [MEMO: Fine-grained Tensor Management For Ultra-long Context LLM Training](https://arxiv.org/abs/2407.12117)

**2025-01-15** | Peking University, Tencent Inc. | M=8 P=5 I=7 *discuss*

*Method:* Fine-grained activation memory management combining token-wise recomputation and swapping with bi-level Mixed Integer Programming (MIP) for memory planning | *LLM role:* none

> Memo enables training 7B LLMs with 1M context on 8 GPUs by combining token-wise activation swapping with a bi-level Mixed Integer Programming (MIP) approach to eliminate memory fragmentation. The results are strong (52% MFU vs ~30% for DeepSpeed) and demonstrate that static memory planning via OR solvers outperforms dynamic allocators for repetitive Transformer workloads. The key takeaway is the bi-level MIP strategy—solving the allocation for one layer and broadcasting it—which makes the NP-hard memory planning tractable. We should adapt this MIP formulation for our own GPU scheduling and inference resource allocation (GPUSched) projects.

### [Guaranteeing Semantic and Performance Determinism in Flexible GPU Sharing](https://arxiv.org/abs/2603.15042)

**2026-03-17** | Shanghai Jiao Tong University, Chinese Academy of Sciences, University of Chinese Academy of Sciences | M=4 P=8 I=6 *discuss*

*Method:* GPU coroutines abstraction decoupling logical execution contexts (vCtx) from physical GPU resources (pCtx) via dynamic context binding and cooperative preemption | *LLM role:* none

> DETSHARE introduces 'GPU coroutines' to decouple logical execution contexts from physical GPU resources, enabling fine-grained spatial sharing without modifying kernels to preserve semantic determinism. The results are highly credible and backed by strong empirical numbers on A800/Hopper GPUs, demonstrating up to 79% higher training throughput and 69% lower inference latency compared to temporal sharing baselines. The most useful takeaway for us is the identification of 'semantic determinism'—specifically how dynamic spatial partitioning alters floating-point reduction trees and ruins training/RLHF stability. While we approach GPU scheduling via OR formulations (MIP/queueing) rather than OS-level CUDA hacking, this paper matters for our GPUSched project. We must incorporate their identified system realities—such as the 12% preemption overhead and the strict requirement for semantic determinism—into our mathematical models to ensure our theoretical schedules are actually deployable in production.

### [Automatic Operator-level Parallelism Planning for Distributed Deep Learning -- A Mixed-Integer Programming Approach](https://arxiv.org/abs/2503.09357)

**2025-03-12** | Huawei | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Mixed-Integer Programming (MIP) formulation with a bi-level solution framework including a heuristic operation merging step | *LLM role:* none

> She et al. formulate distributed LLM training/inference as a Flexible Distributed Job Shop Scheduling Problem (FDJSSP) solved via Mixed-Integer Programming (MIP) combined with a heuristic graph coarsening step. They demonstrate that this automated approach not only reproduces DeepSeek V3's expert-designed "DualPipe" strategy but, when allowed to search longer, discovers a schedule with 50% fewer pipeline bubbles. The primary takeaway is the effectiveness of the bi-level optimization framework (greedy merging + MIP) to handle the scale of operator-level graphs, proving that formal OR methods can outperform manual system design for LLM infrastructure. This is a mandatory read for our GPUSched project, offering a concrete formulation for operator-level constraints we can directly adapt.

### [AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence](https://arxiv.org/abs/2602.16873)

**2026-02-18** | Korea National Open University | M=7 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Task-adaptive topology routing algorithm based on DAG structural properties (parallelism width, critical path depth, coupling density) combined with an adaptive synthesis protocol | *LLM role:* decomposition_guide, executor, arbiter, synthesizer

> AdaptOrch introduces a control layer that dynamically routes tasks to one of four agent topologies (Parallel, Sequential, Hierarchical, Hybrid) by analyzing the task's dependency graph properties (parallelism width, coupling density). The results are strong and credible, showing a 9.8% improvement on SWE-bench over single-model baselines and significantly outperforming static multi-agent architectures like standard MoA. The most valuable takeaway is the **Topology Routing Algorithm**: a linear-time heuristic that maps DAG structure to optimal agent coordination patterns. We should adapt this for AlgoEvo to automatically parallelize search on loosely coupled code components while forcing sequential reasoning on critical paths, potentially improving our sample efficiency and cost scaling.

### [FlexSP: Accelerating Large Language Model Training via Flexible Sequence Parallelism](https://arxiv.org/abs/2412.01523)

**2025-02-11** | Peking University, ByteDance Inc., Beihang University | M=8 P=6 I=7 *discuss*

*Method:* Heterogeneity-adaptive sequence parallelism using MILP and dynamic programming for optimal strategy selection | *LLM role:* none

> FlexSP optimizes distributed LLM training by dynamically assigning varied-length sequences to heterogeneous Sequence Parallelism (SP) groups using a Mixed-Integer Linear Programming (MILP) solver in the loop. The results are solid, showing up to 1.98x speedup on A100 clusters by mitigating communication bottlenecks for short sequences while preventing OOM for long ones. **Key Takeaway:** The authors use Dynamic Programming to 'bucket' similar sequences, drastically reducing the variable count for the MILP solver; this specific technique—reducing problem granularity to make exact solvers feasible in real-time systems—is directly applicable to our 'GPUSched' and inference resource allocation work. While we focus on evolution, this is a definitive reference for our 'OR for AI Systems' track, proving that formal optimization can beat heuristics in dynamic GPU scheduling.

### [Learning to Incentivize: LLM-Empowered Contract for AIGC Offloading in Teleoperation](https://arxiv.org/abs/2508.03464)

**2025-08-05** | University of Houston, The Pennsylvania State University, University of Florida, Kyung Hee University, China University of Petroleum (East China), Prairie View A&M University | M=8 P=5 I=8 **MUST-READ** *discuss*

*Method:* LLM-evolved solver for ASP setting inference (P2) combined with convex optimization for contract derivation (P3) | *LLM role:* evolutionary_search

> Zhan et al. propose an LLM-based evolutionary framework to generate Python solvers for inferring hidden agent parameters in contract design (a bilevel OR problem). While the experiments are toy-scale (N=7 actions) and benchmarks are weak, the methodological architecture is highly relevant: they separate 'short-term reflectors' (analyzing parent pairs) from a 'long-term reflector' (aggregating insights across generations) to guide the Mutation LLM. This is a concrete, transferable implementation of evolutionary memory that we should test to improve sample efficiency in our own code-evolving agents.

### [Predictive Scheduling for Efficient Inference-Time Reasoning in Large Language Models](https://arxiv.org/abs/2602.01237)

**2026-02-01** | Harvard University | M=7 P=8 I=7 **MUST-READ** *discuss*

*Method:* Predictive Scheduling framework using lightweight MLP predictors on intermediate transformer hidden states or LoRA-fine-tuned classifiers on raw text, combined with a greedy batch allocation algorithm | *LLM role:* predictor

> Brown et al. propose a 'Predictive Scheduling' framework that trains lightweight predictors (MLP on hidden states or LoRA) to estimate required CoT length before generation, using a greedy algorithm to allocate tokens under a global budget. Results show a 7.9% accuracy gain on GSM8K over uniform batching, backed by a systematic layer-wise analysis. **The key takeaway for us is that middle transformer layers (12-17)—not the final layer—contain the highest signal-to-noise ratio for predicting reasoning difficulty.** We should immediately test extracting features from these layers for our AlgoEvo value functions to improve sample efficiency. While the greedy scheduling algorithm itself is standard OR, the application to internal model states for pre-run allocation is a valid efficiency win for our serving optimization work.

### [Justitia: Fair and Efficient Scheduling for LLM Applications](https://arxiv.org/abs/2510.17015)

**2025-10-19** | Shanghai Jiao Tong University | M=7 P=9 I=7 **MUST-READ** *discuss*

*Method:* Virtual-time based fair queuing with memory-centric cost modeling and MLP-based demand prediction | *LLM role:* none

> Justitia introduces a scheduler for LLM agents that prioritizes applications based on their 'virtual finish time' (derived from a theoretical fair-sharing model) but executes them with full resource saturation to minimize completion time. The authors demonstrate a ~60% reduction in average job completion time compared to state-of-the-art fair schedulers (VTC) on vLLM, backed by rigorous experiments and theoretical delay bounds. The key takeaway is the 'KV token-time' cost metric (pd + d^2/2) which accurately captures memory bottlenecks in auto-regressive generation, and the insight that 'long-term fairness' allows for short-term resource saturation. This is immediately actionable for your GPUSched project and relevant for optimizing the serving infrastructure of AlgoEvo.

### [AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation](https://arxiv.org/abs/2602.17100)

**2026-02-19** | Shanghai Jiao Tong University, Meituan | M=8 P=7 I=9 **MUST-READ** *changes-thinking* *discuss*

*Method:* Reinforcement learning (GRPO) optimized multi-agent system with LLM-based orchestrator agent for dynamic layered DAG topology generation | *LLM role:* orchestrator

> AgentConductor trains an LLM orchestrator via GRPO to dynamically generate and refine layered DAG interaction topologies (output as YAML) for code generation, optimizing for both correctness and token efficiency. The key innovation is a multi-objective reward that combines execution correctness with a 'difficulty-aware density penalty,' forcing the model to learn a policy that scales graph complexity with task hardness. Results are strong, showing ~14% gains on APPS while reducing token costs. We should immediately steal the **YAML-based topology action space** and the **density-aware reward formulation** to implement RL-driven structure optimization in AlgoEvo and MASPRM, replacing our static or hand-tuned interaction graphs.

### [Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling](https://arxiv.org/abs/2604.17353)

**2026-04-19** | Peking University | M=8 P=8 I=8 **MUST-READ** *changes-thinking* *discuss*

*Method:* Hive multi-agent inference infrastructure with Logits Cache and Agent-Aware Scheduling | *LLM role:* inference_engine_optimization

> Hive is an LLM inference infrastructure that optimizes multi-agent and test-time scaling workloads by introducing Logits Cache for redundant sampling paths and Agent-Aware Scheduling for KV cache eviction. The results are backed by solid empirical evidence, demonstrating a 1.11x-1.76x speedup for re-sampling and a 33%-51% reduction in KV cache miss rates on Qwen3-8B. The key insight is that caching intermediate logits (not just KV states) allows the engine to skip expensive forward passes during stochastic resampling of shared prefixes, while evicting KV cache based on an agent's structural contribution outperforms standard LRU. This is highly relevant for scaling LLM evolutionary search and multi-agent optimization, as it provides concrete systems-level techniques to drastically reduce the inference costs associated with branching generation and complex agent coordination.



## Bridge Papers

Papers connecting multiple research fronts:

### [SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling](https://arxiv.org/abs/2502.14617)

**TRUE SYNTHESIS** | score=0.57 | Front 2 → Front 0

> SageServe optimizes LLM inference resource allocation across regions using an Integer Linear Programming (ILP) model coupled with ARIMA-based traffic forecasting, specifically targeting mixed interact


---

*Generated by Research Intelligence System*
