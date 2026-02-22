# Researcher Profile

## Research by Category

### [LLMs for Algorithm Design] — PRIMARY FOCUS

**LLM Evolutionary Search Innovations for Scientific and Algorithmic Discovery**

Applications of AlphaEvolve-style frameworks:

- Strengthening integer programs via evolution-guided cutting planes (EvoCut)
- Automated algorithm design via continuous latent-space optimization (Latent Heuristic Search)
- Automated control discovery via LLMs with NPU-accelerated cooperative search (AILS-II Enhanced)
- Multi-agent LLM evolutionary search for metaheuristic discovery (AlgoEvo)
- Vehicle routing (automatic extension of VRP variants)
- Scheduling (automatic extension of scheduling problem variants)
- SAT solver improvement via evolved heuristics
- Cloud scheduling and GPU resource allocation for LLM serving

Fundamental improvements to LLM evolutionary search:

- RL-infused evolution: reinforcement learning to guide evolutionary search
- Memory: persistent memory across evolution runs, not one-off executions
- Better signal for search: improved fitness signals and evaluation strategies (process reward models)
- Evolving the Evolver: a planner that decides which code components to evolve, optimizes prompts, or selects better evaluation strategies
- Sample efficiency (ShinkaEvolve or similar): achieving higher-quality results with fewer LLM samples
- Continuous learning: persistent memory/service architecture, not one-off runs
- Scalability: cloud resource management for massively parallel evolution (10,000+ cores), cost/convenience trade-offs
- Observability: expert-in-the-loop injection of human insights during evolution
- Code agent interaction: synergizing evolutionary search with general-purpose coding agents
- Research agents: using auxiliary agents to avoid idea stagnation during evolution

Active projects: AlgoEvo, EvoCut, AILS-II Enhanced, Latent Heuristic Search

### [Generative AI for OR]

**Using LLMs and generative AI for operations research modeling and solving**

- Ambiguity-grounded benchmarking for symbolic operations research modeling (OR-Bench)
- Evaluation methodology for LLM reasoning on OR problems
- Multi-agent systems for optimization modeling
- Vehicle routing & logistics: ALNS for CVRP, VRPTW, PDPTW; heterogeneous fleets; stochastic routing; large-scale instances (1000+ customers)

Active projects: OR-Bench

### [OR for Generative AI]

**Applying formal OR methods to optimize AI and multi-agent systems**

Optimization of multi-agent systems:

- Process reward models for multi-agent systems (MASPRM)
- Hypergraph-based memory architectures for multi-agent debate (HERMES)
- Robust two-stage stochastic optimization of multi-agent system resources (RobustMAS)
- Multi-agent coordination, communication protocols, and role assignment

OR methods for AI infrastructure:

- Integer programming formulations for GPU scheduling
- Resource allocation for LLM inference
- Queueing theory for AI serving systems
- Batch scheduling optimization for model serving

Active projects: MASPRM, HERMES, RobustMAS, GPUSched

## Key Methodologies Used

- ALNS, column generation, Benders decomposition, cutting planes
- Genetic algorithms, evolutionary strategies
- Graph neural networks for routing
- LLM-in-the-loop optimization
- AlphaEvolve-style LLM evolutionary search
- Process reward models for multi-agent evaluation
- Scenario-based stochastic programming
- Hypergraph-based memory architectures

## All Active Projects

1. **AlgoEvo**: Multi-agent LLM evolutionary search for metaheuristic discovery
2. **EvoCut**: Strengthening integer programs via evolution-guided language models
3. **AILS-II Enhanced**: Automated control discovery via LLMs and NPU-accelerated cooperative search
4. **Latent Heuristic Search**: Continuous optimization for automated algorithm design
5. **MASPRM**: Multi-agent system process reward model
6. **HERMES**: Hypergraph-based efficient role-aware memory with elective storage for multi-agent debate systems
7. **RobustMAS**: Robust stochastic two-stage multi-agent systems optimization
8. **GPUSched**: OR formulations for LLM inference scheduling
9. **OR-Bench**: Ambiguity-grounded benchmarking for symbolic operations research modeling

## Reading Priorities

- **Must-read**: Papers that directly improve methods I use, beat my results, advance LLM evolutionary search fundamentals, or propose novel multi-agent architectures for optimization
- **High priority**: Novel ideas transferable to my problems (even from different domains), especially around sample efficiency, reward modeling, memory, scalability of evolutionary search
- **Medium priority**: Incremental improvements in related areas
- **Low priority**: Surveys, tutorials, distant applications
