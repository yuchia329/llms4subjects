# LLMs4Subjects rebuild — local ticket tracker

Tickets derived from [docs/spec.md](../../docs/spec.md). One file per ticket in
`issues/`, numbered in dependency order. A ticket is startable when every ticket
in its **Blocked by** line is done.

## Dependency tree

Levels are earliest-possible start. Everything on the same level can run in parallel.

```
L0   01 vocabulary rebuild            02 repo + environment
      │                                │
      │                                ├──────────────┐
      │                                ▼              │
L1    │                          03 evaluator         │
      │                            │      │           │
      │                 ┌──────────┘      └───────┐   │
      │                 ▼                         ▼   ▼
L2    │           04 kNN end-to-end          14 baseline ──────────┐
      │             │        │                (parallel, off path) │
      ├─────────────┤        │                                      │
      ▼             ▼        ▼                                      │
L3   05 doc→label dense    06 lexical                               │
      │        │             │                                      │
      │        └──────┬──────┘                                      │
      ▼               ▼                                             │
L4   08 translation  07 fusion + ablation                           │
      │               │    │    │                                   │
      └──────┐  ┌─────┘    │    └────────┐                          │
             ▼  ▼          ▼             ▼                          │
L5        09 rung 1   11 group prior  12 reranker                   │
             │             │             │                          │
             ▼             │             ▼                          │
L6        10 rung 2        │        13 LLM adjudication             │
             │             │             │      │                   │
             └──────┬──────┴─────────────┘      │                   │
                    ▼                           ▼                   │
L7              15 rung 3               16 coverage curve            │
                    │                           │                   │
                    └───────────┬───────────────┘                   │
                                ▼                                   │
L8                       17 final test run                          │
                                │                                   │
                                └───────────┬───────────────────────┘
                                            ▼
L9                                    18 writeup
```

Critical path, 10 tickets: **02 → 03 → 04 → 05 → 07 → 09 → 10 → 15 → 17 → 18**

Off the critical path, so free to slot in whenever: **01** (gates 05 and 11), **14** (gates only
the writeup), **06**, **08**, **11**, **12**, **13**, **16**.

## Frontier

Startable immediately: **01**, **02**.

## Notes

- Ticket 14 (baseline re-run) is independent of the pipeline and can run in parallel throughout.
- GPU work goes to the `nlp2` host (`ssh nlp2` → `nlp-gpu-02.be.ucsc.edu`). Only three tickets need it:
  **14** (baseline classifier), **15** (two rung-3 fine-tunes), and **12** only if it recommends a reranker
  fine-tune. Everything else runs on Apple Silicon.
- The dataset is gitignored, so the GPU host rebuilds it with the same command used locally. Ticket 02
  verifies that path before any training run depends on it.
- 17 opens the official test set for the first and only time. Nothing before it may read test data.
