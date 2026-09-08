# 02 — Restructure the repository and define a CUDA-free environment

**What to build:**

Prefactor. The new pipeline is a different architecture from the existing scripts, and the rejected
classifier must keep running because it is a results row. Separate them so a newcomer can tell which
code represents the project's conclusion, and so the pipeline can be built as importable stages rather
than scripts that re-read CSVs.

Establish the stage-module layout described in the spec, move the previous training and evaluation
scripts into a clearly-marked baseline area with their data paths updated, and adopt the cached-artifact
convention where every stage writes an output keyed by its configuration.

Also split the environment definition. The existing dependency pins include CUDA wheels, consistent with
the previous work having run on the GPU host, but indexing, retrieval, reranking and all evaluation must
run on Apple Silicon. Without this split, nothing runs locally.

Two environments, and the split is a hard constraint:

- **Local — Apple Silicon, no CUDA.** Everything except training. This is the loop that runs hundreds of
  times, so it must never require the GPU host to be reachable.
- **GPU — `nlp2`, reached with `ssh nlp2`.** Only the three training runs: two rung-3 fine-tunes and the
  baseline classifier re-run.

Because the dataset is not tracked in version control, the GPU host rebuilds it with the same command used
locally rather than receiving a hand-copied snapshot. Verify that path now, in this ticket, rather than
discovering it mid-training-run.

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] Pipeline stages live in an importable package, one module per stage, with no cross-stage file reads
- [ ] Previous training and evaluation scripts sit in a separate baseline area and still execute against the clean splits
- [ ] A CUDA-free environment definition installs and imports cleanly on Apple Silicon
- [ ] A separate GPU environment definition exists and installs on the `nlp2` host
- [ ] The dataset rebuild command runs successfully on `nlp2`, producing identical record and label counts to local
- [ ] A documented path exists for retrieving training artifacts from `nlp2` back to local
- [ ] No local stage requires `nlp2` to be reachable
- [ ] Experiment configuration is expressed as committed data files, not edited into scripts
- [ ] The artifact directory convention is documented and ignored by git
- [ ] Running the baseline entrypoint reaches its first training step on the Mac or fails with a clear hardware message, not an import error
