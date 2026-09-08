"""One module per stage boundary in docs/spec.md.

A stage takes its inputs as arguments and returns named artifacts; it never
reads the dataset or another stage's files. Persisting a stage's output is the
caller's job, through `llms4subjects.artifacts.ArtifactStore`.

Bodies land with their own tickets: label_text and encoders (04-05), indexes and
retrievers (04-06), fusion (07), group_prior (11), reranker (12), adjudicator
(13), evaluator and submission (03).
"""
