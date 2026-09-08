"""Retrieval-based GND subject tagging for TIBKAT records.

The pipeline is a package of stages (`llms4subjects.stages`), wired together by
`llms4subjects.pipeline.predict`. Reading the dataset is the job of
`llms4subjects.corpus`; reading and writing cached stage outputs is the job of
`llms4subjects.artifacts`. Stages themselves take their inputs as arguments.

The rejected classifier lives in `baseline/`. It is a results row, not a
dependency, and nothing here imports it.
"""
