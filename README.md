# LLM4Subjects

SemEval 2025, Library Tagging Problem, XMLC

## Description

The code use "bert-base-multilingual-cased" or "xlm-roberta-base" for XMTC task.

Code in train.py handles sparse labels with csr_matrix, and hierarchical labels with GCN technique.

The all train set is too large for single GPU, so we only run the experiment on core train set.

Feel free to include more or less label metadata from `label_metadata.concat_subject_metadata` for performance tweak.

## Command

Packages Installation

```
pip install -r requirments.txt
```

Run Train File

```
python train.py
```
