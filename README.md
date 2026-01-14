# Long-Text-Generation-for-D2T

Scaling Up Data-to-Text Generation to Longer Sequences: A New Dataset and Benchmark Results for Generation from Large Triple Sets  https://aclanthology.org/2025.inlg-main.47/

This paper introduces a long-input data-to-text benchmark designed to test whether modern LLMs can generate coherent, faithful longer texts from large RDF triple sets, addressing the fact that most existing datasets are short and English-centric. It builds a new DBpedia based dataset in English and Irish, with 537 triple sets ranging from 8 to 69 triples, and compares outputs from six LLMs, a rule based system (FORGe), and human written texts using LLM based evaluation. The results show clear differences between models and between English and Irish, highlighting both progress and remaining gaps for long-form, structured generation.

## Contents

- `generate.py`: run generation experiments
- `run.sh`: example commands
- `data/`: inputs and related files
- `results/`: generations and evaluation outputs
- `LLM_Eval/`: LLM-based evaluation utilities
- `finetune/`: optional fine-tuning scripts
- `utilities/`: shared helpers

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use API-based models, set your key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

## Run

```bash
bash run.sh
# or
python generate.py
```

## Citation

Please cite the paper above if you use this repository.
```bibtex
@inproceedings{osuji-etal-2025-scaling,
  title = {Scaling Up Data-to-Text Generation to Longer Sequences: A New Dataset and Benchmark Results for Generation from Large Triple Sets},
  author = {Osuji, Chinonso Cynthia and Mille, Simon and O'Connell, Ornait and Castro Ferreira, Thiago and Belz, Anya and Davis, Brian},
  booktitle = {Proceedings of the 18th International Natural Language Generation Conference},
  year = {2025},
  month = oct,
  address = {Hanoi, Vietnam},
  publisher = {Association for Computational Linguistics},
  pages = {810--822},
  url = {https://aclanthology.org/2025.inlg-main.47/}
}
```