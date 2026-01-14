# Long-Text-Generation-for-D2T

Code and experiment artefacts for the INLG 2025 paper:

Scaling Up Data-to-Text Generation to Longer Sequences: A New Dataset and Benchmark Results for Generation from Large Triple Sets  
https://aclanthology.org/2025.inlg-main.47/

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