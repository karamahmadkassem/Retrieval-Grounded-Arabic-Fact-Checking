# Arabic Fact Checking — Thesis Research

Research repository for **Karam Kassem's** thesis on Arabic fact checking, centered on the **ARAFA** (Arabic fact-checking) dataset and related experiments.

## Overview

This project investigates automated fact verification for Arabic text using **ARAFA** (~182K claim–evidence pairs). The current experiment is **in-article evidence localization**: given a claim and a known Wikipedia page, find the gold evidence span.

- **Lexical baseline:** BM25 on overlapping 1–3 sentence windows (`scripts/run_inarticle_bm25_eval.py`).
- **Next neural method:** extractive BERT (one span), not dense ranking.
- **Archived:** open-domain Wikipedia BM25 and e5 bi-encoder / reranker (`archive/`).

## Repository Structure

```
.
├── data/arafa/          # ARAFA, verification, chunks, gold, localization splits
├── scripts/             # Active data + BM25 localization code
├── results/             # Task spec + in-article BM25 findings
├── archive/             # Old tracks (see archive/README.md)
│   └── artifacts/       # Large local indexes/dumps (gitignored)
├── docs/                # Thesis, papers, slides
├── notes/meetings/
├── requirements.txt
└── requirements-ml.txt  # GPU (BERT / transformers)
```

## Getting Started

### Prerequisites

- Python 3.10+ (recommended)
- Git
- [Git LFS](https://git-lfs.com/) — required if you want to version the dataset on GitHub

### Clone and Set Up

```bash
git clone https://github.com/karamahmadkassem/Retrieval-Grounded-Arabic-Fact-Checking.git
cd "Fact Checking"

# If using Git LFS for the dataset:
git lfs install
git lfs pull
```

### Load the Dataset

```python
import json

with open("data/arafa/ARAFA.json", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))  # 181976
```

See [`data/arafa/README.md`](data/arafa/README.md) for the full schema. Summary metrics (judgement counts, claim types, source stats) are in [`data/arafa/ARAFA.metadata.json`](data/arafa/ARAFA.metadata.json).

## Key Documents

| Document | Location |
|----------|----------|
| Thesis proposal | `docs/thesis/Thesis_Proposal_Karam_Kassem.pdf` |
| ARAFA supplementary material | `docs/thesis/ARAFA_Supplementary_Material.pdf` |
| LLM-generated Arabic FC database paper | `docs/papers/llm-generated-arabic-fact-checking-database.pdf` |
| Thesis roadmap | `docs/presentations/ARAFA_Thesis_Roadmap_1.pptx` |
| Thesis framework | `docs/presentations/Thesis_Framework.pptx` |

## Large File: ARAFA.json

`data/arafa/ARAFA.json` is approximately **180 MB** and exceeds GitHub's 100 MB file size limit.

**Recommended approach — Git LFS:**

1. Install Git LFS: `git lfs install`
2. Remove `data/arafa/ARAFA.json` from `.gitignore`
3. Add and commit — `.gitattributes` already configures LFS tracking

**Alternative — exclude from repo:**

Keep the file local only (current `.gitignore` setting) and distribute via cloud storage, Zenodo, or Hugging Face Datasets. Document the download link here when published.

## Development Conventions

- Active Python lives in `scripts/`; retired code goes to `archive/scripts/`
- Write experiment outputs to `results/`
- Add meeting notes under `notes/meetings/`

## Related Work

Meeting notes reference [FEVER](https://fever.ai/) (Fact Extraction and VERification) as a related English fact-checking benchmark.

## License

License TBD — update before public release.

## Author

Karam Kassem
