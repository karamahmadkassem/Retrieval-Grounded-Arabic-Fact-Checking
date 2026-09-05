# Arabic Fact Checking — Thesis Research

Research repository for **Karam Kassem's** thesis on Arabic fact checking, centered on the **ARAFA** (Arabic fact-checking) dataset and related experiments.

## Overview

This project investigates automated fact verification for Arabic text. The core resource is the ARAFA dataset: ~182K claim–evidence pairs with verdicts (`supported`, `refuted`, `nei`) and reasoning annotations. Supporting materials include the thesis proposal, supplementary documentation, reference papers, and presentation slides.

## Repository Structure

```
.
├── data/
│   └── arafa/                  # ARAFA dataset
│       ├── ARAFA.json          # ~180 MB — see Git LFS note below
│       ├── ARAFA.metadata.json # Dataset metrics (judgements, types, etc.)
│       └── README.md           # Dataset schema and statistics
├── scripts/                    # Dataset utilities (format, validate, metadata)
├── docs/
│   ├── thesis/                 # Thesis proposal and supplementary material
│   ├── papers/                 # Reference papers
│   └── presentations/          # Roadmaps and framework slides
├── notes/
│   └── meetings/               # Advisor / team meeting notes
├── src/                        # Source code (experiments, pipelines, utilities)
├── notebooks/                  # Jupyter notebooks for exploration and analysis
├── results/                    # Experiment outputs (gitignored by default)
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
└── .gitattributes              # Git LFS configuration for large dataset
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

- Place reusable Python modules in `src/`
- Use `notebooks/` for exploratory analysis; keep production scripts in `src/`
- Write experiment outputs to `results/` (excluded from git by default)
- Add meeting notes under `notes/meetings/` with descriptive filenames (e.g. `meeting-03-topic.txt`)

## Related Work

Meeting notes reference [FEVER](https://fever.ai/) (Fact Extraction and VERification) as a related English fact-checking benchmark.

## License

License TBD — update before public release.

## Author

Karam Kassem
