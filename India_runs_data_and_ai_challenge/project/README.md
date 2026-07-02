# Senior AI Engineer Candidate Ranker

Production-quality, deterministic Python application that ranks ~100,000 candidates for a Senior AI Engineer role and exports a competition-compliant submission CSV.

## Requirements

- Python 3.11+
- CPU only, no network access during execution
- ~16 GB RAM (tested with streaming JSONL loading)
- Candidate dataset: `candidates.jsonl` or `candidates.jsonl.gz` in the parent directory (not committed to Git)

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repository-name>/project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

No third-party packages are strictly required (stdlib only), but installing from `requirements.txt` keeps the environment documented.

### 4. Add the challenge dataset

Place the competition files in the repository root (one level above `project/`):

```
repository-root/
├── candidates.jsonl          # or candidates.jsonl.gz
├── candidate_schema.json
├── validate_submission.py
└── project/
    └── ...
```

Default paths are configured in `config.json`:

| Setting | Default |
|---------|---------|
| `paths.candidates_file` | `../candidates.jsonl` |
| `paths.schema_file` | `../candidate_schema.json` |
| `paths.output_file` | `output/submission.csv` |

## Running the project

From the `project/` directory:

```bash
python main.py
```

Optional custom config path:

```bash
python main.py config.json
```

Validate the output:

```bash
python ../validate_submission.py output/submission.csv
```

## GitHub repository setup

Before pushing to GitHub:

1. **Never commit secrets** — this project uses no API keys. If you add `.env`, keep it local (listed in `.gitignore`).
2. **Do not commit generated files** — `output/`, `*.csv`, and logs are ignored.
3. **Do not commit the virtual environment** — `.venv/` is ignored.
4. **Do not commit the dataset** — `candidates.jsonl` is large and excluded; document how to obtain it instead.
5. **Copy environment template (optional)**:

   ```bash
   cp .env.example .env
   ```

   The application runs without `.env` using `config.json` defaults.

### Recommended first commit contents

- Python source (`*.py`)
- `config.json`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `.env.example`

## Project structure

```
project/
├── main.py                 # Pipeline entry point
├── config.py               # Configuration loader
├── config.json             # Weights, penalties, paths
├── loader.py               # JSONL / gzip streaming loader
├── parser.py               # Record validation and normalization
├── feature_engineering.py  # Feature extraction
├── resume_score.py         # Skill/resume scoring
├── experience_score.py     # Experience scoring
├── education_score.py      # Education scoring
├── behavior_score.py       # Redrob behavioral signals
├── penalty.py              # Deterministic penalty rules
├── ranking.py              # Final score and top-100 selection
├── export.py               # CSV export
├── utils.py                # Logging and helpers
├── requirements.txt
├── README.md
├── .env.example            # Optional env var template (no secrets)
└── output/                 # Generated at runtime (gitignored)
    └── submission.csv
```

## Configuration

All tunable values live in `config.json`:

| Section | Purpose |
|---------|---------|
| `paths` | Input dataset, schema, and output CSV paths |
| `component_weights` | Resume, experience, education, behavioral blend |
| `skill_weights` | Per-skill importance for AI/ML role fit |
| `proficiency_multipliers` | Skill proficiency scaling |
| `education_field_scores` | Preferred field-of-study scores |
| `behavioral_weights` | Per-signal weights for all 23 Redrob fields (sum to 1.0) |
| `behavioral_normalization` | Per-field normalization bounds |
| `resume_scoring` / `experience_scoring` | Component blend weights and caps |
| `penalty_thresholds` | Deterministic penalty trigger values |
| `experience` | Years and relevance parameters |
| `normalization` | Final score bounds |

Edit `config.json` to change paths or tuning without modifying business logic. Paths are relative to the `project/` directory and work on Windows, macOS, and Linux.

## Algorithm overview

1. **Load** candidates from JSONL (supports `.gz`) one record at a time.
2. **Parse** and validate against `candidate_schema.json` required fields.
3. **Normalize** missing/null values to safe defaults.
4. **Extract features** — skills, career tenure, education, behavioral metrics, penalty flags.
5. **Score components** (each normalized to 0–1):
   - **Resume**: weighted AI/ML skills, proficiency, endorsements, title fit
   - **Experience**: total years, AI role months, continuity, title relevance
   - **Education**: field of study and degree level (not institution prestige)
   - **Behavioral**: Redrob engagement, assessments, activity, verification
6. **Apply penalties** for timeline issues, keyword stuffing, inactive profiles, etc.
7. **Combine** weighted components minus penalties into a final score.
8. **Rank** all candidates; select top 100 with `(-score, candidate_id)` tie-break.
9. **Export** CSV with `candidate_id`, `rank`, `score`, `reasoning`.

### Reasoning format

Each row includes a concise explanation, e.g.:

```
ML Engineer with 6.4 yrs; 4 AI core skills; response rate 0.88.
```

## Expected output

- File: `output/submission.csv`
- Header: `candidate_id,rank,score,reasoning`
- Exactly 100 data rows
- Ranks 1–100, scores non-increasing
- Tie-break: lower `candidate_id` wins when scores are equal
- Passes `validate_submission.py` without modification

## Performance

Designed for ~100k candidates in under 5 minutes on CPU (~55 seconds on a typical laptop):

- Single-pass streaming load with buffered I/O
- Precompiled `SkillMatcher` and `TitleMatcher` (no per-candidate regex)
- Dates parsed once during record normalization
- Behavioral metrics normalized once in feature extraction
- `heapq`-based top-100 selection (constant memory for rankings)
- No external API calls or embeddings

## Error handling

The pipeline fails with clear messages for:

- Missing dataset or config files
- Empty candidate pool
- Invalid configuration (e.g. weights not summing to 1.0)
- Insufficient candidates for top-100 export

Malformed JSONL lines are logged and skipped rather than halting the run.
