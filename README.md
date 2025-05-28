# StarGazers Predictor

A lightweight end-to-end pipeline to **predict GitHub “stars”** for open-source repositories.  
Fetch repo metadata via the GitHub REST API, build tabular features, train and evaluate multiple regression models, and expose a Dockerized FastAPI service—complete with CI/CD and Ansible provisioning.

---

## 📋 Table of Contents

1. [Features](#-features)  
2. [Prerequisites](#-prerequisites)  
3. [Getting Started](#-getting-started)  
4. [Usage](#-usage)  
5. [Folder Structure](#-folder-structure)  
6. [Development Workflow](#-development-workflow)  
7. [Contributing](#-contributing)  

---

## ✨ Features

- **Collector**: paginated GitHub Search + commit-velocity calls  
- **ETL**: flatten JSON → Parquet feature store  
- **Modeling**: OLS, Ridge, RandomForest, XGBoost, LightGBM (GridSearchCV) with log-transform and feature engineering  
- **Serving**: FastAPI endpoint (`/rank`) in Docker  
- **CI/CD**: GitHub Actions + GitHook “train-if-better” + Docker image promotion  
- **Provisioning**: Ansible playbooks for Dev & Prod VMs  
- **Reporting**: automatic R² metrics JSON and scalability charts  

---

## 🛠️ Prerequisites

- **Git** ≥ 2.25  
- **Python** ≥ 3.9  
- **Docker** & **Docker Compose**  
- **Ansible** ≥ 2.10  
- A **GitHub Personal Access Token** with scopes:  
  - `repo`, `read:org` (for data collector)  
  - `write:packages` (for pushing Docker images)  

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone git@github.com:<org>/<repo>.git StarGazers
   cd StarGazers
   
2. Create a Python virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Configure your environment

   Copy ​`.env.example → .env`
   
   Edit `.env` to point `GITHUB_TOKEN_PATH` at your local PAT file:
   ```bash
   GITHUB_TOKEN_PATH=~/.config/star-predictor/token_<your-name>.txt
  
4. Verify your token
   ```bash
   python -c "import os; print(open(os.getenv('GITHUB_TOKEN_PATH')).read().startswith('ghp_'))"

## 🏃‍♂️ Usage

1. Data Collection
   ```bash
   python src/collector/collector.py \
     --query "language:python stars:>50" \
     --output data/raw/repos_raw.jsonl
2. Feature Engineering
   ```bash
   python src/features/build_features.py \
      --input data/raw/repos_raw.jsonl \
      --output data/features/features.parquet
3. Training & Evaluation
   ```bash
   python src/models/train.py \
    --features data/features/features.parquet \
    --model rf \
    --metrics models/metrics/rf_metrics.json
4. Run locally in Docker
   ```bash
   cd infra/docker
    docker-compose up --build
    # Service available at http://localhost:8000/rank

## 📂 Folder Structure
        StarGazers/
        │
        ├── README.md
        ├── .gitignore
        ├── .env.example
        ├── requirements.txt         # (optional, if created)
        │
        ├── data/
        │   ├── raw/                 # 10 JSON files (repos_page_01.json → repos_page_10.json)
        │   └── features/
        │       └── features.parquet
        │
        ├── docs/                    # Documentation folder (if used)
        │
        ├── infra/                   # Infra-related code (e.g., VM, docker, setup) (if used)
        │
        ├── models/
        │   ├── artifacts/           # Trained model files (if saved)
        │   └── metrics/
        │       └── metrics.json     # Performance metrics of all models
        │
        ├── src/
        │   ├── collector/
        │   │   └── collector.py     # Script to collect data from GitHub API
        │   │
        │   ├── features/
        │   │   └── build_features.py # Extracts and transforms features, saves parquet
        │   │
        │   └── models/
        │       └── train.py         # Trains models and saves metrics
        ├── app/
        │   ├── openstack-client/
        │   │   ├── OpenStack instance setup scripts/
        │   │   ├── CloudInit configuration files/
        │   │   ├── Ansible automation/
        │   │   └── Git Hooks
        │   │
        │   ├── githubstar/
        │   │   ├── production_server/
        │   │   └── development_server/ # Extracts and transforms features, saves parquet
        │   │
        │   ├──scalability/
        │   ├──UPPMAX 2025_1-1-openrc.sh
        │   └──README.md
    


## 🔄 Development Workflow
- Branch off `main` for each feature (e.g. `feature/collector`).

- Commit early & often; push to remote.

- Pull Request for review & merge.

- GitHook on `main` triggers `train_and_compare.sh` → promotes Docker image if R² improves.

- GitHub Actions uses `GH_PAT_CI` secret to run workflows and deploy via Ansible.


## 🤝 Contributing
- Fork the repo.

- Create a feature branch.

- Open a Pull Request describing your changes.

- Ensure all checks pass (formatting, tests, training).
