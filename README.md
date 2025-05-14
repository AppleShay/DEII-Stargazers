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
8. [License](#-license)  

---

## ✨ Features

- **Collector**: paginated GitHub Search + commit-velocity calls  
- **ETL**: flatten JSON → Parquet feature store  
- **Modeling**: OLS, Ridge, RandomForest, XGBoost, LightGBM with CV  
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
