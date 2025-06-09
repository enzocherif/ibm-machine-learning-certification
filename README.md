
# IBM Machine Learning Certification · Notebook Portfolio 📚🤖

<p align="center">
  <a href="LICENSE">
    <img alt="Licence"
         src="https://img.shields.io/github/license/enzocherif/ibm-machine-learning-certification?style=flat-square">
  </a>
  <a href="https://github.com/enzocherif/ibm-machine-learning-certification">
    <img alt="Last commit"
         src="https://img.shields.io/github/last-commit/enzocherif/ibm-machine-learning-certification?style=flat-square">
  </a>
  <a href="https://colab.research.google.com/github/enzocherif/ibm-machine-learning-certification">
    <img alt="Open in Colab"
         src="https://img.shields.io/badge/⚡%20Open%20in%20Colab-yellow?logo=google-colab&style=flat-square">
  </a>
</p>

> **Author :** **Enzo CHERIF** — Robotics & Data-Automation Engineer  
> **Programme :** *IBM Machine Learning Professional Certificate*  
> **Progress :** `🏃 In progress — Module X / Y`  
> **Last update :** 2025-06-06


---

## 🗺️ Table of Contents

|   |   |
|---|---|
| 📖 **[About](#about)** | 📂 **[Folder Structure](#structure)** |
| ⚙️ **[Environment & Setup](#setup)** | 📑 **[Notebook Catalogue](#catalogue)** |
| ▶️ **[How to Run](#run)** | 🚀 **[Roadmap](#roadmap)** |
| 🤝 **[Contributing](#contributing)** | 📝 **[License](#license)** |
| 📫 **[Contact](#contact)** | |

<a id="about"></a>
### 📖 About

This repository is the single point of truth for **all** the notebooks, labs, and projects I complete while earning the *IBM Machine Learning Professional Certificate* on Coursera.

**Why make it public?**

| Goal | What it means for you |
|------|----------------------|
| **Transparency** | You can inspect every line of code, model choice, and experiment—no hidden “black-box” demos. |
| **Versioning** | Git history shows how my skills evolve from basic regression to advanced topics like XGBoost, SHAP, and deployment. |
| **Portfolio** | Recruiters, peers, and mentors can clone or open any notebook instantly (🟡 Colab badge above) and reproduce the results. |

**Key Competencies Demonstrated**

- Data wrangling & cleaning with **Pandas** / **NumPy**
- Supervised learning (Regression, Classification) with **scikit-learn**
- Tree-based ensembles & gradient boosting (**Random Forest**, **XGBoost**)
- Unsupervised learning (Clustering, PCA, t-SNE, UMAP)
- Model evaluation, hyper-parameter tuning, and pipeline automation
- Visualisation with **Matplotlib** / **Seaborn**
- Reproducible workflows & environment management (requirements, virtualenv)

> 📌 **TL;DR** – If you want to gauge my hands-on ML ability, just browse the notebooks or click “Open in Colab” to run them live. Feedback and collaboration are welcome!



<a id="structure"></a>
### 📂 Folder Structure

```text
ibm-machine-learning-certification/
├── notebooks/          # 100 % of certification labs & mini-projects (~30 min each)
│   ├── 01_simple-linear-regression.ipynb
│   ├── 02_logistic-regression.ipynb
│   ├── ... (ordre chronologique)
│   └── capstone_project.ipynb
├── utils/              # Helper scripts (e.g. catalogue generator)
├── data/               # Small sample datasets (<100 MB, licence-friendly) – optional
├── env/                # requirements.txt or environment.yml
├── .gitignore          # Ignore checkpoints, virtualenv, etc.
└── README.md
```

> **Why keep almost everything in `notebooks/` ?**  
> IBM’s hands-on tasks are intentionally short and self-contained; grouping them here avoids over-nesting while chronological prefixes (`01_`, `02_`, …) preserve order. Larger case studies stay in the same folder with a suffix like `_project.ipynb` for easy discovery.


