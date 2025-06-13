
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

- 📖 **[About](#about)**  
- 📂 **[Folder Structure](#structure)**  
- ⚙️ **[Environment & Setup](#setup)**  
- 📑 **[Notebook Catalogue](#catalogue)**  
- ▶️ **[How to Run](#run)**  
- 🚀 **[Roadmap](#roadmap)**  
- 🤝 **[Contributing](#contributing)**  
- 📝 **[License](#license)**  
- 📫 **[Contact](#contact)**

---

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

---

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

---

<a id="setup"></a>
### ⚙️ Environment & Setup

#### Prérequis
- **Python ≥ 3.10** (testé sous 3.11)  
- **Git** installé localement  
- Un gestionnaire d’environnements : `venv` (standard) ou **Conda**

---

#### Installation rapide (`venv`)

    # 1️⃣ Cloner le dépôt
    git clone https://github.com/enzocherif/ibm-machine-learning-certification.git
    cd ibm-machine-learning-certification

    # 2️⃣ Créer & activer l’environnement virtuel
    python -m venv .venv
    source .venv/bin/activate      # Windows : .\.venv\Scripts\activate

    # 3️⃣ Installer les dépendances
    pip install -r env/requirements.txt

---

#### Alternative : Conda

    conda env create -f env/environment.yml
    conda activate ibm-ml-cert

---

#### Mettre à jour le projet

    git pull                                # Récupère les nouveaux notebooks
    pip install -U -r env/requirements.txt  # Met à jour les dépendances

> 💡 **Astuce VS Code** : l’extension Python/Jupyter détecte automatiquement l’environnement `.venv` ; ouvrez simplement un notebook et sélectionnez-le comme kernel.

---

<a id="catalogue"></a>
### 📑 Notebook Catalogue

> **Navigation rapide :** cliquez pour déplier la catégorie qui vous intéresse.  
> *(La table est produite automatiquement par `utils/generate_catalogue.py` — exécutez-le après
> avoir ajouté ou renommé un notebook pour mettre cette section à jour.)*

<details>
  <summary>📈 <strong>Regression</strong> — 2 notebooks</summary>

  | Notebook | Points clés | Taille |
  |----------|-------------|--------|
  | `01_simple-linear-regression.ipynb` | OLS · R² · Split train/test | 285 kB |
  | `02_multiple-linear-regression.ipynb` | Multicolinéarité · Pipeline | 428 kB |
</details>

<details>
  <summary>🧮 <strong>Classification</strong> — 2 notebooks</summary>

  | Notebook | Points clés | Taille |
  |----------|-------------|--------|
  | `03_logistic_regression.ipynb` | Sigmoïde · ROC-AUC | 150 kB |
  | `04_knn_classification.ipynb` | K-Fold CV · GridSearch | 297 kB |
</details>

<details>
  <summary>🌳 <strong>Trees & Ensembles</strong> — 2 notebooks</summary>

  | Notebook | Points clés | Taille |
  |----------|-------------|--------|
  | `05_decision_trees.ipynb` | Gini vs Entropy · Pruning | 109 kB |
  | `06_random_forest_xgboost.ipynb` | Feature importance · XGBoost | 512 kB |
</details>

<details>
  <summary>📊 <strong>Clustering</strong> — 1 notebook</summary>

  | Notebook | Points clés | Taille |
  |----------|-------------|--------|
  | `07_kmeans_customer_seg.ipynb` | Elbow · Silhouette | 5 498 kB |
</details>

<details>
  <summary>🔻 <strong>Dimensionality Reduction</strong> — 2 notebooks</summary>

  | Notebook | Points clés | Taille |
  |----------|-------------|--------|
  | `08_pca.ipynb` | Scree plot · Variance expliquée | 170 kB |
  | `09_tsne_umap.ipynb` | Perplexité · UMAP | 603 kB |
</details>

<details>
  <summary>🧪 <strong>Model Evaluation & Workflow</strong> — 2 notebooks</summary>

  | Notebook | Points clés | Taille |
  |----------|-------------|--------|
  | `10_model_evaluation.ipynb` | Confusion matrix · OOB | 409 kB |
  | `11_pipelines_gridsearch.ipynb` | ColumnTransformer · Pipeline | 190 kB |
</details>

---

<a id="run"></a>
### ▶️ How to Run

#### 🚀 Google Colab (zéro installation)
1. Cliquez sur le badge « ⚡ Open in Colab » en haut du README.  
2. Choisissez le notebook voulu ; Colab installe les dépendances courantes automatiquement.

---

#### 💻 Exécution locale (JupyterLab / VS Code)

```bash
# Active l’environnement virtuel (créé à l’étape « Environment & Setup »)
source .venv/bin/activate        # Windows : .\.venv\Scripts\activate

# Lance JupyterLab
jupyter lab
````

* Dans votre navigateur, ouvrez le fichier `.ipynb` souhaité.
* Sous **VS Code**, l’extension *Jupyter* détecte automatiquement l’environnement actif : ouvrez simplement le notebook et sélectionnez le kernel correspondant.

*C’est tout !*

---

<a id="roadmap"></a>
### 🚀 Roadmap

| État | Objectif | Détails / Bénéfice |
|------|----------|--------------------|
| ⏳ | **Terminer les derniers modules IBM** | Time Series Forecasting & Deep Learning (Keras/TensorFlow) |
| ⏳ | **Articles Medium / Dev.to** | Vulgariser 3 notebooks clés (Regression, XGBoost, Clustering) |
| ⏳ | **Binder / Colab stable links** | Un badge supplémentaire pour un environnement exécutable « one-click » |
| ⏳ | **Visualisations interactives** | Passer certains graphiques Matplotlib en Plotly pour l’exploration dynamique |
| ⏳ | **Vignettes automatiques** | Générer des aperçus PNG de chaque notebook via `nbconvert --to webpdf` pour le README |

> *Les cases passent à ✅ au fur et à mesure de l’avancement. N’hésitez pas à ouvrir une issue si vous souhaitez contribuer à l’une de ces étapes !*

---

<a id="contributing"></a>
### 🤝 Contributing

Les suggestions, améliorations et corrections sont les bienvenues !  
Pour proposer une contribution :

1. **Fork** le dépôt puis crée une branche : `git checkout -b feature/ton-idee`  
2. Commits clairs et concis (une fonctionnalité par commit)  
3. **Pull Request** détaillée : contexte, changements majeurs, capture d’écran si pertinent  
4. Attends une rapide code-review ; les tests GitHub Actions doivent passer ✅  

> Besoin d’inspiration ? Consulte la [🚀 Roadmap](#roadmap) ou ouvre une **Issue**.

---

<a id="license"></a>
### 📝 License

⚠️ **Aucune licence open-source n’est définie pour l’instant.**  
La réutilisation, la distribution ou la modification du contenu de ce dépôt nécessite l’accord écrit préalable de l’auteur (Enzo CHERIF).

*Si tu souhaites proposer une licence (MIT, Apache 2.0, GPL…), ouvre une Issue ou contacte-moi directement.*

---

<a id="contact"></a>
### 📫 Contact

| Type | Coordonnée |
|------|------------|
| ✉️ Email | **enzoccherife@gmail.com** |
| 💼 LinkedIn | [linkedin.com/in/enzocherif](https://www.linkedin.com/in/enzo-cherif-0465b5165/) |
| 🌐 Portfolio | <https://enzocherif.github.io> |

_N’hésite pas à me contacter pour toute question, remarque ou opportunité de collaboration._

