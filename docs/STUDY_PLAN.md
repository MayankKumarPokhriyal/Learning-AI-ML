# Study Plan

A week-by-week path through the whole course, designed for someone starting as a beginner and aiming to pass AI/ML engineer interviews. Tick the boxes as you go (GitHub renders them as checkboxes when you edit this file on your own branch).

---

## 1. Set up once

```bash
cd Learning-AI-ML
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
git checkout -b my-learning          # your exercises and notes live here; the course stays clean
tools/start_notebook.sh              # opens Jupyter Notebook with the right kernels
```

- Five notebooks need their own environment and kernel (the launcher lists them): LlamaIndex, Airflow, Monitoring & Drift, Kubeflow, AutoGluon. Install each with its `requirements-<env>.txt` in a separate virtual environment.
- Module 08 and the LLM capstones need a local LLM server (see the README's *Local LLM* section).
- Broke a notebook? `git checkout -- <path>` restores the original.

## 2. How to study one notebook

Each notebook takes roughly **6–8 hours** in total (reading + running + exercises + project). At ~2–3 hours a day, that's one notebook every 2–3 days; full-time, about one a day.

1. **Restart & Clear Output** (Kernel menu), then run the notebook yourself from the top.
2. **Predict before you run** every code cell; then change one thing and re-run.
3. **✍️ Your Turn** — type your answer until the checker shows ✅. Open the solution only after a real attempt.
4. **💡 Interview angle** — say it out loud in one or two sentences.
5. **🔧 Build It From Scratch** — the next morning, re-write it from memory without looking.
6. **🏋️ Practice** and **🚀 Mini Project** — do at least one stretch goal.
7. **🎤 Interview Q&A** — answer each question out loud (90-second timer) *before* opening the answer. Write every miss in your **mistakes log**.
8. **🧪 Quiz**, one **🎥 video** from Resources, and the **📝 cheat sheet**.
9. `git commit -am "day N: <notebook>"` — your learning log.

## 3. Weekly rhythm

| Day | What |
|---|---|
| Mon–Fri | Notebooks from the plan below. First 15 minutes every morning: yesterday's mistakes log + 20 flashcards. |
| Saturday | Redo the week's 🔴 exercises · 2–3 drills from [ML Coding Drills](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/01_ML_Coding_Drills.ipynb) · 2 problems from the DSA notebooks |
| Sunday | Mock interview on the week's topics (ask a friend or Claude to quiz you from the Q&A), then rest |

**Flashcards:** `python tools/export_flashcards.py` exports every interview question in the course to `flashcards/ai_course_interview_flashcards.csv`; import it into Anki and review daily, filtering by the modules you've finished.

---

## 4. The plan (full-time pace: ~18 weeks)

Part-time (2–3 h/day): keep the same order and give each "week" about 2½ weeks (~10 months total). *(optional)* notebooks can be skipped if time is short.

### Week 1 — Python foundations
- [ ] [01 Python Basics](../AI_Full_Stack_Engineer_Course/00_Foundations/01_Python_Basics.ipynb)
- [ ] [02 Python Builtins](../AI_Full_Stack_Engineer_Course/00_Foundations/02_Python_Builtins.ipynb)
- [ ] [03 OOP in Python](../AI_Full_Stack_Engineer_Course/00_Foundations/03_OOP_in_Python.ipynb)
- [ ] [04 Virtual Env & Packaging](../AI_Full_Stack_Engineer_Course/00_Foundations/04_Virtual_Env_and_Packaging.ipynb)
- [ ] [05 Python Internals & Concurrency](../AI_Full_Stack_Engineer_Course/00_Foundations/05_Python_Internals_and_Concurrency.ipynb)

### Week 2 — Data & math
- [ ] [NumPy](../AI_Full_Stack_Engineer_Course/01_Core_Scientific_Computing/01_NumPy.ipynb)
- [ ] [Pandas](../AI_Full_Stack_Engineer_Course/01_Core_Scientific_Computing/02_Pandas.ipynb)
- [ ] [SQL with DuckDB](../AI_Full_Stack_Engineer_Course/01_Core_Scientific_Computing/03_SQL_with_DuckDB.ipynb)
- [ ] [Math for ML](../AI_Full_Stack_Engineer_Course/01_Core_Scientific_Computing/04_Math_for_ML.ipynb)
- [ ] [Statistics & Probability](../AI_Full_Stack_Engineer_Course/01_Core_Scientific_Computing/05_Statistics_and_Probability.ipynb)
- [ ] Weekend: ML Coding Drills §1 (NumPy essentials) · Theory Questions §1–2 (statistics, math)

### Week 3 — Visualization, ML foundations, first algorithm
- [ ] [Matplotlib & Seaborn](../AI_Full_Stack_Engineer_Course/02_Data_Visualization/01_Matplotlib_and_Seaborn.ipynb)
- [ ] [Plotly](../AI_Full_Stack_Engineer_Course/02_Data_Visualization/02_Plotly.ipynb) *(optional)*
- [ ] [ML Fundamentals From Scratch](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/01_ML_Fundamentals_From_Scratch.ipynb)
- [ ] [Scikit-Learn workflow](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/02_Scikit_Learn.ipynb)
- [ ] [Linear Regression](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/03_Linear_Regression.ipynb)
- [ ] Start [DSA Coding Patterns — Part 1](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/06_DSA_Coding_Patterns_Part_1.ipynb): 2 patterns per week from now on

### Week 4 — Supervised learning algorithms
- [ ] [Logistic Regression](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/04_Logistic_Regression.ipynb)
- [ ] [K-Nearest Neighbors](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/05_K_Nearest_Neighbors.ipynb)
- [ ] [Naive Bayes](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/06_Naive_Bayes.ipynb)
- [ ] [Support Vector Machines](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/07_Support_Vector_Machines.ipynb)
- [ ] [Decision Trees](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/08_Decision_Trees.ipynb)

### Week 5 — Ensembles and clustering
- [ ] [Random Forest & Bagging](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/09_Random_Forest_and_Bagging.ipynb)
- [ ] [Gradient Boosting (XGBoost, LightGBM, CatBoost)](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/10_Gradient_Boosting.ipynb)
- [ ] [Ensembles: Voting, Stacking & AdaBoost](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/11_Ensembles_Voting_Stacking_AdaBoost.ipynb)
- [ ] [Clustering: K-Means & Hierarchical](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/12_Clustering_KMeans_and_Hierarchical.ipynb)
- [ ] [Clustering: DBSCAN & Gaussian Mixtures](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/13_Clustering_DBSCAN_and_Gaussian_Mixtures.ipynb)

### Week 6 — Unsupervised learning and practical ML
- [ ] [Dimensionality Reduction: PCA, t-SNE, UMAP](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/14_Dimensionality_Reduction_PCA_tSNE_UMAP.ipynb)
- [ ] [Anomaly Detection](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/15_Anomaly_Detection.ipynb)
- [ ] [Feature Engineering & Selection](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/16_Feature_Engineering_and_Selection.ipynb)
- [ ] [Imbalanced Data & Model Evaluation](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/17_Imbalanced_Data_and_Model_Evaluation.ipynb)
- [ ] [Model Interpretability: SHAP & LIME](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/18_Model_Interpretability_SHAP_LIME.ipynb)
- [ ] Weekend: ML Coding Drills §2 (classical ML) · Theory Questions §3–4 (ML fundamentals, classical models)

### Week 7 — Applied ML and deep learning foundations
- [ ] [Recommender Systems](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/19_Recommender_Systems.ipynb)
- [ ] [Time-Series Forecasting](../AI_Full_Stack_Engineer_Course/03_Classical_Machine_Learning/20_Time_Series_Forecasting.ipynb)
- [ ] [Neural Networks From Scratch](../AI_Full_Stack_Engineer_Course/04_Deep_Learning/01_Neural_Networks_From_Scratch.ipynb)
- [ ] [PyTorch](../AI_Full_Stack_Engineer_Course/04_Deep_Learning/02_PyTorch.ipynb)
- [ ] [Transformers From Scratch](../AI_Full_Stack_Engineer_Course/04_Deep_Learning/03_Transformers_From_Scratch.ipynb)

### Week 8 — NLP and vision
- [ ] [Keras 3 Overview](../AI_Full_Stack_Engineer_Course/04_Deep_Learning/04_Keras_3_Overview.ipynb) *(optional)*
- [ ] [Classical NLP](../AI_Full_Stack_Engineer_Course/05_NLP/01_Classical_NLP.ipynb)
- [ ] [Embeddings & Semantic Search](../AI_Full_Stack_Engineer_Course/05_NLP/02_Embeddings_and_Semantic_Search.ipynb)
- [ ] [Hugging Face Transformers](../AI_Full_Stack_Engineer_Course/05_NLP/03_HuggingFace_Transformers.ipynb)
- [ ] [OpenCV](../AI_Full_Stack_Engineer_Course/06_Computer_Vision/01_OpenCV.ipynb)
- [ ] Weekend: ML Coding Drills §3 (deep learning) · Theory Questions §5 (deep learning) · start [DSA — Part 2](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/07_DSA_Coding_Patterns_Part_2.ipynb)

### Week 9 — Vision, reinforcement learning, LLM basics
- [ ] [CNNs & Transfer Learning](../AI_Full_Stack_Engineer_Course/06_Computer_Vision/02_CNNs_and_Transfer_Learning.ipynb)
- [ ] [YOLO Object Detection](../AI_Full_Stack_Engineer_Course/06_Computer_Vision/03_YOLO_Object_Detection.ipynb)
- [ ] [Gymnasium & Q-Learning](../AI_Full_Stack_Engineer_Course/07_Reinforcement_Learning/01_Gymnasium_and_Q_Learning.ipynb)
- [ ] [Deep RL with Stable-Baselines3](../AI_Full_Stack_Engineer_Course/07_Reinforcement_Learning/02_Deep_RL_with_Stable_Baselines3.ipynb)
- [ ] [LLM APIs & Prompting](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/01_LLM_APIs_and_Prompting.ipynb)

### Week 10 — Generative AI (1)
- [ ] [RAG From Scratch](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/02_RAG_From_Scratch.ipynb)
- [ ] [LangChain & LangGraph](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/03_LangChain_and_LangGraph.ipynb)
- [ ] [LlamaIndex](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/04_LlamaIndex.ipynb)
- [ ] [Agents, Tool Calling & MCP](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/05_Agents_Tool_Calling_and_MCP.ipynb)
- [ ] [LLM Evaluation](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/06_LLM_Evaluation.ipynb)

### Week 11 — Generative AI (2) and MLOps start
- [ ] [Fine-Tuning with LoRA & DPO](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/07_Fine_Tuning_LoRA_and_DPO.ipynb)
- [ ] [LLM Inference & Serving](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/08_LLM_Inference_and_Serving.ipynb)
- [ ] [Distributed Training Overview](../AI_Full_Stack_Engineer_Course/08_Generative_AI_LLM/09_Distributed_Training_Overview.ipynb) *(optional)*
- [ ] [MLflow](../AI_Full_Stack_Engineer_Course/09_MLOps/01_MLflow.ipynb)
- [ ] [Weights & Biases](../AI_Full_Stack_Engineer_Course/09_MLOps/02_Weights_and_Biases.ipynb) *(optional)*
- [ ] Weekend: ML Coding Drills §4 (AI-engineering utilities) · Theory Questions §6 (NLP & LLMs) · [LLM System Design](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/04_LLM_System_Design.ipynb)

### Week 12 — MLOps
- [ ] [DVC](../AI_Full_Stack_Engineer_Course/09_MLOps/03_DVC.ipynb)
- [ ] [Airflow](../AI_Full_Stack_Engineer_Course/09_MLOps/04_Airflow.ipynb)
- [ ] [Model Monitoring & Drift](../AI_Full_Stack_Engineer_Course/09_MLOps/05_Model_Monitoring_and_Drift.ipynb)
- [ ] [Kubeflow Pipelines](../AI_Full_Stack_Engineer_Course/09_MLOps/06_Kubeflow_Pipelines.ipynb) *(optional)*
- [ ] [FastAPI](../AI_Full_Stack_Engineer_Course/10_Model_Serving/01_FastAPI.ipynb)

### Week 13 — Serving and data processing
- [ ] [Docker & CI/CD](../AI_Full_Stack_Engineer_Course/10_Model_Serving/02_Docker_and_CI_CD.ipynb)
- [ ] [BentoML](../AI_Full_Stack_Engineer_Course/10_Model_Serving/03_BentoML.ipynb)
- [ ] [Ray Serve](../AI_Full_Stack_Engineer_Course/10_Model_Serving/04_Ray_Serve.ipynb) *(optional)*
- [ ] [Polars](../AI_Full_Stack_Engineer_Course/11_Data_Processing/01_Polars.ipynb)
- [ ] [PySpark](../AI_Full_Stack_Engineer_Course/11_Data_Processing/02_PySpark.ipynb)
- [ ] Weekend: Theory Questions §7–9 (ranking, causal inference, MLOps) · [ML System Design](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/03_ML_System_Design.ipynb)

### Week 14 — Tuning and the first capstone
- [ ] [Dask](../AI_Full_Stack_Engineer_Course/11_Data_Processing/03_Dask.ipynb) *(optional)*
- [ ] [Optuna](../AI_Full_Stack_Engineer_Course/12_AutoML_Experimentation/01_Optuna.ipynb)
- [ ] [Ray Tune](../AI_Full_Stack_Engineer_Course/12_AutoML_Experimentation/02_Ray_Tune.ipynb)
- [ ] [AutoGluon](../AI_Full_Stack_Engineer_Course/12_AutoML_Experimentation/03_AutoGluon.ipynb) *(optional)*
- [ ] [Capstone 1 — End-to-End ML Project](../AI_Full_Stack_Engineer_Course/13_Capstone_Projects/01_End_to_End_ML_Project.ipynb)

### Week 15 — Capstones
- [ ] [Capstone 2 — End-to-End DL Project](../AI_Full_Stack_Engineer_Course/13_Capstone_Projects/02_End_to_End_DL_Project.ipynb)
- [ ] [Capstone 3 — LLM RAG Assistant](../AI_Full_Stack_Engineer_Course/13_Capstone_Projects/03_LLM_RAG_Assistant_Project.ipynb)
- [ ] [Capstone 4 — AI Agent](../AI_Full_Stack_Engineer_Course/13_Capstone_Projects/04_AI_Agent_Project.ipynb)

### Week 16 — Your own portfolio project
- [ ] Pick the template that matches your goal and adapt it to a dataset *you* care about: [ML](../AI_Full_Stack_Engineer_Course/14_Templates/01_ML_Template.ipynb) · [DL](../AI_Full_Stack_Engineer_Course/14_Templates/02_DL_Template.ipynb) · [LLM](../AI_Full_Stack_Engineer_Course/14_Templates/03_LLM_Template.ipynb)
- [ ] Push it to its own GitHub repository with a README, tests and a short demo.

### Week 17 — Interview drills
- [ ] [ML Coding Drills](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/01_ML_Coding_Drills.ipynb) — all 3 timed mock rounds
- [ ] [ML Theory & Statistics Questions](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/02_ML_Theory_and_Statistics_Questions.ipynb) — all 3 rapid-fire rounds
- [ ] [DSA Coding Patterns — Part 1](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/06_DSA_Coding_Patterns_Part_1.ipynb) and [Part 2](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/07_DSA_Coding_Patterns_Part_2.ipynb) — timed sets
- [ ] System design: one [ML](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/03_ML_System_Design.ipynb) and one [LLM](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/04_LLM_System_Design.ipynb) timed drill per day

### Week 18 — Behavioral and final mocks
- [ ] [Behavioral & Project Storytelling](../AI_Full_Stack_Engineer_Course/15_Interview_Prep/05_Behavioral_and_Project_Storytelling.ipynb) — write your story bank using your capstones and portfolio project
- [ ] Two full mock interview loops (coding + ML theory + system design + behavioral)
- [ ] Update your résumé bullets with the checker in the behavioral notebook

---

## 5. If you're short on time

- **ML engineer:** weeks 1–7, then MLOps (weeks 12–13), capstone 1, interview prep.
- **AI / LLM engineer:** weeks 1–4, week 7 (deep learning), NLP embeddings + Hugging Face, weeks 10–11, FastAPI + Docker, capstones 3–4, interview prep.
- **Skip first if needed:** every *(optional)* notebook, module 07 (reinforcement learning), Keras.
