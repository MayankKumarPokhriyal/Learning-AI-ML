# Course Map (v2 — interview-ready)

Beginner-first order. ⭐ = new notebook, 🔀 = merged from older notebooks, (optional) = skim or skip if short on time.

| Module | Notebook | Status |
|---|---|---|
| **00 Foundations** | 01_Python_Basics | rewrite |
| | 02_Python_Builtins | rewrite |
| | 03_OOP_in_Python | rewrite |
| | 04_Virtual_Env_and_Packaging | rewrite (uv, pyproject) |
| | 05_Python_Internals_and_Concurrency | ⭐ mutability, closures, GIL, threads/processes/asyncio |
| **01 Core Scientific Computing** | 01_NumPy | rewrite (pilot) |
| | 02_Pandas | rewrite |
| | 03_SQL_with_DuckDB | ⭐ joins, window functions, SQL ↔ pandas |
| | 04_Math_for_ML | ⭐ vectors, matrices, derivatives, gradients, intuition |
| | 05_Statistics_and_Probability | rewrite of 03_SciPy (CLT, tests, power, bootstrap, A/B) |
| **02 Data Visualization** | 01_Matplotlib_and_Seaborn | 🔀 Matplotlib + Seaborn |
| | 02_Plotly | rewrite (optional) |
| **03 Classical ML** | 01_ML_Fundamentals_From_Scratch | ⭐ regression, logistic, bias-variance, regularization, metrics |
| | 02_Scikit_Learn | rewrite |
| | 03_Gradient_Boosting | 🔀 XGBoost + LightGBM + CatBoost, boosting from scratch |
| **04 Deep Learning** | 01_Neural_Networks_From_Scratch | ⭐ autograd + MLP in NumPy |
| | 02_PyTorch | rewrite |
| | 03_Transformers_From_Scratch | ⭐ attention, positional encoding, mini-GPT, KV cache |
| | 04_Keras_3_Overview | 🔀 replaces TensorFlow + Keras (optional) |
| **05 NLP** | 01_Classical_NLP | 🔀 NLTK + spaCy + Gensim |
| | 02_Embeddings_and_Semantic_Search | rewrite of SentenceTransformers |
| | 03_HuggingFace_Transformers | rewrite (fine-tuning with current Trainer API) |
| **06 Computer Vision** | 01_OpenCV | rewrite |
| | 02_CNNs_and_Transfer_Learning | rewrite of Torchvision |
| | 03_YOLO_Object_Detection | rewrite (real training, IoU/NMS/mAP from scratch) |
| **07 Reinforcement Learning** | 01_Gymnasium_and_Q_Learning | rewrite |
| | 02_Deep_RL_with_Stable_Baselines3 | rewrite (+ REINFORCE from scratch) |
| **08 Generative AI & LLMs** | 01_LLM_APIs_and_Prompting | rewrite of OpenAI SDK (multi-provider, structured output, streaming, cost) |
| | 02_RAG_From_Scratch | ⭐ chunking, hybrid search, reranking, citations |
| | 03_LangChain_and_LangGraph | rewrite |
| | 04_LlamaIndex | rewrite |
| | 05_Agents_Tool_Calling_and_MCP | ⭐ |
| | 06_LLM_Evaluation | ⭐ golden sets, retrieval metrics, LLM-as-judge |
| | 07_Fine_Tuning_LoRA_and_DPO | ⭐ |
| | 08_LLM_Inference_and_Serving | rewrite of vLLM (KV cache math, batching, quantization) |
| | 09_Distributed_Training_Overview | 🔀 DeepSpeed/FSDP condensed (optional) |
| **09 MLOps** | 01_MLflow | rewrite (aliases, tracing) |
| | 02_Weights_and_Biases | rewrite (optional) |
| | 03_DVC | rewrite (real pipeline) |
| | 04_Airflow | rewrite (Airflow 3, TaskFlow) |
| | 05_Model_Monitoring_and_Drift | ⭐ |
| | 06_Kubeflow_Pipelines | rewrite (KFP v2, local runner) (optional) |
| **10 Model Serving** | 01_FastAPI | 🔀 FastAPI (+ Flask comparison) |
| | 02_Docker_and_CI_CD | ⭐ |
| | 03_BentoML | rewrite (1.2+ API) |
| | 04_Ray_Serve | rewrite (optional) |
| **11 Data Processing** | 01_Polars | rewrite |
| | 02_PySpark | rewrite (Spark 4) |
| | 03_Dask | rewrite (optional) |
| **12 AutoML & Experimentation** | 01_Optuna | rewrite |
| | 02_Ray_Tune | rewrite (Tuner API) |
| | 03_AutoGluon | rewrite (optional) |
| **13 Capstone Projects** | 01_End_to_End_ML_Project | rewrite + real service code |
| | 02_End_to_End_DL_Project | rewrite + real code |
| | 03_LLM_RAG_Assistant_Project | rewrite + real service code |
| | 04_AI_Agent_Project | ⭐ + real code |
| **14 Templates** | 01_ML_Template · 02_DL_Template · 03_LLM_Template | rewrite |
| **15 Interview Prep** | 01_ML_Coding_Drills | ⭐ |
| | 02_ML_Theory_and_Statistics_Questions | ⭐ |
| | 03_ML_System_Design | ⭐ |
| | 04_LLM_System_Design | ⭐ |
| | 05_Behavioral_and_Project_Storytelling | ⭐ |

**Removed** (low interview value or outdated): Bokeh, Altair, TensorFlow and JAX notebooks (Keras 3 overview covers the idea), Detectron2, RLlib, H2O, standalone Flask (now a section of FastAPI).
