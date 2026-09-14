# Course Map (v2 — interview-ready)

Beginner-first order. Every notebook follows [NOTEBOOK_TEMPLATE.md](NOTEBOOK_TEMPLATE.md), was executed end to end with real outputs, and passes `tools/nb.py check`, `run --solutions`, and link checks.

**Kinds:** topic (standard template) · capstone (walkthrough + real project folder) · template (copy-and-fill starter) · prep (interview prep).
**Environments:** `core` = `requirements.txt`; others have their own `requirements-<env>.txt` and a separate virtual environment.
*(optional)* = useful but rarely needed for interviews.

| Module | Notebook | Kind | Env |
|---|---|---|---|
| **00 Foundations** | 01_Python_Basics | topic | core |
| | 02_Python_Builtins | topic | core |
| | 03_OOP_in_Python | topic | core |
| | 04_Virtual_Env_and_Packaging | topic | core |
| | 05_Python_Internals_and_Concurrency | topic | core |
| **01 Core Scientific Computing** | 01_NumPy | topic | core |
| | 02_Pandas | topic | core |
| | 03_SQL_with_DuckDB | topic | core |
| | 04_Math_for_ML | topic | core |
| | 05_Statistics_and_Probability | topic | core |
| **02 Data Visualization** | 01_Matplotlib_and_Seaborn | topic | core |
| | 02_Plotly *(optional)* | topic | core |
| **03 Classical ML** | 01_ML_Fundamentals_From_Scratch | topic | core |
| | 02_Scikit_Learn | topic | core |
| | 10_Gradient_Boosting | topic | core |
| **04 Deep Learning** | 01_Neural_Networks_From_Scratch | topic | core |
| | 02_PyTorch | topic | core |
| | 03_Transformers_From_Scratch | topic | core |
| | 04_Keras_3_Overview *(optional)* | topic | core |
| **05 NLP** | 01_Classical_NLP | topic | core |
| | 02_Embeddings_and_Semantic_Search | topic | core |
| | 03_HuggingFace_Transformers | topic | core |
| **06 Computer Vision** | 01_OpenCV | topic | core |
| | 02_CNNs_and_Transfer_Learning | topic | core |
| | 03_YOLO_Object_Detection | topic | core |
| **07 Reinforcement Learning** | 01_Gymnasium_and_Q_Learning | topic | core |
| | 02_Deep_RL_with_Stable_Baselines3 | topic | core |
| **08 Generative AI & LLMs** | 01_LLM_APIs_and_Prompting | topic | core + local LLM |
| | 02_RAG_From_Scratch | topic | core + local LLM |
| | 03_LangChain_and_LangGraph | topic | core + local LLM |
| | 04_LlamaIndex | topic | llamaindex + local LLM |
| | 05_Agents_Tool_Calling_and_MCP | topic | core + local LLM |
| | 06_LLM_Evaluation | topic | core + local LLM |
| | 07_Fine_Tuning_LoRA_and_DPO | topic | core |
| | 08_LLM_Inference_and_Serving | topic | core + llama.cpp |
| | 09_Distributed_Training_Overview *(optional)* | topic | core |
| **09 MLOps** | 01_MLflow | topic | core |
| | 02_Weights_and_Biases *(optional)* | topic | core |
| | 03_DVC | topic | core |
| | 04_Airflow | topic | airflow |
| | 05_Model_Monitoring_and_Drift | topic | monitoring |
| | 06_Kubeflow_Pipelines *(optional)* | topic | kfp |
| **10 Model Serving** | 01_FastAPI | topic | core |
| | 02_Docker_and_CI_CD | topic | core + Docker |
| | 03_BentoML | topic | core |
| | 04_Ray_Serve *(optional)* | topic | core |
| **11 Data Processing** | 01_Polars | topic | core |
| | 02_PySpark | topic | core + Java 17 |
| | 03_Dask *(optional)* | topic | core |
| **12 AutoML & Experimentation** | 01_Optuna | topic | core |
| | 02_Ray_Tune | topic | core |
| | 03_AutoGluon *(optional)* | topic | autogluon |
| **13 Capstone Projects** | 01_End_to_End_ML_Project + `churn_service/` | capstone | core |
| | 02_End_to_End_DL_Project + `image_classifier/` | capstone | core |
| | 03_LLM_RAG_Assistant_Project + `rag_assistant/` | capstone | core + local LLM |
| | 04_AI_Agent_Project + `ai_agent/` | capstone | core + local LLM |
| **14 Templates** | 01_ML_Template | template | core |
| | 02_DL_Template | template | core |
| | 03_LLM_Template | template | core + local LLM |
| **15 Interview Prep** | 01_ML_Coding_Drills | prep | core |
| | 02_ML_Theory_and_Statistics_Questions | prep | core |
| | 03_ML_System_Design | prep | core |
| | 04_LLM_System_Design | prep | core + local LLM |
| | 05_Behavioral_and_Project_Storytelling | prep | core |

**Removed from v1** (low interview value or outdated): Bokeh, Altair, TensorFlow, standalone Keras 2, JAX, Detectron2, RLlib, H2O, standalone Flask (now a section of FastAPI), DeepSpeed (condensed into Distributed Training Overview). The v1 notebooks remain available on the `main` branch.

**Honest skips:** a few cells print a `⏭️ Skipped` message instead of running, because they need hardware or services this laptop didn't have: CUDA GPUs (QLoRA, vLLM, DeepSpeed), hosted-LLM API keys (OpenAI/Anthropic/Gemini/LangSmith/W&B sweeps), a running Docker daemon (capstone container builds), or a Kubernetes cluster. Each message says exactly what to set up to run it.
