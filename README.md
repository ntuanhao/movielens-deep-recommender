# 🎬 Movie Recommendation System (MovieLens 25M)
**Graduation Thesis 2026 - Ho Chi Minh City University of Natural Resources and Environment (HCMUNRE)**

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-yellow.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightgrey.svg)
![Git LFS](https://img.shields.io/badge/Git_LFS-Enabled-green.svg)

## 📌 Project Overview
This repository contains the complete implementation of a **Deep Learning-based Movie Recommendation System**, trained on the large-scale **MovieLens 25M** dataset. The project explores various neural network architectures, from traditional Matrix Factorization approaches to advanced Deep & Cross Networks (DCN) and hybrid Wide & Deep models, incorporating rich contextual features such as genres, release years, genome tags, and user/movie text tags.

This project was developed as a Graduation Thesis at HCMUNRE.

## 📁 Repository Structure
```text
.
├── models/                   # Contains trained model weights (.keras) via Git LFS
├── .gitattributes            # Git LFS tracking configuration for large model files
├── ratings-phim.ipynb        # Main Jupyter Notebook for data processing, training, and evaluation
├── requirements.txt          # Python dependencies
└── run_demo.py               # Inference script for real-time recommendations
```

## 📊 Dataset & Features
- **Dataset:** MovieLens 25M
- **Users:** 162,541
- **Movies:** 59,047
- **Genres:** 20 (MultiLabel Binarized)
- **Genome Tags:** 1,128 dimensions (Sparse Matrix)
- **Text Tags:** Top 500 features extracted via TF-IDF Vectorizer

**Feature Engineering:**
- **Categorical:** User IDs & Movie IDs (Label Encoded & Embedded).
- **Numerical:** Release Year (MinMax Scaled).
- **Multi-hot:** Genres.
- **Text/NLP:** TF-IDF representation of user and movie tags to capture nuanced item semantics and user preferences.

## 🧠 Model Architectures
The project implements and evaluates 7 different architectures:
- **Baseline Model:** Simple Embeddings concatenated with Dense layers.
- **Model V1:** Deeper architecture with 128-dim embeddings.
- **Wide & Deep (V2):** Combines Dot product (Wide) and Multi-Layer Perceptron (Deep) paths.
- **Wide & Deep Enhanced (V2_Enhanced):** Adds Layer Normalization and L2 Regularization (1e-5) to mitigate overfitting.
- **Deep & Cross Network (DCN):** Utilizes 3 Custom Cross Layers to explicitly model bounded-degree feature interactions.
- **Neural Collaborative Filtering (NCF++):** Fuses Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) with LayerNorm.
- **Genome Model (V3):** Incorporates 1,128-dimensional dense genome features.
- **Tag-Aware Model (V4):** Integrates TF-IDF user and movie text tags with DCN architecture.

## 🏆 Evaluation & Results
All models were evaluated on a strictly isolated test set using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

| Model | RMSE | MAE |
| :--- | :--- | :--- |
| V2_Enhanced | 0.77598 | 0.58560 |
| NCF++ | 0.77611 | 0.58687 |
| DCN | 0.77625 | 0.58795 |
| V3_Genome | 0.76260 | 0.57310 |
| V4_Tags | 0.76412 | 0.57447 |
| **Ensemble (Average)** | **0.75140** | **0.56734** |

The Ensemble Average approach yielded the best predictive performance, significantly outperforming individual models.

## 🚀 How to Run
### 1. Installation
Clone the repository and install the required dependencies:
```bash

git lfs install
git lfs pull
pip install -r requirements.txt

```

### 2. Training (Optional)
To retrain the models from scratch, execute the Jupyter Notebook:
```bash
jupyter notebook ratings-phim.ipynb
```
*Note: Training on the full 25M dataset requires a GPU (e.g., Tesla T4).*

### 3. Inference / Demo
The notebook packages necessary encoders and metadata into `demo_artifacts.joblib`. You can use `run_demo.py` to get top 10 movie recommendations for a specific user:
```bash
python run_demo.py
```
*(Example output for User 123 provides scores predicting highly rated documentaries and classic animations like 'The Blue Planet' and 'Wallace & Gromit'.)*

## 👨‍💻 Author
**Nguyen Tuan Hao**
- **Role:** Software Engineer / AI Researcher
- **University:** Ho Chi Minh City University of Natural Resources and Environment (HCMUNRE)
- **Email:** tuanhao050403@gmail.com
- **LinkedIn:** www.linkedin.com/in/tuấn-hào-a34b9b218

If you find this repository helpful, please consider leaving a ⭐!
