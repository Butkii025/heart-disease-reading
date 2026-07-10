# Heart Disease Classification using Tree-Based ML Models

This repository contains a comprehensive step-by-step implementation and evaluation of tree-based machine learning models for classification, applied to a heart disease medical dataset. The project demonstrates the full lifecycle of training, visualizing, optimizing, and validating decision trees and ensemble methods using Scikit-learn.

## 📌 Project Overview
The goal of this project is to accurately predict the presence of heart disease (`target`) using a patient's clinical and medical profiles. 

### Dataset Features
* **Source:** `heart.csv` (1,025 observations, 14 medical metrics)
* **Target Variable:** `target` (0 = No Disease, 1 = Disease Presence)
* **Key Inputs:** Age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), max heart rate achieved (`thalach`), and more.

---
# Tree Image

<img width="4710" height="2433" alt="heart_tree" src="https://github.com/user-attachments/assets/616778d8-4b28-40c6-8b46-cb4b8d8480d6" />


---
## 🛠️ What We Did

### 1. Trained and Visualized a Baseline Decision Tree
* Built an unconstrained `DecisionTreeClassifier` using Scikit-learn.
* Bypassed system-level Graphviz requirements by plotting the tree nodes cleanly and directly with `matplotlib.pyplot` and `plot_tree`, saving the structure layout as `heart_tree.png`.

### 2. Controlled Tree Depth & Analyzed Overfitting
* Compared performance metrics between an unconstrained deep tree and a pre-pruned tree to observe the bias-variance tradeoff.
* **Unconstrained Tree:** Achieved perfect training metrics but overfit the training split.
* **Pruned Tree (`max_depth=3`):** Restricted tree splits to control depth, dropping noise and ensuring tighter generalization between train and test splits.

### 3. Scaled up to a Random Forest Ensemble
* Implemented a `RandomForestClassifier` consisting of 100 estimators (`n_estimators=100`, `max_depth=5`).
* Explored how averaging predictions across multiple random subsets of data and features mitigates model variance.

### 4. Interpreted Feature Importances
* Evaluated and extracted the architectural weights calculated by the tree splits.
* Sorted and identified the top contributing risk indicators for clinical decision-making.

### 5. Validated Performance via Cross-Validation
* Applied **5-Fold Cross-Validation** (`cross_val_score`) across the full dataset to get a stable, split-independent model evaluation.

---

## 📈 What We Gained (Key Takeaways & Results)

* **Overfitting Mitigation:** Learned how unconstrained decision trees aggressively grow until all leaf nodes are pure (**100.0% Train Accuracy** vs **98.5% Test Accuracy** on this data split). Applying `max_depth=3` pruned unnecessary branches and kept the model computationally lightweight.
* **The Power of Ensembles:** Witnessed the architectural advantage of Random Forests. Scaling up to a 100-tree ensemble stabilized predictions, yielding a highly dependable test accuracy of **87.32%**.
* **Model Explainability (Top Indicators):** Extracted clean feature metrics showing that the top 5 clinical drivers for predicting cardiac complications in this dataset are:
  1. **`cp` (Chest Pain Type)** — 16.45% importance
  2. **`thal` (Thalassemia type)** — 14.73% importance
  3. **`ca` (Number of major vessels colored by fluoroscopy)** — 14.62% importance
  4. **`oldpeak` (ST depression induced by exercise)** — 12.99% importance
  5. **`thalach` (Maximum heart rate achieved)** — 11.18% importance
* **Robust Evaluation Stability:** A single train/test split can sometimes be a statistical anomaly. Running a 5-Fold Cross-Validation confirmed a highly resilient mean accuracy of **92.98%** across the entire population dataset.

---
