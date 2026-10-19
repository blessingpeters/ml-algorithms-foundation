# ML — Machine Learning Coursework & Practice

This repository contains my hands-on machine learning assignments. Each folder
represents a session/topic covered in class, containing only my own assignment
work based on what was taught.

## 📁 Repository Structure

```
ML/
├── 2.Classification Metrics & ML Workflow/
│   └── Assignment_2_Classification_Metrics.ipynb
├── 3.KNN/
│   └── Assignment_3_KNN_Diabetes_Experiment.ipynb
├── 4.SVM/
│   └── Assignment_4_Model_Comparison.ipynb
├── 5.Decision Tree Investigation/
│   └── Assignment_5_Decision_Tree_Investigation.ipynb
├── 6.Decision Trees for Regression/
│   └── Assignment_6_Tree_Based_Prediction.ipynb
├── 7.Random Forest/
│   └── Assignment_7_Decision_Tree_vs_Random_Forest.ipynb
├── 8.Gradient Boosting(XGBoost)/
│   └── Assignment_8_Model_Comparison_LogReg_RF_XGBoost.ipynb
├── 9.K-Means Clustering/
│   └── Assignment_9_Customer_Segmentation.ipynb
├── 10.Hierarchical Clustering and Principal Component Analysis (PCA)/
│   └── Assignment_10_PCA_Breast_Cancer_Dataset.ipynb
├── 11.Feature Engineering/
│   └── Assignment_11_Feature_Engineering_Titanic.ipynb
├── 12.Feature Selection and pipelines/
│   └── Assignment_12_Reusable_ML_Pipeline_Churn.ipynb
└── README.md
```

## 📚 Topics Covered

### 2. Classification Metrics & ML Workflow
- Covered the full supervised machine learning workflow: data loading, train/test
  split, model training, prediction, and evaluation.
- Explored key classification metrics: **Accuracy, Precision, Recall, F1-Score,
  and Confusion Matrix**.
- Discussed when to prioritize Precision vs Recall depending on the real-world
  cost of false positives vs false negatives.
- Practiced interpreting a confusion matrix and classification report.

### 3. K-Nearest Neighbors (KNN)
- Trained a KNN classifier on the **Pima Indians Diabetes dataset**.
- Compared performance across different values of **K** (1, 3, 5, 7, 9).
- Investigated overfitting (K=1) vs underfitting (large K).
- Identified the best K based on Testing Accuracy and F1-Score.
- Demonstrated the importance of **feature scaling** for distance-based models.

### 4. Support Vector Machines (SVM)
- Compared **Logistic Regression, KNN, and SVM (RBF kernel)** on the same dataset.
- Used the `make_circles` dataset to visualize non-linear decision boundaries.
- Generated a full metrics comparison table: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Explained the Kernel Trick, margin maximization, and why SVM generalizes well on non-linear data.

### 5. Decision Trees (Classification)
- Built Decision Tree classifiers on the **Iris dataset**.
- Trained multiple trees using different values of `max_depth`.
- Compared Training vs Testing accuracy to identify the point where overfitting begins.
- Visualized the best-performing tree and explained why unrestricted trees overfit.

### 6. Decision Trees for Regression
- Built a Decision Tree Regressor to predict a continuous target variable.
- Compared model performance across different `max_depth` values.
- Evaluated regression performance using metrics such as MAE, MSE, and R².
- Investigated how tree depth affects underfitting/overfitting in a regression context.

### 7. Random Forest
- Compared a single **Decision Tree** against a **Random Forest** on the same dataset.
- Explored how ensembling multiple trees (bagging) reduces overfitting and variance.
- Compared model performance and stability between the two approaches.
- Discussed why Random Forest generally generalizes better than a single Decision Tree.

### 8. Gradient Boosting (XGBoost)
- Compared **Logistic Regression, Random Forest, and XGBoost** on the same dataset.
- Explored how gradient boosting builds trees sequentially to correct previous errors.
- Generated a full classification metrics comparison table across all three models.
- Discussed the trade-offs between boosting and bagging-based ensemble methods.

### 9. K-Means Clustering
- Segmented customers from the **UCI Wholesale Customers dataset** using K-Means.
- Selected and scaled features before clustering.
- Used the **elbow method** (WCSS) to choose the number of clusters (K=4).
- Profiled and named each cluster based on its spending pattern.

### 10. Hierarchical Clustering & PCA
- Applied **PCA** to the **Breast Cancer Wisconsin dataset** (30 numerical features).
- Determined that **7 components** retain at least 90% of the variance, using a cumulative explained variance curve.
- Built a **2D projection** (PC1 vs PC2, 63% of variance) that still separates malignant from benign tumours.
- Explained what information is retained and what is lost when reducing dimensions.

### 11. Feature Engineering
- Engineered features on the raw **Titanic dataset**, learning every fill value from the training split only.
- Handled missing values with group-median imputation, a missing-value flag, mode filling, and an "Unknown" category.
- Encoded categorical variables (binary and one-hot encoding).
- Created new features: `family_size`, `fare_per_person`, `is_child`, and `female_in_upper_class`.
- Removed duplicate, redundant, and **target-leaking** features (`alive`).
- Raised test accuracy from ~68–70% to ~78–79% and explained how each transformation improves performance.

### 12. Feature Selection & Pipelines
- Built a reusable scikit-learn **Pipeline** on the raw **Telco Customer Churn dataset**.
- Automated data-type fixing, missing-value imputation (`SimpleImputer`), scaling (`StandardScaler`),
  and one-hot encoding (`OneHotEncoder`) with a `ColumnTransformer`, followed by model training and prediction.
- Predicted on raw test data with no manual preprocessing (80.6% accuracy, 0.842 ROC-AUC).
- Demonstrated reusability: swapping models, predicting for new customers with missing or unseen values,
  and saving/reloading the fitted pipeline with `joblib`.

## 🛠️ Tools & Libraries
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`sklearn`)
- xgboost
- joblib

## 🎯 Purpose
This repository is a personal learning log for practicing supervised and unsupervised
machine learning concepts — classification metrics, model evaluation, hyperparameter
tuning, ensemble methods, model comparison, clustering, dimensionality reduction,
feature engineering, and ML pipelines — through hands-on Jupyter notebook
assignments.