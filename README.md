## Logistic Regression: A Machine Learning Approach for Predicting Bank Term Deposits Subscriptions

### Introduction

Machine learning (ML) is a branch of artificial intelligence that focuses on the development and study of statistical algorithms capable of learning from data. These algorithms can generalize to unseen data, enabling them to perform tasks without explicit instructions. One of the classical problems in machine learning is classification, where the goal is to categorize instances based on various parameters. Examples include spam detection and flower species classification. Among the numerous models used for classification, logistic regression stands out as one of the most prevalent and interpretable.

Logistic regression is a supervised machine learning algorithm designed for classification tasks. Its primary purpose is to predict the probability that a given instance belongs to a specific class. This model analyzes the relationship between one dependent binary variable and one or more independent variables by estimating probabilities using a logistic function.

### Project Description

This project aims to develop a robust logistic regression model capable of accurately predicting whether a client will subscribe ('yes') or not ('no') to a term deposit offered by a bank. Marketing campaigns in the banking sector can be resource-intensive, both in terms of time and financial investment. It is crucial to identify potential clients who are likely to be interested in these term deposits to optimize marketing efforts and reduce unnecessary expenditures. Therefore, an effective ML model that classifies target audiences into probable interested and uninterested parties is essential.

The dataset for this project consists of various features related to the characteristics of clients, their interactions with the bank, and the outcomes of previous marketing efforts. By analyzing these features, the model can discern patterns and relationships conducive to predicting subscription outcomes. Given the nature of the classification task, developing an effective logistic regression model is of paramount importance. Through meticulous analysis of the dataset and iterative model refinement, the goal is to achieve optimal predictive performance. Additionally, logistic regression's interpretability allows for the identification of key factors influencing subscription decisions, providing actionable insights for future marketing strategies.

### Dataset Overview

The dataset, sourced from 'bank-additional-full.csv', includes 41,188 instances and 20 input features, spanning the period from May 2008 to November 2010. This dataset mirrors data examined in prior research by Moro et al. (2014), facilitating comparative analysis and validation of model performance.

### Exploratory Data Analysis (EDA)

The dataset is in a semi-colon separated format with dimensions of 41,199 rows and 21 columns. During preprocessing, 76 missing values were identified and handled appropriately—missing categorical values were filled with the mode, and numerical values with the mean.

Outliers, which can significantly influence model behavior, were identified using box plots. Extreme values were removed to ensure the model's robustness. Next, categorical variables were transformed into numerical values using "LabelEncoding" from the "sklearn.preprocessing" module.

Multicollinearity, the occurrence of high intercorrelations among independent variables, was addressed using the "variable-inflation-factor" (VIF) from "statsmodels.stats.outliers_influence". Variables with VIF scores above 6 were iteratively removed to mitigate multicollinearity and enhance model stability.

### Data Preprocessing

After preprocessing, the dataset was divided into independent variables (df_ind) and the dependent variable (df_def). Using the "train_test_split" method from "sklearn.model_selection", the dataset was split into training and testing sets:

- Shape of x_train: (22,770, 12)
- Shape of y_train: (22,770,)
- Shape of x_test: (7,590, 12)
- Shape of y_test: (7,590,)

### Model Building

The logistic regression model was implemented using "LogisticRegression" from the "sklearn.linear_model" module. The model was trained on the training data (x_train and y_train). Predictions were generated using the test data (test_pred = logisticRegr.predict(x_test)).

### Model Performance Evaluation

The model's performance was evaluated using a classification report, which provides detailed metrics including precision, recall, F1-score, and support, alongside the confusion matrix.

#### Classification Report

|                         | Precision | Recall | F1-Score | Support |
|-------------------------|-----------|--------|----------|---------|
| No Subscription (0)     | 0.95      | 0.99   | 0.97     | 7,176   |
| Subscription (1)        | 0.47      | 0.16   | 0.24     | 414     |
| **Accuracy**            |           |        | 0.94     | 7,590   |
| **Macro Avg**           | 0.71      | 0.58   | 0.61     | 7,590   |
| **Weighted Avg**        | 0.93      | 0.94   | 0.93     | 7,590   |

#### Interpretation

- **Precision**: The model shows high precision (0.95) for predicting instances of no subscription but relatively lower precision (0.47) for predicting subscriptions.
- **Recall**: The recall is excellent (0.99) for no subscription instances but significantly lower (0.16) for subscription instances.
- **F1-Score**: The F1-score is strong (0.97) for no subscription instances but lower (0.24) for subscription instances.
- **Accuracy**: The model's overall accuracy is high (0.94), indicating its proficiency in predicting subscription outcomes.
- **Macro Avg**: The macro-average scores provide a balanced view across classes, with precision, recall, and F1-score at 0.71, 0.58, and 0.61, respectively.
- **Weighted Avg**: The weighted average metrics account for class imbalances, reaffirming the model's strong overall performance.

### Conclusion

This comprehensive evaluation highlights the logistic regression model's predictive capabilities, strengths, and areas for improvement. By identifying key factors influencing subscription decisions, the model provides valuable insights for enhancing future marketing strategies, thereby optimizing resource allocation and increasing the efficiency of direct marketing campaigns.
