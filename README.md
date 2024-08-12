# Probability-of-Default

This project aims to predict the probability of loan defaults using historical data to assist financial institutions in managing risk.


# Introduction
The objective of this project is to develop a predictive model to estimate the probability of default (PD) on loans. Accurate PD predictions are essential for effective risk management and strategic planning in financial institutions. This project utilizes historical loan data provided by Claxon Excellence Academy for Professionals to build and evaluate various machine learning models.

# Data
The dataset is provided by Claxon Excellence Academy for Professionals as part of their competition. It includes historical borrower information with various features that impact the probability of loan default. Key details about the data include:

Source: Claxon Excellence Academy for Professionals competition dataset
Description: Contains historical loan and borrower information with multiple features influencing default probability.

- Features:

loan_id
gender
disbursemet_date
currency
country
sex
is_employed
job
location
loan_amount
number_of_defaults
outstanding_balance
interest_rate
age
remaining term
salary
marital_status
Loan Status

# Methods

- Data Cleaning
  
The first step was to clean and preprocess the dataset. This involved handling missing values, encoding categorical features, and engineering new features. Outlier detection and treatment techniques were also applied to address extreme values.

- Exploratory Data Analysis (EDA)
  
Extensive EDA was conducted to gain insights into the dataset. Visualization techniques were used to explore feature distributions and correlations. Feature importance analysis, including statistical tests, helped identify the key drivers of loan default.

- Feature Selection
  
Based on the EDA insights, a feature selection process was implemented. This included correlation analysis, Recursive Feature Elimination, and the application of domain knowledge to identify the most relevant features for model training.

- Hyperparameter Tuning
  
To optimize the machine learning models, hyperparameter tuning was conducted using techniques such as grid search and random search, coupled with cross-validation to ensure the selected hyperparameters generalized well.

- Cross-Validation
  
A robust 5-fold cross-validation strategy was employed to assess the models' performance and generalization capabilities. This provided a reliable estimate of the models' expected performance on new, unseen data.

- Feature Scaling and Transformation
  
Appropriate scaling and transformation techniques were applied to the features to ensure they were on a common scale and to address any non-linearity or skewness in the data.

- Model Building
  
At least 5 machine learning models were trained, with the choice of algorithm and any model assumptions and limitations discussed.

- Model Evaluation

The trained models were evaluated on a separate validation set, and the performance metrics were interpreted to assess their effectiveness in predicting loan defaults.

- Endpoint Development for Inference

API endpoints were created using FastAPI to enable model training and inference. Clear documentation was provided on how to use these endpoints.

# Results

After training and evaluating the models, the Random Forest classifier achieved the best performance on the validation set:

AUC: 85.93%, indicating strong discriminatory ability.
Validation Accuracy: 0.90, the highest among the models.
Balance: Maintains a good balance between precision and recall for both classes, particularly for the negative class, making it a reliable choice.
The Random Forest model stands out as the best performer, demonstrating the effectiveness of the developed model in accurately predicting the probability of loan defaults. This information can be valuable for financial institutions to make more informed lending decisions and implement appropriate risk management strategies.


