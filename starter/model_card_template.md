# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

It is a RandomForestClassifier from the sklearn.ensable. It uses the default configuration - no hyperparameter tuning:

- n_estimators=100

## Intended Use

This model can be used to predict the salary (categorical data) of a person.

## Training Data

The data is from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The original set has 32561 rows and 15 columns.Train/Test split has training data set for 80% of the set.

## Evaluation Data

20 % of data is used to test.

## Metrics

The overall model performance was evaluated using Fbeta score, and Precision-Recall Score

Script used to compute performance : model_performance.py

- Precision Score : 0.9524765729585006
- Recall Score : 0.9270358306188925
- FBeta Score : 0.9395840211290856

## Ethical Considerations

Considering the data is from census, it reflects on real people from the United States. Ethical implications could therefore apply - handle with care.

Privacy: The use of this machine learning model can raise privacy concerns as it involves the collection of census data which in parts comprises of personal data. This data can be misused for malicious purposes. 

Accountability: It can be difficult to determine who is responsible for the decisions made by a machine learning model and how these decisions can be challenged or corrected. This lack of accountability can make it difficult to ensure that the model is being used ethically - handle with care.

## Caveats and Recommendations

This model and data were used for a project course from Udacity called "Machine Learning DevOps Engineer".

Fairness: Since the data was taken from census at a certain point in time, there may be bias present now, since that data might now be entirely different, leading to bias in the model's predictions.

It is recommended to use diverse up to date data to train the model and to mitigate possible biases.
