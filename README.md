# fraud-binary-classification

The main objective of this use case is to develop a classification model using machine learning algorithms to help the compliance team identify fraudulent transactions.
It is expected to predict if a transaction is fraudulent or not using a classification model with a given degree of confidence. The result will help to support decision-making in the team.

## Data Understanding
The bank provided two datasets: the transaction dataset and the customer dataset. The transaction dataset contains 4,260,904 observations and 10 variables. The customer dataset contains 1,000 observations and 15 variables.

To prepare the data to develop the classification model, the following transformations of the variables were implemented:

- The categorical variables were transformed by one hot encoding and ordinal encoder, depending on their nature and the number of unique values for each variable.
In cases where the variables contained many unique values, a new feature was obtained that indicates the percentage of registered fraud cases for each category of the variable in question. This variable makes it possible to highlight cases with a higher frequency of fraud.

- feature engineering: new features were obtained based on the transaction date, such as a month, day of the week, and time.
A purchase type variable is also created: online, pos and others to highlight categories grouped by this characteristic.
On the other hand, the precision of the coordinates is decreased, and the distance between the customer and the point of sale is obtained.

- The age variable is divided into age groups, and an order encoder is assigned according to the order of each category created.

<img width="414" alt="image1" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/6cfde99c-0503-46ea-ae53-e772f2191535">


## Training the model

We trained several models based on decision trees, random forests, extra trees, and XGboost classifiers to predict if a transaction is a fraud. We assumed that decision trees-based algorithms will perform better based on this dataset's predominant categorical features.

The data was divided between training, validation and testing, using Stratify=y to preserve the distribution of the two classes (fraud and non-fraud) due to a strongly unbalanced sample. 

This use case  uses an F2 score. The reason behind the metric selection is based on the highly imbalanced dataset and the fact that we care more about positive class for the purpose of the prediction (Czakon, 2022) (Malato, 2021), (Olugbenga, 2022).

Due to the number of variables and the presence of overfitting in the above models, it was decided to implement xg boosts, seeking to correct the learning pattern. With this algorithm, a lower degree of overfitting is obtained, and therefore hyperparameter tuning is used to improve the model's score under training.
The parameter scale_pos_weight is configured for the XGBoost model, a value used to scale the gradient of the positive class (fraud events).
Finally, oversampling of the positive class and undersampling of the negative class are applied, using the SMOTE technique, seeking to improve the final score of the model.

## Evaluation

The model based on xgboost and using the SMOTE oversampling and undersampling technique was the best resulting model with an f2 score value of 88% in training and around 70% in tests. This model included hyperparameter tuning through grid search and selecting the variables to be used in the training process through iterative training for different groups of variables.

<img width="517" alt="image2" src="https://github.com/amedinabe/cancer-mortality-regression/assets/51183046/e1fb512a-1650-49ad-9eae-a16df873f523">

## References

●	Brownlee, J. (2020a, January 5). ROC Curves and Precision-Recall Curves for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/

●	Brownlee, J. (2020b, January 7). Tour of Evaluation Metrics for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

●	Brownlee, J. (2020c, January 12). How to Fix k-Fold Cross-Validation for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

●	Brownlee, J. (2020d, January 16). SMOTE for Imbalanced Classification with Python. MachineLearningMastery.Com. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

●	Brownlee, J. (2020e, February 4). How to Configure XGBoost for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/xgboost-for-imbalanced-classification/

●	Brownlee, J. (2020f, June 11). Ordinal and One-Hot Encodings for Categorical Data. MachineLearningMastery.Com. https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/

●	Brownlee, J. (2021, March 21). A Gentle Introduction to XGBoost Loss Functions. MachineLearningMastery.Com. https://machinelearningmastery.com/xgboost-loss-functions/

●	Brownlee, J. (2020a, January 5). ROC Curves and Precision-Recall Curves for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/

●	Brownlee, J. (2020b, January 7). Tour of Evaluation Metrics for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

●	Brownlee, J. (2020c, January 12). How to Fix k-Fold Cross-Validation for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

●	Brownlee, J. (2020d, January 16). SMOTE for Imbalanced Classification with Python. MachineLearningMastery.Com. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

●	Brownlee, J. (2020e, February 4). How to Configure XGBoost for Imbalanced Classification. MachineLearningMastery.Com. https://machinelearningmastery.com/xgboost-for-imbalanced-classification/

●	Brownlee, J. (2020f, June 11). Ordinal and One-Hot Encodings for Categorical Data. MachineLearningMastery.Com. https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/

●	Brownlee, J. (2021, March 21). A Gentle Introduction to XGBoost Loss Functions. MachineLearningMastery.Com. https://machinelearningmastery.com/xgboost-loss-functions/

●	Filho, M. (2023, March 24). Do Decision Trees Need Feature Scaling Or Normalization? https://forecastegy.com/posts/do-decision-trees-need-feature-scaling-or-normalization/

●	hmong.wiki. (2023). Grados decimales PrecisiónyEjemplo. https://hmong.es/wiki/Decimal_degrees

●	Hotz, N. (2018, September 10). What is CRISP DM? Data Science Process Alliance. https://www.datascience-pm.com/crisp-dm-2/

●	imbalanced-learn developers. (2023a). RandomUnderSampler—Version 0.10.1. https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html

●	imbalanced-learn developers. (2023b). SMOTE — Version 0.10.1. https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html

●	Mazumder, S. (2021, June 21). 5 techniques to handle imbalanced data for a classification problem. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/

●	LOCPHAM201. (2022). Fraud Detection using RandomForest+SMOTE+Tuning. https://kaggle.com/code/locpham2001/fraud-detection-using-randomforest-smote-tuning

●	Malato, G. (2021, June 7). Precision, recall, accuracy. How to choose? Your Data Teacher. https://www.yourdatateacher.com/2021/06/07/precision-recall-accuracy-how-to-choose/

●	Muralidhar, K. S. V. (2021, March 31). The right way of using SMOTE with Cross-validation. Medium. https://towardsdatascience.com/the-right-way-of-using-smote-with-cross-validation-92a8d09d00c7

●	Olugbenga, M. (2022, July 22). Balanced Accuracy: When Should You Use It? Neptune.Ai. https://neptune.ai/blog/balanced-accuracy

●	prashant111 (2020) A guide on XGBoost hyperparameters tuning, Kaggle. Available at: https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning (Accessed: 27 May 2023). 

●	Sklearn. Ensemble. Gradientboostingclassifier. (n.d.). Scikit-Learn. Retrieved April 28, 2023, from https://scikit-learn/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

●	Sklearn. Ensemble. Randomforestclassifier. (n.d.). Scikit-Learn. Retrieved April 28, 2023, from https://scikit-learn/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

●	Mazumder, S. (2021, June 21). 5 techniques to handle imbalanced data for a classification problem. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/

●	LOCPHAM201. (2022). Fraud Detection using RandomForest+SMOTE+Tuning. https://kaggle.com/code/locpham2001/fraud-detection-using-randomforest-smote-tuning

●	Malato, G. (2021, June 7). Precision, recall, accuracy. How to choose? Your Data Teacher. https://www.yourdatateacher.com/2021/06/07/precision-recall-accuracy-how-to-choose/

●	Muralidhar, K. S. V. (2021, March 31). The right way of using SMOTE with Cross-validation. Medium. https://towardsdatascience.com/the-right-way-of-using-smote-with-cross-validation-92a8d09d00c7

●	Olugbenga, M. (2022, July 22). Balanced Accuracy: When Should You Use It? Neptune.Ai. https://neptune.ai/blog/balanced-accuracy

●	prashant111 (2020) A guide on XGBoost hyperparameters tuning, Kaggle. Available at: https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning (Accessed: 27 May 2023). 

●	Sklearn. Ensemble. Gradientboostingclassifier. (n.d.). Scikit-Learn. Retrieved April 28, 2023, from https://scikit-learn/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

●	Sklearn. Ensemble. Randomforestclassifier. (n.d.). Scikit-Learn. Retrieved April 28, 2023, from https://scikit-learn/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


