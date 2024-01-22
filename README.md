# credit-risk-classification
Module Challenge 20
---
In this challenge, use various techniques to train and evaluate a model based on loan risk (`loan_status`). Use the provided dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

To do this...

1) Create labels set `y` from the `loan_status` where `0` is healthy and `1` is high-risk.
2) Separate the features from the labels into a DataFrame.
3) Split the data into `Training` and `Test` sets.
4) Create a `Logistic Regression` Model using the original data.
5) Evaluate the model's performance using `Confusion Matrix` and print out a `Classification Report`.
6) Create a new `Logistic Regression` Model using the `Random Over Sample` module to help balance out the minority class.
7) Evaluate and determine the best fitted model between the two models created.

## Overview of the Analysis
The purpose of this analysis is to identify future creditworthiness of borrowers using a provided dataset `lending_data.csv` and `Logistic Regression` model -- a supervised machine learning model. The dataset contains 77,536 observations with 7 features and 1 column with labels. The features are:

-   loan_size
-   interest_rate
-   borrower_income
-   debt_to_income
-   num_of_accounts
-   derogatory_marks
-   total_debt

The labels column is the `loan_status` which having two labels: (1) 0 or healthy loan, and (2) 1 or high-risk loan.

Using general observation by `value_counts()` function, we have more observations of heathly loans over high-risk loans (75,036 versus 2,500). In looking at this, we can qualitatively predict that future worthiness of a borrower. However, it's always best do this quantitatively using a model that can help predict and categorize the borrower into either one of the two labels above. This is where Logistic Regression model comes in to play.

By using the Logistic Regression model to predict by splitting the dataset into 75% for training and 25% for testing, we can accurately predict with high precision the credit worthiness of a future loan borrower.

As seen in the `credit_risk_classification.jpynb` file located under the `Credit_Risk` directory, we can see that in the first Logistic Regression model using the default dataset yielded the following report:

```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384
```

This indicated that the model was yielded a `100%` in precision and `99%` in recall for healthy loans versus `85%` in precision and `91%` in recall for high-risk loans. We suspected the reason for a lower percentage yield of high-risk is due to the ratio of observations for this label versus the label for healthy loans. However, the model yielded an overall of `99%` accuracy and when looking at the `Balanced Accuracy Score`, which is the calculated average of recall obtained on each class, we were able to obtain a `95.20%` which is considered to be very good. Overall, by using this model, we can confidentally say that this Logistic Regression model is a good model to use in predicting the future borrower.

Although we can further improve the precision and recall of the model for high-risk loans by simply providing the training to the model with more observations of high-risk loans added. The next best method is simply to use the `RandomOverSampler` module from `imblearn` library to let the program automatically over-sample the minority class(es) (in this case the high-risk loans) by picking samples at random with replacement. This will create a more balance ratio between the two labels which should help improve the precisions and recalls, in which it did.

As a result of utilizing the `RandomOverSampler` module, we were able to yield the following report output:

```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384
```

Even though there was a `1%` drop in precision, there was a signifiant increase of recall for the high-risk loans of `99%` from the previous `91%` in the original data while retaining a `99%` of accuracy. Additionally, the `Balanced Accuracy Score` also increased to about `99.37%` from the previous `99.20%` using the original data which further reinforces the usefulness of this model to help predict the future creditworthniess of borrowers.

## Results
As mentioned in the above analysis, the balanced accuracy, precisions, recalls, and accuracies of all the machine learning models (Logistic Regression) used in this challenge proved to be the right choice of model to use in predicting the creditworthiness of borrowers, especially with the model 2 where resampling was used.

## Summary
In summary, either models would be good to use in this scenario. The first model (using the original data) can be improved in term of the high-risk loans' precision and recall simply by collecting/feeding more observations of high-risk loans. However, it would require more data gathering and time comsuming whereas the second model took advantage of the `RandomOverSampler` to do the same thing but let the program decides instead.

For this machine learning, it would be important to be able to predict a high-risk accurately since it would not be ideal to approve loans to those who may not deemed qualify for a loan.

There may be other better models to use out there such as the `Support Vector Machines (SVM)`. However, it all really boils down to the needs and complexity of the problem. The SVM model may be an overkill to this dataset when the `Logistic Regression` model already provides a good accuracy.