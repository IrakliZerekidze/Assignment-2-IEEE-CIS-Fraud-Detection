# IEEE-CIS Fraud Detection

## კონკურსის და პროექტის მიმოხილვა

კონკურსის მიზანია ონლაინ ტრანზაქციებში fraud ოპერაციების აღმოჩენა სხვადასხვა მომხმარებლის, ტრანზაქციისა და მოწყობილობის მახასიათებლების საფუძველზე.
ამ პროექტში განხორციელდა მონაცემთა გაწმენდა, Feature Engineering, Feature Selection და რამდენიმე Machine Learning მოდელის ტრენინგი.
საბოლოოდ შეირჩა საუკეთესო მოდელი, რომელმაც ყველაზე მაღალი ROC-AUC შედეგი აჩვენა.

## რეპოზიტორიის სტრუქტურა

```
├── model-experiment.ipynb     ← EDA, preprocessing, ექსპერიმენტები
├── model-inference.ipynb      ← საუკეთესო მოდელის ჩატვირთვა, inference, submission
└── README.md
```

| ფაილი                    | აღწერა                                                                            |
| ------------------------ | --------------------------------------------------------------------------------- |
| `model-experiment.ipynb` | მთავარი notebook EDA-სთვის, preprocessing-ისთვის და მოდელების ექსპერიმენტებისთვის |
| `model-inference.ipynb`  | MLflow-იდან საუკეთესო pipeline-ის ჩატვირთვა და Kaggle submission-ის გენერაცია     |

---

## Pipeline Overview

```
Drop high-missing columns
    ↓
Drop high-dominance columns
    ↓
Impute missing values
    ↓
Feature engineering
    ↓
Encoding
    ↓
Feature selection
    ↓
Train model
```

---

# მონაცემთა გაწმენდა და დამუშავება (Cleaning & Preprocessing)

## 1. High Missing Columns Removal

წავშალე ის სვეტები, რომლებშიც ძალიან მაღალი რაოდენობის missing მნიშვნელობები იყო.

* ძირითადად გამოვიყენე `0.98` threshold
* სვეტები, რომლებშიც 98%-ზე მეტი მნიშვნელობა იყო დაკარგული, მოდელისთვის პრაქტიკულად არაინფორმატიული აღმოჩნდა

Feature Cleaning განსაკუთრებით ეფექტური აღმოჩნდა XGBoost-ზე.

---

## 2. High Dominance Columns Removal

წავშალე ისეთი სვეტები, სადაც ერთი მნიშვნელობა თითქმის მთლიან სვეტს იკავებდა.

გამოვიყენე რამდენიმე threshold:

* 0.99
* 0.995
* 0.98

თუმცა შედეგებმა აჩვენა, რომ aggressive dominance filtering უმეტეს მოდელებზე უარყოფითად მოქმედებდა, განსაკუთრებით XGBoost-ზე.

საბოლოოდ საუკეთესო შედეგი მივიღე მხოლოდ High Missing Cleaning-ის გამოყენებით.

---

## 3. Missing Value Imputation

### რიცხვითი მნიშვნელობები

რიცხვითი სვეტები შეივსო median მნიშვნელობებით.

### კატეგორიული მნიშვნელობები

კატეგორიული სვეტები შეივსო ყველაზე ხშირი მნიშვნელობით (`most_frequent`).

ეს მიდგომა სტაბილური აღმოჩნდა ყველა მოდელისთვის.

---

## 4. Encoding

გამოვიყენე:

* One-Hot Encoding
* WOE Encoding

WOE Encoding გამოვცადე როგორც Logistic Regression-ზე, ასევე XGBoost-ზე.

შედეგებმა აჩვენა, რომ:

* Logistic Regression-ზე WOE-ს მნიშვნელოვანი გაუმჯობესება არ მოუტანია
* XGBoost-ზე კი შედეგი პრაქტიკულად უცვლელი დარჩა

ეს მიუთითებს, რომ tree-based boosting მოდელები თავად ახერხებენ კატეგორიული ინფორმაციის ეფექტურად გამოყენებას.

---

# Feature Engineering

პროექტში გამოვცადე სხვადასხვა ტიპის Feature Engineering.

საბოლოოდ საუკეთესო შედეგი behavioral feature-ებმა აჩვენა.

---

## 1. Log Transformation

### Log Transaction Amount

```python
log(TransactionAmt + 1)
```

ლოგარითმული ტრანსფორმაცია გამოვიყენე Transaction Amount-ის skewness-ის შესამცირებლად.

Log-transform-მა განსაკუთრებით გააუმჯობესა Logistic Regression-ის შედეგები.

---

## 2. Behavioral Features

### UID Amount Deviation

```python
TransactionAmt - mean(TransactionAmt per UID)
```

ეს feature აღწერს რამდენად განსხვავდება კონკრეტული ტრანზაქცია მომხმარებლის ჩვეულებრივი ქცევისგან.

სწორედ ამ feature-მა აჩვენა ყველაზე მნიშვნელოვანი გაუმჯობესება XGBoost-ზე.

საბოლოოდ ეს გახდა საუკეთესო Feature Engineering იდეა მთელ პროექტში.

---

## 3. Count Features

გამოვცადე:

* UID transaction counts
* Product interaction counts
* სხვადასხვა aggregation features

თუმცა უმეტეს შემთხვევაში ამ feature-ებმა მნიშვნელოვანი გაუმჯობესება ვერ მოიტანა.

---

## 4. Interaction Features

გამოვიყენე სხვადასხვა interaction feature-ები:

* ProductCD + TransactionAmt
* categorical combinations
* grouped behavioral features

თუმცა XGBoost უკვე თავად სწავლობდა მსგავს ურთიერთობებს და დამატებითი interaction feature-ები ხშირად noise-ს ამატებდა.

---

# Feature Selection

პროექტში გამოვიყენე რამდენიმე Feature Selection მეთოდი.

---

## 1. Correlation Filter

Correlation Filter გამოვიყენე Logistic Regression-ზე.

Tree-based მოდელებზე correlation filtering არაეფექტური აღმოჩნდა.

გარდა ამისა, correlation filtering საკმაოდ slow იყო დიდი რაოდენობის feature-ების გამო.

---

## 2. IVValidator

Information Value Filtering გამოვცადე:

* Logistic Regression-ზე
* Decision Tree-ზე
* XGBoost-ზე

IV Filtering-მა ზოგიერთ შემთხვევაში შეამცირა noise, თუმცა საუკეთესო ROC-AUC ვერ აჩვენა.

---

## 3. RFE (Recursive Feature Elimination)

RFE აღმოჩნდა ყველაზე წარმატებული Feature Selection მეთოდი.

გამოვიყენე XGBoost estimator-ით.

სწორედ RFE-მ უზრუნველყო საბოლოო საუკეთესო შედეგი.

---

# მოდელები

## Logistic Regression

Logistic Regression გამოიყენებოდა როგორც baseline მოდელი.

გამოვიყენე:

* L1 Regularization
* L2 Regularization
* Class Balancing
* Log Transformations
* Feature Selection

Logistic Regression-ზე Log Transaction Amount feature-მა მნიშვნელოვანი გაუმჯობესება მოიტანა.

| Experiment        | Val ROC-AUC | Precision | Recall | F1    |
| ----------------- | ----------- | --------- | ------ | ----- |
| Baseline          | ~0.818      | 0.126     | 0.665  | 0.212 |
| L2 Regularization | ~0.881      | 0.152     | 0.753  | 0.253 |
| Log Transform     | ~0.879      | 0.149     | 0.751  | 0.249 |

---

## Decision Tree

Decision Tree-მა საკმაოდ სწრაფად დაიწყო overfitting.

გამოვიყენე:

* max_depth tuning
* feature engineering
* IV filtering

თუმცა Decision Tree მნიშვნელოვნად ჩამორჩა სხვა მოდელებს.

| Experiment | Val ROC-AUC | Precision | Recall | F1   |
| ---------- | ----------- | --------- | ------ | ---- |
| Baseline   | ~0.80       | 0.18      | 0.72   | 0.29 |
| Tuned Tree | ~0.87       | 0.17      | 0.75   | 0.28 |

---

## Random Forest

Random Forest-მა მნიშვნელოვნად გააუმჯობესა შედეგები Decision Tree-სთან შედარებით.

გამოვიყენე:

* n_estimators tuning
* depth tuning
* class balancing
* feature engineering

Random Forest გახდა ბევრად უფრო სტაბილური და ნაკლებად overfit.

| Experiment | Val ROC-AUC | Precision | Recall | F1    |
| ---------- | ----------- | --------- | ------ | ----- |
| Baseline   | ~0.884      | 0.189     | 0.726  | 0.300 |
| Tuned RF   | ~0.911      | 0.328     | 0.693  | 0.445 |

---

## XGBoost

XGBoost-მ აჩვენა საუკეთესო შედეგი ყველა მოდელს შორის.

Boosting-ის საშუალებით მან შეძლო რთული fraud pattern-ების დაჭერა და მაღალი ROC-AUC მიღწევა.

გამოვიყენე:

* High Missing Cleaning
* UIDAmountDeviation Feature
* RFE
* WOE experiments
* Hyperparameter tuning

WOE Encoding-მა მნიშვნელოვანი გაუმჯობესება არ მოიტანა.

RFE კი გახდა ყველაზე ეფექტური Feature Selection მეთოდი.

---

## საუკეთესო XGBoost მოდელი

| Metric             | Value  |
| ------------------ | ------ |
| Validation ROC-AUC | 0.9427 |
| Precision          | 0.9247 |
| Recall             | 0.4875 |
| F1 Score           | 0.6384 |
| Overfit Gap        | 0.0145 |

---

## საბოლოო XGBoost Pipeline

```
HighMissingDropper
    ↓
UIDAmountDeviationAdder
    ↓
AutoPreprocessor
    ↓
RFE Feature Selection
    ↓
XGBoost
```

---

# MLflow Tracking

ყველა ექსპერიმენტი დავარეგისტრირე MLflow-ში DagsHub-ის საშუალებით.

ექსპერიმენტებში ვინახავდი:

* ROC-AUC
* Log Loss
* Precision
* Recall
* F1 Score
* Overfit Gap
* Hyperparameters
* Feature Engineering ინფორმაციას
* Cleaning ინფორმაციას
* Feature Selection ინფორმაციას

---

## დალოგილი მეტრიკები

* train_roc_auc
* val_roc_auc
* train_log_loss
* val_log_loss
* precision
* recall
* f1_score
* overfit_gap
* overfit_loss_gap

---

## დალოგილი პარამეტრები

* model
* feature_engineering
* engineered_feature
* feature_selection
* encoding
* cleaning
* threshold
* max_depth
* learning_rate
* n_estimators
* subsample
* colsample_bytree

---

# Model Inference

საბოლოო inference notebook-ში:

* MLflow-იდან იტვირთება საუკეთესო pipeline
* Kaggle test set ტრანსფორმირდება
* გენერირდება საბოლოო submission.csv

გამოყენებულია:

* DagsHub
* MLflow
* Registered Model Loading

---

# Kaggle შედეგი

| Score Type    | Result |
| ------------- | ------ |
| Public Score  | 0.9288 |
| Private Score | 0.9089 |

საბოლოოდ საუკეთესო შედეგი აჩვენა:

```text
XGBoost + UIDAmountDeviation + RFE
```

რომელიც გახდა პროექტის საბოლოო მოდელი.

<img width="1370" height="162" alt="image" src="https://github.com/user-attachments/assets/eaed7c41-bf6a-4dcc-9704-0aeaebac7630" />
