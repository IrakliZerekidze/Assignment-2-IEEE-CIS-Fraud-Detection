# IEEE-CIS Fraud Detection

## კონკურსის და პროექტის მიმოხილვა

კონკურსის მიზანია ონლაინ ტრანზაქციებში fraud ოპერაციების აღმოჩენა სხვადასხვა მომხმარებლის, ტრანზაქციისა და მოწყობილობის მახასიათებლების საფუძველზე.
ამ პროექტში განხორციელდა მონაცემთა გაწმენდა, Feature Engineering, Feature Selection და რამდენიმე Machine Learning მოდელის ტრენინგი.
საბოლოოდ შეირჩა საუკეთესო მოდელი, რომელმაც ყველაზე მაღალი ROC-AUC შედეგი აჩვენა.

## რეპოზიტორიის სტრუქტურა

```
├── model-experiment-decision-tree.ipynb    
├── model-experiment-logistic-regression.ipynb
├── model-experiment-random-forest.ipynb     
├── model-experiment-xgboost.ipynb     
├── model-inference.ipynb
└── README.md
```
---

## პროექტის სტრუქტურა

| ფაილი | აღწერა |
|---|---|
| `model-experiment-decision-tree.ipynb` | Decision Tree მოდელის ექსპერიმენტები და ჰიპერპარამეტრების ტესტირება, preprocessing-ის, cleaning-ის და feature engineering-ის სხვადასხვა მიდგომებით. |
| `model-experiment-logistic-regression.ipynb` | Logistic Regression მოდელის ექსპერიმენტები რეგულარიზაციის, encoding-ის, feature engineering-ის და feature selection-ის სხვადასხვა მეთოდებით. |
| `model-experiment-random-forest.ipynb` | Random Forest მოდელის ტრენინგი და ტესტირება სხვადასხვა depth-ის, estimator-ის და preprocessing-ის პარამეტრებით. |
| `model-experiment-xgboost.ipynb` | XGBoost მოდელის ექსპერიმენტები, პარამეტრების tuning-ი, feature engineering და feature selection მეთოდების გამოყენებით. |
| `model-inference.ipynb` | საბოლოო inference pipeline, რომელიც გამოიყენება გაწვრთნილი მოდელების ჩასატვირთად და test მონაცემებზე პროგნოზების გასაკეთებლად. |
| `README.md` | პროექტის დოკუმენტაცია, ექსპერიმენტების აღწერა, შედეგები და გამოყენების ინსტრუქცია. |

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
* ვცადე სხვა threshold-ებიც თუმცა ყველაზე წარმატებული ეს აღმოჩნდა.

---

## 2. High Dominance Columns Removal

წავშალე ისეთი სვეტები, სადაც ერთი მნიშვნელობა თითქმის მთლიან სვეტს იკავებდა.

გამოვიყენე რამდენიმე threshold:

* 0.99
* 0.995
* 0.98

თუმცა შედეგებმა აჩვენა, რომ dominance filtering უმეტეს მოდელებზე უარყოფითად მოქმედებდა, შეაბამისად
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

ორივე გამოვცადე ყველა მოდელზე თუმცა ძირითადად One-Hot Encoding უკეთეს შედეგს გვაძლევდა, ჰიბრიდულმა
encoding-მა XGBoost-ზე იგივენაირი შედეგი აჩვენა რაც One-Hot Encoding-მა

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

Correlation Filter ვცადე მოდელთა უმეტესობაზე თუმცა ძალიან დიდი დრო მიჰქონდა და იქრაშებოდა kaggle ამიტომ
აღარ მივაქციე ყურადღება

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

## Logistic Regression Experiments

| Experiment | Encoding | Cleaning / Preprocessing | Feature Engineering | Feature Selection | Regularization | C | ROC-AUC | F1 Score | Precision | Recall | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Raw Baseline | OneHotEncoder | Minimal preprocessing | None | None | None | - | 0.7403 | 0.1223 | 0.0673 | 0.6685 | Initial baseline model |
| L2 Regularization | OneHotEncoder | Minimal preprocessing | None | None | L2 | 0.1 | 0.7909 | 0.1644 | 0.0934 | 0.6872 | Large improvement over baseline |
| L2 Regularization (Best Basic) | OneHotEncoder | Minimal preprocessing | None | None | L2 | 5 | 0.8181 | 0.2128 | 0.1267 | 0.6654 | Best result before feature engineering |
| L1 Regularization | OneHotEncoder | Minimal preprocessing | None | None | L1 | 1 | 0.5774 | 0.1454 | 0.1002 | 0.2647 | Sparse model, weaker performance |
| Cleaning High NaN Columns | OHE | Removed high-missing columns | None | None | L2 | 0.1 | 0.8123 | 0.2094 | 0.1248 | 0.6516 | Slight drop after aggressive cleaning |
| Cleaning + Dominance Filtering | OHE | Removed high NaN & dominant cols | None | None | L2 | 0.1 | 0.8116 | 0.2091 | 0.1246 | 0.6487 | Similar to previous cleaning strategy |
| WOE + OHE Encoding | WOE + OHE | Minimal preprocessing | None | None | L2 | 0.1 | 0.7377 | 0.1488 | 0.0855 | 0.5725 | Encoding combination underperformed |
| Log Feature Engineering | OHE | Dominance + missing filtering | Log(amount) transform | None | L2 | 0.1 | 0.8809 | 0.2532 | 0.1522 | 0.7534 | Significant improvement after feature engineering |
| Log Feature Engineering (Repeat) | OHE | Dominance + missing filtering | Log(amount) transform | None | L2 | 0.1 | 0.8793 | 0.2493 | 0.1495 | 0.7518 | Stable repeatable results |
| L1 Feature Selection + Log FE | OHE | Dominance + missing filtering | Log(amount) transform | L1-based selection | L2 | 0.1 | 0.8756 | 0.2430 | 0.1452 | 0.7445 | Slight reduction after feature selection |


## Decision Tree

Decision Tree-მა საკმაოდ სწრაფად დაიწყო overfitting.

გამოვიყენე:

* max_depth tuning
* feature engineering
* IV filtering

თუმცა Decision Tree მნიშვნელოვნად ჩამორჩა სხვა მოდელებს.

## Decision Tree Experiments

| Experiment | Encoding | Cleaning / Preprocessing | Feature Engineering | Feature Selection | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Raw Baseline | OneHotEncoder | Minimal preprocessing | None | None | 0.8011 | 0.4504 | 0.8273 | 0.3095 |
| Tuned Baseline | OneHotEncoder | Minimal preprocessing | None | None | 0.8695 | 0.2832 | 0.1740 | 0.7595 |
| WOE + OHE Encoding | WOE + OHE | Minimal preprocessing | None | None | 0.8693 | 0.2750 | 0.1675 | 0.7682 |
| Cleaning High NaN + Dominance | OHE | Removed high-missing & dominant cols | None | None | 0.8699 | 0.2830 | 0.1739 | 0.7595 |
| FE: UID | OHE | Minimal preprocessing | UID feature | None | 0.8698 | 0.2887 | 0.1784 | 0.7564 |
| FE: Log Adder | OHE | Minimal preprocessing | Log-transformed feature | None | 0.8697 | 0.2832 | 0.1740 | 0.7597 |
| RFE Feature Selection | OHE | Minimal preprocessing | None | RFE (100 features) | 0.8698 | 0.2817 | 0.1729 | 0.7609 |
| Final Decision Tree | OHE | High missing + dominance filtering | None | None | **0.8706** | **0.2835** | **0.1742** | **0.7607** |

---

## Random Forest

Random Forest-მა მნიშვნელოვნად გააუმჯობესა შედეგები Decision Tree-სთან შედარებით.

გამოვიყენე:

* n_estimators tuning
* depth tuning
* class balancing
* feature engineering

Random Forest გახდა ბევრად უფრო სტაბილური და ნაკლებად overfit.

## Random Forest Experiments

| Experiment | Encoding | Cleaning / Preprocessing | Feature Engineering | Max Depth | Trees | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| RF Baseline | OHE | Minimal preprocessing | None | 12 | 200 | 0.8841 | 0.3007 | 0.1896 | 0.7263 |
| RF Baseline (500 Trees) | OHE | Minimal preprocessing | None | 12 | 500 | 0.8847 | 0.3018 | 0.1902 | 0.7293 |
| RF High Missing Cleaning | OHE | Removed high-missing columns | None | 12 | 500 | 0.8838 | 0.2989 | 0.1881 | 0.7273 |
| RF High Dominance Cleaning | OHE | Removed dominant columns | None | 12 | 500 | 0.8838 | 0.2989 | 0.1881 | 0.7273 |
| RF Tuned Depth 14 | OHE | Minimal preprocessing | None | 14 | 700 | 0.8942 | 0.3399 | 0.2221 | 0.7232 |
| RF Tuned Depth 16 | OHE | Minimal preprocessing | None | 16 | 700 | 0.9030 | 0.3876 | 0.2663 | 0.7116 |
| RF Tuned Depth 18 | OHE | Minimal preprocessing | None | 18 | 700 | 0.9111 | 0.4455 | 0.3282 | 0.6932 |
| RF Feature Engineering (UID) | OHE | Minimal preprocessing | UID feature | 18 | 700 | **0.9121** | **0.4494** | **0.3328** | 0.6920 |

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

## XGBoost Experiments

| Experiment | Encoding | Cleaning / Preprocessing | Feature Engineering | Feature Selection | Learning Rate | Max Depth | Trees | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGB Baseline | OHE | Minimal preprocessing | None | None | 0.05 | 6 | 300 | 0.9291 | 0.5893 | 0.9140 | 0.4348 |
| XGB Depth 5 LR 0.03 | OHE | Minimal preprocessing | None | None | 0.03 | 5 | 500 | 0.9199 | 0.5520 | 0.9018 | 0.3978 |
| XGB Depth 6 LR 0.05 | OHE | Minimal preprocessing | None | None | 0.05 | 6 | 500 | 0.9387 | 0.6248 | 0.9276 | 0.4711 |
| XGB Regularized | OHE | Minimal preprocessing | None | None | 0.05 | 6 | 500 | 0.9368 | 0.6176 | 0.9171 | 0.4655 |
| XGB High Missing + Dominance Cleaning | OHE | Removed high-missing & dominant cols | None | None | 0.05 | 6 | 500 | 0.9403 | 0.6248 | 0.9255 | 0.4716 |
| XGB Log Amount Feature | OHE | Minimal preprocessing | Log(amount) | None | 0.05 | 6 | 500 | 0.9390 | 0.6256 | 0.9243 | 0.4728 |
| XGB UID Features | OHE | Minimal preprocessing | UID-based features | None | 0.05 | 6 | 500 | 0.9411 | 0.6267 | 0.9228 | 0.4745 |
| XGB Feature Selection (RFE) | OHE | Minimal preprocessing | UID features | RFE | 0.05 | 6 | 500 | **0.9427** | **0.6385** | **0.9247** | **0.4875** |
| XGB WOE + OHE | OHE + WOE | Minimal preprocessing | UID features | RFE | 0.05 | 6 | 500 | **0.9427** | **0.6385** | **0.9247** | **0.4875** |

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

🔗 [https://dagshub.com/](https://dagshub.com/izere23/Assignment-2-IEEE-CIS-Fraud-Detection.mlflow)

<img width="1370" height="162" alt="image" src="https://github.com/user-attachments/assets/eaed7c41-bf6a-4dcc-9704-0aeaebac7630" />
