# MEDICAL-APPOINTMENT-NO-SHOW-PREDICTION
# Medical Appointment No-Show Prediction

## Project Summary
This project predicts whether a patient will miss a scheduled medical appointment (“No-show”).  
It demonstrates an end-to-end machine learning workflow: preprocessing, feature engineering, feature selection, model training, and model evaluation on an imbalanced dataset.

## Dataset
- Source: Kaggle — *Medical Appointment No Shows*
- Size: 110,527 rows, 14 original columns
- Target: `No-show` (encoded as 0 = Show, 1 = No-show)
- Data quality: no missing values observed

## Approach
### Preprocessing & Feature Engineering
- Removed identifiers: `PatientId`, `AppointmentID`
- Encoded categorical variables:
  - `Gender`: one-hot encoding
  - `Neighbourhood`: frequency encoding
- Date-time processing (`ScheduledDay`, `AppointmentDay`):
  - extracted hour, weekday, and month features
  - dropped original date columns after extraction

### Feature Selection
Top 8 features selected using **SelectKBest (ANOVA F-test)**:
- Age, Scholarship, Hipertension, Diabetes, SMS_received  
- ScheduledHour, ScheduledMonth, AppointmentMonth

### Models
Models compared using cross-validation (GridSearchCV, cv=3):
- Logistic Regression (with scaling)
- Linear SVM (with scaling)
- Random Forest

## Results (Test Set)
Given class imbalance, performance is interpreted with emphasis on the **No-show class** (precision/recall/F1).

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Logistic Regression | 0.797 | 0.351 | 0.009 | 0.017 |
| Linear SVM | 0.798 | 0.000 | 0.000 | 0.000 |
| Random Forest | 0.765 | 0.326 | 0.153 | 0.208 |

**Best model (by F1-score): Random Forest (F1 = 0.208)**  
Saved model (optional): `models/Lara_Gatigbene_best_model.joblib`

## How to Run
1. Clone the repository
2. Install dependencies (example): `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib`
3. Open and run the notebook:
   - `medical_apointment_no_show_predict.ipynb`

## Repo Structure (recommended)
- `medical_apointment_no_show_predict.ipynb` — main notebook
- `models/` — saved model (optional)
- `figures/` — plots/images (optional)
- `README.md`

## Limitations & Improvements
- The dataset is imbalanced; accuracy is close to the majority-class baseline.
- Improve recall for No-show prediction via:
  - class weighting, threshold tuning, resampling (e.g., SMOTE)
  - alternative metrics (PR-AUC)
- Add explainability (feature importance / SHAP) and a lightweight inference script.

## Author
Gatigbene Bomboma, Lara Abou-Arraj  

