# Stroke Prediction ML Pipeline

A machine learning project to predict stroke risk using patient clinical data. The main challenge here is the severe class imbalance: only about 5% of the dataset had a stroke. Because of that, a standard 95% accuracy usually means the model just predicted "No Stroke" for everyone and caught zero patients. 

This project focuses on fixing that by prioritizing **Recall** over raw accuracy, so we don't miss actual patients who need medical attention.

---

## What We Did

1. **Data Cleaning:** 
   - Dropped the `id` column since it's just an arbitrary number.
   - Removed a single row labeled `Other` under gender.
   - Filled missing `bmi` values with the median.

2. **Preprocessing:** 
   - Used `pd.get_dummies(drop_first=True)` for categorical columns.
   - Split the data (80% train / 20% test) using `stratify=y` so both splits keep the same stroke ratio.
   - Scaled the numeric features using `StandardScaler` after the split to avoid data leakage.

3. **Models Tested:**
   - **KNN:** Tested default 0.50 threshold vs a lower 0.15 threshold using distance weighting.
   - **Random Forest:** Configured with `class_weight='balanced'` and capped tree depth to prevent overfitting.
   - **Logistic Regression:** Also trained with `class_weight='balanced'`.
   - **Decision Tree & AdaBoost:** For comparison.

---

## Results on Test Data (1,022 rows, 50 stroke cases)

| Model | Accuracy | Recall (Caught Stroke) | F1-Score | True Positives | Missed Cases |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN (0.50 threshold)** | 94.32% | 8.00% | 12.12% | 4 | 46 |
| **KNN (0.15 threshold)** | 84.83% | 24.00% | 13.41% | 12 | 38 |
| **Random Forest (Balanced)** | 88.06% | 54.00% | 30.68% | 27 | 23 |
| **Logistic Regression (Balanced)** | 73.87% | **80.00%** | 23.05% | **40** | **10** |

---

## Takeaways

- **Default models fail here:** KNN with default settings scored 94.3% accuracy, but it missed 46 out of 50 stroke patients. That makes it useless for any medical screening.
- **The best screening model:** **Balanced Logistic Regression** caught 40 out of 50 stroke cases (80% recall). It has a lower accuracy (around 74%) because of more false alarms, but in a medical context, doing an extra test on a healthy person is far better than sending a sick person home.
- **The best all-round model:** **Random Forest** had the highest F1-score (30.68%) and gave the best compromise between catching cases and keeping false alarms down.

---
##  Contributors & Team

* **Yahya Eltagredy** - [GitHub](https://github.com/eltagredy122)
* **Yousry Oraby** - [GitHub](https://github.com/YousryOraby)
* **Menna Zoghla** - [GitHub](https://github.com/Menna-Khaled9)
* **Sandy Makram** - [GitHub](https://github.com/SandyMakram12)
* **Hesham Mohamed** - [GitHub](https://github.com/Hesham2006)
