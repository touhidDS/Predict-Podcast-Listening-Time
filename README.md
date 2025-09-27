# Podcast Listening Time Prediction

## Overview
This repository contains a Jupyter Notebook for the [Kaggle Playground Series - Season 5, Episode 4](https://www.kaggle.com/competitions/playground-series-s5e4) competition. The goal is to predict the listening time (in minutes) for podcast episodes based on features like episode length, genre, host/guest popularity, publication details, and more.

The notebook performs data preprocessing, feature engineering, model training using LightGBM, and generates a submission file for the competition.

## Dataset
- **Training Data**: `/kaggle/input/playground-series-s5e4/train.csv` (750,000 entries)
- **Test Data**: `/kaggle/input/playground-series-s5e4/test.csv` (250,000 entries)
- **Features**:
  - `Podcast_Name` (categorical)
  - `Episode_Title` (converted to episode number)
  - `Episode_Length_minutes` (numerical)
  - `Genre` (categorical)
  - `Host_Popularity_percentage` (numerical)
  - `Publication_Day` (categorical)
  - `Publication_Time` (categorical)
  - `Guest_Popularity_percentage` (numerical)
  - `Number_of_Ads` (numerical)
  - `Episode_Sentiment` (categorical)
- **Target**: `Listening_Time_minutes` (numerical, regression task)
- **Missing Values**: Handled by filling with median for numerical columns (`Episode_Length_minutes`, `Guest_Popularity_percentage`, `Number_of_Ads`).

## Approach
1. **Data Loading & Exploration**:
   - Load train and test datasets using Pandas.
   - Drop unnecessary `id` column.
   - Check data info, null percentages, and skewness for numerical features.

2. **Preprocessing**:
   - Impute missing values with median.
   - Label encode categorical features: `Publication_Day`, `Publication_Time`, `Episode_Sentiment`, `Genre`, `Podcast_Name`.
   - Extract episode number from `Episode_Title` (e.g., "Episode 71" → 71).

3. **Feature Engineering**:
   - No additional features created beyond encoding and extraction.
   - Commented sections for one-hot encoding and visualizations (boxplots, KDE plots).

4. **Modeling**:
   - Use LightGBM Regressor with hyperparameters:
     - `n_iter=1000`
     - `max_depth=-1`
     - `num_leaves=1024`
     - `colsample_bytree=0.7`
     - `learning_rate=0.03`
     - `objective='l2'`
     - `metric='rmse'`
     - `verbosity=-1`
     - `max_bin=1024`
   - Train on full training data.
   - Predict on test data.

5. **Evaluation**:
   - No cross-validation in the final script (commented train-test split available for testing).
   - Metrics (for local evaluation): RMSE, MAE, R-squared.

6. **Submission**:
   - Generate `submission.csv` with predictions.

## Requirements
- Python 3.11+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn` (for `LabelEncoder`, metrics, and splits)
  - `lightgbm`

Install via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
```

## Results
- The model predicts listening time with a focus on RMSE minimization.
- Sample predictions (first 10 from submission):
  - `id 750000`: ~54.56
  - `id 750001`: ~18.99
  - (See notebook output for full details)

## Notes
- This is designed for the Kaggle environment 
- For local runs, update file paths accordingly.
- Potential improvements: Hyperparameter tuning, ensemble models, additional feature engineering.


