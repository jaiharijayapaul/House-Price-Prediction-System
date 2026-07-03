# 🏠 House Price Prediction
A machine learning web application that predicts house sale prices using **Linear Regression** trained on the **Ames Housing Dataset**. Built with Python and Streamlit, the app provides an end-to-end interactive experience — from raw data exploration to real-time price prediction.
---
## 📌 Project Overview
This project implements a complete supervised machine learning pipeline to predict residential house prices based on two key features:
- **GrLivArea** — Above-grade living area (square feet)
- **TotRmsAbvGrd** — Total rooms above grade (excludes bathrooms)
The model is trained, evaluated, and served interactively through a multi-page Streamlit dashboard.
---
## 🗂️ Project Structure
House Price Prediction/ │ ├── Data/ │ ├── train.csv # Ames Housing training dataset │ └── data_description.txt # Feature descriptions │ ├── House_Price_Prediction.ipynb # Jupyter notebook (full ML pipeline) ├── app.py # Streamlit web application └── README.md # Project documentation

---
## 🔧 Tech Stack
| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas & NumPy | Data loading and manipulation |
| Scikit-learn | Linear Regression, StandardScaler, metrics |
| Matplotlib | Visualization and charts |
| Streamlit | Interactive web application |
| Jupyter Notebook | Step-by-step ML pipeline documentation |
---
## 🚀 Getting Started
### 1. Prerequisites
Make sure Python 3.8+ is installed. Then install the required packages:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn
2. Dataset
Place the Ames Housing dataset (train.csv) inside a Data/ folder:

House Price Prediction/
└── Data/
    └── train.csv
You can download the dataset from Kaggle — House Prices: Advanced Regression Techniques

3. Run the Streamlit App
bash
streamlit run app.py
Then open your browser at: http://localhost:8501

4. Run the Notebook
bash
jupyter notebook House_Price_Prediction.ipynb
📊 ML Pipeline (7 Steps)
Step	Description
1️⃣ Load & Explore	Read CSV, inspect columns, check missing values, plot distributions
2️⃣ Preprocessing	Remove anomalous outliers, 80/20 train-test split, apply StandardScaler (fit on train only)
3️⃣ Model Training	Fit LinearRegression on scaled training features
4️⃣ Evaluation	Compute MAE, MSE, RMSE, R² on both train and test sets
5️⃣ Visualization	Area vs Price scatter, Actual vs Predicted chart, Residual analysis
6️⃣ Prediction	Accept custom inputs (area + rooms) and predict sale price
7️⃣ Conclusion	Findings, limitations, and future improvement directions
📈 Model Performance
Metric	Training Set	Test Set
R² Score	~0.53	~0.52
MAE	~$37,000	~$38,000
RMSE	~$52,000	~$53,000
Exact values are computed at runtime from the dataset.

🖥️ App Pages
Page	Content
🏡 Home	Project overview, live metrics, pipeline summary
📂 Dataset Explorer	Raw data table, statistics, missing value analysis, distribution charts
⚙️ Preprocessing	Outlier removal visualization, scaler statistics, split info
🤖 Model & Equation	LaTeX equation, coefficients, feature importance chart
📊 Evaluation	Train vs Test metrics dashboard, Actual vs Predicted table
📈 Visualizations	4 interactive Matplotlib charts (selectable)
🎯 Predict My House	Slider inputs + quick presets → instant price prediction
📝 Conclusion	Key findings, limitations, future improvements
🔑 Key Findings
Living area (GrLivArea) is the strongest predictor — each standard unit increase raises predicted price significantly.
Room count (TotRmsAbvGrd) is a secondary contributor.
The model shows no significant overfitting (Train R² ≈ Test R²).
Residuals are approximately normally distributed, confirming Linear Regression assumptions are met.
A 2-feature model explains approximately 52% of price variation — a strong baseline.
⚠️ Limitations
Struggles with luxury homes (>$400K) where quality and location dominate pricing.
Only 2 features used — many important columns (e.g., OverallQual, Neighborhood, YearBuilt) are excluded.
Trained on Ames, Iowa data — may not generalize to other cities or markets.
🚀 Future Improvements
Improvement	Expected Benefit
Add OverallQual	Highest single-feature correlation with SalePrice
Add Neighborhood	Location is a top pricing driver
Add YearBuilt	Newer homes command higher prices
Log-transform SalePrice	Reduces right-skew, improves model fit
Polynomial Regression	Captures non-linear area–price curves
Ridge / Lasso Regression	Regularization for many features
K-Fold Cross Validation	More reliable accuracy estimate
XGBoost / Gradient Boosting	Typically achieves R² > 0.90 on this dataset
📄 License
This project is for educational purposes. The Ames Housing Dataset is publicly available via Kaggle under their competition terms.

