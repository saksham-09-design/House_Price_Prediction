# 🏡 House Price Prediction using Random Forest Regressor

Welcome to the **House Price Prediction** project!
This repository demonstrates a complete **Machine Learning pipeline** built using the **California Housing Dataset**.
The goal is to **predict house prices** using a **Random Forest Regressor**, along with a well-structured pipeline that handles:

*  **Data Preprocessing**
*  **Scaling Numerical Features** using `StandardScaler`
*  **Encoding Categorical Features** using `OneHotEncoder`
*  **Model Training**
*  **Model & Pipeline Saving** using `Joblib`

---

## 🚀 Project Overview

This project builds a regression model capable of predicting housing prices using various features such as:

* Median income
* House age
* Latitude / Longitude
* Population
* And other relevant attributes from the **California Housing Dataset**

A **Random Forest Regressor** is used because it is:

* 🌲 Robust
* ⚡ Fast
* 🧠 Accurate
* 🛡️ Less prone to overfitting

---

## 🧠 Machine Learning Pipeline

The project uses a **complete end-to-end pipeline**, ensuring data transformation and prediction are seamless and reproducible.

### 🔧 Pipeline Components

* **StandardScaler** → For numerical features
* **OneHotEncoder** → For categorical features
* **ColumnTransformer** → To combine both transformations
* **RandomForestRegressor** → Final ML model

This pipeline guarantees:

* No data leakage
* Cleaner code
* Easier reproducibility
* Direct `.predict()` on raw input

---

## 💾 Saving the Model

Using `joblib`, both the trained **model** and the **pipeline** are saved as:

```
model.pkl
pipeline.pkl
```

This allows fast loading and deployment without retraining.

---

## 📊 Dataset

The dataset used is the **California Housing Prices** dataset from the California census.
It includes information about:

* 🏘️ Housing blocks
* 👨‍👩‍👧 Population
* 💰 Median income
* 🧱 House age
* 🌎 Geographical coordinates

---

## 🛠️ Technologies Used

* 🐍 **Python**
* 🧮 **NumPy**
* 🧹 **Pandas**
* 📊 **Matplotlib / Seaborn**
* 🤖 **Scikit-Learn**
* 💾 **Joblib**

---

## 📈 Results

* ✔️ Model trained using RandomForestRegressor
* ✔️ Full preprocessing + training pipeline
* ✔️ High accuracy on test dataset
* ✔️ Ready for deployment

---

## 🤝 Contributing

Feel free to create an issue or submit a pull request if you'd like to contribute!

---

## ⭐ Show Support


Just tell me!
