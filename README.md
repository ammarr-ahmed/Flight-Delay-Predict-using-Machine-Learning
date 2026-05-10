✈️ Flight Delay Prediction
An AI-powered Flight Delay Prediction system leveraging Random Forest and XGBoost models. The project processes historical flight data, integrates live weather data from the Open-Meteo API, trains models, compares performance, and provides real-time delay predictions with a clean, modular structure.

🚀 Features
Flight delay prediction using ML models
Data balancing, cleaning, and preprocessing pipeline
Model training with Random Forest & XGBoost
Accuracy comparison with visual graphs
Integration with Open-Meteo API for live weather inputs
Saved trained models for fast predictions
Interactive app for real-time predictions
Organized and scalable project structure


🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Joblib
Open-Meteo API


📂 Project Structure
bash
FLIGHT DELAY/
│
├── artifacts/
│   ├── data/                # Raw, cleaned, and split datasets
│   ├── graph/               # Model comparison visualizations
│   └── models/              # Saved trained models (.pkl)
│
├── src/
│   ├── 1-balance_data.py    # Handle class imbalance
│   ├── 2-clean_data.py      # Clean and preprocess dataset
│   ├── 3-preprocessed.py    # Feature engineering & scaling
│   ├── 4-train_random_forest.py
│   ├── 5-train_xgboost.py
│   ├── 6-compare_models.py  # Evaluate and compare accuracy
│   ├── 7-predict.py         # Generate predictions
│   └── 8-app.py             # Interactive prediction app (with Open-Meteo API)
│
└── README.md



📊 Workflow
Run src/1-balance_data.py to balance dataset
Run src/2-clean_data.py for cleaning
Run src/3-preprocessed.py for preprocessing
Train models with src/4-train_random_forest.py 
src/5-train_xgboost.py
Compare accuracy with src/6-compare_models.py
Predict delays using src/7-predict.py
Launch interactive app with src/8-app.py (fetches live weather data from Open-Meteo API)


📈 Results
Accuracy comparison graphs stored in artifacts/graph/
Trained models saved in artifacts/models/ for reuse
Real-time predictions enhanced with live weather inputs

👨‍💻 Author
Developed by Muhammad Ammar Ahmad (https://github.com/ammarr-ahmed))
