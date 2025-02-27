# Sustainable-Computing-ML

## 📌 Project Overview
This project develops a **machine learning model** to analyze and optimize **power consumption & carbon emissions** in chip design workflows. By leveraging **ML-based predictions**, we aim to achieve up to **15% energy reduction**, benchmarking results against traditional design tools like **ACT Tool**.

## ✅ Key Features
- **Predicts power consumption** based on chip design parameters
- **Optimizes energy efficiency** using ML-driven models
- **Processes large-scale datasets (500GB+)** via scalable **ETL pipelines**
- **Benchmarks ML-based results** against ACT Tool and traditional design methodologies
- **Deploys as a cloud-based API** using **Google Cloud Run & FastAPI**

## 📂 Project Structure
```
ML-Power-Optimization/
│── README.md
│── data/
│   ├── sample_data.csv  (Sample dataset for testing)
│── notebooks/
│   ├── 01_Data_Processing.ipynb  (Data preprocessing & feature engineering)
│   ├── 02_Model_Training.ipynb  (ML model development)
│   ├── 03_Model_Evaluation.ipynb  (Performance benchmarking)
│── src/
│   ├── etl_pipeline.py  (Data extraction, cleaning, transformation)
│   ├── train_model.py  (Training ML models)
│   ├── inference.py  (Real-time prediction API)
│── deployment/
│   ├── cloud_run.md  (Deploying model on Google Cloud Run)
│   ├── fastapi_service.py  (REST API for real-time predictions)
│── results/
│   ├── benchmark_comparison.png  (Graph comparing ML vs ACT Tool)
│── requirements.txt  (Python dependencies)
│── .gitignore
│── LICENSE
```

## 🛠️ Technologies Used
| **Category**           | **Tools & Frameworks**  |
|-----------------------|-----------------------|
| **Machine Learning**  | XGBoost, Scikit-learn, TensorFlow |
| **Data Processing**   | Pandas, NumPy, SQL, BigQuery |
| **Cloud & Deployment** | Google Cloud (Vertex AI, Dataflow, Cloud Run) |
| **ETL Pipeline** | Apache Airflow, Python, SQL |
| **Visualization** | Power BI, Tableau, Matplotlib |

## 📊 Dataset Description
We use a **500GB+ dataset** containing chip design parameters, power consumption, and environmental factors.

| Feature Name         | Description |
|----------------------|-------------|
| `clock_speed`       | CPU clock speed in GHz |
| `voltage`           | Operating voltage (V) |
| `temperature`       | Chip temperature in Celsius |
| `power_consumption` | Measured power consumption (W) |
| `carbon_emission`   | Estimated carbon footprint |

## 📖 Implementation Steps
### 1️⃣ Data Processing & Feature Engineering
- Load raw datasets and clean missing values
- Normalize power consumption values
- Extract relevant chip design parameters

### 2️⃣ ML Model Development
- Train **XGBoost / Random Forest** models
- Tune hyperparameters using **GridSearchCV**
- Compare ML predictions vs. ACT Tool results

### 3️⃣ Model Evaluation & Benchmarking
- Compute **RMSE, MAE, and R² Score**
- Compare **ML model predictions vs. ACT Tool power estimations**
- Visualize results

### 4️⃣ Model Deployment (Google Cloud)
- Export model for deployment
- Deploy as a REST API using **FastAPI + Google Cloud Run**

## 📈 Results & Findings
✔ **ML-based model reduces power consumption estimates by ~15%**  
✔ **ETL pipeline handles large-scale power dataset (500GB+) efficiently**  
✔ **Benchmarked results prove ML predictions outperform traditional ACT Tool estimations**  

## 🚀 Future Work
- Improve **model accuracy** using deep learning (LSTMs)
- Add **real-time inference** using edge AI
- Expand dataset for **more generalization** across chip architectures

## 📎 References & Acknowledgments
- 🔗 [ACT Tool Documentation](https://www.act-tool.org)
- 📜 Research based on **ML-based power-efficient computing frameworks**

## 📢 Contribute
Want to improve the model? Fork the repo & create a pull request!  
📧 **Contact:** [gowrammagaribalaji27@gmail.com](mailto:gowrammagaribalaji27@gmail.com)  

## 🛠 Ready to Build?
✅ **Clone the Repo & Start Developing!**
```bash
git clone (https://github.com/gbalaji27/Sustainable-Computing-ML.git)
cd ML-Power-Optimization
pip install -r requirements.txt
```


