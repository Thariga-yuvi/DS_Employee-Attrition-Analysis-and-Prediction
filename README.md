# 👩‍💼 Employee Attrition Prediction Dashboard

An interactive **Streamlit web application** that helps HR teams analyze employee data, understand attrition trends, and predict the likelihood of employee attrition using a **Machine Learning model** (Random Forest Classifier).

---

## 🚀 Project Overview

Employee retention is a key challenge for organizations.  
This dashboard enables **data-driven decision-making** by combining **Exploratory Data Analysis (EDA)**, **Model Evaluation**, and **Predictive Analytics** in one place.

The app is built using **Python**, **Streamlit**, **Plotly**, **Matplotlib**, and **Seaborn**.

---

## 🧠 Features

### 🩵 1. Exploratory Data Analysis (EDA)
Gain insights into employee behavior and attrition patterns:
- 📊 **Attrition distribution** (count + pie chart)
- 🔍 **Pairplot visualization** for Age, Income, and Tenure vs Attrition
- 💰 **Monthly income distribution** with key insights
- Clean, responsive layout using Streamlit columns

### 💡 2. Model Evaluation Metrics
Performance summary of the **Random Forest** model:
- Accuracy: `94%`
- Precision: `96%`
- Recall: `92%`
- F1 Score: `94%`
- ROC AUC: `98%`

> Model parameters: `{ 'n_estimators': 200, 'max_depth': None }`

### 🎯 3. Attrition Prediction Tool
A dynamic prediction form for HR teams to simulate different employee profiles.

- Input 20+ features such as Age, Department, Job Role, Income, and OverTime
- Predict attrition likelihood in real-time
- Visualize satisfaction metrics in a **Radar Chart**
- Get explanations for high-risk or low-risk predictions:
  - 🚨 *“High Attrition Risk Detected!”* → Detailed contributing factors
  - ✅ *“Low Attrition Risk Detected”* → Positive retention indicators

### 📈 4. Employee Insights
- View **Top 10 High-Income Employees**
- Analyze **Attrition by Department** (via pie or donut charts)
- 
| Metric          | Observation                     |
| --------------- | ------------------------------- |
| Attrition Rate  | 16.1% employees left            |
| Top Factor      | Overtime frequency              |
| High-Risk Group | Employees < 2 years at company  |
| Positive Signal | High satisfaction, low overtime |

---

## 🧩 Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Frontend** | Streamlit |
| **Visualization** | Seaborn, Matplotlib, Plotly |
| **Machine Learning** | Scikit-learn (Random Forest Classifier) |
| **Data Handling** | Pandas, NumPy |
| **Deployment** | Streamlit Cloud / Localhost |

<img width="1503" height="882" alt="image" src="https://github.com/user-attachments/assets/27a483c5-7113-4059-9e45-e1e88182d1c2" />
<img width="1403" height="827" alt="image" src="https://github.com/user-attachments/assets/82d23d4a-f994-4413-9957-c80dc22a8eba" />
<img width="1617" height="713" alt="image" src="https://github.com/user-attachments/assets/8936de5a-48e3-45a5-92ae-fafafee781e1" />
<img width="1460" height="803" alt="image" src="https://github.com/user-attachments/assets/1476c2e4-e021-408e-bfc4-fab09ecdd230" />
<img width="1763" height="872" alt="image" src="https://github.com/user-attachments/assets/df0d72ae-21f4-4a4c-9373-0a12eaab77dc" />
<img width="1217" height="883" alt="image" src="https://github.com/user-attachments/assets/c2cbe74e-5291-44c8-869d-033c533a07fe" />
<img width="1690" height="877" alt="image" src="https://github.com/user-attachments/assets/d9a1022a-c1a2-4261-8c55-ba2440b65307" />
---

## 🧰 Installation & Setup

### 1️⃣ Clone this Repository

git clone https://github.com/Thariga-yuvi/DS_Employee-Attrition-Analysis-and-Prediction.git
cd employee-attrition-dashboard

2️⃣ Create and Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate     # On Windows
source .venv/bin/activate  # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py

🗂️ Folder Structure
employee-insights-dashboard/
│
├── app.py                # Main Streamlit app
├── data/                 # Dataset folder (e.g., employees.csv)
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation

🪪 License
This project is licensed under the MIT License — you’re free to use and modify it for educational or commercial purposes.

🌟 Acknowledgements 

Special thanks to:
⦁	The IBM HR Analytics dataset for employee attrition.
⦁	Streamlit community for easy web app deployment.
⦁	Guvi IITM Data Science program for project guidance.

👩‍💻 Author
Thariga Charles
🎓 Data Science Professional | 📊 AI-Augmented Data Scientist Aspirant |
