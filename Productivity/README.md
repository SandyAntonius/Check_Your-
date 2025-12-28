# 📊 Social Media vs Productivity – Deep Learning Project

This project analyzes the relationship between social media usage and individual productivity using a **Multilayer Perceptron (MLP)** regression model.  
It predicts an **actual productivity score** based on digital habits, wellbeing indicators, and work behavior.

---

## 🚀 Project Overview

- **Type:** Regression problem
- **Model:** Multilayer Perceptron (MLP)
- **Framework:** TensorFlow / Keras
- **Deployment:** Flask REST API
- **Target Variable:** `actual_productivity_score`

---

## 🧠 Features Used

### 📱 Digital Habits
- Daily social media time
- Number of notifications
- Screen time before sleep

### 💚 Wellbeing
- Sleep hours
- Stress level
- Burnout days per month
- Weekly offline hours

### 💼 Work Habits
- Work hours per day
- Breaks during work
- Job satisfaction
- Perceived productivity

### 🧩 Engineered Features
- Social media × notifications
- Stress × burnout
- Total screen time
- Wellbeing score

---

## ⚙️ Model Architecture


- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Early stopping + learning rate scheduling applied

---

## 📈 Model Performance

| Dataset     | RMSE  | MAE  | R² Score |
|------------|------|------|----------|
| Training   | 0.56 | 0.42 | 0.91     |
| Validation | 0.57 | 0.43 | 0.91     |
| Test       | 0.58 | 0.44 | 0.90     |

✔️ Strong generalization  
✔️ No overfitting detected  

---

## 🌍 Real-World Insight

The model captures realistic scenarios where **high productivity can coexist with stress or burnout**, reflecting real workplace behavior rather than ideal assumptions.

---

## 🖥️ API Usage

### Run the server & installing the requirements:
```bash
pip install -r requirements.txt
python app.py


---

## 🔹 Step 5: Commit & Push (safe order)

```bash
git init
git add .
git commit -m "Initial commit - Social Media vs Productivity ML project"
git branch -M main
git remote add origin https://github.com/SandyAntonius/Check_Your📈.git
git push -u origin main



