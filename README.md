# Credit Scoring Model – Bati Bank BNPL Partnership

##  Project Overview

Bati Bank is collaborating with a fast-growing eCommerce platform to launch a **Buy-Now-Pay-Later (BNPL)** service. This project focuses on building a **credit scoring system** that predicts the **creditworthiness** of customers using eCommerce behavioral data. The goal is to enable the bank to make **data-driven decisions** about extending short-term credit, with minimal default risk.

---

## Objectives

- Define a **proxy target variable** to simulate default risk.
- Select features that correlate well with customer credit behavior.
- Develop a model to:
  - Predict **risk probability**
  - Translate risk into an interpretable **credit score**
  - Recommend the **optimal loan amount and duration**
- Ensure the model is **interpretable, explainable, and Basel II-compliant**

---

## Credit Scoring Business Understanding

### What is Credit Risk?

**Credit Risk** is the risk that a borrower will fail to repay a loan, either in part or full, resulting in financial loss to the lender. In the BNPL context, credit risk arises when a customer makes a purchase on credit but doesn’t pay it back within the agreed period.

Key components:
- **Probability of Default (PD):** Likelihood a customer defaults.
- **Loss Given Default (LGD):** The portion of the loan the bank will lose if the customer defaults.
- **Exposure at Default (EAD):** The total credit amount owed at the time of default.

---

### Basel II Compliance & Model Interpretability

The **Basel II Capital Accord** emphasizes accurate, consistent, and explainable risk modeling to ensure **financial stability and transparency**. This means:
- The model must be **interpretable**, so regulators and bank officers can understand **why** a decision was made.
- All modeling steps must be **well-documented**, including assumptions, thresholds, and feature logic.
- The risk assessment must be based on **quantitative evidence**, not gut feeling.

In our case, this pushes us toward **transparent models** (like Logistic Regression with Weight of Evidence) and clear explanations of how risk scores are calculated.

---

### Why We Need a Proxy Variable

Since we **lack historical BNPL repayment/default data**, we can't train a model directly on real default labels. To move forward, we engineer a **proxy target variable** based on **RFM behavior**:

- **Recency:** How recently a user made a purchase
- **Frequency:** How often a user shops
- **Monetary:** How much money a user spends

We define a rule (e.g., customers with long recency, low frequency, and low spending = high risk) to label customers as **"likely to default" (1)** or **"low risk" (0)**.

#### Business Risks of Using a Proxy:
- **Misclassification:** We might falsely label a good customer as risky.
- **Bias:** Proxies may introduce bias based on shopping patterns unrelated to actual financial behavior.
- **Regulatory risk:** If the proxy is flawed and causes wrongful rejections or discrimination, it could lead to audits or legal action.

---

### Model Selection: Trade-Offs

| Model Type                   | Pros                                                | Cons                                                      | Suitability |
|-----------------------------|-----------------------------------------------------|-----------------------------------------------------------|-------------|
| Logistic Regression (with WoE) | - Interpretable<br>- Basel II-friendly<br>- Fast   | - May underperform on complex patterns                   |  Best for deployment |
| Gradient Boosting (e.g. XGBoost) | - High accuracy<br>- Handles nonlinearity & interactions | - Hard to explain<br>- Needs post-hoc interpretability tools |  Use for internal scoring or ensemble |

>  Recommended: Start with an interpretable baseline model. Enhance later with more complex models once validated.

---

## 🛠️ Project Structure

```plaintext
credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
## Author
**Meron Muluye**
Analytics Engineer @ Bati Bank

