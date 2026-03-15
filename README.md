# Trader Behavior vs Fear & Greed Analysis

A data analytics and machine learning project that explores how trader behavior and profitability change with market sentiment (Fear vs Greed).
The project includes end-to-end data preparation, behavioral analysis, trader segmentation, predictive modeling, and an interactive Streamlit dashboard.

---

## Project Overview

This project analyzes historical crypto trading data alongside the Fear & Greed Index to answer key questions:

* Do traders perform differently during Fear vs Greed?
* Does trader behavior change with market sentiment?
* Can we identify behavioral trader archetypes?
* Can we predict trader profitability using sentiment + behavior?

The final output is an interactive dashboard for exploration and insights.

---

## Repository Structure

```
   trader-behavior-analysis
 ┣ app.py                     # Streamlit dashboard
 ┣ analysis.ipynb             # exploratory notebook
 ┣ historical_data.csv        # trader data
 ┣ fear_greed_index.csv       # sentiment data
 ┗ README.md
```

---

## Tech Stack

* Python (Pandas, NumPy)
* Streamlit
* Plotly
* Scikit-learn
* Machine Learning (Random Forest)
* Data Visualization

---

## Setup Instructions

###  Clone the repository

```bash
git clone https://github.com/your-username/trader-behavior-analysis.git
cd trader-behavior-analysis
```

---


## How to Run the Dashboard

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually):

```
http://localhost:8501
```
### Screenshot of Dashboard:

#### Overview Tab
<img width="965" height="966" alt="image" src="https://github.com/user-attachments/assets/dfb306f2-aabc-45ad-ac7e-b9cda0aeb93f" />

#### Behavior Analysis Tab
<img width="1173" height="583" alt="image" src="https://github.com/user-attachments/assets/78b12321-c899-4dea-a023-51ba9a1d9fb5" />

#### Trader Archetypes Tab
<img width="1162" height="630" alt="image" src="https://github.com/user-attachments/assets/4c827f30-188d-4197-a977-c66b53671ae2" />

#### ML Prediction Tab
<img width="1147" height="675" alt="image" src="https://github.com/user-attachments/assets/b5d1dad7-2507-4b69-968c-03fc37f2e9b9" />



---

## Key Features

* KPI overview (PnL, Win Rate, Trade Size)
* Performance comparison: Fear vs Greed
* Trader behavior analysis
* Behavioral archetype segmentation
* ML model to predict win probability
* Interactive visual dashboard

---

## Methodology (Short Summary)

1. **Data Preparation**

   * Cleaned trading data
   * Converted timestamps to daily level
   * Merged with Fear & Greed index
   * Engineered key metrics (win rate, trade size, etc.)

2. **Behavioral Analysis**

   * Compared trader performance across sentiment regimes
   * Analyzed trade frequency and side bias
   * Built trader-level aggregates

3. **Segmentation**

   * Classified traders into archetypes:

     * Consistent Winners
     * High Leverage Gamblers
     * Overtraders
     * Smart Money
     * Casual Traders

4. **Predictive Modeling**

   * Random Forest classifier
   * Features: trade size + sentiment value
   * Target: trade win probability

---

## Key Insights

* Trader performance varies significantly across sentiment regimes.
* High trade frequency does not always correlate with profitability.
* A small group of consistent winners drives most profits.
* Sentiment combined with position sizing helps predict trade outcomes.

---

## Strategy Recommendations

**Rule 1:** During Fear periods, reduce position size for high-risk traders.
**Rule 2:** Consistent winners can increase activity during neutral/greed regimes.

---

## Future Improvements

* Real-time data pipeline
* Advanced time-series modeling
* Portfolio-level risk metrics
* Deployment to Streamlit Cloud
* Clustering with KMeans/DBSCAN

---

## Author

**Niyati Kalia**
Aspiring Data Analyst | ML Enthusiast

* SQL | Python | Power BI | Tableau | Machine Learning
* Passionate about data-driven decision making

---

## If you found this useful

Give the repo a star and feel free to fork!
