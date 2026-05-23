# Restaurant Chain Analysis

A data analytics and machine learning project built on a simulated restaurant chain operating across 5 major cities in Pakistan. The goal was to analyze business performance across multiple branches, understand customer behavior, predict weekly revenue, and present everything in an interactive Power BI dashboard.

---

## The Problem

A restaurant chain owner running 15 branches across Lahore, Karachi, Islamabad, Faisalabad, and Multan has no centralized way to monitor performance. Which branch is underperforming? Which food category drives the most revenue? What hours bring the most customers? And can we predict next week's revenue before it happens? This project answers all of that.

---

## Dataset

Since real restaurant chain data is not publicly available, a realistic synthetic dataset was generated using Python. The data was designed to reflect actual business patterns — peak hours, seasonal trends, city-wise differences, and customer behavior.

The dataset consists of 5 tables:

- **Branches** — 15 branches across 5 cities with details like area, seating capacity, and opening year
- **Menu Items** — 40 items across 8 categories (Burger, Pizza, Deal, Roll, Sides, Drinks, Dessert, Breakfast) with price and cost
- **Orders** — 100,000 orders from 2022 to 2024 with date, time, customer type, and payment method
- **Order Details** — 225,263 line items showing exactly what was ordered, quantity, discount, and profit
- **Master Table** — all tables joined into one for analysis and Power BI

---

## What Was Done

**Data Generation** — Realistic data was generated using NumPy and Pandas with controlled probability distributions to simulate real business patterns like peak hours, city-wise traffic, and seasonal trends.

**Data Preprocessing** — Date columns were converted to proper datetime format. New time features were extracted including week number, quarter, and weekend flag. Order-level revenue was aggregated and merged back into the master table.

**Exploratory Data Analysis** — Five key areas were analyzed with visualizations: revenue trends, branch performance, city comparison, customer behavior, and food category breakdown.

**Feature Engineering** — Weekly revenue was aggregated per branch. Lag features were created — previous week revenue and two weeks prior — so the model could learn from historical patterns without data leakage.

**Machine Learning** — Three models were trained to predict weekly branch revenue: Linear Regression, Random Forest, and Gradient Boosting. Data leakage was identified and fixed during the process. Gradient Boosting was selected as the final model.

**Power BI Dashboard** — An interactive two-page dashboard was built connecting all five data tables with proper relationships and DAX measures.

---

## Key Insights from EDA

**Revenue is stable** — Monthly revenue stayed between Rs. 5.4M and 6.4M consistently across 3 years with no major seasonal drops.

**Lahore dominates** — With 5 branches, Lahore generated Rs. 106M total — nearly half of all revenue. Karachi was second at Rs. 67M.

**Older branches earn more** — BR001 (Lahore Gulberg, opened 2018) was the top branch with Rs. 25M. Newer branches are still growing.

**Deals are the top category** — Rs. 73M revenue came from combo deals, followed by Pizza at Rs. 43M. Customers prefer value deals over individual items.

**8 PM is peak hour** — Evening dinner rush (7–10 PM) and lunch (12–2 PM) were clearly the busiest periods. 3–5 PM was the slowest.

**Cash still dominates** — Rs. 96.8M paid in cash. Card payments were second. JazzCash and EasyPaisa combined made Rs. 53M showing digital payments are growing.

**Feedback is consistent** — All 15 branches maintained a 3.9–4.0 average feedback score showing consistent quality across the chain.

---

## Machine Learning — Weekly Revenue Prediction

**Problem:** Predict how much revenue a branch will generate in the coming week.

**Type:** Regression

**Challenge — Data Leakage:** The first model runs produced R²=1.0 which looked perfect but was wrong. Two features were causing leakage — `total_rev_ever` contained future data and `revenue_trend` was directly calculated from the target variable. Both were removed.

**Models Compared:**

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.9818 | Rs. 5,212 |
| Random Forest | 0.9871 | Rs. 4,137 |
| **Gradient Boosting** | **0.9891** | **Rs. 3,952** |

**Final Result:** Gradient Boosting with R²=0.9891 and MAE of Rs. 3,952 — meaning predictions are off by only Rs. 3,952 on a weekly revenue of ~Rs. 140,000, which is a 2.8% error rate.

**Most Important Features:** Total items sold, total orders, and average order value were the strongest predictors of weekly revenue.

---

## Power BI Dashboard

Two-page interactive dashboard built in Power BI Desktop.

**Page 1 — Executive Overview**
- KPI cards showing Total Revenue, Total Profit, Total Orders, Profit Margin, and Avg Feedback
- Monthly Revenue trend line chart by city
- Map visualization showing revenue by city with bubble sizes
- Branch performance bar chart
- Revenue by category donut chart
- City and Area slicers for filtering

**Page 2 — Branch and ML Insights**
- Actual vs Predicted weekly revenue line chart
- Peak hours column chart
- Customer feedback gauge
- Branch and Year slicers

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Power BI Desktop

---

## How to Reproduce

Run `notebooks/data_generate.ipynb` first to generate all CSV files in the `data/` folder, then run `notebooks/script.ipynb` for preprocessing, EDA, and ML. Open `dashboard/restaurant_dashboard.pdf` to view the dashboard.
