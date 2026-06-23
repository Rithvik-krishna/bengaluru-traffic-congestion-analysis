# Bengaluru Traffic Congestion Analysis

An end-to-end data science analysis of historical traffic metrics across Bengaluru, diagnosing geographical hotspots, weather-induced delays, volume-speed dynamics, and weekly cycles.

## Project Overview

Bengaluru, India's preeminent tech metropolis, is home to a rapidly growing population and a massive transportation network that faces extreme daily traffic congestion. This project performs a detailed data science analysis of historical traffic metrics, diagnosing congestion hotspots, weather-induced delays, and volume-speed dynamics to propose a series of data-driven, strategic urban mobility conclusions.

## Problem Statement

Bengaluru experiences severe traffic congestion during peak hours, which significantly impacts daily commute times, increases fuel consumption, and degrades localized environmental air quality. Rapid economic expansion has led to vehicular demand outstripping the capacity of historical roadways, creating chronic gridlock across major commercial and IT corridors.

This analysis aims to identify the underlying structural causes of congestion by investigating the relationship between traffic volume, average speed, capacity utilization, and weather conditions. Understanding these patterns enables urban planners to move from reactive capacity extension to proactive, intelligent traffic management and demand-side solutions.

## Dataset

- **Total Records:** 8,936
- **Total Features:** 16
- **Time Range:** 2022-01-01 to 2024-08-09
- **Coverage:** 8 areas, 16 roads/intersections

### Features:
1. `date` - Date of traffic observation (YYYY-MM-DD).
2. `area_name` - The locality/area in Bengaluru (e.g., Koramangala, Indiranagar).
3. `road_intersection_name` - The specific intersection or road name.
4. `traffic_volume` - Estimated vehicle count during the observation interval.
5. `average_speed` - Average speed of vehicles traversing the road in km/h.
6. `travel_time_index` - Ratio of travel time during congestion compared to free-flow conditions.
7. `congestion_level` - Estimated percentage level of congestion (0% - 100%).
8. `road_capacity_utilization` - The ratio of volume to maximum road capacity.
9. `incident_reports` - Number of active traffic incidents (crashes, breakdowns) reported.
10. `environmental_impact` - Estimated emissions index calculated from volume and speed.
11. `public_transport_usage` - Estimated share of public transport commuters on this corridor.
12. `traffic_signal_compliance` - Average signal compliance index.
13. `parking_usage` - Average parking slot utilization rate.
14. `pedestrian_and_cyclist_count` - Non-motorized user count in the zone.
15. `weather_conditions` - Weather at observation (Clear, Rain, Fog, Overcast, Windy).
16. `roadwork_and_construction_activity` - Active construction markers (Yes / No).

## Project Methodology

1. **Dataset Collection:** Daily traffic observations across major routes.
2. **Data Cleaning:** Validating nulls, duplicates, and standardizing schemas.
3. **Feature Engineering:** Extracting day of week, month, quarter, and year temporal features.
4. **Exploratory Data Analysis:** Analyzing statistical properties and correlations.
5. **Visualization:** Generating high-resolution plots mapping urban trends.
6. **Synthesis:** Drawing structured, data-grounded conclusions.

## Data Cleaning

- **Missing Value Handling:** Verified raw dataset had 0 missing values initially. The pipeline ensures robust imputation where numeric missing values are forward-filled (and backward-filled if still missing) and categorical missing values are mode-imputed.
- **Duplicate Removal:** Checked for duplicate records (0 duplicates found) and enforced deduplication in-place.
- **Column Standardization:** Converted all 16 raw column headers to lowercase and replaced spaces and slashes with underscores (e.g., "Road/Intersection Name" became "road_intersection_name").
- **DateTime Conversion:** Parsed the `date` string to datetime objects and extracted standard components: `day_of_week`, `month`, `quarter`, and `year` to support temporal trend analysis. Hourly analysis was omitted since the dataset contains daily-level granularity.

## Exploratory Data Analysis

### Visualizations:

1. **Traffic Volume Distribution (Histogram)**
   ![Traffic Volume Distribution](outputs/histogram_traffic_volume.png)
   Insight: Volume follows a right-skewed normal distribution, averaging 29,236 vehicles, with a wide standard deviation.

2. **Congestion Level Distribution (Boxplot)**
   ![Congestion Level Distribution](outputs/boxplot_congestion_level.png)
   Insight: Median congestion level is high at 92.39%, representing chronic gridlock as the standard baseline state.

3. **Chronological Monthly Traffic Trend (Line Chart)**
   ![Monthly Traffic Trend](outputs/lineplot_monthly_traffic.png)
   Insight: Captures monthly average traffic volume fluctuations over the 2022-2024 period, capturing long-term growth and seasonal trends.

4. **Traffic Volume vs Speed (Scatter Plot)**
   ![Traffic Volume vs Speed](outputs/scatterplot_volume_speed.png)
   Insight: Average speed decays rapidly and non-linearly as traffic volume increases, showing a correlation coefficient of -0.3411.

5. **Top 10 Most Congested Areas (Bar Chart)**
   ![Top 10 Most Congested Areas](outputs/barchart_top_congested_areas.png)
   Insight: Congestion is highly concentrated around major commercial and IT hubs, with Koramangala (93.99%), M.G. Road (90.58%), and Indiranagar (87.64%) being the top spots.

6. **Weather Impact Analysis (Subplots)**
   ![Weather Impact Analysis](outputs/weather_impact_analysis.png)
   Insight: Congestion level averages remain uniform across clear and rainy weather conditions (-0.23% difference), showing that capacity constraints drive gridlock more than weather.

7. **Traffic Volume by Day of Week (Bar Chart)**
   ![Traffic Volume by Day of Week](outputs/traffic_by_day_of_week.png)
   Insight: Weekday traffic exceeds weekend traffic marginally (0.73% difference), showing congestion is persistent throughout the entire week.

## Research Findings & Conclusions

- **Finding 1 (Weekly Persistence):** Weekend volume is only 0.73% lower than weekdays, showing congestion is persistent throughout the entire week.
- **Finding 2 (Geographic Hotspots):** Congestion is heavily concentrated in major IT/commercial hubs: Koramangala (93.99%), M.G. Road (90.58%), and Indiranagar (87.64%).
- **Finding 3 (Weather Impact Assessment):** Weather conditions have negligible impact (rain averages 80.54% vs clear at 80.72%, a difference of -0.23%).
- **Finding 4 (Volume-Speed Correlation):** Strong negative correlation (-0.3411) showing average speed collapses to a floor of 20 km/h under high volumes.
- **Future Work:** Deployment of sub-daily telemetric traffic logs and integration of public transit metro/bus passenger flows.

## Assumptions & Limitations

### Assumptions:
1. Dataset accurately represents Bengaluru traffic patterns.
2. Weather information is correctly recorded and consistent.
3. Traffic volume directly correlates with congestion levels.
4. Data collection methodology remained constant.
5. External factors are randomly distributed.

### Limitations:
1. NO HOURLY TIMESTAMPS: The dataset contains daily-level dates (YYYY-MM-DD), preventing sub-daily morning/evening peak hour trend calculations.
2. Geographical coverage is restricted to 8 key areas in Bengaluru.
3. Specific traffic incidents (crashes, construction) are represented as daily aggregate counts.
4. Air quality indices are calculated metrics rather than directly measured.
5. Integration with city public transport network volumes is not included.

## Technologies Used

- Python 3.11
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)
- Jupyter Notebook (analysis)
- HTML5, CSS3, & Vanilla JS (insights dashboard website)

## Project Structure

```
bengaluru-traffic-congestion-analysis/
├── raw_data/
│   └── Banglore_traffic_Dataset.csv
├── processed_data/
│   └── cleaned_traffic_data.csv
├── notebooks/
│   └── traffic_analysis.ipynb
├── outputs/
│   ├── histogram_traffic_volume.png
│   ├── boxplot_congestion_level.png
│   ├── lineplot_monthly_traffic.png
│   ├── scatterplot_volume_speed.png
│   ├── barchart_top_congested_areas.png
│   ├── weather_impact_analysis.png
│   └── traffic_by_day_of_week.png
├── website/
│   ├── index.html
│   └── style.css
├── docs/
│   ├── key_findings.txt
│   ├── assumptions_limitations.txt
│   └── executive_summary.txt
├── README.md
└── requirements.txt
```

## Author

Rithvik Krishna DK  
June 2026

## License

This project is for educational purposes.

---

## Machine Learning Extension

A machine learning module has been added to predict **Congestion Level** from the existing traffic features. All ML work is contained in a new branch (`feature/ml-extension`) and does not modify the original analysis.

### Problem Statement

Given a set of daily traffic observations — including volume, speed, weather, road capacity utilisation, and area — predict the **congestion level** (a continuous percentage value from 0–100%). This is a **supervised regression** problem because the target variable is continuous.

### Models Used

| Model | Type | Role |
|---|---|---|
| Linear Regression | Parametric | Baseline model |
| Random Forest Regressor | Ensemble (200 trees) | Primary predictive model |

### Evaluation Metrics

Three standard regression metrics were used:
- **MAE** (Mean Absolute Error) — average absolute prediction error in percentage points.
- **RMSE** (Root Mean Squared Error) — penalises large errors more heavily.
- **R² Score** — proportion of variance in congestion level explained by the model.

### Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 5.3636 | 6.6590 | 0.9180 |
| **Random Forest** | **2.8543** | **4.3355** | **0.9653** |

The Random Forest model achieved a higher R² score and lower prediction error, making it the preferred model for congestion prediction.

### Feature Importance

Top 10 most important features (Random Forest Gini importance):

| Rank | Feature | Importance |
|---|---|---|
| 1 | Traffic Volume | 0.5270 |
| 2 | Environmental Impact | 0.4388 |
| 3 | Road Capacity Utilization | 0.0071 |
| 4 | Parking Usage | 0.0040 |
| 5 | Public Transport Usage | 0.0039 |
| 6 | Average Speed | 0.0038 |
| 7 | Pedestrian and Cyclist Count | 0.0036 |
| 8 | Traffic Signal Compliance | 0.0035 |
| 9 | Incident Reports | 0.0027 |
| 10 | Travel Time Index | 0.0026 |

Traffic Volume and Environmental Impact together account for ~96.5% of total feature importance.

### ML Outputs

- `outputs/ml_actual_vs_predicted.png` — Scatter plot: Actual vs Predicted congestion levels.
- `outputs/ml_feature_importance.png` — Horizontal bar chart: Top 10 feature importances.
- `docs/ml_metrics.json` — Machine-readable metrics for website integration.
- `docs/ml_viva_notes.txt` — Professor-ready Q&A documentation.
- `notebooks/ml_congestion_prediction.py` — Standalone ML training script.

### Future Improvements

1. **Hourly Data Integration**: Sub-daily timestamps would enable peak-hour prediction models.
2. **Cross-Validation**: K-fold CV would yield more robust generalization estimates.
3. **Gradient Boosting**: XGBoost or LightGBM could further reduce RMSE.
4. **Time-Series Models**: LSTM or ARIMA could capture day-to-day temporal dependencies.
5. **Hyperparameter Tuning**: GridSearchCV / RandomizedSearchCV on Random Forest parameters.
6. **Real-Time Deployment**: Wrap the model in a FastAPI endpoint for live congestion prediction.

### Updated Project Structure

```
bengaluru-traffic-congestion-analysis/
├── raw_data/
│   └── Banglore_traffic_Dataset.csv
├── processed_data/
│   └── cleaned_traffic_data.csv
├── notebooks/
│   ├── traffic_analysis.ipynb
│   └── ml_congestion_prediction.py          ← NEW
├── outputs/
│   ├── histogram_traffic_volume.png
│   ├── boxplot_congestion_level.png
│   ├── lineplot_monthly_traffic.png
│   ├── scatterplot_volume_speed.png
│   ├── barchart_top_congested_areas.png
│   ├── weather_impact_analysis.png
│   ├── traffic_by_day_of_week.png
│   ├── ml_actual_vs_predicted.png           ← NEW
│   └── ml_feature_importance.png            ← NEW
├── website/
│   ├── index.html                           ← UPDATED (ML section added)
│   └── style.css                            ← UPDATED (ML styles added)
├── docs/
│   ├── key_findings.txt
│   ├── assumptions_limitations.txt
│   ├── executive_summary.txt
│   ├── insights_data.json
│   ├── ml_metrics.json                      ← NEW
│   └── ml_viva_notes.txt                    ← NEW
├── README.md                                ← UPDATED
└── requirements.txt                         ← UPDATED (scikit-learn added)
```
