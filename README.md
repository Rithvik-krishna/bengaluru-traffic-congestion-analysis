# Bengaluru Traffic Congestion Analysis and Peak Hour Detection

## 📋 Project Overview

Bengaluru, India's preeminent tech metropolis, is home to a rapidly growing population and a massive transportation network that faces extreme daily traffic congestion. This project performs an end-to-end data science analysis of historical traffic metrics, diagnosing peak hourly commute patterns, geographical hotspots, weather-induced delays, and volume-speed dynamics to propose a series of data-driven, strategic urban mobility recommendations.

## 🎯 Problem Statement

Bengaluru experiences severe traffic congestion during peak hours, which significantly impacts daily commute times, increases fuel consumption, and degrades localized environmental air quality. Rapid economic expansion has led to vehicular demand outstripping the capacity of historical roadways, creating chronic gridlock across major commercial and IT corridors. 

This analysis aims to identify the underlying structural causes of congestion by investigating the relationship between traffic volume, average speed, capacity utilization, and weather conditions. Understanding these patterns enables urban planners to move from reactive capacity extension to proactive, intelligent traffic management and demand-side solutions.

## 📊 Dataset

- **Total Records:** 8,936
- **Total Features:** 16
- **Time Range:** 2022-01-01 to 2024-08-09
- **Coverage:** 8 areas, 16 roads/intersections

### Features:
1. `date` - Date and hour of traffic observation.
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

## 🔧 Data Cleaning

- **Missing Value Handling:** The raw dataset had 0 missing values initially. However, our pipeline ensures robust imputation where numeric missing values are forward-filled (and backward-filled if still missing) and categorical missing values are mode-imputed.
- **Duplicate Removal:** Checked for duplicate records (0 duplicates found) and enforced deduplication in-place.
- **Column Standardization:** Converted all 16 raw column headers to lowercase and replaced spaces and slashes with underscores (e.g., "Road/Intersection Name" became "road_intersection_name").
- **DateTime Conversion:** Parsed the `date` string to datetime objects and extracted standard components: `hour`, `day_of_week`, `month`, and `date_only` to support temporal trend analysis.

## 📈 Exploratory Data Analysis

### Key Findings:
1. **Peak Traffic Hours:** N/A (Daily date granularity limitation). The source dataset contains only daily-level dates (YYYY-MM-DD), so sub-daily hourly peak commute volumes cannot be computed.
2. **Hotspot Areas:** Top congested areas are Koramangala (94.0%), M.G. Road (90.6%), and Indiranagar (87.6%).
3. **Weather Impact:** Average congestion is 80.54% under Rainy conditions vs 80.72% under Clear conditions (a difference of -0.23%).
4. **Weekday vs Weekend:** Weekday traffic volume averages 29,297.2 vehicles, which is 0.73% higher than weekend volume of 29,084.0 vehicles.
5. **Volume-Speed Relationship:** Moderate/weak negative correlation (-0.3411) showing average speeds drop by ~0.28 km/h for every 1000 vehicle increase.

## 📊 Visualizations

### 1. Traffic Volume Distribution (Histogram)
![Traffic Volume Distribution](outputs/histogram_traffic_volume.png)
Insight: Shows a right-skewed normal distribution centered around 29,000 vehicles, indicating stable average flow with significant outlier periods.

### 2. Congestion Level Distribution (Boxplot)
![Congestion Level Distribution](outputs/boxplot_congestion_level.png)
Insight: High median congestion across records with outlier events representing complete intersection lockup.

### 3. Chronological Monthly Traffic Trend (Line Chart)
![Monthly Traffic Trend](outputs/lineplot_monthly_traffic.png)
Insight: Displays monthly average traffic volume fluctuations over the 2022-2024 period, capturing long-term growth and seasonal trends. Hourly analysis could not be performed because the dataset only contains daily timestamps.

### 4. Traffic Volume vs Speed (Scatter Plot)
![Traffic Volume vs Speed](outputs/scatterplot_volume_speed.png)
Insight: Reveals speed collapsing from over 40 km/h to under 20 km/h once volume crosses a critical capacity threshold.

### 5. Top 10 Most Congested Areas (Bar Chart)
![Top 10 Most Congested Areas](outputs/barchart_top_congested_areas.png)
Insight: Koramangala (94.0%) and M.G. Road (90.6%) lead the city in gridlock density.

### 6. Weather Impact Analysis (Subplots)
![Weather Impact Analysis](outputs/weather_impact_analysis.png)
Insight: Average speed drops slightly under rainy conditions, while average congestion levels remain highly uniform (-0.23% difference between rainy and clear conditions).

### 7. Traffic Volume by Day of Week (Bar Chart)
![Traffic Volume by Day of Week](outputs/traffic_by_day_of_week.png)
Insight: Weekday traffic exceeds weekend traffic, indicating job commutes as the primary driver of city congestion.

## 💡 Key Insights

- **Peak Commute Limitation:** The dataset contains daily-level dates (YYYY-MM-DD), which prevents sub-daily peak commute hour modeling. Peak hours are marked as N/A.
- **Economic Hotspots:** Koramangala, M.G. Road, and Indiranagar are severe bottleneck hubs, requiring prioritized infrastructure and mass transit routing.
- **Weather Resilience:** Rainy conditions see traffic volume and congestion remain high and steady, indicating that demand does not drop, but average speeds decay slightly.

## 🎯 Recommendations

### Short-term (0-6 months):
- Deploy additional traffic personnel at the top 3 hotspot junctions during evening peaks.
- Implement weather-adaptive signal timing plans that adjust cycle length during rain.

### Medium-term (6-12 months):
- Encourage tech companies to implement staggered login/logout schedules or hybrid work policies.
- Establish dedicated bus lanes on major IT corridors.

### Long-term (1+ years):
- Expand metro lines to connect major residential hubs with Koramangala and M.G. Road.
- Implement congestion pricing for private vehicles entering the Central Business District during peak hours.

## ⚠️ Assumptions & Limitations

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

## 🚀 Future Improvements

1. Real-time traffic prediction using Machine Learning (ML).
2. Route optimization algorithms.
3. Integration with public transport data.
4. Incident detection system.
5. External events correlation.

## 📚 Technologies Used

- Python 3.11
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)
- Jupyter Notebook (analysis)
- HTML5, CSS3, & Vanilla JS (insights dashboard website)

## 📁 Project Structure

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

## 👨💻 Author

Applied Data Science Student  
June 2026

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests.

## 📧 Contact

For questions or suggestions, please contact the project maintainers.
