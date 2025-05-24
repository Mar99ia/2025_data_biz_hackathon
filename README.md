# OOH Delivery Infrastructure Analysis

## Project Overview
This project analyzes the Out-of-Home (OOH) delivery infrastructure in Poland, focusing on identifying market gaps and opportunities for expansion. The analysis includes various types of pickup points and their characteristics, with a special focus on identifying underserved areas with high market potential.

## Project Structure
```
├── data/                  # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── src/                  # Source code
│   ├── data/            # Data processing scripts
│   ├── analysis/        # Analysis scripts
│   ├── visualization/   # Visualization scripts
│   └── models/          # Machine learning models
├── notebooks/           # Jupyter notebooks for exploration
├── dashboard/           # Streamlit dashboard
└── docs/               # Documentation
```

## Setup Instructions
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Features
- Interactive map visualization of OOH delivery points
- Market gap analysis using demographic and economic data
- Location profiling and scoring system
- Executive dashboard with key insights
- Machine learning models for market potential prediction

## Data Sources
- InPost API data
- OpenStreetMap
- GUS (Polish Central Statistical Office)
- Eurostat
- Other open-source data sources

## Technologies Used
- Python
- Pandas & GeoPandas
- Folium for map visualization
- Streamlit for interactive dashboard
- Scikit-learn for machine learning
- PostgreSQL/PostGIS for spatial data (optional)

## Team
[Your Team Name]

## License
MIT License