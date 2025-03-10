# RetentionLens: HR Analytics
## Project Overview

This project implements an HR analytics solution to quantify and reduce the financial impact of employee attrition. Using data science and machine learning techniques, the project aims to:

1. Quantify the full financial impact of employee attrition
2. Identify at-risk employees before they leave
3. Determine key drivers of turnover
4. Recommend targeted interventions with measurable ROI

## Project Structure

The project follows a modular structure:

- `notebooks/`: Jupyter notebooks for exploration and development
- `src/`: Source code modules for data processing, modeling, financial analysis, and visualization
- `data/`: Data storage (raw, processed, and models)
- `docs/`: Project documentation
- `streamlit/`: Streamlit dashboard application

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/hr-attrition-analytics.git
   cd hr-attrition-analytics
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

To run the Streamlit dashboard:

```
cd streamlit
streamlit run app.py
```

## Implementation Plan

The project implementation follows an 8-week plan:

- Weeks 1-2: Foundation & Data Preparation
- Weeks 3-4: Feature Engineering & Modeling
- Weeks 5-6: Financial Analysis & Prototype Development
- Weeks 7-8: Validation & Refinement

## Documentation

For detailed information, see the documentation in the `docs/` directory:

- Business Case
- Data Dictionary
- Model Methodology
- User Guide