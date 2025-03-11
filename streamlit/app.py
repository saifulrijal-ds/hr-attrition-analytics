import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="BFI Finance - Retention Analytics",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    employees = pd.read_csv('../data/processed/employees_engineered.csv')
    risk_scores = pd.read_csv('../data/processed/employee_risk_scores.csv')
    costs = pd.read_csv('../data/processed/attrition_costs.csv')
    interventions = pd.read_csv('../data/processed/intervention_plans.csv')
    
    # Merge datasets
    analysis_df = pd.merge(
        employees,
        risk_scores[['EmployeeID', 'AttritionProbability', 'RiskCategory']],
        on='EmployeeID',
        how='left'
    )
    
    analysis_df = pd.merge(
        analysis_df,
        costs[['EmployeeID', 'TotalCost', 'ExpectedCost']],
        on='EmployeeID',
        how='left'
    )
    
    return {
        'employees': employees,
        'risk_scores': risk_scores,
        'costs': costs,
        'interventions': interventions,
        'analysis_df': analysis_df
    }

data = load_data()

# Sidebar
st.sidebar.title("RetentionLens")
page = st.sidebar.radio("Navigation", ["Executive Summary", "Risk Analysis", "Financial Impact", "Intervention Planner"])

# Title
st.title("BFI Finance Indonesia HR Analytics")
st.markdown("## Quantifying and Reducing the Impact of Employee Attrition")

# Executive Summary Page
if page == "Executive Summary":
    # Layout with 3 columns for KPIs
    col1, col2, col3 = st.columns(3)
    
    # Calculate key metrics
    current_employees = len(data['employees'][~data['employees']['Attrition']])
    attrition_rate = data['employees']['Attrition'].mean() * 100
    high_risk_count = len(data['risk_scores'][data['risk_scores']['RiskCategory'].isin(['High', 'Very High'])])
    expected_annual_cost = data['costs']['ExpectedCost'].sum() / 1e9  # in billions
    
    with col1:
        st.metric("Current Employees", f"{current_employees:,}")
        st.metric("Historical Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col2:
        st.metric("High-Risk Employees", f"{high_risk_count:,}")
        st.metric("% of Workforce at High Risk", f"{high_risk_count/current_employees*100:.1f}%")
    
    with col3:
        st.metric("Expected Annual Attrition Cost", f"{expected_annual_cost:.2f} B IDR")
        
        # Calculate potential savings from interventions
        if 'interventions' in data and not data['interventions'].empty:
            potential_savings = data['interventions']['PotentialSavings'].sum() / 1e9  # in billions
            st.metric("Potential Savings from Interventions", f"{potential_savings:.2f} B IDR")
    
    # Department Overview
    st.markdown("## Department Overview")
    
    dept_metrics = data['analysis_df'].groupby('Department').agg(
        EmployeeCount=('EmployeeID', 'count'),
        AvgAttritionRisk=('AttritionProbability', 'mean'),
        HighRiskCount=('RiskCategory', lambda x: (x.isin(['High', 'Very High'])).sum()),
        ExpectedAnnualCost=('ExpectedCost', 'sum')
    ).sort_values('ExpectedAnnualCost', ascending=False)
    
    dept_metrics['HighRiskPercentage'] = dept_metrics['HighRiskCount'] / dept_metrics['EmployeeCount'] * 100
    
    # Create bar chart for department risk
    fig = px.bar(
        dept_metrics.reset_index(), 
        x='Department', 
        y='AvgAttritionRisk',
        color='HighRiskPercentage',
        color_continuous_scale='Reds',
        hover_data=['EmployeeCount', 'HighRiskCount', 'ExpectedAnnualCost'],
        title='Average Attrition Risk by Department',
        labels={'AvgAttritionRisk': 'Average Risk Score', 'HighRiskPercentage': 'High Risk %'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Drivers of Attrition
    st.markdown("## Key Drivers of Attrition")
    
    # Based on your SHAP analysis, show the key factors
    drivers_data = {
        'Factor': ['Years Since Hire', 'Years At Company', 'Distance From Home', 
                  'Engagement Score', 'Salary Ratio To Level', 'Performance Rating'],
        'Impact': [0.41, 0.36, 0.34, 0.32, 0.27, 0.20]
    }
    
    drivers_df = pd.DataFrame(drivers_data)
    
    fig = px.bar(
        drivers_df,
        x='Impact',
        y='Factor',
        orientation='h',
        title='Key Attrition Risk Factors (SHAP Values)',
        color='Impact',
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Risk Analysis Page
elif page == "Risk Analysis":
    st.markdown("## Employee Attrition Risk Analysis")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_filter = st.multiselect(
            "Department",
            options=sorted(data['employees']['Department'].unique()),
            default=sorted(data['employees']['Department'].unique())
        )
    
    with col2:
        level_filter = st.multiselect(
            "Job Level",
            options=sorted(data['employees']['JobLevel'].unique()),
            default=sorted(data['employees']['JobLevel'].unique())
        )
    
    with col3:
        risk_filter = st.multiselect(
            "Risk Category",
            options=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            default=['High', 'Very High']
        )
    
    # Filter data
    filtered_data = data['analysis_df'][
        (data['analysis_df']['Department'].isin(dept_filter)) &
        (data['analysis_df']['JobLevel'].isin(level_filter)) &
        (data['analysis_df']['RiskCategory'].isin(risk_filter))
    ]
    
    # Display risk distribution
    st.markdown(f"### Risk Distribution ({len(filtered_data)} employees)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk by department
        dept_risk = filtered_data.groupby('Department')['AttritionProbability'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=dept_risk.index,
            y=dept_risk.values,
            title='Average Risk by Department',
            labels={'x': 'Department', 'y': 'Average Risk Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by job level
        level_risk = filtered_data.groupby('JobLevel')['AttritionProbability'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=level_risk.index,
            y=level_risk.values,
            title='Average Risk by Job Level',
            labels={'x': 'Job Level', 'y': 'Average Risk Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Employee table with risk scores
    st.markdown("### High-Risk Employees")
    
    employee_table = filtered_data[[
        'EmployeeID', 'FirstName', 'LastName', 'Department', 'JobLevel', 
        'AttritionProbability', 'RiskCategory', 'MonthlyIncome', 'YearsAtCompany'
    ]].sort_values('AttritionProbability', ascending=False)
    
    employee_table['AttritionProbability'] = employee_table['AttritionProbability'].apply(lambda x: f"{x:.1%}")
    employee_table.columns = [
        'ID', 'First Name', 'Last Name', 'Department', 'Job Level', 
        'Risk Score', 'Risk Category', 'Monthly Salary (IDR M)', 'Years at Company'
    ]
    
    st.dataframe(employee_table, height=400)

# Financial Impact Page
elif page == "Financial Impact":
    st.markdown("## Financial Impact of Attrition")
    
    # Cost breakdown
    cost_breakdown = data['costs'][[
        'RecruitmentCost', 'VacancyCost', 'RampUpCost', 
        'KnowledgeTransferCost', 'TeamImpactCost', 'CustomerImpactCost',
        'OpportunityCost'
    ]].sum()
    
    cost_breakdown = cost_breakdown / 1e9  # Convert to billions
    
    # Create pie chart
    fig = px.pie(
        names=cost_breakdown.index,
        values=cost_breakdown.values,
        title='Attrition Cost Breakdown (Billions IDR)',
        labels={
            'names': 'Cost Category',
            'values': 'Amount (B IDR)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost by department and job level
    col1, col2 = st.columns(2)
    
    with col1:
        dept_cost = data['costs'].groupby('Department')['ExpectedCost'].sum().sort_values(ascending=False) / 1e9
        fig = px.bar(
            x=dept_cost.index,
            y=dept_cost.values,
            title='Expected Annual Attrition Cost by Department (B IDR)',
            labels={'x': 'Department', 'y': 'Expected Cost (B IDR)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        level_cost = data['costs'].groupby('JobLevel')['ExpectedCost'].sum().sort_values(ascending=False) / 1e9
        fig = px.bar(
            x=level_cost.index,
            y=level_cost.values,
            title='Expected Annual Attrition Cost by Job Level (B IDR)',
            labels={'x': 'Job Level', 'y': 'Expected Cost (B IDR)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost calculator for what-if scenarios
    st.markdown("### Attrition Cost Calculator")
    st.markdown("Estimate the cost impact of changing attrition rates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_select = st.selectbox(
            "Department",
            options=['All Departments'] + sorted(data['employees']['Department'].unique())
        )
    
    with col2:
        current_rate = data['employees']['Attrition'].mean() * 100
        new_rate = st.slider(
            "New Attrition Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(current_rate) * 0.75,  # 25% reduction as default target
            step=0.5
        )
    
    with col3:
        current_cost = data['costs']['ExpectedCost'].sum()
        if dept_select != 'All Departments':
            current_cost = data['costs'][data['costs']['Department'] == dept_select]['ExpectedCost'].sum()
        
        # Calculate new cost based on rate change
        new_cost = current_cost * (new_rate / current_rate) if current_rate > 0 else 0
        savings = current_cost - new_cost
        
        st.metric(
            "Estimated Annual Savings",
            f"{savings/1e9:.2f} B IDR",
            f"{(1 - new_rate/current_rate)*100:.1f}%"
        )

# Intervention Planner Page
elif page == "Intervention Planner":
    st.markdown("## Retention Intervention Strategies")
    
    if 'interventions' in data and not data['interventions'].empty:
        # Strategy effectiveness
        strategy_metrics = data['interventions'].groupby('Strategy').agg(
            EmployeeCount=('EmployeeID', 'nunique'),
            AvgROI=('ROI', 'mean'),
            TotalCost=('InterventionCost', 'sum'),
            TotalSavings=('PotentialSavings', 'sum')
        ).sort_values('TotalSavings', ascending=False)
        
        # Convert to millions
        strategy_metrics['TotalCost'] = strategy_metrics['TotalCost'] / 1e6
        strategy_metrics['TotalSavings'] = strategy_metrics['TotalSavings'] / 1e6
        
        # Plot ROI by strategy
        fig = px.bar(
            strategy_metrics.reset_index(),
            x='Strategy',
            y='AvgROI',
            color='TotalSavings',
            color_continuous_scale='Greens',
            title='Intervention Strategy ROI Analysis',
            labels={'AvgROI': 'Average ROI', 'Strategy': 'Intervention Strategy'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost-benefit analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                strategy_metrics.reset_index(),
                x='Strategy',
                y='TotalCost',
                title='Total Cost by Strategy (M IDR)',
                labels={'TotalCost': 'Total Cost (M IDR)', 'Strategy': 'Intervention Strategy'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                strategy_metrics.reset_index(),
                x='Strategy',
                y='TotalSavings',
                title='Expected Savings by Strategy (M IDR)',
                labels={'TotalSavings': 'Expected Savings (M IDR)', 'Strategy': 'Intervention Strategy'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Intervention plans by priority
        st.markdown("### Recommended Intervention Plans")
        priority_filter = st.selectbox(
            "Priority Level",
            options=['All Priorities', 'High', 'Medium', 'Low'],
            index=0
        )
        
        filtered_interventions = data['interventions']
        if priority_filter != 'All Priorities':
            filtered_interventions = data['interventions'][data['interventions']['Priority'] == priority_filter]
        
        # Get employee details
        employee_details = data['employees'][['EmployeeID', 'FirstName', 'LastName', 'Department', 'JobLevel']]
        intervention_details = pd.merge(
            filtered_interventions,
            employee_details,
            on='EmployeeID',
            how='left'
        )
        
        # Format for display
        display_cols = [
            'EmployeeID', 'FirstName', 'LastName', 'Department', 'JobLevel',
            'Strategy', 'Intervention', 'CurrentRisk', 'NewRisk',
            'InterventionCost', 'PotentialSavings', 'ROI', 'Priority'
        ]
        
        display_table = intervention_details[display_cols].sort_values(['Priority', 'ROI'], ascending=[True, False])
        
        # Format percentages and currency
        display_table['CurrentRisk'] = display_table['CurrentRisk'].apply(lambda x: f"{x:.1%}")
        display_table['NewRisk'] = display_table['NewRisk'].apply(lambda x: f"{x:.1%}")
        display_table['InterventionCost'] = display_table['InterventionCost'].apply(lambda x: f"{x/1e6:.1f}M IDR")
        display_table['PotentialSavings'] = display_table['PotentialSavings'].apply(lambda x: f"{x/1e6:.1f}M IDR")
        display_table['ROI'] = display_table['ROI'].apply(lambda x: f"{x:.2f}")
        
        # Rename columns for display
        display_table.columns = [
            'ID', 'First Name', 'Last Name', 'Department', 'Job Level',
            'Strategy', 'Intervention', 'Current Risk', 'New Risk',
            'Cost', 'Savings', 'ROI', 'Priority'
        ]
        
        st.dataframe(display_table, height=400)
    else:
        st.warning("Intervention data is not available. Please generate intervention plans first.")
