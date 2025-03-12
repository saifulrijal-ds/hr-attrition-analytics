# Business Understanding: HR Analytics at BFI Finance Indonesia

## Background

BFI Finance Indonesia, one of the leading multi-finance companies in Indonesia, has experienced significant growth over the past decade, expanding its branch network and service offerings across the archipelago. This growth has led to a substantial increase in workforce size and complexity, with operations spanning urban centers like Jakarta to smaller regional offices.

Despite its market success, BFI Finance faces a persistent challenge with employee attrition. Over the past four years, the company has experienced a cumulative attrition rate of 22.9%, which exceeds industry benchmarks and creates significant operational and financial strain. This turnover is particularly pronounced in customer-facing departments, with Customer Service (25.0%), Collections (24.5%), and Sales (24.0%) experiencing the highest departure rates.

The company's HR department has traditionally relied on exit interviews and manager observations to understand attrition. However, these reactive approaches have proven insufficient for developing effective retention strategies. Leadership recognized the need for a more sophisticated, data-driven approach that could identify at-risk employees before they leave and quantify the full financial impact of turnover.

## Problem Statement

BFI Finance Indonesia faces several interconnected challenges related to employee attrition:

1. **Financial Burden**: Each employee departure costs an average of 105 million IDR when accounting for recruitment, training, lost productivity, and knowledge transfer costs. With hundreds of departures annually, this represents over 93 billion IDR in avoidable expenses.

2. **Knowledge Drain**: Employees most likely to leave are those with 2-3 years of experience—precisely when they've developed valuable institutional knowledge and client relationships but before they've fully realized their potential value to the organization.

3. **Customer Impact**: In client-facing roles, employee departures directly affect customer relationships, potentially impacting loan renewals, customer satisfaction, and ultimately revenue generation.

4. **Intervention Inefficiency**: Without data-driven insights, retention efforts have been applied broadly rather than targeted at specific risk factors or high-value employees, resulting in suboptimal resource allocation.

5. **Reactive Posture**: The current approach to retention is reactive—addressing departures after they occur rather than proactively identifying and mitigating attrition risks.

## Project Objectives

The HR Analytics project aims to transform BFI Finance's approach to employee retention through these specific objectives:

1. **Quantify Financial Impact**: Develop a comprehensive attrition cost calculator that accounts for all direct and indirect costs associated with employee turnover, providing a clear business case for retention investments.

2. **Predict Flight Risk**: Build predictive models that identify employees at risk of leaving with meaningful accuracy (our LightGBM model currently achieves 34.5% recall), enabling proactive intervention.

3. **Understand Root Causes**: Apply advanced analytics techniques like SHAP analysis to uncover the true drivers of attrition—revealing that tenure (YearsSinceHire, YearsAtCompany), commute distance (DistanceFromHome), engagement metrics, and compensation equity (SalaryRatioToLevel) are the most significant factors.

4. **Develop Targeted Interventions**: Create data-informed intervention strategies with ROI projections, showing that engagement initiatives (9.05 ROI) and tenure experience programs (8.33 ROI) offer the highest returns while compensation adjustments (0.83 ROI) remain necessary despite lower efficiency.

5. **Enable Strategic Decision-Making**: Provide an interactive dashboard that guides HR and management through the analytical journey from descriptive understanding to prescriptive action.

## Expected Business Outcomes

The implementation of this HR Analytics solution is expected to deliver several concrete business benefits:

1. **Cost Reduction**: A targeted 5% reduction in overall attrition rates would save approximately 20.1 billion IDR annually in direct costs.

2. **Performance Continuity**: Retention of high-performing employees (particularly those with performance ratings of 4+) will preserve the 3× productivity advantage these individuals demonstrate.

3. **Revenue Protection**: Maintaining stability in customer-facing roles will protect an estimated 15.3 billion IDR in revenue that would otherwise be at risk from relationship disruption.

4. **Talent Development ROI**: Improved retention will increase the return on the approximately 2.8 billion IDR invested annually in employee training and development.

5. **Strategic Resource Allocation**: Data-driven interventions will allow more efficient allocation of the HR budget, with expected intervention costs of 5.8 billion IDR generating 19.95 billion IDR in savings.

## Analytics Approach

To address these challenges, the project employs a comprehensive analytical approach spanning the full spectrum from descriptive to prescriptive analytics:

1. **Descriptive Analytics**: Analyzing historical attrition patterns, department-specific trends, and financial impacts to understand "what happened"

2. **Diagnostic Analytics**: Applying SHAP values, correlation analysis, and exit interview examination to determine "why it happened"

3. **Predictive Analytics**: Developing machine learning models (LogisticRegression, XGBoost, and LightGBM) to forecast "what will happen" regarding future attrition

4. **Prescriptive Analytics**: Formulating targeted intervention strategies with ROI projections to address "what should be done" to optimize retention

This end-to-end analytical framework transforms raw HR data into actionable business intelligence, enabling BFI Finance Indonesia to address its attrition challenges comprehensively and strategically, with clear financial and operational benefits.