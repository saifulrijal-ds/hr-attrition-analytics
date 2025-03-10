# HR Analytics Data Dictionary

This document describes the data fields available in each dataset generated for the HR Analytics project.

## Employee Data

The `employees.csv` file contains the main employee information.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| EmployeeID | Unique identifier for each employee | Integer | 10001, 10002 |
| FirstName | Employee's first name | String | Budi, Dewi |
| LastName | Employee's last name | String | Wijaya, Setiawan |
| Age | Employee's age in years | Integer | 28, 35, 42 |
| Gender | Employee's gender | String | Male, Female |
| OfficeLocation | Office location | String | Jakarta, Surabaya, Bandung |
| DistanceFromHome | Distance from home to office in kilometers | Float | 5.2, 12.8 |
| Department | Department the employee works in | String | Sales, Collections, HR |
| JobRole | Specific job title | String | Sales Representative, Collections Manager |
| JobLevel | Job level category | String | Entry Level, Junior, Mid-Level, Senior, Manager, Director, Executive |
| Education | Highest education level | String | High School, Diploma, Bachelor's Degree, Master's Degree, PhD |
| FieldOfStudy | Field of study | String | Business Administration, Finance, Economics |
| HireDate | Date the employee was hired | Date | 2020-05-15 |
| YearsAtCompany | Years the employee has been at the company | Float | 2.5, 4.7 |
| YearsSinceLastPromotion | Years since the employee's last promotion | Integer | 0, 1, 3 |
| JobSatisfaction | Job satisfaction rating (1-5 scale) | Float | 3.5, 4.0 |
| EnvironmentSatisfaction | Environment satisfaction rating (1-5 scale) | Integer | 3, 4, 5 |
| WorkLifeBalance | Work-life balance rating (1-5 scale) | Integer | 2, 3, 4 |
| PerformanceRating | Performance rating (1-5 scale) | Integer | 3, 4, 5 |
| Overtime | Whether the employee works overtime | Boolean | True, False |
| MonthlyIncome | Monthly income in IDR millions | Float | 7.5, 15.2, 40.6 |
| AnnualIncome | Annual income in IDR millions (including 13th month) | Float | 97.5, 197.6 |
| TrainingTimesLastYear | Hours of training in the last year | Integer | 25, 40, 60 |
| Attrition | Whether the employee has left the company | Boolean | True, False |
| ExitDate | Date the employee left (if applicable) | Date | 2022-08-30 |

## Engagement Survey Data

The `surveys.csv` file contains employee engagement survey responses.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| SurveyID | Unique identifier for each survey response | Integer | 1, 2, 3 |
| EmployeeID | Employee identifier (foreign key) | Integer | 10001, 10002 |
| SurveyDate | Date the survey was completed | Date | 2022-06-15 |
| SurveyItem | Survey question/statement | String | "I am satisfied with my job" |
| Score | Response score (1-5 scale) | Integer | 3, 4, 5 |

## Performance Data

The `performance.csv` file contains quarterly performance metrics for employees.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| EmployeeID | Employee identifier (foreign key) | Integer | 10001, 10002 |
| Year | Performance review year | Integer | 2021, 2022 |
| Quarter | Performance review quarter | Integer | 1, 2, 3, 4 |
| TenureYears | Employee tenure at time of review | Float | 1.25, 3.5 |
| PerformanceScore | Overall performance score (1-5 scale) | Float | 3.5, 4.2 |
| GoalAchievement | Percentage of goals achieved | Float | 85.5, 102.3 |
| AbsentDays | Days absent in the quarter | Integer | 0, 1, 3 |
| LatenessInstances | Number of late arrivals in the quarter | Integer | 0, 2, 5 |
| SalesTarget | Sales target in IDR millions (Sales roles) | Float | 800.0, 1500.0 |
| SalesAchievement | Actual sales in IDR millions (Sales roles) | Float | 750.0, 1650.0 |
| CustomersTarget | Target number of customers (Sales roles) | Integer | 15, 25 |
| CustomersAcquired | Actual number of customers (Sales roles) | Integer | 12, 28 |
| DefaultRate | Loan default rate percentage (Sales roles) | Float | 1.5, 3.2 |
| RecoveryRate | Percentage of debts recovered (Collections roles) | Float | 72.5, 85.3 |
| AmountCollected | Amount collected in IDR millions (Collections roles) | Float | 500.0, 950.0 |
| ProductivityScore | Productivity score (Operations/CS/IT/Finance roles) | Float | 85.2, 92.5 |
| ErrorRate | Error rate percentage (Operations/CS/IT/Finance roles) | Float | 1.2, 3.5 |

## Promotion History

The `promotions.csv` file contains employee promotion records.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| EmployeeID | Employee identifier (foreign key) | Integer | 10001, 10002 |
| PromotionDate | Date of promotion | Date | 2021-04-15 |
| PreviousJobLevel | Job level before promotion | String | Junior, Mid-Level |
| NewJobLevel | Job level after promotion | String | Mid-Level, Senior |
| PreviousJobRole | Job role before promotion | String | Sales Representative |
| NewJobRole | Job role after promotion | String | Senior Sales Representative |
| SalaryIncreasePct | Percentage salary increase | Float | 12.5, 18.3 |

## Training Data

The `training.csv` file contains employee training records.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| TrainingID | Unique identifier for each training record | Integer | 1, 2, 3 |
| EmployeeID | Employee identifier (foreign key) | Integer | 10001, 10002 |
| CourseName | Name of training course | String | "Sales Techniques for Financial Products" |
| TrainingFormat | Format of the training | String | Online Course, Classroom Training, Workshop |
| CompletionDate | Date training was completed | Date | 2022-03-10 |
| DurationHours | Duration of training in hours | Integer | 4, 8, 16 |
| HasCertificate | Whether a certificate was earned | Boolean | True, False |
| Score | Training score if applicable | Float | 85.5, 92.0 |
| Cost | Training cost in IDR | Float | 1500000, 3000000 |

## Exit Interview Data

The `exit_interviews.csv` file contains data from exit interviews with former employees.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| EmployeeID | Employee identifier (foreign key) | Integer | 10001, 10002 |
| ExitDate | Date employee left the company | Date | 2022-08-30 |
| InterviewDate | Date of exit interview | Date | 2022-09-02 |
| PrimaryExitReason | Primary reason for leaving | String | Better Compensation Elsewhere, Career Advancement Opportunity |
| SecondaryFactors | Secondary factors contributing to decision | String | "Inadequate salary, Limited growth opportunities" |
| DestinationCompany | Company employee is moving to | String | Bank Mandiri, CIMB Niaga |
| WouldRecommendCompany | Likelihood to recommend (1-5 scale) | Float | 3.5, 4.0 |
| OverallExperience | Overall experience rating (1-5 scale) | Float | 3.0, 4.5 |
| ManagerEffectiveness | Manager effectiveness rating (1-5 scale) | Float | 2.5, 4.0 |
| CompensationSatisfaction | Compensation satisfaction rating (1-5 scale) | Float | 2.0, 3.5 |
| BenefitsSatisfaction | Benefits satisfaction rating (1-5 scale) | Float | 2.5, 3.5 |
| WorkLifeBalanceSatisfaction | Work-life balance satisfaction rating (1-5 scale) | Float | 2.0, 4.0 |

## Recruitment Cost Data

The `recruitment_costs.csv` file contains recruitment cost benchmarks by job level.

| Field | Description | Data Type | Example Values |
|-------|-------------|-----------|---------------|
| JobLevel | Job level category | String | Entry Level, Junior, Mid-Level, Senior, Manager, Director, Executive |
| AverageAnnualSalary | Average annual salary in IDR | Float | 78000000, 195000000 |
| TotalRecruitmentCost | Total recruitment cost in IDR | Float | 11700000, 58500000 |
| RecruitmentCostPct | Recruitment cost as percentage of annual salary | Float | 15.0, 30.0 |
| AdvertisingCost | Advertising cost in IDR | Float | 1755000, 8775000 |
| AgencyFees | Agency fees in IDR | Float | 4680000, 23400000 |
| InterviewCosts | Interview costs in IDR | Float | 1755000, 8775000 |
| AssessmentCosts | Assessment costs in IDR | Float | 877500, 8775000 |
| AdministrativeCosts | Administrative costs in IDR | Float | 877500, 5850000 |
| RelocationCosts | Relocation costs in IDR | Float | 0, 5850000 |
| SigningBonus | Signing bonus in IDR | Float | 0, 8775000 |