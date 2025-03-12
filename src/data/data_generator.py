"""
Data Generator for BFI Finance Indonesia HR Analytics

This module creates realistic synthetic HR data for BFI Finance Indonesia,
with characteristics typical for an Indonesian lending company.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class HRDataGenerator:
    """
    Generate synthetic HR data for analytics and modeling.
    
    This class creates realistic employee data with attributes relevant
    for attrition prediction and analysis in an Indonesian context.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the data generator with seed for reproducibility.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define constants for data generation
        self.current_date = datetime.now()
        
        # Indonesian names
        self.indonesian_first_names = [
            "Andi", "Budi", "Citra", "Dewi", "Eko", "Fitri", "Gatot", "Hadi", 
            "Indah", "Joko", "Kartini", "Lukman", "Maya", "Nia", "Putri", "Rina", 
            "Sari", "Tono", "Umar", "Vina", "Wati", "Yanto", "Zainal", "Agus", 
            "Bambang", "Dedi", "Endang", "Faisal", "Gita", "Hendra", "Irfan", 
            "Johan", "Kurnia", "Laila", "Muhamad", "Nurul", "Oki", "Pratiwi", 
            "Ridwan", "Siti", "Tuti", "Ujang", "Wahyu", "Yuli", "Zainab", "Ahmad",
            "Bagus", "Dian", "Edi", "Farida", "Gunawan", "Hartono", "Iwan"
        ]
        
        self.indonesian_last_names = [
            "Wijaya", "Suharto", "Sukarno", "Habibie", "Widodo", "Sudiarto", 
            "Kusuma", "Hidayat", "Nugroho", "Santoso", "Wibowo", "Sutanto", 
            "Budiman", "Susanto", "Setiawan", "Kurniawan", "Agustina", "Purnama", 
            "Utama", "Saputra", "Siregar", "Tampubolon", "Hutapea", "Hutagalung", 
            "Aruan", "Sinaga", "Lubis", "Sihombing", "Situmorang", "Purba", 
            "Siagian", "Manullang", "Panjaitan", "Aritonang", "Hutabarat", 
            "Simanjuntak", "Nasution", "Batubara", "Harahap", "Ritonga", 
            "Nainggolan", "Hasibuan", "Pakpahan", "Simatupang", "Samosir", 
            "Tamba", "Siahaan", "Naibaho", "Marpaung", "Pardede"
        ]
        
        # Indonesian cities for office locations
        self.indonesian_cities = [
            "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", 
            "Makassar", "Palembang", "Tangerang", "Depok", "Bekasi",
            "Bogor", "Yogyakarta", "Malang", "Bali", "Batam"
        ]
        
        # Population distribution skews toward Jakarta and major cities
        self.city_weights = [
            0.25, 0.15, 0.10, 0.10, 0.08, 
            0.07, 0.05, 0.05, 0.04, 0.03,
            0.03, 0.02, 0.01, 0.01, 0.01
        ]
        
        # Departments typical for a finance company
        self.departments = {
            'Sales': 0.25,
            'Collections': 0.20,
            'Operations': 0.15,
            'Customer Service': 0.10,
            'Finance': 0.08,
            'Risk Management': 0.08,
            'IT': 0.06,
            'HR': 0.04,
            'Legal': 0.02,
            'Executive': 0.02
        }
        
        # Job roles by department
        self.job_roles = {
            'Sales': [
                'Sales Representative', 'Account Manager', 'Sales Manager', 
                'Branch Sales Supervisor', 'Regional Sales Director'
            ],
            'Collections': [
                'Collections Officer', 'Collections Supervisor', 'Collections Manager',
                'Recovery Specialist', 'Recovery Manager'
            ],
            'Operations': [
                'Loan Processor', 'Operations Staff', 'Operations Supervisor',
                'Branch Operations Manager', 'Regional Operations Director'
            ],
            'Customer Service': [
                'Customer Service Representative', 'Customer Service Supervisor',
                'Customer Experience Manager', 'Call Center Agent', 'Call Center Supervisor'
            ],
            'Finance': [
                'Finance Officer', 'Accountant', 'Financial Analyst',
                'Finance Manager', 'Financial Controller'
            ],
            'Risk Management': [
                'Credit Analyst', 'Risk Analyst', 'Compliance Officer',
                'Risk Manager', 'Chief Risk Officer'
            ],
            'IT': [
                'IT Support', 'Software Developer', 'System Administrator',
                'IT Project Manager', 'IT Director'
            ],
            'HR': [
                'HR Assistant', 'HR Officer', 'Recruitment Specialist',
                'HR Manager', 'HR Director'
            ],
            'Legal': [
                'Legal Assistant', 'Legal Officer', 'Corporate Lawyer',
                'Legal Manager', 'General Counsel'
            ],
            'Executive': [
                'Executive Assistant', 'Department Head', 'Director',
                'Vice President', 'C-Level Executive'
            ]
        }
        
        # Education levels with distribution for Indonesia's finance sector
        self.education_levels = {
            'High School': 0.15,
            'Diploma': 0.25,
            'Bachelor\'s Degree': 0.50,
            'Master\'s Degree': 0.09,
            'PhD': 0.01
        }
        
        # Fields of study
        self.fields_of_study = {
            'Business Administration': 0.25,
            'Finance': 0.20,
            'Economics': 0.15,
            'Accounting': 0.15,
            'Information Technology': 0.10,
            'Engineering': 0.05,
            'Law': 0.05,
            'Other': 0.05
        }
        
        # Performance ratings
        self.performance_ratings = {
            1: 0.05,  # Below Expectations
            2: 0.15,  # Meets Some Expectations
            3: 0.50,  # Meets Expectations
            4: 0.25,  # Exceeds Expectations
            5: 0.05   # Exceptional
        }
        
        # Salary ranges by job level (in IDR millions per month)
        self.salary_ranges = {
            'Entry Level': (3.5, 8),       # Entry level
            'Junior': (8, 15),             # Junior level
            'Mid-Level': (15, 25),         # Mid-level
            'Senior': (25, 40),            # Senior level
            'Manager': (40, 60),           # Manager level
            'Director': (60, 100),         # Director level
            'Executive': (100, 200)        # Executive level
        }
        
        # Attrition factors with weights
        self.attrition_factors = {
            'low_performance': 3.0,
            'low_salary': 2.5,
            'long_commute': 1.5,
            'low_job_satisfaction': 2.7,
            'limited_growth': 2.3,
            'work_life_balance': 2.0,
            'years_since_promotion': 1.8,
            'overtime': 1.5,
            'age_factor': 0.5,  # Younger employees tend to leave more
            'tenure_factor': 1.0,  # Short tenure increases attrition risk
        }

        self.desired_attrition_rate=0.5

    def generate_employee_data(self, num_employees=1000, historical_years=3):
        """
        Generate employee data for the specified number of employees.
        
        Args:
            num_employees (int): Number of employees to generate
            historical_years (int): Years of historical data to generate
            
        Returns:
            pandas.DataFrame: DataFrame containing employee data
        """
        # Historical start date
        historical_start = self.current_date - timedelta(days=365 * historical_years)
        
        # Generate base employee data
        employees = []
        
        # Employee ID counter
        employee_id = 10001
        
        # Generate current employees and past employees (who have left)
        total_to_generate = int(num_employees * 1.5)  # Generate 50% more to account for attrition
        
        current_employees_count = 0
        departed_employees_count = 0

        for i in range(total_to_generate):
            # Basic employee info
            first_name = random.choice(self.indonesian_first_names)
            last_name = random.choice(self.indonesian_last_names)
            
            # Gender (slightly more males in Indonesian finance sector)
            gender = random.choices(['Male', 'Female'], weights=[0.55, 0.45])[0]
            
            # Age distribution skewed toward younger workforce
            if random.random() < 0.7:  # 70% younger employees
                age = random.randint(22, 35)
            else:
                age = random.randint(36, 55)
            
            # Location - office city
            office_location = random.choices(self.indonesian_cities, weights=self.city_weights)[0]
            
            # Calculate distance from home (in km)
            if office_location == "Jakarta":
                # Jakarta has longer commutes
                distance_from_home = random.choices(
                    [random.uniform(1, 5), random.uniform(5, 15), random.uniform(15, 40)],
                    weights=[0.2, 0.5, 0.3]
                )[0]
            else:
                # Other cities have shorter commutes
                distance_from_home = random.choices(
                    [random.uniform(1, 3), random.uniform(3, 10), random.uniform(10, 25)],
                    weights=[0.3, 0.5, 0.2]
                )[0]
            
            # Department
            department = random.choices(
                list(self.departments.keys()),
                weights=list(self.departments.values())
            )[0]
            
            # Job role based on department
            job_role = random.choice(self.job_roles[department])
            
            # Job level
            if 'Director' in job_role or 'C-Level' in job_role:
                job_level = 'Director' if 'Director' in job_role else 'Executive'
            elif 'Manager' in job_role:
                job_level = 'Manager'
            elif 'Supervisor' in job_role or 'Senior' in job_role:
                job_level = 'Senior'
            elif any(word in job_role for word in ['Specialist', 'Analyst', 'Officer']):
                job_level = 'Mid-Level'
            elif any(word in job_role for word in ['Assistant', 'Support']):
                job_level = 'Junior'
            else:
                job_level = 'Entry Level'
            
            # Education
            education = random.choices(
                list(self.education_levels.keys()),
                weights=list(self.education_levels.values())
            )[0]
            
            # Field of study (dependent on education)
            if education == 'High School':
                field_of_study = 'High School'
            else:
                field_of_study = random.choices(
                    list(self.fields_of_study.keys()),
                    weights=list(self.fields_of_study.values())
                )[0]
            
            # Hire date
            max_tenure_days = (self.current_date - historical_start).days
            tenure_days = random.randint(1, max_tenure_days)
            hire_date = self.current_date - timedelta(days=tenure_days)
            
            # Initial job satisfaction and environment satisfaction (1-5 scale)
            initial_job_satisfaction = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.05, 0.10, 0.25, 0.40, 0.20]
            )[0]
            
            initial_environment_satisfaction = random.choices(
                [1, 2, 3, 4, 5],
                weights=[0.05, 0.15, 0.30, 0.35, 0.15]
            )[0]
            
            # Calculate job satisfaction (decreases over time if not promoted)
            years_since_promotion = random.randint(0, min(5, max(1, int(tenure_days / 365))))
            job_satisfaction = max(1, initial_job_satisfaction - 0.5 * (years_since_promotion / 2))
            
            # Environment satisfaction
            environment_satisfaction = initial_environment_satisfaction
            
            # Performance rating (1-5 scale)
            performance_rating = random.choices(
                list(self.performance_ratings.keys()),
                weights=list(self.performance_ratings.values())
            )[0]
            
            # Overtime (more common in Sales and Collections)
            if department in ['Sales', 'Collections']:
                overtime_probability = 0.4
            else:
                overtime_probability = 0.2
            
            overtime = random.random() < overtime_probability
            
            # Work-life balance rating (1-5 scale, lower if overtime)
            if overtime:
                work_life_balance = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            else:
                work_life_balance = random.choices([3, 4, 5], weights=[0.3, 0.5, 0.2])[0]
            
            # Calculate monthly salary in IDR millions
            min_salary, max_salary = self.salary_ranges[job_level]
            
            # Adjust salary based on tenure and performance
            tenure_years = tenure_days / 365
            tenure_multiplier = min(1.5, 1 + tenure_years * 0.05)  # Up to 50% increase for tenure
            performance_multiplier = 0.8 + performance_rating * 0.1  # 0.9 to 1.3 based on rating
            
            # Base salary randomized within range, then adjusted
            base_salary = random.uniform(min_salary, max_salary)
            monthly_salary = base_salary * tenure_multiplier * performance_multiplier
            
            # Annual salary (IDR millions)
            annual_salary = monthly_salary * 13  # 13 months including THR (Religious Holiday Allowance)
            
            # Training time in last year (hours)
            if job_level in ['Entry Level', 'Junior']:
                training_times = random.randint(20, 80)
            elif job_level in ['Mid-Level', 'Senior']:
                training_times = random.randint(15, 60)
            else:
                training_times = random.randint(10, 40)
            
            # Calculate attrition risk factors
            attrition_score = (
                (6 - performance_rating) * self.attrition_factors['low_performance'] + 
                (min_salary / monthly_salary) * self.attrition_factors['low_salary'] +
                (distance_from_home / 10) * self.attrition_factors['long_commute'] +
                (6 - job_satisfaction) * self.attrition_factors['low_job_satisfaction'] +
                (years_since_promotion) * self.attrition_factors['years_since_promotion'] +
                (6 - work_life_balance) * self.attrition_factors['work_life_balance'] +
                (overtime * 1.0) * self.attrition_factors['overtime'] +
                (max(0, (30 - age) / 10)) * self.attrition_factors['age_factor'] +
                (max(0, (2 - tenure_years) / 2)) * self.attrition_factors['tenure_factor']
            )
            
            # Normalize attrition score to probability (0-1)
            max_possible_score = 30 # Increased from 30 to reduce probabilities
            attrition_probability = min(0.90, attrition_score / max_possible_score)

            # Further reduce attrition probability to get more realistic rates
            attrition_probability *= 0.8
            
            # Determine if employee has left
            attrition = False
            exit_date = None

            # Check if we should consider this employee for attrition based on current counts
            current_total_employees = current_employees_count + departed_employees_count
            current_attrition_rate = departed_employees_count / max(1, current_total_employees)

            desired_attrition_rate=self.desired_attrition_rate

            # Skip creating any more departed employees if we're already over the target attrition rate
            # and we have enough current employees
            if current_employees_count >= num_employees and current_attrition_rate >= desired_attrition_rate:
                attrition = False
            # Otherwhise, apply the calculated attrition probability    
            if random.random() < attrition_probability:
                # This employee has left
                attrition = True
                # Exit date before current date
                max_exit_days = min(tenure_days, 365 * 2)  # Most leave within 2 years
                
                # Ensure max_exit_days is at least 30
                if max_exit_days < 30:
                    exit_days = max_exit_days  # If less than 30, just use the max available
                else:
                    exit_days = random.randint(30, max_exit_days)
                
                exit_date = self.current_date - timedelta(days=exit_days)
            
            # Only include this employee if they're still active or left within our timeframe
            if not attrition or exit_date >= historical_start:
                employee = {
                    'EmployeeID': employee_id,
                    'FirstName': first_name,
                    'LastName': last_name,
                    'Age': age,
                    'Gender': gender,
                    'OfficeLocation': office_location,
                    'DistanceFromHome': round(distance_from_home, 2),
                    'Department': department,
                    'JobRole': job_role,
                    'JobLevel': job_level,
                    'Education': education,
                    'FieldOfStudy': field_of_study,
                    'HireDate': hire_date,
                    'YearsAtCompany': round(tenure_days / 365, 2),
                    'YearsSinceLastPromotion': years_since_promotion,
                    'JobSatisfaction': round(job_satisfaction, 1),
                    'EnvironmentSatisfaction': environment_satisfaction,
                    'WorkLifeBalance': work_life_balance,
                    'PerformanceRating': performance_rating,
                    'Overtime': overtime,
                    'MonthlyIncome': round(monthly_salary, 2),
                    'AnnualIncome': round(annual_salary, 2),
                    'TrainingTimesLastYear': training_times,
                    'Attrition': attrition
                }
                
                if exit_date:
                    employee['ExitDate'] = exit_date
                    departed_employees_count += 1
                else:
                    current_employees_count += 1
                
                employees.append(employee)
                employee_id += 1
                
                # Break if we've reached the desired number of current employees and a reasonable number of departed employees
                target_departed = int(num_employees * desired_attrition_rate)
                if current_employees_count >= num_employees and departed_employees_count >= target_departed:
                    break
        
        # Convert to DataFrame
        df = pd.DataFrame(employees)

        print(f"Generated {len(df)} total employees:")
        print(f" - Current employees: {current_employees_count}")
        print(f" - Departed employees: {departed_employees_count}")
        print(f" - Attrition rate: {departed_employees_count / len(df):.2%}")
        
        return df
    
    def generate_engagement_survey_data(self, employee_df):
        """
        Generate engagement survey data for employees.
        
        Args:
            employee_df (pandas.DataFrame): Employee data DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame containing survey results
        """
        # Survey items
        survey_items = [
            "I am satisfied with my job",
            "I am proud to work for BFI Finance",
            "I would recommend BFI Finance as a great place to work",
            "My work gives me a sense of accomplishment",
            "I have opportunities to learn and grow at BFI Finance",
            "My manager provides useful feedback about my performance",
            "I believe there are good career opportunities for me at BFI Finance",
            "The leadership team communicates a clear vision for the company",
            "I feel valued as an employee",
            "My opinions and ideas are respected",
            "I have the resources I need to do my job effectively",
            "The company's benefits package meets my needs",
            "My salary is fair compared to similar roles in other companies",
            "I can maintain a healthy work-life balance",
            "I plan to be working at BFI Finance in two years"
        ]
        
        # Generate survey responses for each employee
        survey_data = []
        survey_id = 1
        
        for _, employee in employee_df.iterrows():
            # Skip employees who left more than 6 months ago
            if employee['Attrition'] and hasattr(employee, 'ExitDate'):
                if (self.current_date - employee['ExitDate']).days > 180:
                    continue
            
            # Base sentiment affected by job satisfaction and attrition risk
            base_sentiment = (
                employee['JobSatisfaction'] + 
                employee['EnvironmentSatisfaction'] + 
                employee['WorkLifeBalance']
            ) / 3
            
            # Survey date - random date in last year for active employees,
            # or before exit date for departed employees
            if employee['Attrition'] and 'ExitDate' in employee:
                max_days = min(180, (employee['ExitDate'] - timedelta(days=30) - 
                                   (self.current_date - timedelta(days=365))).days)
                survey_days_ago = random.randint(0, max(1, max_days))
                survey_date = employee['ExitDate'] - timedelta(days=30) - timedelta(days=survey_days_ago)
            else:
                survey_days_ago = random.randint(0, 365)
                survey_date = self.current_date - timedelta(days=survey_days_ago)
            
            # Response for each survey item
            for item in survey_items:
                # Adjust score based on item content and employee attributes
                score_adjustment = 0
                
                # Satisfaction related
                if 'satisfied' in item.lower() or 'satisfaction' in item.lower():
                    score_adjustment += (employee['JobSatisfaction'] - 3) * 0.5
                
                # Career/growth related
                if any(word in item.lower() for word in ['opportunities', 'career', 'grow']):
                    score_adjustment -= (employee['YearsSinceLastPromotion'] * 0.2)
                
                # Manager/leadership related
                if any(word in item.lower() for word in ['manager', 'leadership']):
                    mgmt_sentiment = random.uniform(-0.5, 0.5)  # Random manager-specific adjustment
                    score_adjustment += mgmt_sentiment
                
                # Salary related
                if any(word in item.lower() for word in ['salary', 'compensation', 'benefits']):
                    # High variation in compensation satisfaction
                    if employee['JobLevel'] in ['Entry Level', 'Junior'] and employee['MonthlyIncome'] < 10:
                        score_adjustment -= random.uniform(0.5, 1.5)
                    elif employee['PerformanceRating'] >= 4 and employee['YearsSinceLastPromotion'] >= 2:
                        score_adjustment -= random.uniform(0.5, 1.0)
                
                # Work-life balance related
                if 'work-life balance' in item.lower():
                    score_adjustment += (employee['WorkLifeBalance'] - 3) * 0.5
                    if employee['Overtime']:
                        score_adjustment -= random.uniform(0.5, 1.5)
                
                # Future intent related
                if 'plan to be working' in item.lower():
                    if employee['Attrition']:
                        score_adjustment -= random.uniform(1.0, 2.0)
                    else:
                        loyalty_factor = min(employee['YearsAtCompany'] * 0.2, 1.0)
                        score_adjustment += loyalty_factor
                
                # Calculate final score (1-5 scale)
                raw_score = base_sentiment + score_adjustment + random.uniform(-0.75, 0.75)
                score = max(1, min(5, round(raw_score)))
                
                survey_data.append({
                    'SurveyID': survey_id,
                    'EmployeeID': employee['EmployeeID'],
                    'SurveyDate': survey_date,
                    'SurveyItem': item,
                    'Score': score
                })
                
                survey_id += 1
        
        # Convert to DataFrame
        survey_df = pd.DataFrame(survey_data)
        
        return survey_df
    
    def generate_performance_data(self, employee_df, quarters=12):
        """
        Generate quarterly performance data for employees.
        
        Args:
            employee_df (pandas.DataFrame): Employee data DataFrame
            quarters (int): Number of quarters of historical data to generate
            
        Returns:
            pandas.DataFrame: DataFrame containing performance data
        """
        performance_data = []
        
        for _, employee in employee_df.iterrows():
            hire_date = employee['HireDate']
            
            # Calculate exit date if employee has left
            exit_date = None
            if employee['Attrition'] and 'ExitDate' in employee:
                exit_date = employee['ExitDate']
            
            # Performance baseline is influenced by performance rating
            performance_baseline = employee['PerformanceRating']
            
            # Generate quarterly data
            for q in range(quarters):
                # Quarter end date
                quarter_date = self.current_date - timedelta(days=90 * (quarters - q - 1))
                quarter_year = quarter_date.year
                quarter_num = (quarter_date.month - 1) // 3 + 1
                
                # Skip if employee wasn't hired yet or had already left
                if quarter_date < hire_date:
                    continue
                if exit_date and quarter_date > exit_date:
                    continue
                
                # Calculate tenure at this quarter
                tenure_days = (quarter_date - hire_date).days
                tenure_years = tenure_days / 365
                
                # Performance metrics
                
                # Performance score (influenced by baseline but with random variation)
                performance_trend = min(1.0, tenure_years * 0.05)  # Improvement with experience
                random_factor = random.uniform(-0.5, 0.5)
                performance_score = max(1, min(5, performance_baseline + performance_trend + random_factor))
                
                # Goal achievement (percentage, based on performance)
                goal_achievement = min(120, max(60, (performance_score / 5) * 100 + random.uniform(-10, 10)))
                
                # Sales metrics for sales roles
                if 'Sales' in employee['Department'] or 'sales' in employee['JobRole'].lower():
                    # Sales performance (% of target)
                    if performance_score >= 4:
                        sales_factor = random.uniform(1.0, 1.3)  # Exceeds target
                    elif performance_score >= 3:
                        sales_factor = random.uniform(0.85, 1.1)  # Meets or slightly exceeds target
                    else:
                        sales_factor = random.uniform(0.6, 0.9)  # Below target
                        
                    # Target depends on job level
                    if employee['JobLevel'] == 'Entry Level':
                        sales_target = random.uniform(300, 500)  # IDR millions
                    elif employee['JobLevel'] == 'Junior':
                        sales_target = random.uniform(500, 800)
                    elif employee['JobLevel'] == 'Mid-Level':
                        sales_target = random.uniform(800, 1200)
                    elif employee['JobLevel'] == 'Senior':
                        sales_target = random.uniform(1200, 2000)
                    else:  # Manager+
                        sales_target = random.uniform(2000, 5000)
                    
                    sales_achievement = sales_target * sales_factor
                    
                    # Customer acquisition
                    customers_target = int(sales_target / 50)  # Rough estimate of customers per IDR 50M
                    customers_acquired = int(customers_target * sales_factor)
                    
                    # Default rates depend on performance (lower is better)
                    if performance_score >= 4:
                        default_rate = random.uniform(0.5, 2.0)
                    elif performance_score >= 3:
                        default_rate = random.uniform(1.5, 3.5)
                    else:
                        default_rate = random.uniform(3.0, 6.0)
                else:
                    sales_target = None
                    sales_achievement = None
                    customers_target = None
                    customers_acquired = None
                    default_rate = None
                
                # Collections metrics for collections roles
                if 'Collections' in employee['Department'] or 'collections' in employee['JobRole'].lower():
                    # Recovery rate
                    if performance_score >= 4:
                        recovery_rate = random.uniform(75, 95)
                    elif performance_score >= 3:
                        recovery_rate = random.uniform(60, 80)
                    else:
                        recovery_rate = random.uniform(40, 65)
                    
                    # Amount collected
                    if employee['JobLevel'] == 'Entry Level':
                        target_collection = random.uniform(200, 400)  # IDR millions
                    elif employee['JobLevel'] == 'Junior':
                        target_collection = random.uniform(400, 700)
                    elif employee['JobLevel'] == 'Mid-Level':
                        target_collection = random.uniform(700, 1000)
                    elif employee['JobLevel'] == 'Senior':
                        target_collection = random.uniform(1000, 1500)
                    else:  # Manager+
                        target_collection = random.uniform(1500, 3000)
                    
                    collection_factor = recovery_rate / 70  # Normalized to 1.0 at 70% recovery
                    amount_collected = target_collection * collection_factor
                else:
                    recovery_rate = None
                    amount_collected = None
                
                # Productivity metrics for operations and other roles
                if employee['Department'] in ['Operations', 'Customer Service', 'IT', 'Finance']:
                    # Productivity score (1-100)
                    productivity_base = (performance_score / 5) * 100
                    productivity_score = max(50, min(100, productivity_base + random.uniform(-15, 15)))
                    
                    # Error rate (inverse of performance)
                    error_rate_base = 10 - (performance_score * 1.5)
                    error_rate = max(0.5, min(15, error_rate_base + random.uniform(-2, 2)))
                else:
                    productivity_score = None
                    error_rate = None
                
                # Attendance metrics for all employees
                absent_days_base = 3 - (performance_score * 0.4)  # Base absence days per quarter
                absent_days = max(0, round(absent_days_base + random.uniform(-1, 2)))
                
                # Lateness instances
                lateness_base = 4 - (performance_score * 0.5)
                lateness_instances = max(0, round(lateness_base + random.uniform(-1, 3)))
                
                performance_record = {
                    'EmployeeID': employee['EmployeeID'],
                    'Year': quarter_year,
                    'Quarter': quarter_num,
                    'TenureYears': round(tenure_years, 2),
                    'PerformanceScore': round(performance_score, 1),
                    'GoalAchievement': round(goal_achievement, 1),
                    'AbsentDays': absent_days,
                    'LatenessInstances': lateness_instances
                }
                
                # Add role-specific metrics if applicable
                if sales_target is not None:
                    performance_record.update({
                        'SalesTarget': round(sales_target, 2),
                        'SalesAchievement': round(sales_achievement, 2),
                        'CustomersTarget': customers_target,
                        'CustomersAcquired': customers_acquired,
                        'DefaultRate': round(default_rate, 2)
                    })
                
                if recovery_rate is not None:
                    performance_record.update({
                        'RecoveryRate': round(recovery_rate, 2),
                        'AmountCollected': round(amount_collected, 2)
                    })
                
                if productivity_score is not None:
                    performance_record.update({
                        'ProductivityScore': round(productivity_score, 1),
                        'ErrorRate': round(error_rate, 2)
                    })
                
                performance_data.append(performance_record)
        
        # Convert to DataFrame
        performance_df = pd.DataFrame(performance_data)
        
        return performance_df
    
    def generate_promotion_history(self, employee_df):
        """
        Generate promotion history for employees.
        
        Args:
            employee_df (pandas.DataFrame): Employee data DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame containing promotion history
        """
        promotion_data = []
        
        for _, employee in employee_df.iterrows():
            hire_date = employee['HireDate']
            years_at_company = employee['YearsAtCompany']
            
            # Skip if employee has been at company less than 1 year
            if years_at_company < 1:
                continue
            
            # Number of promotions based on years at company and performance
            max_promotions = min(5, int(years_at_company / 1.5))
            
            # Adjust based on performance rating
            if employee['PerformanceRating'] >= 4:
                promotion_factor = 1.2
            elif employee['PerformanceRating'] == 3:
                promotion_factor = 0.9
            else:
                promotion_factor = 0.6
            
            num_promotions = max(0, min(max_promotions, round(max_promotions * promotion_factor)))
            
            # For higher performers, ensure at least one promotion if they've been there 2+ years
            if employee['PerformanceRating'] >= 4 and years_at_company >= 2 and num_promotions == 0:
                num_promotions = 1
            
            # Generate promotion records
            if num_promotions > 0:
                # Spread promotions across tenure
                tenure_days = years_at_company * 365
                
                # Generate promotion dates
                promotion_days = sorted(random.sample(
                    range(180, int(tenure_days)), 
                    min(num_promotions, int(tenure_days / 180))
                ))
                
                for i, days in enumerate(promotion_days):
                    promotion_date = hire_date + timedelta(days=days)
                    
                    # Skip if promotion date is in the future
                    if promotion_date > self.current_date:
                        continue
                    
                    # Determine promotion details based on job level progression
                    if i == 0:
                        # First promotion
                        prev_level = "Entry Level" if employee['JobLevel'] != "Entry Level" else "Junior"
                        prev_role = employee['JobRole'].replace("Senior ", "").replace("Manager", "Officer")
                    else:
                        # Subsequent promotions
                        prev_record = promotion_data[-1]
                        prev_level = prev_record['NewJobLevel']
                        prev_role = prev_record['NewJobRole']
                    
                    # Level progression
                    level_progression = {
                        "Entry Level": "Junior",
                        "Junior": "Mid-Level",
                        "Mid-Level": "Senior",
                        "Senior": "Manager",
                        "Manager": "Director",
                        "Director": "Executive"
                    }
                    
                    # Determine new level
                    if i == len(promotion_days) - 1:
                        # Last promotion should match current level
                        new_level = employee['JobLevel']
                    else:
                        # Intermediate promotion
                        new_level = level_progression.get(prev_level, prev_level)
                    
                    # Determine new role based on level
                    if "Manager" in employee['JobRole'] and new_level != "Manager":
                        new_role = employee['JobRole'].replace("Manager", "Supervisor")
                    elif "Senior" in employee['JobRole'] and new_level != "Senior":
                        new_role = employee['JobRole'].replace("Senior ", "")
                    elif new_level == "Manager" and "Manager" not in employee['JobRole']:
                        new_role = prev_role.replace("Supervisor", "Manager")
                    elif new_level == "Senior" and "Senior" not in employee['JobRole']:
                        new_role = "Senior " + prev_role
                    else:
                        new_role = prev_role
                    
                    # Salary change (10-30% increase)
                    salary_increase_pct = random.uniform(10, 30)
                    
                    promotion_record = {
                        'EmployeeID': employee['EmployeeID'],
                        'PromotionDate': promotion_date,
                        'PreviousJobLevel': prev_level,
                        'NewJobLevel': new_level,
                        'PreviousJobRole': prev_role,
                        'NewJobRole': new_role,
                        'SalaryIncreasePct': round(salary_increase_pct, 2)
                    }
                    
                    promotion_data.append(promotion_record)
        
        # Convert to DataFrame
        promotion_df = pd.DataFrame(promotion_data)
        
        return promotion_df
    
    def generate_training_data(self, employee_df):
        """
        Generate training history for employees.
        
        Args:
            employee_df (pandas.DataFrame): Employee data DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame containing training history
        """
        # List of training courses by department
        training_courses = {
            'Sales': [
                'Sales Techniques for Financial Products', 
                'Customer Relationship Management', 
                'Negotiation Skills',
                'Product Knowledge: Leasing Solutions',
                'Product Knowledge: Working Capital Solutions',
                'Client Needs Assessment',
                'Closing Techniques',
                'Digital Selling Strategies'
            ],
            'Collections': [
                'Effective Collections Strategies',
                'Negotiation Skills for Collections',
                'Legal Aspects of Debt Collection',
                'Customer Communication in Collections',
                'Telephone Collections Techniques',
                'Handling Difficult Customers',
                'Payment Arrangement Strategies',
                'Collections Ethics and Compliance'
            ],
            'Operations': [
                'Loan Processing Procedures',
                'Document Verification Techniques',
                'Operational Risk Management',
                'Process Optimization',
                'Workflow Management',
                'Quality Assurance in Operations',
                'Regulatory Compliance for Operations',
                'Digital Document Management'
            ],
            'Customer Service': [
                'Customer Service Excellence',
                'Handling Customer Complaints',
                'Communication Skills',
                'Problem Resolution Techniques',
                'Customer Retention Strategies',
                'Telephone Etiquette',
                'Digital Customer Service Channels',
                'Emotional Intelligence in Customer Service'
            ],
            'Finance': [
                'Financial Statement Analysis',
                'Budgeting and Forecasting',
                'Cost Control Management',
                'Financial Reporting Standards',
                'Tax Regulations and Compliance',
                'Financial Risk Management',
                'Treasury Management',
                'Internal Controls and Audit'
            ],
            'Risk Management': [
                'Credit Risk Assessment',
                'Risk Mitigation Strategies',
                'Fraud Detection and Prevention',
                'Portfolio Risk Management',
                'Regulatory Compliance in Lending',
                'Basel Framework Overview',
                'Stress Testing Methodologies',
                'Risk Reporting and Analytics'
            ],
            'IT': [
                'Cybersecurity Essentials',
                'IT Service Management',
                'Database Management',
                'Network Administration',
                'Cloud Computing Solutions',
                'Software Development Lifecycle',
                'IT Project Management',
                'Digital Transformation in Finance'
            ],
            'HR': [
                'Talent Acquisition Strategies',
                'Performance Management',
                'Employee Engagement',
                'HR Analytics',
                'Compensation and Benefits',
                'Employee Relations',
                'HR Compliance and Regulations',
                'Learning and Development Programs'
            ],
            'Legal': [
                'Financial Services Regulations',
                'Contract Management',
                'Legal Risk Management',
                'Compliance Framework',
                'Corporate Governance',
                'Intellectual Property Protection',
                'Dispute Resolution',
                'Legal Aspects of Debt Collection'
            ],
            'Executive': [
                'Strategic Leadership',
                'Executive Decision Making',
                'Corporate Strategy',
                'Change Management',
                'Innovation in Financial Services',
                'Digital Transformation Leadership',
                'Crisis Management',
                'Corporate Social Responsibility'
            ]
        }
        
        # Generic courses available to all departments
        general_courses = [
            'Business Ethics and Compliance',
            'Effective Communication',
            'Time Management',
            'Leadership Essentials',
            'Microsoft Office Skills',
            'Presentation Skills',
            'Project Management Fundamentals',
            'Emotional Intelligence in the Workplace',
            'Diversity and Inclusion',
            'Stress Management and Well-being',
            'Problem Solving and Decision Making',
            'Team Collaboration Skills',
            'Financial Literacy Basics',
            'Anti-Money Laundering (AML) Training',
            'GDPR and Data Protection'
        ]
        
        # Training formats
        training_formats = ['Online Course', 'Classroom Training', 'Workshop', 'Webinar', 'Conference']
        format_weights = [0.5, 0.25, 0.15, 0.08, 0.02]  # More online, less conference
        
        # Course durations in hours
        course_durations = {
            'Online Course': (2, 8),
            'Classroom Training': (6, 16),
            'Workshop': (4, 8),
            'Webinar': (1, 3),
            'Conference': (8, 24)
        }
        
        # Certificate options
        certificate_options = [True, False]
        certificate_weights = [0.7, 0.3]  # 70% have certificates
        
        # Generate training records
        training_data = []
        training_id = 1
        
        for _, employee in employee_df.iterrows():
            # Determine how many trainings this employee has completed
            department = employee['Department']
            tenure_years = employee['YearsAtCompany']
            job_level = employee['JobLevel']
            
            # Tranings per year based on level and department
            if job_level in ['Entry Level', 'Junior']:
                trainings_per_year = random.uniform(2, 5)
            elif job_level in ['Mid-Level', 'Senior']:
                trainings_per_year = random.uniform(2, 4)
            elif job_level == 'Manager':
                trainings_per_year = random.uniform(1.5, 3.5)
            else:  # Director, Executive
                trainings_per_year = random.uniform(1, 3)
            
            # Adjust for certain departments
            if department in ['IT', 'Risk Management']:
                trainings_per_year *= 1.2  # More training in technical departments
            
            # Calculate number of trainings
            num_trainings = int(tenure_years * trainings_per_year)
            
            # Exit date if employee has left
            exit_date = None
            if employee['Attrition'] and 'ExitDate' in employee:
                exit_date = employee['ExitDate']
            
            # Generate training records
            for _ in range(num_trainings):
                # Course selection
                if random.random() < 0.7 and department in training_courses:
                    # Department-specific course
                    course_name = random.choice(training_courses[department])
                else:
                    # General course
                    course_name = random.choice(general_courses)
                
                # Training format
                training_format = random.choices(training_formats, weights=format_weights)[0]
                
                # Duration
                min_hours, max_hours = course_durations[training_format]
                duration_hours = random.randint(min_hours, max_hours)
                
                # Completion date
                if tenure_years <= 1:
                    days_ago = random.randint(1, int(tenure_years * 365))
                else:
                    # Recency bias - more recent trainings are more likely to be recorded
                    days_ago_weights = [3, 2, 1.5, 1, 0.8, 0.6, 0.4, 0.2]
                    weights_to_use = days_ago_weights[:min(8, int(tenure_years))]
                    year_segment = random.choices(
                        range(min(8, int(tenure_years))), 
                        weights=weights_to_use
                    )[0]
                    
                    min_days = year_segment * 365
                    max_days = min((year_segment + 1) * 365, int(tenure_years * 365))
                    days_ago = random.randint(min_days, max_days)
                
                completion_date = self.current_date - timedelta(days=days_ago)
                
                # Skip if completion date is after exit date
                if exit_date and completion_date > exit_date:
                    continue
                
                # Certification
                has_certificate = random.choices(certificate_options, weights=certificate_weights)[0]
                
                # Score (if applicable)
                if has_certificate and random.random() < 0.8:
                    # Score correlates with performance but with variation
                    base_score = 60 + (employee['PerformanceRating'] * 7)
                    score = min(100, max(60, base_score + random.uniform(-15, 15)))
                else:
                    score = None
                
                # Training cost
                if training_format == 'Online Course':
                    cost = random.uniform(50, 300) * 15000  # Convert to IDR
                elif training_format == 'Webinar':
                    cost = random.uniform(30, 150) * 15000
                elif training_format == 'Classroom Training':
                    cost = random.uniform(300, 800) * 15000
                elif training_format == 'Workshop':
                    cost = random.uniform(200, 600) * 15000
                else:  # Conference
                    cost = random.uniform(800, 2500) * 15000
                
                training_record = {
                    'TrainingID': training_id,
                    'EmployeeID': employee['EmployeeID'],
                    'CourseName': course_name,
                    'TrainingFormat': training_format,
                    'CompletionDate': completion_date,
                    'DurationHours': duration_hours,
                    'HasCertificate': has_certificate,
                    'Cost': round(cost, 2)
                }
                
                if score:
                    training_record['Score'] = round(score, 1)
                
                training_data.append(training_record)
                training_id += 1
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        
        return training_df

    def generate_exit_interview_data(self, employee_df):
        """
        Generate exit interview data for employees who have left.
        
        Args:
            employee_df (pandas.DataFrame): Employee data DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame containing exit interview data
        """
        # Exit reasons
        primary_exit_reasons = {
            'Better Compensation Elsewhere': 0.25,
            'Career Advancement Opportunity': 0.20,
            'Work-Life Balance': 0.15,
            'Relocation': 0.10,
            'Job Dissatisfaction': 0.08,
            'Conflict with Management': 0.07,
            'Health or Family Reasons': 0.06,
            'Company Culture': 0.05,
            'Retirement': 0.02,
            'Education': 0.02
        }
        
        # Secondary factors
        secondary_factors = [
            'Inadequate salary',
            'Limited growth opportunities',
            'Excessive workload',
            'Poor management',
            'Lack of recognition',
            'Inadequate benefits',
            'Long commute',
            'Workplace stress',
            'Better opportunity elsewhere',
            'Family obligations',
            'Health issues',
            'Pursuing education',
            'Relocation',
            'Career change',
            'Retirement',
            'Company restructuring',
            'Poor work environment',
            'Conflict with colleagues',
            'Lack of job security',
            'Pursuing entrepreneurship'
        ]
        
        # Destination companies (Indonesian financial institutions)
        destination_companies = [
            'Bank Mandiri',
            'Bank BRI',
            'Bank BCA',
            'Bank BNI',
            'CIMB Niaga',
            'Bank Danamon',
            'Bank Permata',
            'Adira Finance',
            'OJK (Financial Services Authority)',
            'Mandiri Tunas Finance',
            'ACC Finance',
            'FIF Group',
            'WOM Finance',
            'Bank Mega',
            'Bank OCBC NISP',
            'Maybank Indonesia',
            'Other Financial Institution',
            'Company Outside Financial Sector',
            'Self-Employed',
            'Not Specified'
        ]
        
        # Destination company weights (higher for bigger/more prestigious companies)
        destination_weights = [
            0.12, 0.12, 0.12, 0.10, 0.08, 
            0.07, 0.06, 0.05, 0.04, 0.04,
            0.03, 0.03, 0.02, 0.02, 0.02,
            0.02, 0.03, 0.02, 0.01, 0.01
        ]
        
        # Generate exit interview data
        exit_data = []
        
        for _, employee in employee_df.iterrows():
            if not employee['Attrition'] or 'ExitDate' not in employee:
                continue
            
            exit_date = employee['ExitDate']
            
            # Skip if exit date is too old (more than 3 years ago)
            if (self.current_date - exit_date).days > 365 * 3:
                continue
            
            # Interview date (usually within 1-5 days after exit date)
            interview_date = exit_date + timedelta(days=random.randint(1, 5))
            
            # Primary reason selection (influenced by employee characteristics)
            reason_weights = list(primary_exit_reasons.values())
            
            # Adjust weights based on employee characteristics
            if employee['JobSatisfaction'] <= 2:
                # Increase weight for job dissatisfaction
                idx = list(primary_exit_reasons.keys()).index('Job Dissatisfaction')
                reason_weights[idx] *= 2.0
            
            if employee['MonthlyIncome'] < 10 and employee['JobLevel'] in ['Entry Level', 'Junior', 'Mid-Level']:
                # Increase weight for compensation
                idx = list(primary_exit_reasons.keys()).index('Better Compensation Elsewhere')
                reason_weights[idx] *= 2.0
            
            if employee['WorkLifeBalance'] <= 2 or employee['Overtime']:
                # Increase weight for work-life balance
                idx = list(primary_exit_reasons.keys()).index('Work-Life Balance')
                reason_weights[idx] *= 2.0
            
            if employee['YearsSinceLastPromotion'] >= 3:
                # Increase weight for career advancement
                idx = list(primary_exit_reasons.keys()).index('Career Advancement Opportunity')
                reason_weights[idx] *= 1.8
            
            if employee['Age'] >= 50:
                # Increase weight for retirement
                idx = list(primary_exit_reasons.keys()).index('Retirement')
                reason_weights[idx] *= 10.0
            
            # Normalize weights
            total_weight = sum(reason_weights)
            reason_weights = [w / total_weight for w in reason_weights]
            
            # Select primary reason
            primary_reason = random.choices(
                list(primary_exit_reasons.keys()),
                weights=reason_weights
            )[0]
            
            # Secondary factors (2-4 factors)
            num_factors = random.randint(2, 4)
            
            # Filter secondary factors to be consistent with primary reason
            if primary_reason == 'Better Compensation Elsewhere':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['salary', 'benefits', 'opportunity', 'compensation'])]
            elif primary_reason == 'Career Advancement Opportunity':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['growth', 'opportunity', 'career', 'advancement'])]
            elif primary_reason == 'Work-Life Balance':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['workload', 'stress', 'commute', 'balance', 'family'])]
            elif primary_reason == 'Relocation':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['relocation', 'family', 'commute'])]
            elif primary_reason == 'Job Dissatisfaction':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['recognition', 'management', 'environment', 'stress'])]
            elif primary_reason == 'Conflict with Management':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['management', 'conflict', 'colleagues', 'recognition'])]
            elif primary_reason == 'Health or Family Reasons':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['health', 'family', 'stress', 'balance'])]
            elif primary_reason == 'Company Culture':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['environment', 'culture', 'colleagues', 'management'])]
            elif primary_reason == 'Retirement':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['retirement', 'age', 'health'])]
            elif primary_reason == 'Education':
                relevant_factors = [f for f in secondary_factors if any(word in f.lower() for word in 
                                    ['education', 'career', 'change'])]
            else:
                relevant_factors = secondary_factors
            
            # Ensure we have enough factors
            if len(relevant_factors) < num_factors:
                relevant_factors.extend([f for f in secondary_factors if f not in relevant_factors])
            
            selected_factors = random.sample(relevant_factors, num_factors)
            
            # Destination company
            if primary_reason == 'Career Advancement Opportunity' or primary_reason == 'Better Compensation Elsewhere':
                # More likely to go to a top financial institution
                dest_weights = destination_weights.copy()
                # Increase weight for top companies
                for i in range(5):
                    dest_weights[i] *= 1.5
                dest_weights = [w / sum(dest_weights) for w in dest_weights]
                destination = random.choices(destination_companies, weights=dest_weights)[0]
            elif primary_reason == 'Retirement':
                destination = 'Retirement'
            elif primary_reason == 'Education':
                destination = 'Education Institution'
            else:
                destination = random.choices(destination_companies, weights=destination_weights)[0]
            
            # Would recommend company (1-5 scale, influenced by reason for leaving)
            if primary_reason in ['Better Compensation Elsewhere', 'Career Advancement Opportunity']:
                recommend_base = random.uniform(2.5, 4.0)
            elif primary_reason in ['Work-Life Balance', 'Job Dissatisfaction', 'Conflict with Management', 'Company Culture']:
                recommend_base = random.uniform(1.0, 3.0)
            elif primary_reason in ['Relocation', 'Health or Family Reasons', 'Retirement', 'Education']:
                recommend_base = random.uniform(3.0, 4.5)
            else:
                recommend_base = random.uniform(2.0, 4.0)
            
            # Adjust based on job satisfaction
            recommend_score = min(5, max(1, recommend_base + (employee['JobSatisfaction'] - 3) * 0.5))
            
            # Overall experience (1-5 scale)
            experience_score = min(5, max(1, recommend_score + random.uniform(-0.5, 0.5)))
            
            # Manager effectiveness (1-5 scale)
            if 'Conflict with Management' in primary_reason:
                manager_score = random.uniform(1.0, 2.5)
            else:
                manager_score = min(5, max(1, 3 + random.uniform(-1.5, 1.5)))
            
            # Compensation satisfaction (1-5 scale)
            if primary_reason == 'Better Compensation Elsewhere':
                compensation_score = random.uniform(1.0, 2.5)
            else:
                compensation_score = min(5, max(1, 3 + random.uniform(-1.5, 1.5)))
            
            # Benefits satisfaction (1-5 scale)
            benefits_score = min(5, max(1, compensation_score + random.uniform(-0.5, 0.5)))
            
            # Work-life balance satisfaction (1-5 scale)
            if primary_reason == 'Work-Life Balance':
                worklife_score = random.uniform(1.0, 2.5)
            else:
                worklife_score = min(5, max(1, employee['WorkLifeBalance'] + random.uniform(-0.5, 0.5)))
            
            # Create exit interview record
            exit_record = {
                'EmployeeID': employee['EmployeeID'],
                'ExitDate': exit_date,
                'InterviewDate': interview_date,
                'PrimaryExitReason': primary_reason,
                'SecondaryFactors': ', '.join(selected_factors),
                'DestinationCompany': destination,
                'WouldRecommendCompany': round(recommend_score, 1),
                'OverallExperience': round(experience_score, 1),
                'ManagerEffectiveness': round(manager_score, 1),
                'CompensationSatisfaction': round(compensation_score, 1),
                'BenefitsSatisfaction': round(benefits_score, 1),
                'WorkLifeBalanceSatisfaction': round(worklife_score, 1),
            }
            
            exit_data.append(exit_record)
        
        # Convert to DataFrame
        exit_df = pd.DataFrame(exit_data)
        
        return exit_df
    
    def generate_recruitment_cost_data(self):
        """
        Generate recruitment cost benchmarks by job level.
        
        Returns:
            pandas.DataFrame: DataFrame containing recruitment cost data
        """
        job_levels = list(self.salary_ranges.keys())
        
        recruitment_data = []
        
        for level in job_levels:
            min_salary, max_salary = self.salary_ranges[level]
            avg_salary = (min_salary + max_salary) / 2 * 13  # Annual salary (13 months)
            
            # Recruitment costs as percentage of annual salary
            if level in ['Executive', 'Director']:
                pct_range = (25, 35)
            elif level == 'Manager':
                pct_range = (20, 30)
            elif level == 'Senior':
                pct_range = (18, 25)
            elif level == 'Mid-Level':
                pct_range = (15, 22)
            else:  # Junior, Entry Level
                pct_range = (10, 18)
            
            # Components of recruitment cost
            advertising_pct = random.uniform(0.1, 0.2)
            agency_pct = random.uniform(0.3, 0.5)
            interview_pct = random.uniform(0.1, 0.2)
            assessment_pct = random.uniform(0.05, 0.15)
            admin_pct = random.uniform(0.05, 0.1)
            relocation_pct = 0 if level in ['Entry Level', 'Junior'] else random.uniform(0, 0.1)
            signing_bonus_pct = 0 if level in ['Entry Level', 'Junior'] else random.uniform(0, 0.15)
            
            # Normalize percentages
            total_pct = advertising_pct + agency_pct + interview_pct + assessment_pct + admin_pct + relocation_pct + signing_bonus_pct
            advertising_pct /= total_pct
            agency_pct /= total_pct
            interview_pct /= total_pct
            assessment_pct /= total_pct
            admin_pct /= total_pct
            relocation_pct /= total_pct
            signing_bonus_pct /= total_pct
            
            # Total cost as percentage of annual salary
            total_pct = random.uniform(pct_range[0], pct_range[1]) / 100
            total_cost = avg_salary * total_pct * 1000000  # Convert to IDR
            
            recruitment_record = {
                'JobLevel': level,
                'AverageAnnualSalary': round(avg_salary * 1000000, 2),  # Convert to IDR
                'TotalRecruitmentCost': round(total_cost, 2),
                'RecruitmentCostPct': round(total_pct * 100, 2),
                'AdvertisingCost': round(total_cost * advertising_pct, 2),
                'AgencyFees': round(total_cost * agency_pct, 2),
                'InterviewCosts': round(total_cost * interview_pct, 2),
                'AssessmentCosts': round(total_cost * assessment_pct, 2),
                'AdministrativeCosts': round(total_cost * admin_pct, 2),
                'RelocationCosts': round(total_cost * relocation_pct, 2),
                'SigningBonus': round(total_cost * signing_bonus_pct, 2)
            }
            
            recruitment_data.append(recruitment_record)
        
        # Convert to DataFrame
        recruitment_df = pd.DataFrame(recruitment_data)
        
        return recruitment_df
    
    def generate_all_data(self, num_employees=1000, historical_years=3):
        """
        Generate all datasets needed for the HR analytics project.
        
        Args:
            num_employees (int): Number of current employees to generate
            historical_years (int): Years of historical data to generate
            
        Returns:
            dict: Dictionary containing all generated DataFrames
        """
        print("Generating employee data...")
        employee_df = self.generate_employee_data(num_employees, historical_years)
        
        print("Generating survey data...")
        survey_df = self.generate_engagement_survey_data(employee_df)
        
        print("Generating performance data...")
        performance_df = self.generate_performance_data(employee_df, quarters=historical_years * 4)
        
        print("Generating promotion history...")
        promotion_df = self.generate_promotion_history(employee_df)
        
        print("Generating training data...")
        training_df = self.generate_training_data(employee_df)
        
        print("Generating exit interview data...")
        exit_df = self.generate_exit_interview_data(employee_df)
        
        print("Generating recruitment cost data...")
        recruitment_df = self.generate_recruitment_cost_data()
        
        # Return all datasets in a dictionary
        return {
            'employees': employee_df,
            'surveys': survey_df,
            'performance': performance_df,
            'promotions': promotion_df,
            'training': training_df,
            'exit_interviews': exit_df,
            'recruitment_costs': recruitment_df
        }
    
    def save_datasets(self, data_dict, output_dir='data/raw'):
        """
        Save all datasets to CSV files.
        
        Args:
            data_dict (dict): Dictionary containing DataFrames
            output_dir (str): Directory to save the files
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each DataFrame to a CSV file
        for name, df in data_dict.items():
            file_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} records to {file_path}")


def main():
    """
    Main function to generate and save HR data.
    """
    print("Initializing HR data generator...")
    generator = HRDataGenerator(seed=42)
    
    print("Generating data for BFI Finance Indonesia...")
    data_dict = generator.generate_all_data(num_employees=1000, historical_years=3)
    
    print("Saving datasets to CSV files...")
    generator.save_datasets(data_dict)
    
    print("Data generation complete!")
    
    # Print summary statistics
    for name, df in data_dict.items():
        print(f"\nDataset: {name}")
        print(f"Records: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()