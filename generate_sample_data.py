"""Run this once to regenerate sample_data/employees.csv"""
import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

departments = {
    "Engineering": ["Junior Software Engineer", "Software Engineer", "Senior Software Engineer",
                    "Staff Engineer", "Principal Engineer", "Engineering Manager"],
    "Data": ["Data Analyst", "Senior Data Analyst", "Data Scientist", "Senior Data Scientist",
             "ML Engineer", "Head of Data"],
    "Product": ["Product Manager", "Senior Product Manager", "Director of Product"],
    "Marketing": ["Marketing Coordinator", "Marketing Manager", "Senior Marketing Manager",
                  "VP of Marketing"],
    "Sales": ["Sales Representative", "Account Executive", "Senior Account Executive",
              "Sales Manager", "VP of Sales"],
    "Finance": ["Financial Analyst", "Senior Financial Analyst", "Finance Manager", "CFO"],
    "HR": ["HR Coordinator", "HR Manager", "Senior HR Manager", "VP of People"],
    "Operations": ["Operations Analyst", "Operations Manager", "Director of Operations"],
}

base_salaries = {
    "Junior Software Engineer": 72000,
    "Software Engineer": 105000,
    "Senior Software Engineer": 140000,
    "Staff Engineer": 175000,
    "Principal Engineer": 210000,
    "Engineering Manager": 185000,
    "Data Analyst": 75000,
    "Senior Data Analyst": 100000,
    "Data Scientist": 120000,
    "Senior Data Scientist": 155000,
    "ML Engineer": 145000,
    "Head of Data": 190000,
    "Product Manager": 115000,
    "Senior Product Manager": 145000,
    "Director of Product": 185000,
    "Marketing Coordinator": 55000,
    "Marketing Manager": 85000,
    "Senior Marketing Manager": 110000,
    "VP of Marketing": 170000,
    "Sales Representative": 60000,
    "Account Executive": 90000,
    "Senior Account Executive": 115000,
    "Sales Manager": 130000,
    "VP of Sales": 180000,
    "Financial Analyst": 78000,
    "Senior Financial Analyst": 105000,
    "Finance Manager": 130000,
    "CFO": 250000,
    "HR Coordinator": 52000,
    "HR Manager": 80000,
    "Senior HR Manager": 105000,
    "VP of People": 160000,
    "Operations Analyst": 70000,
    "Operations Manager": 100000,
    "Director of Operations": 150000,
}

locations = ["New York", "San Francisco", "Austin", "Chicago", "Remote", "Seattle", "Boston"]
employment_types = ["Full-time", "Full-time", "Full-time", "Full-time", "Contract"]

first_names = ["Alice", "Bob", "Carlos", "Diana", "Ethan", "Fiona", "George", "Hannah",
               "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina", "Omar", "Priya",
               "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
               "Yara", "Zach", "Amelia", "Ben", "Clara", "David", "Elena", "Frank",
               "Grace", "Henry", "Iris", "James", "Kate", "Leo", "Maya", "Noah",
               "Olivia", "Paul", "Qian", "Rosa", "Steve", "Tara", "Uri", "Vera",
               "Will", "Xena"]

last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
              "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
              "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
              "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
              "Campbell", "Mitchell", "Carter", "Roberts"]

rows = []
emp_id = 1001

for dept, titles in departments.items():
    for title in titles:
        n_employees = random.randint(3, 8)
        for _ in range(n_employees):
            first = random.choice(first_names)
            last = random.choice(last_names)
            hire_year = random.randint(2018, 2022)
            hire_date = date(hire_year, random.randint(1, 12), random.randint(1, 28))
            location = random.choice(locations)
            emp_type = random.choice(employment_types)
            base = base_salaries[title]

            for year in [2021, 2022, 2023, 2024]:
                if year >= hire_year:
                    annual_raise = random.uniform(0.03, 0.08)
                    years_exp = year - hire_year
                    salary = int(base * (1 + annual_raise) ** years_exp * random.uniform(0.92, 1.12))
                    rows.append({
                        "employee_id": emp_id,
                        "first_name": first,
                        "last_name": last,
                        "full_name": f"{first} {last}",
                        "department": dept,
                        "job_title": title,
                        "salary": salary,
                        "year": year,
                        "hire_date": hire_date.isoformat(),
                        "location": location,
                        "employment_type": emp_type,
                    })
            emp_id += 1

df = pd.DataFrame(rows)
df.to_csv("sample_data/employees.csv", index=False)
print(f"Generated {len(df)} rows, {df['employee_id'].nunique()} unique employees")
print(df.head())
