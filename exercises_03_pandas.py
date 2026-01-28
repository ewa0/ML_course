import numpy as np
import pandas as pd

sal = pd.read_csv(
    "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/04-Pandas-Exercises/Salaries.csv")

print("initial dataframe:", sal.head())
print("\ninfo about the dataframe:", sal.info())
print("\naverage base pay:", sal["BasePay"].mean())
print("\nthe highest amount of overtime pay:", sal["OvertimePay"].max())
print("\nJOSEPH DRISCOLL's job title:", sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]["JobTitle"])
print("\nJOSEPH DRISCOLL makes:", sal[sal["EmployeeName"] == "JOSEPH DRISCOLL"]["TotalPayBenefits"])
# index of max value
print("\nhighest paid person:", sal.loc[sal["TotalPayBenefits"].idxmax(), "EmployeeName"])
print("\nlowest paid person:", sal.loc[sal["TotalPayBenefits"].idxmin(), "EmployeeName"])
print("\naverage BasePay of all employees per year (2011-2014):", sal.groupby("Year")["BasePay"].mean())
print("\nnumber of unique job titles", sal["JobTitle"].nunique())
print("\nthe top 5 most common jobs:", sal["JobTitle"].value_counts().head(5))
print("\njob titles represented by only 1 person in 2013:",
      sum(sal[sal["Year"] == 2013]["JobTitle"].value_counts() == 1))
print("\nnumber of people with word <<Chief>> in their title:",
      sum(sal["JobTitle"].apply(lambda title: "chief" in title.lower())))

job_title = sal["JobTitle"].apply(lambda l: len(l))
salary = sal["TotalPay"]
print("\nis there a correlation between length of the job title string and salary?", job_title.corr(salary, method="pearson"))