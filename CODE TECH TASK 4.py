import pandas as pd
from pulp import *

data = pd.read_excel(
    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\code alpha\financial dashboard\Financial Sample.xlsx"
)

print("\nFinancial Dataset")
print(data.head())

print("\nDataset Shape :", data.shape)

print("\nColumns in Dataset")
print(data.columns)

model = LpProblem(
    "Business_Profit_Optimization",
    LpMaximize
)

product_A = LpVariable(
    "Product_A",
    lowBound=0,
    cat='Integer'
)

product_B = LpVariable(
    "Product_B",
    lowBound=0,
    cat='Integer'
)

profit_A = 40
profit_B = 30

model += (
    profit_A * product_A +
    profit_B * product_B
)

model += (
    2 * product_A +
    1 * product_B <= 100
)

model += (
    1 * product_A +
    1 * product_B <= 80
)

model += (
    1 * product_A +
    2 * product_B <= 90
)

print("\nOptimization Problem")

print(model)

solution_status = model.solve()

print("\nSolver Status :")
print(LpStatus[solution_status])

print("\nOptimal Production Values")

print(
    "Units of Product A :",
    product_A.varValue
)

print(
    "Units of Product B :",
    product_B.varValue
)

maximum_profit = value(model.objective)

print(
    "\nMaximum Profit : ₹",
    maximum_profit
)

print("\nConstraints Information")

for name, constraint in model.constraints.items():

    print(
        name,
        ":",
        constraint,
        "=",
        constraint.value()
    )

results = pd.DataFrame({

    "Product": [
        "Product A",
        "Product B"
    ],

    "Optimal Units": [
        product_A.varValue,
        product_B.varValue
    ]

})

print("\nFinal Optimized Results")

print(results)

results.to_csv(

    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\code alpha\financial dashboard\optimization_results.csv",

    index=False
)

print(
    "\nOptimization Results Saved Successfully"
)