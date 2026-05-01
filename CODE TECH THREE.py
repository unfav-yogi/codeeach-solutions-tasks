
import pandas as pd
from pulp import *

data = pd.read_excel(
    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\code alpha\financial dashboard\Financial Sample.xlsx"
)

print(data.head())

model = LpProblem(
    "Profit_Maximization",
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

model += 40 * product_A + 30 * product_B

model += 2 * product_A + 1 * product_B <= 100

model += 1 * product_A + 1 * product_B <= 80

model += 1 * product_A + 2 * product_B <= 90

print("\nOptimization Problem")

print(model)

model.solve()

print("\nStatus :", LpStatus[model.status])

print("\nOptimal Production")

print("Units of Product A :", product_A.varValue)

print("Units of Product B :", product_B.varValue)

print("\nMaximum Profit : ₹", value(model.objective))

print("\nConstraints Information")

for name, constraint in model.constraints.items():

    print(name, ":", constraint, "=", constraint.value())

