import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

closed_deals = pd.read_csv(
    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\future intern\TASK2 DISESTER\olist_closed_deals_dataset.csv"
)

marketing_leads = pd.read_csv(
    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\future intern\TASK2 DISESTER\olist_marketing_qualified_leads_dataset.csv"
)

print("Closed Deals Dataset Shape :", closed_deals.shape)
print("Marketing Leads Dataset Shape :", marketing_leads.shape)

print("\nClosed Deals Dataset")
print(closed_deals.head())

print("\nMarketing Leads Dataset")
print(marketing_leads.head())

print("\nMissing Values in Closed Deals")
print(closed_deals.isnull().sum())

print("\nMissing Values in Marketing Leads")
print(marketing_leads.isnull().sum())

closed_deals.drop_duplicates(inplace=True)
marketing_leads.drop_duplicates(inplace=True)

print("\nDuplicates Removed Successfully")

num_cols_closed = closed_deals.select_dtypes(include=np.number).columns
num_cols_marketing = marketing_leads.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='mean')

if len(num_cols_closed) > 0:
    closed_deals[num_cols_closed] = imputer.fit_transform(
        closed_deals[num_cols_closed]
    )

if len(num_cols_marketing) > 0:
    marketing_leads[num_cols_marketing] = imputer.fit_transform(
        marketing_leads[num_cols_marketing]
    )

for col in closed_deals.select_dtypes(include='object').columns:
    closed_deals[col] = closed_deals[col].fillna(
        closed_deals[col].mode()[0]
    )

for col in marketing_leads.select_dtypes(include='object').columns:
    marketing_leads[col] = marketing_leads[col].fillna(
        marketing_leads[col].mode()[0]
    )

print("\nMissing Values Handled Successfully")

date_columns_closed = ['won_date']

for col in date_columns_closed:
    if col in closed_deals.columns:
        try:
            closed_deals[col] = pd.to_datetime(closed_deals[col])
        except:
            pass

print("\nDate Conversion Completed")

label_encoder = LabelEncoder()

for col in closed_deals.select_dtypes(include='object').columns:
    closed_deals[col] = label_encoder.fit_transform(
        closed_deals[col].astype(str)
    )

for col in marketing_leads.select_dtypes(include='object').columns:
    marketing_leads[col] = label_encoder.fit_transform(
        marketing_leads[col].astype(str)
    )

print("\nCategorical Encoding Completed")

scaler = StandardScaler()

if len(num_cols_closed) > 0:
    closed_deals[num_cols_closed] = scaler.fit_transform(
        closed_deals[num_cols_closed]
    )

if len(num_cols_marketing) > 0:
    marketing_leads[num_cols_marketing] = scaler.fit_transform(
        marketing_leads[num_cols_marketing]
    )

print("\nFeature Scaling Completed")

common_columns = list(
    set(closed_deals.columns).intersection(set(marketing_leads.columns))
)

if len(common_columns) > 0:

    merge_column = common_columns[0]

    final_data = pd.merge(
        closed_deals,
        marketing_leads,
        on=merge_column,
        how='inner'
    )

    print("\nDatasets Merged on :", merge_column)

else:

    final_data = pd.concat(
        [closed_deals, marketing_leads],
        axis=0,
        ignore_index=True
    )

    print("\nNo Common Column Found")
    print("Datasets Concatenated")

final_data.to_csv(
    r"C:\Users\yoeshwar\OneDrive\Desktop\internships\future intern\TASK2 DISESTER\cleaned_olist_dataset.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully")

print("\nFinal Dataset Shape :", final_data.shape)

print("\nFirst 5 Rows of Final Dataset")
print(final_data.head())