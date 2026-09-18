 Project OverviewThe raw dataset contains 10,000 customer records across 12 feature columns. It exhibits significant data quality issues, including sentinel missing values, invalid numeric outliers, unstandardized categorical fields, corrupt date logic, and invalid contact placeholders.Dataset ProfileColumn NameRaw Data TypeIdentified Quality IssuesCustomerIDStringMissing values, duplicated entries, mixed alphanumeric formatting (e.g., C2 vs 3).NameString1,658 missing entries, leading/trailing whitespace.GenderStringInconsistent casing and abbreviations (Male, male, M, Female, female, F).AgeFloatExtreme outliers and sentinel values (-5.0, 200.0), missing values.City / CountryStringMissing values, unstripped whitespace.Signup_DateString / DateString placeholders (not_a_date), mixed date formats (DD/MM/YYYY).Last_purchase_dateString / DateMissing values (missing), chronologically invalid entries preceding Signup_Date.purchase_amountFloatNegative placeholder values (-999.0), missing data.feedback_scoreFloatOut-of-range sentinel values (-1.0), missing scores.emailStringInvalid formats missing domain symbols (e.g., user1mail.com), missing data.Phone_numberStringNon-numeric placeholders (0, abc123), missing values.Data Cleaning StepsSentinel Value Conversion: Replaced implicit missing string flags ("missing", "not_a_date", "") across all fields with explicit NaN/NaT values.Categorical Standardization: Stripped whitespace from text fields; mapped Gender variations (male, m, female, f) to standardized 'Male' and 'Female' categories.Numeric Outlier Handling: Replaced invalid Age values ($< 18$ or $> 100$) with NaN and imputed missing entries using the median age ($43.0$).Date Parsing & Logic Verification: Parsed date strings using day-first datetime formatting (%d/%m/%Y); set Last_purchase_date to NaT whenever it occurred prior to Signup_Date.Contact & Value Auditing: Identified invalid placeholder phone numbers (abc123, 0) and ill-formed email strings for downstream filtering.UsagePythonimport pandas as pd
import numpy as np

# Load dataset and standardise nulls
df = pd.read_csv("messy_customer_data.csv")
df = df.replace(["missing", "not_a_date", ""], np.nan)

# Standardise gender categories
df['Gender'] = df['Gender'].str.lower().map({
    'male': 'Male', 'm': 'Male',
    'female': 'Female', 'f': 'Female'
})

# Correct Age boundaries and impute median
df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median())

# Parse dates and correct chronological errors
df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], errors='coerce', dayfirst=True)
df['Last_purchase_date'] = pd.to_datetime(df['Last_purchase_date'], errors='coerce', dayfirst=True)
df.loc[df['Last_purchase_date'] < df['Signup_Date'], 'Last_purchase_date'] = pd.NaT
