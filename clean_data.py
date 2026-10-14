
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("It_Ticket.csv")

print("========== ORIGINAL DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn Names:")
print(df.columns.tolist())

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Remove extra spaces from column names and text values
df.columns = df.columns.str.strip()

text_columns = df.select_dtypes(include=["object"]).columns

for col in text_columns:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# 4. Remove duplicate ticket IDs, if available
if "Ticket_ID" in df.columns:
    df = df.drop_duplicates(
        subset=["Ticket_ID"], keep="first"
    )

# 5. Convert date column
if "Created_Date" in df.columns:
    df["Created_Date"] = pd.to_datetime(
        df["Created_Date"], errors="coerce"
    )

# 6. Convert numeric columns
numeric_columns = [
    "Resolution_Time_hrs",
    "Customer_Rating"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

# 7. Handle missing values in text columns
fill_values = {
    "Department": "Unknown",
    "Priority": "Unknown",
    "Status": "Unknown",
    "Assigned_To": "Unassigned",
    "Tags": "Not Specified",
    "Body": "No Description"
}

for col, value in fill_values.items():
    if col in df.columns:
        df[col] = df[col].fillna(value)
        df[col] = df[col].replace("", value)

# 8. Remove records without essential information
required_columns = [
    col for col in [
        "Ticket_ID", "Created_Date", "Priority", "Status"
    ] if col in df.columns
]

if required_columns:
    df = df.dropna(subset=required_columns)

# 9. Handle invalid resolution times
if "Resolution_Time_hrs" in df.columns:
    df.loc[
        df["Resolution_Time_hrs"] < 0,
        "Resolution_Time_hrs"
    ] = float("nan")

# 10. Validate customer ratings (1 to 5)
if "Customer_Rating" in df.columns:
    invalid_rating = ~df["Customer_Rating"].between(1, 5)
    df.loc[invalid_rating, "Customer_Rating"] = float("nan")

# 11. Add date-based columns
if "Created_Date" in df.columns:
    df["Created_Year"] = df["Created_Date"].dt.year
    df["Created_Month"] = df["Created_Date"].dt.month
    df["Month_Name"] = df["Created_Date"].dt.strftime("%B")
    df["Day_of_Week"] = df["Created_Date"].dt.strftime("%A")

# 12. Save cleaned dataset
df.to_csv("cleaned_helpdesk_tickets.csv", index=False)

# 13. Display cleaning results
print("\n========== CLEANING COMPLETED ==========")
print("Final rows:", df.shape[0])
print("Final columns:", df.shape[1])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

if "Ticket_ID" in df.columns:
    print(
        "Duplicate Ticket IDs:",
        df["Ticket_ID"].duplicated().sum()
    )

print("\nCleaned dataset saved as:")
print("cleaned_helpdesk_tickets.csv")
