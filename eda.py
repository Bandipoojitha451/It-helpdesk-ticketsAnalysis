
import pandas as pd
import matplotlib

# Use a non-interactive backend so execution never pauses for graphs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import redirect_stdout

# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

input_file = "cleaned_helpdesk_tickets.csv"
output_dir = Path("EDA_Results")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "eda_report.txt", "w", encoding="utf-8") as report_file:
    with redirect_stdout(report_file):

        df = pd.read_csv(input_file)

        print("=" * 50)
        print("IT HELP DESK TICKET ANALYSIS - EDA REPORT")
        print("=" * 50)

        # Clean column names
        df.columns = df.columns.str.strip()

        print("\n1. DATASET OVERVIEW")
        print("Total tickets:", len(df))
        print("Total columns:", len(df.columns))
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\n2. DATASET INFORMATION")
        print(df.info())

        # ==========================================
        # 2. MISSING VALUES
        # ==========================================

        print("\n3. MISSING VALUES")
        print(df.isnull().sum().to_string())

        # ==========================================
        # 3. STATISTICAL SUMMARY
        # ==========================================

        print("\n4. STATISTICAL SUMMARY")
        print(df.describe(include="all").to_string())

        # ==========================================
        # 4. DATE CONVERSION
        # ==========================================

        if "Created_Date" in df.columns:
            df["Created_Date"] = pd.to_datetime(
                df["Created_Date"], errors="coerce"
            )

        # ==========================================
        # 5. TICKET STATUS ANALYSIS
        # ==========================================

        if "Status" in df.columns:
            print("\n5. TICKETS BY STATUS")

            status_counts = df["Status"].value_counts()
            print(status_counts.to_string())

            status_counts.plot(
                kind="bar",
                title="Tickets by Status",
                figsize=(8, 5)
            )
            plt.xlabel("Ticket Status")
            plt.ylabel("Number of Tickets")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "eda_status.png", dpi=150)
            plt.close()

        # ==========================================
        # 6. PRIORITY ANALYSIS
        # ==========================================

        if "Priority" in df.columns:
            print("\n6. TICKETS BY PRIORITY")

            priority_counts = df["Priority"].value_counts()
            print(priority_counts.to_string())

            priority_counts.plot(
                kind="bar",
                title="Tickets by Priority",
                figsize=(8, 5)
            )
            plt.xlabel("Priority")
            plt.ylabel("Number of Tickets")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "eda_priority.png", dpi=150)
            plt.close()

        # ==========================================
        # 7. DEPARTMENT ANALYSIS
        # ==========================================

        if "Department" in df.columns:
            print("\n7. TICKETS BY DEPARTMENT")

            department_counts = df["Department"].value_counts()
            print(department_counts.to_string())

            department_counts.plot(
                kind="bar",
                title="Tickets by Department",
                figsize=(10, 5)
            )
            plt.xlabel("Department")
            plt.ylabel("Number of Tickets")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "eda_department.png", dpi=150)
            plt.close()

        # ==========================================
        # 8. MONTHLY TICKET TREND
        # ==========================================

        if "Created_Date" in df.columns:
            valid_dates = df.dropna(subset=["Created_Date"]).copy()

            if not valid_dates.empty:
                valid_dates["Month"] = (
                    valid_dates["Created_Date"].dt.to_period("M")
                )

                monthly_tickets = valid_dates.groupby("Month").size()

                print("\n8. MONTHLY TICKET TREND")
                print(monthly_tickets.to_string())

                monthly_tickets.plot(
                    kind="line",
                    marker="o",
                    title="Monthly Ticket Trend",
                    figsize=(10, 5)
                )
                plt.xlabel("Month")
                plt.ylabel("Number of Tickets")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(
                    output_dir / "eda_monthly_trend.png", dpi=150
                )
                plt.close()

        # ==========================================
        # 9. RESOLUTION TIME ANALYSIS
        # ==========================================

        if "Resolution_Time_hrs" in df.columns:
            resolution = pd.to_numeric(
                df["Resolution_Time_hrs"], errors="coerce"
            ).dropna()

            resolution = resolution[resolution >= 0]

            print("\n9. RESOLUTION TIME ANALYSIS")

            if not resolution.empty:
                print("Average resolution time:",
                      round(resolution.mean(), 2), "hours")
                print("Median resolution time:",
                      round(resolution.median(), 2), "hours")
                print("Minimum resolution time:",
                      round(resolution.min(), 2), "hours")
                print("Maximum resolution time:",
                      round(resolution.max(), 2), "hours")

                resolution.plot(
                    kind="hist",
                    bins=20,
                    title="Resolution Time Distribution",
                    figsize=(8, 5)
                )
                plt.xlabel("Resolution Time (Hours)")
                plt.ylabel("Number of Tickets")
                plt.tight_layout()
                plt.savefig(
                    output_dir / "eda_resolution_time.png", dpi=150
                )
                plt.close()

        # ==========================================
        # 10. CUSTOMER RATING ANALYSIS
        # ==========================================

        if "Customer_Rating" in df.columns:
            ratings = pd.to_numeric(
                df["Customer_Rating"], errors="coerce"
            ).dropna()

            ratings = ratings[ratings.between(1, 5)]

            print("\n10. CUSTOMER RATING ANALYSIS")

            if not ratings.empty:
                print("Average customer rating:",
                      round(ratings.mean(), 2), "/ 5")
                print("\nRating counts:")
                print(ratings.value_counts().sort_index().to_string())

                ratings.value_counts().sort_index().plot(
                    kind="bar",
                    title="Customer Rating Distribution",
                    figsize=(8, 5)
                )
                plt.xlabel("Rating")
                plt.ylabel("Number of Tickets")
                plt.tight_layout()
                plt.savefig(
                    output_dir / "eda_customer_rating.png", dpi=150
                )
                plt.close()

        # ==========================================
        # 11. RESOLUTION TIME BY PRIORITY
        # ==========================================

        if all(col in df.columns for col in [
            "Priority", "Resolution_Time_hrs"
        ]):
            df["Resolution_Time_hrs"] = pd.to_numeric(
                df["Resolution_Time_hrs"], errors="coerce"
            )

            valid = df[
                df["Resolution_Time_hrs"].notna()
                & (df["Resolution_Time_hrs"] >= 0)
            ]

            avg_resolution = valid.groupby(
                "Priority"
            )["Resolution_Time_hrs"].mean().sort_values(
                ascending=False
            )

            print("\n11. AVERAGE RESOLUTION TIME BY PRIORITY")
            print(avg_resolution.round(2).to_string())

            if not avg_resolution.empty:
                avg_resolution.plot(
                    kind="bar",
                    title="Average Resolution Time by Priority",
                    figsize=(8, 5)
                )
                plt.xlabel("Priority")
                plt.ylabel("Average Resolution Time (Hours)")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(
                    output_dir / "eda_resolution_by_priority.png",
                    dpi=150
                )
                plt.close()

        # ==========================================
        # 12. ASSIGNED AGENT ANALYSIS
        # ==========================================

        if "Assigned_To" in df.columns:
            print("\n12. TOP 10 ASSIGNED AGENTS")
            print(
                df["Assigned_To"]
                .value_counts()
                .head(10)
                .to_string()
            )

        # ==========================================
        # 13. ISSUE CATEGORY ANALYSIS
        # ==========================================

        if "Issue_Category" in df.columns:
            print("\n13. TICKETS BY ISSUE CATEGORY")

            category_counts = df["Issue_Category"].value_counts()
            print(category_counts.to_string())

            category_counts.plot(
                kind="bar",
                title="Tickets by Issue Category",
                figsize=(10, 5)
            )
            plt.xlabel("Issue Category")
            plt.ylabel("Number of Tickets")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                output_dir / "eda_issue_category.png", dpi=150
            )
            plt.close()

        # ==========================================
        # 14. SLA ANALYSIS
        # ==========================================

        if "SLA_Status" in df.columns:
            print("\n14. SLA STATUS ANALYSIS")

            sla_counts = df["SLA_Status"].value_counts()
            print(sla_counts.to_string())

            sla_counts.plot(
                kind="bar",
                title="SLA Status Distribution",
                figsize=(8, 5)
            )
            plt.xlabel("SLA Status")
            plt.ylabel("Number of Tickets")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(
                output_dir / "eda_sla_status.png", dpi=150
            )
            plt.close()

        print("\n" + "=" * 50)
        print("EDA COMPLETED SUCCESSFULLY")
        print("=" * 50)

# ==========================================
# 15. SHOW COMPLETION MESSAGE
# ==========================================

print("EDA COMPLETED SUCCESSFULLY!")
print("All graphs saved in the EDA_Results folder.")
print("Text report saved as EDA_Results/eda_report.txt")
print("Open the folder to view your graphs and report.")
