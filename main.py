import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_style("whitegrid")  # Clean background
plt.rcParams.update({'figure.autolayout': True})  # Better spacing

# ------------------------------
# Load and Clean Data
# ------------------------------
df = pd.read_csv("preorder_food_data.csv")
df = df[df['delivery_status'].notnull()]  # Drop cancelled orders

# Convert datetime
df['order_time'] = pd.to_datetime(df['order_time'])
df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

# Time-based features
df['pre_order_lead_minutes'] = (df['scheduled_time'] - df['order_time']).dt.total_seconds() / 60
df['delay_minutes'] = (df['actual_delivery_time'] - df['scheduled_time']).dt.total_seconds() / 60
df['delay_minutes'] = df['delay_minutes'].apply(lambda x: max(x, 0))

df['hour'] = df['order_time'].dt.hour
df['day_of_week'] = df['order_time'].dt.day_name()
df['month'] = df['order_time'].dt.month_name()

# ------------------------------
# Console Output
# ------------------------------
print("\n🔍 Dataset Preview:\n", df.head())
print("\n📊 Summary Statistics:\n", df.describe())
print("\n📋 Delivery Status Count:\n", df['delivery_status'].value_counts())
print("\n🗓️ Date Range:", df['order_time'].min(), "to", df['order_time'].max())

# ------------------------------
# Plot 1: Order Value Distribution
# ------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df['order_value'], bins=30, kde=True, color='#36B37E')
plt.title("💸 Order Value Distribution", fontsize=14)
plt.xlabel("Order Value (INR)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------
# Plot 2: Delivery Delay Distribution
# ------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(df['delay_minutes'], bins=30, kde=True, color='#FF6F61')
plt.title("⏰ Delivery Delay (Minutes)", fontsize=14)
plt.xlabel("Delay in Minutes")
plt.ylabel("Number of Orders")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------
# Plot 3: Delay by Zone
# ------------------------------
plt.figure(figsize=(6, 4))
sns.barplot(x='location_zone', y='delay_minutes', data=df, palette='viridis', estimator=np.mean)
plt.title("📍 Avg Delay by Zone", fontsize=14)
plt.ylabel("Avg Delay (min)")
plt.xlabel("Zone")
plt.tight_layout()
plt.show()

# ------------------------------
# Plot 4: Ratings Distribution
# ------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="rating", data=df, palette="pastel")
plt.title("🌟 Customer Ratings", fontsize=14)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------------
# Plot 5: Coupon Use vs Delay
# ------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="coupon_used", y="delay_minutes", data=df, palette='coolwarm')
plt.title("Delay by Coupon Use", fontsize=14)
plt.xlabel("Coupon Used")
plt.ylabel("Delay (min)")
plt.tight_layout()
plt.show()

# ------------------------------
#  T-Test: Delay with/without Coupon
# ------------------------------
coupon_delay = df[df['coupon_used'] == 'Yes']['delay_minutes']
no_coupon_delay = df[df['coupon_used'] == 'No']['delay_minutes']
t_stat, p_value = stats.ttest_ind(coupon_delay, no_coupon_delay)

print("\n T-Test Result: Delay by Coupon Use")
print(f"  T-Statistic: {t_stat:.3f}")
print(f"  P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("   Statistically significant difference in delay due to coupon.")
else:
    print("  ❌ No significant difference in delay between groups.")

# ------------------------------
# 📈 Correlation
# ------------------------------
correlation = df['order_value'].corr(df['delay_minutes'])
print(f"\n Correlation between Order Value and Delay: {correlation:.3f}")

# ------------------------------
# Plot 6: Heatmap of Delay by Time of Day
# ------------------------------
heat_df = df.groupby(['day_of_week', 'hour'])['delay_minutes'].mean().unstack().fillna(0)
heat_df = heat_df.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

plt.figure(figsize=(12, 5))
sns.heatmap(heat_df, cmap="YlGnBu", linewidths=0.5, linecolor='gray')
plt.title(" Heatmap: Avg Delay by Day and Hour", fontsize=14)
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.show()

# ------------------------------
# Plot 7: Weekday vs Weekend Delay
# ------------------------------
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
plt.figure(figsize=(6, 4))
sns.boxplot(x='is_weekend', y='delay_minutes', data=df, palette='Set2')
plt.title(" Delay: Weekday vs Weekend", fontsize=14)
plt.xlabel("Is Weekend?")
plt.ylabel("Delay (min)")
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.tight_layout()
plt.show()

# ------------------------------
# Plot 8: Monthly Seasonality
# ------------------------------
monthly_avg = df.groupby('month')['delay_minutes'].mean().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])

plt.figure(figsize=(10, 4))
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', color='#0088CC')
plt.title(" Seasonality: Monthly Average Delay", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Avg Delay (min)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
