import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("preorder_food_data.csv")

# Basic info
print(df.head())
print(df.info())
print(df.describe())

# Order value distribution
sns.histplot(df['order_value'], bins=30, kde=True)
plt.title("Order Value Distribution")
plt.show()

# Delivery status count
sns.countplot(x='delivery_status', data=df)
plt.title("Delivery Status: On-Time vs Late")
plt.show()
