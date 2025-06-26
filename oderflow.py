import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
np.random.seed(42)

def generate_data(n=1000):
    data = []
    food_items_list = ['Pizza', 'Burger', 'Pasta', 'Biryani', 'Salad', 'Fries', 'Noodles', 'Tacos']
    cancellation_reasons = [None, 'User changed mind', 'Late delivery', 'Out of stock', 'Wrong address']
    zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D']

    for i in range(n):
        order_time = fake.date_time_this_year()
        scheduled_time = order_time + timedelta(minutes=random.randint(30, 120))
        actual_delivery_time = scheduled_time + timedelta(minutes=random.randint(-10, 40))
        is_cancelled = random.choices([True, False], weights=[0.1, 0.9])[0]
        delivery_status = 'Late' if actual_delivery_time > scheduled_time else 'On-Time'

        data.append({
            'order_id': f'ORD{i:05}',
            'customer_id': f'CUST{random.randint(1, 300)}',
            'restaurant_id': f'RES{random.randint(1, 50)}',
            'order_time': order_time,
            'scheduled_time': scheduled_time,
            'actual_delivery_time': None if is_cancelled else actual_delivery_time,
            'location_zone': random.choice(zones),
            'food_items': ', '.join(random.choices(food_items_list, k=random.randint(1, 3))),
            'order_value': round(random.uniform(100, 800), 2),
            'delivery_status': None if is_cancelled else delivery_status,
            'cancellation_status': 'Cancelled' if is_cancelled else 'Not Cancelled',
            'cancellation_reason': random.choice(cancellation_reasons) if is_cancelled else None,
            'rating': None if is_cancelled else random.choice([1, 2, 3, 4, 5]),
            'coupon_used': random.choice(['Yes', 'No']),
            'device_type': random.choice(['Android', 'iOS', 'Web']),
            'subscription_user': random.choice(['Yes', 'No']),
        })

    return pd.DataFrame(data)

df = generate_data(1000)
df.to_csv("preorder_food_data.csv", index=False)
print("✅ Dataset 'preorder_food_data.csv' generated.")
