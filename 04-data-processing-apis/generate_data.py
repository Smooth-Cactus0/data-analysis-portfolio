"""
============================================================
DATA GENERATOR - Multi-Source Retail Data
============================================================
Generates realistic synthetic data simulating multiple business systems:
- Sales transactions (CSV)
- Product catalog (JSON)
- Customer data (CSV)
- Inventory levels (XML)
- Store locations (JSON)
- Supplier data (Excel)

Author: Alexy Louis
============================================================
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

print("="*70)
print("MULTI-SOURCE DATA GENERATOR")
print("="*70)

# ============================================================
# 1. STORE LOCATIONS (JSON)
# ============================================================
print("\n[1/6] Generating store locations...")

stores = []
regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
store_types = ['Flagship', 'Standard', 'Express', 'Outlet']

cities = {
    'Northeast': [('New York', 'NY', 40.7128, -74.0060), ('Boston', 'MA', 42.3601, -71.0589), 
                  ('Philadelphia', 'PA', 39.9526, -75.1652), ('Pittsburgh', 'PA', 40.4406, -79.9959)],
    'Southeast': [('Miami', 'FL', 25.7617, -80.1918), ('Atlanta', 'GA', 33.7490, -84.3880),
                  ('Charlotte', 'NC', 35.2271, -80.8431), ('Orlando', 'FL', 28.5383, -81.3792)],
    'Midwest': [('Chicago', 'IL', 41.8781, -87.6298), ('Detroit', 'MI', 42.3314, -83.0458),
                ('Minneapolis', 'MN', 44.9778, -93.2650), ('Cleveland', 'OH', 41.4993, -81.6944)],
    'Southwest': [('Dallas', 'TX', 32.7767, -96.7970), ('Houston', 'TX', 29.7604, -95.3698),
                  ('Phoenix', 'AZ', 33.4484, -112.0740), ('Austin', 'TX', 30.2672, -97.7431)],
    'West': [('Los Angeles', 'CA', 34.0522, -118.2437), ('San Francisco', 'CA', 37.7749, -122.4194),
             ('Seattle', 'WA', 47.6062, -122.3321), ('Denver', 'CO', 39.7392, -104.9903)]
}

store_id = 1001
for region, city_list in cities.items():
    for city, state, lat, lon in city_list:
        num_stores = random.randint(1, 3)
        for i in range(num_stores):
            stores.append({
                'store_id': f'STR-{store_id}',
                'store_name': f'{city} {"Downtown" if i == 0 else "Mall" if i == 1 else "Plaza"}',
                'store_type': random.choice(store_types),
                'region': region,
                'city': city,
                'state': state,
                'latitude': lat + random.uniform(-0.05, 0.05),
                'longitude': lon + random.uniform(-0.05, 0.05),
                'square_feet': random.randint(5000, 25000),
                'employees': random.randint(15, 80),
                'opening_date': (datetime(2015, 1, 1) + timedelta(days=random.randint(0, 2500))).strftime('%Y-%m-%d'),
                'manager_id': f'EMP-{random.randint(10000, 99999)}',
                'phone': f'+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
                'is_active': True
            })
            store_id += 1

with open(os.path.join(RAW_DIR, 'stores.json'), 'w') as f:
    json.dump({'stores': stores, 'generated_at': datetime.now().isoformat()}, f, indent=2)

print(f"   ✓ Generated {len(stores)} stores across {len(regions)} regions")

# ============================================================
# 2. PRODUCT CATALOG (JSON)
# ============================================================
print("\n[2/6] Generating product catalog...")

categories = {
    'Electronics': {
        'subcategories': ['Smartphones', 'Laptops', 'Tablets', 'Accessories', 'Audio'],
        'price_range': (49.99, 1999.99),
        'margin_range': (0.15, 0.35)
    },
    'Clothing': {
        'subcategories': ['Men\'s Wear', 'Women\'s Wear', 'Kids', 'Footwear', 'Accessories'],
        'price_range': (19.99, 299.99),
        'margin_range': (0.40, 0.65)
    },
    'Home & Garden': {
        'subcategories': ['Furniture', 'Decor', 'Kitchen', 'Outdoor', 'Bedding'],
        'price_range': (14.99, 999.99),
        'margin_range': (0.35, 0.55)
    },
    'Sports & Outdoors': {
        'subcategories': ['Fitness', 'Camping', 'Team Sports', 'Water Sports', 'Cycling'],
        'price_range': (9.99, 599.99),
        'margin_range': (0.30, 0.50)
    },
    'Beauty & Health': {
        'subcategories': ['Skincare', 'Makeup', 'Hair Care', 'Supplements', 'Personal Care'],
        'price_range': (4.99, 149.99),
        'margin_range': (0.50, 0.75)
    }
}

products = []
brands = ['TechPro', 'StyleMax', 'HomeElite', 'SportFit', 'GlowUp', 'PrimeLine', 
          'ValueBest', 'LuxeLife', 'EcoSmart', 'TrendSet']

product_id = 10001
for category, cat_info in categories.items():
    for subcategory in cat_info['subcategories']:
        num_products = random.randint(15, 30)
        for _ in range(num_products):
            base_price = round(random.uniform(*cat_info['price_range']), 2)
            margin = random.uniform(*cat_info['margin_range'])
            
            products.append({
                'product_id': f'PRD-{product_id}',
                'sku': f'{category[:3].upper()}-{product_id}-{random.randint(100,999)}',
                'product_name': f'{random.choice(brands)} {subcategory} Item {product_id}',
                'category': category,
                'subcategory': subcategory,
                'brand': random.choice(brands),
                'unit_price': base_price,
                'cost_price': round(base_price * (1 - margin), 2),
                'margin_percent': round(margin * 100, 1),
                'weight_kg': round(random.uniform(0.1, 15.0), 2),
                'is_active': random.random() > 0.05,
                'launch_date': (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))).strftime('%Y-%m-%d'),
                'supplier_id': f'SUP-{random.randint(100, 150)}',
                'reorder_level': random.randint(10, 100),
                'reorder_quantity': random.randint(50, 500),
                'lead_time_days': random.randint(3, 21)
            })
            product_id += 1

with open(os.path.join(RAW_DIR, 'product_catalog.json'), 'w') as f:
    json.dump({
        'products': products, 
        'categories': list(categories.keys()),
        'total_products': len(products),
        'generated_at': datetime.now().isoformat()
    }, f, indent=2)

print(f"   ✓ Generated {len(products)} products across {len(categories)} categories")

# ============================================================
# 3. CUSTOMER DATA (CSV)
# ============================================================
print("\n[3/6] Generating customer data...")

first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
               'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
               'Thomas', 'Sarah', 'Charles', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Lisa']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
              'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
              'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White']

tiers = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
tier_weights = [0.40, 0.30, 0.18, 0.09, 0.03]

customers = []
for i in range(8000):
    tier = np.random.choice(tiers, p=tier_weights)
    signup_date = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2200))
    
    # Higher tier = more spending
    tier_multiplier = {'Bronze': 1, 'Silver': 2, 'Gold': 4, 'Platinum': 8, 'Diamond': 15}[tier]
    
    customer = {
        'customer_id': f'CUS-{100000 + i}',
        'first_name': random.choice(first_names),
        'last_name': random.choice(last_names),
        'email': f'customer{100000+i}@{"gmail.com" if random.random() > 0.3 else "yahoo.com" if random.random() > 0.5 else "outlook.com"}',
        'phone': f'+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
        'date_of_birth': (datetime(1950, 1, 1) + timedelta(days=random.randint(0, 20000))).strftime('%Y-%m-%d'),
        'gender': random.choice(['M', 'F', 'Other', None]),
        'city': random.choice([c[0] for cities_list in cities.values() for c in cities_list]),
        'state': random.choice([c[1] for cities_list in cities.values() for c in cities_list]),
        'loyalty_tier': tier,
        'signup_date': signup_date.strftime('%Y-%m-%d'),
        'total_orders': int(random.expovariate(1/10) * tier_multiplier),
        'total_spent': round(random.expovariate(1/500) * tier_multiplier, 2),
        'avg_order_value': round(random.uniform(30, 200) * (tier_multiplier ** 0.3), 2),
        'preferred_store': random.choice([s['store_id'] for s in stores]),
        'email_opt_in': random.random() > 0.2,
        'sms_opt_in': random.random() > 0.6,
        'is_active': random.random() > 0.1,
        'last_purchase_date': (signup_date + timedelta(days=random.randint(0, 500))).strftime('%Y-%m-%d') if random.random() > 0.15 else None
    }
    customers.append(customer)

# Add some data quality issues (realistic!)
for i in random.sample(range(len(customers)), 200):
    issue = random.choice(['missing_email', 'invalid_phone', 'future_date', 'duplicate_phone'])
    if issue == 'missing_email':
        customers[i]['email'] = None
    elif issue == 'invalid_phone':
        customers[i]['phone'] = 'INVALID'
    elif issue == 'future_date':
        customers[i]['date_of_birth'] = '2030-01-15'  # Invalid future date

customers_df = pd.DataFrame(customers)
customers_df.to_csv(os.path.join(RAW_DIR, 'customers.csv'), index=False)

print(f"   ✓ Generated {len(customers)} customers with realistic data quality issues")

# ============================================================
# 4. SALES TRANSACTIONS (CSV)
# ============================================================
print("\n[4/6] Generating sales transactions...")

transactions = []
transaction_id = 1000000

# Generate 18 months of transactions
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 6, 30)
current_date = start_date

while current_date <= end_date:
    # More transactions on weekends
    is_weekend = current_date.weekday() >= 5
    # Holiday boost
    is_holiday_season = current_date.month in [11, 12]
    
    base_transactions = random.randint(150, 300)
    daily_transactions = int(base_transactions * (1.4 if is_weekend else 1.0) * (1.6 if is_holiday_season else 1.0))
    
    for _ in range(daily_transactions):
        customer = random.choice(customers)
        store = random.choice(stores)
        
        # Generate 1-5 items per transaction
        num_items = np.random.choice([1, 2, 3, 4, 5], p=[0.35, 0.30, 0.20, 0.10, 0.05])
        
        for item_num in range(num_items):
            product = random.choice(products)
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.60, 0.25, 0.10, 0.03, 0.02])
            
            # Apply random discounts
            discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], p=[0.50, 0.20, 0.15, 0.08, 0.05, 0.02])
            unit_price = product['unit_price']
            discount_amount = round(unit_price * quantity * discount_pct / 100, 2)
            
            transaction = {
                'transaction_id': f'TXN-{transaction_id}',
                'line_item': item_num + 1,
                'transaction_date': current_date.strftime('%Y-%m-%d'),
                'transaction_time': f'{random.randint(9, 21):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}',
                'store_id': store['store_id'],
                'customer_id': customer['customer_id'] if random.random() > 0.2 else None,  # 20% anonymous
                'product_id': product['product_id'],
                'quantity': quantity,
                'unit_price': unit_price,
                'discount_percent': discount_pct,
                'discount_amount': discount_amount,
                'line_total': round(unit_price * quantity - discount_amount, 2),
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'Cash', 'Mobile Pay', 'Gift Card']),
                'employee_id': f'EMP-{random.randint(10000, 99999)}'
            }
            transactions.append(transaction)
        
        transaction_id += 1
    
    current_date += timedelta(days=1)

# Add some data quality issues
for i in random.sample(range(len(transactions)), 500):
    issue = random.choice(['negative_qty', 'missing_store', 'invalid_date', 'price_zero'])
    if issue == 'negative_qty':
        transactions[i]['quantity'] = -1
    elif issue == 'missing_store':
        transactions[i]['store_id'] = None
    elif issue == 'invalid_date':
        transactions[i]['transaction_date'] = 'INVALID'
    elif issue == 'price_zero':
        transactions[i]['unit_price'] = 0

transactions_df = pd.DataFrame(transactions)
transactions_df.to_csv(os.path.join(RAW_DIR, 'sales_transactions.csv'), index=False)

print(f"   ✓ Generated {len(transactions)} transaction line items")
print(f"   ✓ Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# ============================================================
# 5. INVENTORY LEVELS (XML)
# ============================================================
print("\n[5/6] Generating inventory data (XML)...")

xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
xml_content += '<inventory_snapshot>\n'
xml_content += f'  <generated_at>{datetime.now().isoformat()}</generated_at>\n'
xml_content += '  <items>\n'

for store in stores:
    # Each store has subset of products
    store_products = random.sample(products, k=min(len(products), random.randint(200, 400)))
    
    for product in store_products:
        current_stock = random.randint(0, 200)
        
        xml_content += '    <item>\n'
        xml_content += f'      <store_id>{store["store_id"]}</store_id>\n'
        xml_content += f'      <product_id>{product["product_id"]}</product_id>\n'
        xml_content += f'      <sku>{product["sku"]}</sku>\n'
        xml_content += f'      <current_stock>{current_stock}</current_stock>\n'
        xml_content += f'      <reorder_level>{product["reorder_level"]}</reorder_level>\n'
        xml_content += f'      <needs_reorder>{"true" if current_stock < product["reorder_level"] else "false"}</needs_reorder>\n'
        xml_content += f'      <last_restock_date>{(datetime.now() - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d")}</last_restock_date>\n'
        xml_content += f'      <warehouse_location>WH-{random.randint(1,5)}-{random.choice("ABCDEF")}{random.randint(1,50)}</warehouse_location>\n'
        xml_content += '    </item>\n'

xml_content += '  </items>\n'
xml_content += '</inventory_snapshot>\n'

with open(os.path.join(RAW_DIR, 'inventory.xml'), 'w') as f:
    f.write(xml_content)

print(f"   ✓ Generated inventory data for {len(stores)} stores")

# ============================================================
# 6. SUPPLIER DATA (Excel)
# ============================================================
print("\n[6/6] Generating supplier data (Excel)...")

suppliers = []
supplier_names = ['Global Supply Co', 'FastTrack Logistics', 'Premier Distributors', 'Elite Imports',
                  'Quality First Inc', 'Direct Source Ltd', 'Wholesale Partners', 'Supply Chain Pro',
                  'International Traders', 'Bulk Goods Corp', 'Metro Distribution', 'National Suppliers']

countries = ['USA', 'China', 'Germany', 'Japan', 'South Korea', 'Vietnam', 'Mexico', 'India', 'Italy', 'UK']

for i in range(50):
    country = random.choice(countries)
    suppliers.append({
        'supplier_id': f'SUP-{100 + i}',
        'supplier_name': f'{random.choice(supplier_names)} {random.choice(["LLC", "Inc", "Corp", "Ltd"])}',
        'contact_name': f'{random.choice(first_names)} {random.choice(last_names)}',
        'email': f'supplier{i}@{random.choice(["supply.com", "wholesale.net", "imports.com"])}',
        'phone': f'+{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000000, 9999999)}',
        'country': country,
        'payment_terms': random.choice(['Net 30', 'Net 45', 'Net 60', 'Net 90']),
        'min_order_value': random.choice([500, 1000, 2500, 5000, 10000]),
        'avg_lead_time_days': random.randint(5, 45),
        'reliability_score': round(random.uniform(0.70, 0.99), 2),
        'contract_start': (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000))).strftime('%Y-%m-%d'),
        'contract_end': (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 730))).strftime('%Y-%m-%d'),
        'is_active': random.random() > 0.1,
        'categories_supplied': ', '.join(random.sample(list(categories.keys()), k=random.randint(1, 3))),
        'currency': 'USD' if country == 'USA' else random.choice(['USD', 'EUR', 'GBP', 'CNY', 'JPY'])
    })

suppliers_df = pd.DataFrame(suppliers)

# Create Excel with multiple sheets
with pd.ExcelWriter(os.path.join(RAW_DIR, 'suppliers.xlsx'), engine='openpyxl') as writer:
    suppliers_df.to_excel(writer, sheet_name='Suppliers', index=False)
    
    # Add a summary sheet
    summary = pd.DataFrame({
        'Metric': ['Total Suppliers', 'Active Suppliers', 'Countries', 'Avg Lead Time'],
        'Value': [len(suppliers), sum(1 for s in suppliers if s['is_active']), 
                  len(set(s['country'] for s in suppliers)),
                  round(np.mean([s['avg_lead_time_days'] for s in suppliers]), 1)]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"   ✓ Generated {len(suppliers)} suppliers from {len(set(s['country'] for s in suppliers))} countries")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("DATA GENERATION COMPLETE")
print("="*70)
print(f"\nFiles created in: {RAW_DIR}")
print(f"  • stores.json          - {len(stores)} stores")
print(f"  • product_catalog.json - {len(products)} products")
print(f"  • customers.csv        - {len(customers)} customers")
print(f"  • sales_transactions.csv - {len(transactions)} line items")
print(f"  • inventory.xml        - Inventory for all stores")
print(f"  • suppliers.xlsx       - {len(suppliers)} suppliers")
print("\n✅ All data sources ready for ETL pipeline!")
