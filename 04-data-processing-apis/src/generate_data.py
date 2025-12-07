"""
Generate Multi-Source E-commerce Data for ETL Pipeline Demo
This creates realistic data across multiple formats and sources
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# =============================================================================
# SOURCE 1: Sales Transactions (CSV) - Primary transactional data
# =============================================================================

n_transactions = 5000
start_date = datetime(2024, 1, 1)

# Generate transaction IDs
transaction_ids = [f"TXN-{str(i).zfill(7)}" for i in range(1, n_transactions + 1)]

# Generate dates (weighted toward recent)
days_back = np.random.exponential(scale=60, size=n_transactions).astype(int)
days_back = np.clip(days_back, 0, 300)
transaction_dates = [start_date + timedelta(days=int(300-d)) for d in days_back]

# Customer IDs (some repeat customers)
n_customers = 1200
customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
transaction_customers = np.random.choice(customer_ids, n_transactions, 
    p=np.random.dirichlet(np.ones(n_customers) * 0.5))

# Product IDs
n_products = 150
product_ids = [f"PROD-{str(i).zfill(4)}" for i in range(1, n_products + 1)]
transaction_products = np.random.choice(product_ids, n_transactions)

# Quantities (mostly 1-3, sometimes more)
quantities = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], n_transactions)

# Unit prices (will be validated against product catalog)
unit_prices = np.round(np.random.uniform(9.99, 499.99, n_transactions), 2)

# Payment methods
payment_methods = np.random.choice(
    ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay', 'Bank Transfer'],
    n_transactions, p=[0.40, 0.25, 0.15, 0.10, 0.07, 0.03]
)

# Order status
statuses = np.random.choice(
    ['Completed', 'Completed', 'Completed', 'Completed', 'Shipped', 'Processing', 'Refunded', 'Cancelled'],
    n_transactions
)

# Shipping costs
shipping_costs = np.where(
    unit_prices * quantities > 100, 0,
    np.random.choice([5.99, 7.99, 9.99, 12.99], n_transactions)
)

# Discount codes (20% have discounts)
discount_codes = np.where(
    np.random.random(n_transactions) < 0.2,
    np.random.choice(['SAVE10', 'SUMMER20', 'WELCOME15', 'VIP25', 'FLASH30'], n_transactions),
    ''
)

# Discount amounts
discount_amounts = np.where(
    discount_codes == 'SAVE10', unit_prices * quantities * 0.10,
    np.where(discount_codes == 'SUMMER20', unit_prices * quantities * 0.20,
    np.where(discount_codes == 'WELCOME15', unit_prices * quantities * 0.15,
    np.where(discount_codes == 'VIP25', unit_prices * quantities * 0.25,
    np.where(discount_codes == 'FLASH30', unit_prices * quantities * 0.30, 0)))))

# Sales channels
channels = np.random.choice(
    ['Website', 'Mobile App', 'Marketplace', 'Social Media', 'In-Store'],
    n_transactions, p=[0.45, 0.30, 0.12, 0.08, 0.05]
)

# Create DataFrame
sales_df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'transaction_date': transaction_dates,
    'customer_id': transaction_customers,
    'product_id': transaction_products,
    'quantity': quantities,
    'unit_price': unit_prices,
    'payment_method': payment_methods,
    'order_status': statuses,
    'shipping_cost': shipping_costs,
    'discount_code': discount_codes,
    'discount_amount': np.round(discount_amounts, 2),
    'sales_channel': channels
})

# Add some data quality issues for cleaning demonstration
# Missing values
missing_idx = np.random.choice(len(sales_df), 50, replace=False)
sales_df.loc[missing_idx[:20], 'payment_method'] = np.nan
sales_df.loc[missing_idx[20:35], 'shipping_cost'] = np.nan
sales_df.loc[missing_idx[35:], 'sales_channel'] = np.nan

# Duplicate transactions (for deduplication demo)
dup_idx = np.random.choice(len(sales_df), 25, replace=False)
duplicates = sales_df.iloc[dup_idx].copy()
sales_df = pd.concat([sales_df, duplicates], ignore_index=True)

# Invalid data (negative quantities, future dates)
sales_df.loc[np.random.choice(len(sales_df), 10), 'quantity'] = -1
sales_df.loc[np.random.choice(len(sales_df), 5), 'unit_price'] = -99.99

# Save
sales_df.to_csv('/home/claude/data-analysis-portfolio/04-data-processing-apis/data/raw/sales_transactions.csv', index=False)
print(f"✅ Sales transactions: {len(sales_df)} records")

# =============================================================================
# SOURCE 2: Product Catalog (JSON) - Product master data
# =============================================================================

categories = {
    'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Accessories', 'Audio'],
    'Clothing': ['Men', 'Women', 'Kids', 'Accessories', 'Footwear'],
    'Home & Garden': ['Furniture', 'Decor', 'Kitchen', 'Garden', 'Bedding'],
    'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports', 'Cycling']
}

products = []
for i, prod_id in enumerate(product_ids):
    category = random.choice(list(categories.keys()))
    subcategory = random.choice(categories[category])
    
    base_price = random.uniform(15, 400)
    
    product = {
        'product_id': prod_id,
        'product_name': f"{subcategory} Item {i+1}",
        'category': category,
        'subcategory': subcategory,
        'base_price': round(base_price, 2),
        'cost_price': round(base_price * random.uniform(0.4, 0.7), 2),
        'supplier_id': f"SUP-{random.randint(1, 20):03d}",
        'brand': random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'Generic']),
        'weight_kg': round(random.uniform(0.1, 15), 2),
        'dimensions': {
            'length_cm': random.randint(5, 100),
            'width_cm': random.randint(5, 80),
            'height_cm': random.randint(2, 50)
        },
        'is_active': random.choice([True, True, True, True, False]),
        'created_date': (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))).isoformat(),
        'tags': random.sample(['bestseller', 'new', 'sale', 'limited', 'eco-friendly', 'premium'], 
                             k=random.randint(0, 3)),
        'rating': round(random.uniform(3.0, 5.0), 1),
        'review_count': random.randint(0, 500)
    }
    products.append(product)

# Add some products with missing/invalid data
products[5]['base_price'] = None
products[12]['category'] = ''
products[25]['product_name'] = None

with open('/home/claude/data-analysis-portfolio/04-data-processing-apis/data/raw/product_catalog.json', 'w') as f:
    json.dump({'products': products, 'last_updated': datetime.now().isoformat()}, f, indent=2)

print(f"✅ Product catalog: {len(products)} products")

# =============================================================================
# SOURCE 3: Customer Data (CSV) - Customer master data
# =============================================================================

# Customer segments
segments = ['Bronze', 'Silver', 'Gold', 'Platinum']
segment_probs = [0.50, 0.30, 0.15, 0.05]

# Regions
regions = ['North', 'South', 'East', 'West', 'Central']

customers_data = []
for cust_id in customer_ids:
    signup_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1700))
    
    customer = {
        'customer_id': cust_id,
        'email': f"customer_{cust_id.split('-')[1]}@email.com",
        'first_name': random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Emily', 'Chris', 'Lisa']),
        'last_name': random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']),
        'signup_date': signup_date.strftime('%Y-%m-%d'),
        'segment': np.random.choice(segments, p=segment_probs),
        'region': random.choice(regions),
        'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                               'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Austin']),
        'age': random.randint(18, 75),
        'gender': random.choice(['M', 'F', 'Other', 'Prefer not to say']),
        'marketing_opt_in': random.choice([True, True, True, False]),
        'preferred_channel': random.choice(['Email', 'SMS', 'Push', 'None']),
        'lifetime_value': round(random.uniform(50, 5000), 2),
        'total_orders': random.randint(1, 50)
    }
    customers_data.append(customer)

customers_df = pd.DataFrame(customers_data)

# Add data quality issues
customers_df.loc[np.random.choice(len(customers_df), 30), 'email'] = 'invalid-email'
customers_df.loc[np.random.choice(len(customers_df), 20), 'age'] = -1
customers_df.loc[np.random.choice(len(customers_df), 15), 'region'] = np.nan

customers_df.to_csv('/home/claude/data-analysis-portfolio/04-data-processing-apis/data/raw/customers.csv', index=False)
print(f"✅ Customer data: {len(customers_df)} customers")

# =============================================================================
# SOURCE 4: External Market Data (JSON - simulating API response)
# =============================================================================

# Simulating external API data like market trends, competitor prices, etc.
market_data = {
    'api_version': '2.0',
    'request_timestamp': datetime.now().isoformat(),
    'data_source': 'MarketAnalytics API',
    'market_trends': [
        {
            'category': 'Electronics',
            'trend_score': 8.5,
            'yoy_growth': 12.3,
            'market_size_millions': 450,
            'top_brands': ['Apple', 'Samsung', 'Sony'],
            'avg_price_index': 1.05
        },
        {
            'category': 'Clothing',
            'trend_score': 7.2,
            'yoy_growth': 5.8,
            'market_size_millions': 380,
            'top_brands': ['Nike', 'Adidas', 'Zara'],
            'avg_price_index': 0.98
        },
        {
            'category': 'Home & Garden',
            'trend_score': 8.0,
            'yoy_growth': 15.2,
            'market_size_millions': 290,
            'top_brands': ['IKEA', 'Wayfair', 'HomeDepot'],
            'avg_price_index': 1.02
        },
        {
            'category': 'Sports',
            'trend_score': 7.8,
            'yoy_growth': 9.5,
            'market_size_millions': 220,
            'top_brands': ['Nike', 'UnderArmour', 'Adidas'],
            'avg_price_index': 1.00
        }
    ],
    'competitor_pricing': [
        {'competitor': 'CompetitorA', 'price_index': 1.05, 'market_share': 15.2},
        {'competitor': 'CompetitorB', 'price_index': 0.95, 'market_share': 12.8},
        {'competitor': 'CompetitorC', 'price_index': 1.10, 'market_share': 8.5},
        {'competitor': 'CompetitorD', 'price_index': 0.92, 'market_share': 6.2}
    ],
    'seasonal_factors': {
        'Q1': 0.85,
        'Q2': 0.95,
        'Q3': 1.05,
        'Q4': 1.25
    }
}

with open('/home/claude/data-analysis-portfolio/04-data-processing-apis/data/external/market_data.json', 'w') as f:
    json.dump(market_data, f, indent=2)

print(f"✅ Market data: External API simulation")

# =============================================================================
# SOURCE 5: Inventory Data (CSV) - Warehouse inventory
# =============================================================================

inventory_data = []
warehouses = ['WH-EAST', 'WH-WEST', 'WH-CENTRAL']

for prod_id in product_ids:
    for warehouse in warehouses:
        inv = {
            'product_id': prod_id,
            'warehouse_id': warehouse,
            'quantity_on_hand': random.randint(0, 500),
            'quantity_reserved': random.randint(0, 50),
            'reorder_point': random.randint(10, 100),
            'reorder_quantity': random.randint(50, 200),
            'last_restock_date': (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%Y-%m-%d'),
            'unit_cost': round(random.uniform(5, 200), 2),
            'location_code': f"{warehouse}-{random.choice(['A', 'B', 'C'])}-{random.randint(1, 50):02d}"
        }
        inventory_data.append(inv)

inventory_df = pd.DataFrame(inventory_data)

# Add some data issues
inventory_df.loc[np.random.choice(len(inventory_df), 20), 'quantity_on_hand'] = -10
inventory_df.loc[np.random.choice(len(inventory_df), 10), 'warehouse_id'] = 'INVALID'

inventory_df.to_csv('/home/claude/data-analysis-portfolio/04-data-processing-apis/data/raw/inventory.csv', index=False)
print(f"✅ Inventory data: {len(inventory_df)} records")

# =============================================================================
# SOURCE 6: Supplier Data (XML-like structure in JSON)
# =============================================================================

suppliers = []
for i in range(1, 21):
    supplier = {
        'supplier_id': f"SUP-{i:03d}",
        'company_name': f"Supplier Company {i}",
        'contact_name': f"Contact Person {i}",
        'contact_email': f"contact{i}@supplier{i}.com",
        'phone': f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        'address': {
            'street': f"{random.randint(100, 9999)} Business Ave",
            'city': random.choice(['New York', 'Chicago', 'Los Angeles', 'Miami', 'Seattle']),
            'state': random.choice(['NY', 'IL', 'CA', 'FL', 'WA']),
            'zip_code': f"{random.randint(10000, 99999)}",
            'country': 'USA'
        },
        'payment_terms': random.choice(['Net 30', 'Net 45', 'Net 60', 'COD']),
        'rating': round(random.uniform(3.0, 5.0), 1),
        'lead_time_days': random.randint(3, 21),
        'minimum_order_value': random.randint(100, 1000),
        'active': random.choice([True, True, True, False])
    }
    suppliers.append(supplier)

with open('/home/claude/data-analysis-portfolio/04-data-processing-apis/data/raw/suppliers.json', 'w') as f:
    json.dump({'suppliers': suppliers}, f, indent=2)

print(f"✅ Supplier data: {len(suppliers)} suppliers")

print("\n" + "="*60)
print("PHASE 1 COMPLETE: All data sources created!")
print("="*60)
