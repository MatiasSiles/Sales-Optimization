This dataset contains comprehensive sales transaction data from a business, a multi-channel retail company specializing in consumer electronics, clothing, home goods, and lifestyle products. it operates across five major regions globally and sells through multiple channels including online platforms, physical retail stores, mobile applications, and third-party marketplaces.
The dataset spans from January 2020 to December 2024, capturing the company's sales performance through various market conditions including the COVID-19 pandemic, supply chain disruptions, and economic recovery periods. This rich dataset provides insights into customer behavior, product performance, seasonal trends, and operational efficiency across different business segments.

Dataset Specifications:


Total Records: 12120 
Time Period: January 2020 - December 2024
Business Type: Multi-channel retail
Geographic Coverage: Global (5 regions)
Product Categories: 8 main categories


Column Definitions
Transaction Identifiers

order_id (string): Unique identifier for each sales transaction. Format: ORD_YYYYMMDD_XXXX
date (datetime): Date when the transaction occurred

Customer Information:

customer_id (string): Unique customer identifier
customer_age (integer): Customer's age in years (18-80)
customer_gender (string): Customer's gender (Male, Female, Other)
customer_segment (string): Customer classification based on purchasing behavior and value

Premium: High-value customers with frequent purchases
Standard: Regular customers with moderate purchase frequency
Budget: Price-sensitive customers seeking deals
Enterprise: B2B customers with bulk purchases



Product Information:

product_category (string): Main product classification

Electronics: Computers, phones, gadgets, accessories
Clothing: Apparel, shoes, fashion accessories
Home & Garden: Furniture, decor, gardening supplies
Sports & Outdoors: Fitness equipment, outdoor gear
Books: Physical and digital books, educational materials
Health & Beauty: Personal care, cosmetics, wellness products
Automotive: Car accessories, tools, maintenance products
Toys & Games: Children's toys, board games, video games


product_name (string): Specific product name/model

Financial Data:

unit_price (float): Price per individual item in USD
quantity (integer): Number of items purchased in the transaction
subtotal (float): Total before discounts and taxes (unit_price × quantity)
discount_rate (float): Percentage discount applied (0.0 to 1.0)
discount_amount (float): Dollar amount of discount applied
tax_rate (float): Tax percentage applied based on region
tax_amount (float): Tax amount in USD
shipping_cost (float): Shipping and handling charges
total_amount (float): Final transaction amount (subtotal - discount + tax + shipping)

Sales & Marketing:

sales_channel (string): Channel through which the sale was made

Online: Company website
Retail Store: Physical brick-and-mortar locations
Phone: Telephone sales
Mobile App: Mobile application purchases
Third Party: External marketplaces and partners


payment_method (string): Method used for payment

Credit Card, Debit Card, PayPal, Bank Transfer, Cash, Cryptocurrency


region (string): Geographic region of the sale

North America: USA, Canada, Mexico
Europe: EU countries, UK, Norway, Switzerland
Asia Pacific: Japan, Australia, South Korea, Singapore
Latin America: Brazil, Argentina, Chile, Colombia
Middle East & Africa: UAE, Saudi Arabia, South Africa


sales_rep (string): Name of the sales representative handling the transaction
lead_source (string): Original source that brought the customer

Organic Search, Paid Search, Social Media, Email, Referral, Direct, Advertisement



Customer Experience:

is_returned (boolean): Whether the item was returned (True/False)
return_reason (string): Reason for return if applicable

Defective, Wrong Item, Not as Described, Changed Mind, Damaged in Shipping


satisfaction_score (integer): Customer satisfaction rating (1-10 scale). Note: Contains missing values

Derived Fields:

year (integer): Year extracted from transaction date
month (integer): Month extracted from transaction date (1-12)
quarter (integer): Quarter of the year (1-4)
day_of_week (string): Day name (Monday, Tuesday, etc.)
is_weekend (boolean): Whether transaction occurred on weekend
profit_margin (float): Estimated profit margin percentage (0.15-0.45)
profit (float): Estimated profit amount in USD
