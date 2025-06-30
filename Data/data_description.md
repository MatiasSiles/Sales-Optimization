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

Customer Information

customer_id (string): Unique customer identifier
customer_age (integer): Customer's age in years (18-80)
customer_gender (string): Customer's gender (Male, Female, Other)
customer_segment (string): Customer classification based on purchasing behavior and value

Premium: High-value customers with frequent purchases
Standard: Regular customers with moderate purchase frequency
Budget: Price-sensitive customers seeking deals
Enterprise: B2B customers with bulk purchases



Product Information

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

Financial Data

unit_price (float): Price per individual item in USD
quantity (integer): Number of items purchased in the transaction
subtotal (float): Total before discounts and taxes (unit_price × quantity)
discount_rate (float): Percentage discount applied (0.0 to 1.0)
discount_amount (float): Dollar amount of discount applied
tax_rate (float): Tax percentage applied based on region
tax_amount (float): Tax amount in USD
shipping_cost (float): Shipping and handling charges
total_amount (float): Final transaction amount (subtotal - discount + tax + shipping)

Sales & Marketing

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



Customer Experience

is_returned (boolean): Whether the item was returned (True/False)
return_reason (string): Reason for return if applicable

Defective, Wrong Item, Not as Described, Changed Mind, Damaged in Shipping


satisfaction_score (integer): Customer satisfaction rating (1-10 scale). Note: Contains missing values

Derived Fields

year (integer): Year extracted from transaction date
month (integer): Month extracted from transaction date (1-12)
quarter (integer): Quarter of the year (1-4)
day_of_week (string): Day name (Monday, Tuesday, etc.)
is_weekend (boolean): Whether transaction occurred on weekend
profit_margin (float): Estimated profit margin percentage (0.15-0.45)
profit (float): Estimated profit amount in USD

TechnoMart Sales Dataset - Data Dictionary
Dataset Overview
This dataset contains comprehensive sales transaction data from TechnoMart, a multi-channel retail company specializing in consumer electronics, clothing, home goods, and lifestyle products. TechnoMart operates across five major regions globally and sells through multiple channels including online platforms, physical retail stores, mobile applications, and third-party marketplaces.
The dataset spans from January 2020 to December 2024, capturing the company's sales performance through various market conditions including the COVID-19 pandemic, supply chain disruptions, and economic recovery periods. This rich dataset provides insights into customer behavior, product performance, seasonal trends, and operational efficiency across different business segments.
Dataset Specifications

Total Records: ~12,000 transactions
Time Period: January 2020 - December 2024
Business Type: Multi-channel retail
Geographic Coverage: Global (5 regions)
Product Categories: 8 main categories

Column Definitions
Transaction Identifiers
ColumnTypeDescriptionorder_idstringUnique identifier for each sales transaction. Format: ORD_YYYYMMDD_XXXXdatedatetimeDate when the transaction occurred
Customer Information
ColumnTypeDescriptioncustomer_idstringUnique customer identifier. Format: CUST_XXXXXcustomer_ageintegerCustomer's age in years (18-80). Note: Contains missing valuescustomer_genderstringCustomer's gender (Male, Female, Other). Note: Contains missing valuescustomer_segmentstringCustomer classification based on purchasing behavior and value
Customer Segment Values:

Premium: High-value customers with frequent purchases
Standard: Regular customers with moderate purchase frequency
Budget: Price-sensitive customers seeking deals
Enterprise: B2B customers with bulk purchases

Product Information
ColumnTypeDescriptionproduct_categorystringMain product classificationproduct_namestringSpecific product name/model
Product Category Values:

Electronics: Computers, phones, gadgets, accessories
Clothing: Apparel, shoes, fashion accessories
Home & Garden: Furniture, decor, gardening supplies
Sports & Outdoors: Fitness equipment, outdoor gear
Books: Physical and digital books, educational materials
Health & Beauty: Personal care, cosmetics, wellness products
Automotive: Car accessories, tools, maintenance products
Toys & Games: Children's toys, board games, video games

Financial Data
ColumnTypeDescriptionunit_pricefloatPrice per individual item in USDquantityintegerNumber of items purchased in the transactionsubtotalfloatTotal before discounts and taxes (unit_price × quantity)discount_ratefloatPercentage discount applied (0.0 to 1.0)discount_amountfloatDollar amount of discount appliedtax_ratefloatTax percentage applied based on regiontax_amountfloatTax amount in USDshipping_costfloatShipping and handling chargestotal_amountfloatFinal transaction amount (subtotal - discount + tax + shipping)
Sales & Marketing
ColumnTypeDescriptionsales_channelstringChannel through which the sale was madepayment_methodstringMethod used for paymentregionstringGeographic region of the salesales_repstringName of the sales representative handling the transactionlead_sourcestringOriginal source that brought the customer
Sales Channel Values:

Online: Company website
Retail Store: Physical brick-and-mortar locations
Phone: Telephone sales
Mobile App: Mobile application purchases
Third Party: External marketplaces and partners

Payment Method Values:

Credit Card, Debit Card, PayPal, Bank Transfer, Cash, Cryptocurrency

Region Values:

North America: USA, Canada, Mexico
Europe: EU countries, UK, Norway, Switzerland
Asia Pacific: Japan, Australia, South Korea, Singapore
Latin America: Brazil, Argentina, Chile, Colombia
Middle East & Africa: UAE, Saudi Arabia, South Africa

Lead Source Values:

Organic Search, Paid Search, Social Media, Email, Referral, Direct, Advertisement

Customer Experience
ColumnTypeDescriptionis_returnedbooleanWhether the item was returned (True/False)return_reasonstringReason for return if applicablesatisfaction_scoreintegerCustomer satisfaction rating (1-10 scale). Note: Contains missing values
Return Reason Values:

Defective, Wrong Item, Not as Described, Changed Mind, Damaged in Shipping

Derived Fields
ColumnTypeDescriptionyearintegerYear extracted from transaction datemonthintegerMonth extracted from transaction date (1-12)quarterintegerQuarter of the year (1-4)day_of_weekstringDay name (Monday, Tuesday, etc.)is_weekendbooleanWhether transaction occurred on weekendprofit_marginfloatEstimated profit margin percentage (0.15-0.45)profitfloatEstimated profit amount in USD
Data Quality Notes
This dataset intentionally includes realistic data quality issues commonly found in business datasets:

Missing Values: Some records have missing customer demographics and satisfaction scores
Duplicates: Approximately 1% duplicate records (common in real-world data)
Outliers: Occasional extreme values in quantity and pricing
Data Entry Errors: Rare instances of unusual price points
Seasonal Patterns: Built-in seasonality effects for holidays, summer, back-to-school periods

Business Context
TechnoMart's business model includes:

Multi-channel Strategy: Strong online presence with physical retail support
Global Operations: Diverse geographic footprint with region-specific tax structures
Customer Segmentation: Differentiated service levels based on customer value
Seasonal Business: Significant holiday season sales spikes
Return Policy: Liberal return policy resulting in ~8% return rate
Discount Strategy: Regular promotional campaigns with varying discount levels
