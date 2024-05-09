# DSC-511 PROJECT
## Classification of Argentina properties (rent/sale)

Data link -> https://drive.google.com/drive/folders/1rj2_1tq-roAhYP5mDtbFghAWh6P21vx7?usp=sharing

Dataset Description:
--------------------
The dataset contains information about house sales and rent. The columns are:
1. id: Notice identifier.
2. ad_type: It contains one unique value, 'propiedad'. It means that it is a property.
3. start_date: The start date of the advertisment of the property.
4. end_date: Date of termination of the advertisement.
5. created_on: Date when the first version of the notice was created.
6. lat: Latitude of the property.
7. lon: Longitude of the property.
8. l1: The highest or broadest level of location detail. In the context of Argentina, it represents the country itself or major regions within the country.
9. l2: Sub-regions or states/provinces.
10. l3: Cities or municipalities within a province.
11. l4, l5 & l6: Subdivisions of the location, such as neighborhoods, zones, or even streets.
12. rooms: Number of rooms.
13. bedrooms: Number of bedrooms.
14. bathrooms: Number of bathrooms.
15. surface_total: Total area in m².
16. surface_covered: Covered area in m².
17. price: Price published in the ad.
18. currency: Currency of the price published in the ad.
19. price_period: Frequency at which a payment is made or expected for a property.
20. title: Title of the advertisement.
21. description: Description of the advertisement.
22. property_type: Type of property (House, Apartment, PH).
23. operation_type: Type of real estate transaction being referenced for each property listed.

Our goals:
----------
- Create a classifier that distinguishes between properties in Argentina that are either up for sale or available for rent. Leveraging a dataset brimming with property specifics—such as room count, location coordinates, pricing details, and more—the aim is to develop machine learning models capable of automatically categorizing each listing. This classification tool will streamline the process for individuals seeking to buy or rent properties, enhancing their search experience with quick and accurate results.
- In addition to property classification, the project involves leveraging text analytics on the property descriptions. By analyzing the textual content, we aim to extract meaningful insights that further refine our understanding of each listing. This text analysis enables us to uncover key features and unique selling points of the properties, helping the classification process.
- Lastly, the project extends to the development of a recommendation system. Leveraging the insights gained from both structured data (property attributes) and unstructured data (textual descriptions), the recommendation system will suggest similar properties to users based on their preferences. This personalized recommendation mechanism enhances user experience by offering tailored suggestions, ultimately aiding individuals in finding properties that closely match their preferences and requirements.

1) PREPROSSESING
----------------
Firstly, we addressed missing values in the dataset. Since columns l4, l5, and l6 had over 80% missing data, we dropped them. Additionally, we removed the 'id' column since it served as a unique identifier with no additional information. Null values in the 'currency' column, accounting for less than 10%, were also dropped. The 'ad_type' column, containing only one unique value (propiedad), was deemed redundant and thus removed. Upon inspecting the 'l1' column, we found entries for 'Uruguay' and 'Estados Unidos', which were not relevant to our analysis of Argentine properties; thus, we eliminated these rows and subsequently dropped the 'l1' column. Erroneous values in the 'bedrooms' column (negative and >100) were excluded, while instances with fewer bedrooms than rooms were retained. Missing values in the 'price' and 'description' columns were also discarded. Recognizing the high correlation between latitude and longitude, we combined them into a single feature, filling null values with the mean. Additionally, we dropped NA values from the 'l3' column due to its low percentage of missing data (~5%). Subsequently, a correlation matrix revealed significant correlations between surface_covered and surface_total (86%), bedrooms and bathrooms (66%), bedrooms and rooms (81%), and bathrooms and rooms (57%). Next, we converted the dataset to pandas and utilized KNN to impute missing values for 'bedrooms', 'rooms', 'bathrooms', 'surface_total', and 'surface_covered'. To ensure integer values, we rounded the float numbers resulting from KNN imputation. Additionally, we filtered out entries where the price was zero.

2) EDA
-------
