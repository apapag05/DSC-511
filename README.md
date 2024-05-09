# DSC-511 PROJECT
## Classification of Argentina properties (rent/sale)

Original data link -> https://www.kaggle.com/datasets/msorondo/argentina-venta-de-propiedades?select=uy_properties_crude.csv
Datasets link -> https://drive.google.com/drive/folders/1rj2_1tq-roAhYP5mDtbFghAWh6P21vx7?usp=sharing
----------------------------------------------------------------------------------------------------------------------------------

The jupyter files follow the followin order:
--------------------------------------------
1. preprocessing_done_final
2. EDA_Argentina_house_pynb
3. Text_Analytics_final
4. MachineLearning_Rent & MachineLearning_Sale
5. RS_FINAL
6. rs_2

Dataset Description:
--------------------
The dataset contains information about house sales and rent from Argentina. The columns are:
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

Main Goal/Objective:
--------------------
The primary aim of this project is to perform a multiclass classification problem to predict the price buckets of properties in Argentina, distinguishing between properties that are for sale and those available for rent. This involves extensive preprocessing and analysis to prepare the dataset for effective machine learning modeling.

Preprocessing:
--------------
Our primary focus is to clean and prepare a real estate dataset for analysis, ensuring optimal data quality for predictive modeling. The dataset, initially containing 1 million rows, is loaded into Apache Spark where we establish a session with the necessary memory configurations. We define a schema to maintain correct data types and promptly address data quality issues by calculating and analyzing the percentage of missing values across columns. Key identifiers like 'id', 'ad_type', and 'operation_type' exhibit no missing data, ensuring the integrity of these essential records. However, detailed locations and features such as 'l6' and 'l5' have nearly complete data absence, indicating their limited usability without additional context. Other columns like 'l4', 'surface_total', 'surface_covered', and 'price_period' also show significant data gaps, affecting over 60% of entries, which could undermine detailed property analysis or model accuracy. For properties with moderate missing data in 'rooms', 'bedrooms', and 'bathrooms', which could impact size and capacity analysis, we implement KNN imputation to estimate missing values effectively, based on observed correlations among these attributes. We also undertake currency conversion to standardize financial data across regions by converting all values to euros, facilitating uniform price comparisons. Further steps include refining data types for room and surface measurements, filtering out temporary rental listings, and manually translating columns like 'property_type' and 'operation_type' from Spanish to English to enhance data accessibility and comprehension. After extensive cleaning, including the removal of irrelevant rows and the adjustment of data inconsistencies like ensuring 'surface_total' is logically consistent with 'surface_covered', our dataset is reduced to 259,910 rows. 

EDA:
----
In our exploratory data analysis (EDA), we convert our data into a Pandas DataFrame to facilitate visualization and explored the correlations between numeric variables, noting moderate connections between bedrooms and bathrooms and a perfect correlation between surface_total and surface_covered. Further investigation into the variable 'operation_type' reveals an imbalance between sales and rentals, leading us to examine price trends over time for both categories, where we observe notable variations indicative of market anomalies. We also analyze room distributions, identifying a preference for smaller properties which influences the type of listings. Additionally, we examine property types, finding that apartments and houses dominate the listings while other types serve to niche markets or specific buyer preferences. Through various visualizations, we gain insights into price distributions and the presence of outliers, helping us understand market dynamics. To further refine our analysis, we partition the dataset into two subsets for sales and rental properties, respectively, to develop distinct classification models tailored to each market segment.

Text Analysis:
---------------
In the text analytics phase of our project, we focus on preprocessing and analyzing text to predict price buckets for properties, distinguishing between those for rent and for sale. Initially, the entire dataset is preprocessed in Spanish as automated translation attempts were unsuccessful. We combine titles and descriptions of properties to streamline preprocessing, which includes lowercasing text, removing URLs, mentions, emails, and any special characters, then tokenizing and stemming the text. This reduces our token count by 10 million, significantly refining our dataset. We convert the cleaned text into numerical representations using n-grams and TF-IDF transformations to highlight the most informative terms rather than just the frequent ones. We generate unigrams, bigrams, and trigrams and apply an IDF model to downscale less informative tokens. The resulting TF-IDF vectors are then prepared for integration into our classification models. Word clouds generated from IDF scores emphasize words that are significant but not necessarily frequent, offering insights into unique terms relevant to different property types. These visualizations help identify the thematic focus within the segments of our dataset. Additionally, we separate the dataset based on operation type (rent or sale) and perform the text analysis for each, ensuring that our approach aligns with specific market dynamics. The processed data is saved in Parquet format, optimizing both storage and computation, crucial for managing large datasets and also keeping the schema.

Machine Learning:
-----------------
In the machine learning phase, we implemented a robust process of scaling, encoding, and model evaluation to optimize our datasets for rental and sales price prediction. We updated the data by removing non-essential columns and focusing on crucial numerical data labeled "final_feature," which included categorizing prices into logarithmic buckets for better range management and using visualizations to detect market trends and inconsistencies between rental and sale prices. We normalized geographic coordinates and applied MinMaxScaler to room counts and surface areas to ensure uniform feature contributions. Feature selection was conducted using Random Forest analysis, which highlighted subtle yet significant factors influencing prices in both datasets. While no single factor dominated, the interplay of variables like 'start_year', 'l2_one_hot', and 'bedrooms_scaled' was noted for rentals, and features like 'encoded_end_day_of_week' and 'bathrooms_scaled' were significant for sales. Our modeling began with Decision Trees as well as with Random Forest and appreciated for their handling of uneven data and simplicity but shifted to LinearSVC due to its superior performance, especially when using the complete feature set. Despite initial challenges with overfitting and feature relevance in other models, LinearSVC managed the complexity well, showcasing the need for comprehensive feature utilization. We refined this approach through Optuna-driven hyperparameter tuning, significantly enhancing model performance. The optimal configurations achieved an F1-score of 0.675 for rentals (50-trials) and an accuracy of about 0.74 for sales (10-trials), underlining the effectiveness of our methodological choices and the critical role of thorough hyperparameter tuning in achieving reliable predictive outcomes.

Recommendation systems:
-----------------------
We also implemented a Recommendation System for properties for sale in Argentina. This system uses cosine similarity to suggest 10 properties similar to a given property based on its unique ID. We utilize two versions of our dataset: one with numeric representations for calculating similarities and another with human-readable characteristics for presenting recommendations. After, we assigned a unique ID to each dataset entry using the Spark window function. The next step involves constructing a feature vector for each property using VectorAssembler, which merges relevant features into a single vector. Due to resource constraints, we initially work with only 10% of the complete dataset. Before similarity calculations, we normalize these vectors using Spark's Normalizer to speed up computations and simplify the process. For demonstration, we select a sample property to show how the system identifies and suggests similar listings effectively. We optimize the performance by caching the dataframe and repartitioning it by 'id', which improves the efficiency of our queries across the cluster. A user-defined function (UDF) calculates cosine similarity between properties, and another function recommends properties based on these similarity scores. The feature vector of a specified property is broadcast to all cluster nodes, minimizing network overhead during the similarity computation. The system then computes these similarities, selects the top matches, and demonstrates that the recommendation system can effectively suggest properties closely matching user preferences or needs, as indicated by the high similarity scores of the recommended properties. 

Recommendation systems (2):
---------------------------
We also developed a second recommendation system utilizing KMeans Clustering to establish a "ground truth" for evaluating property recommendations. By clustering similar properties based on their normalized feature vectors and specifying the number of clusters, we organize the properties into sensible groups. This clustering serves as a basis for our recommendations, allowing us to test whether the recommended properties fall into the same cluster as the specified property ID. Consequently, our evaluation function focuses on calculating the accuracy by determining the percentage of recommended properties that share the same cluster as the original property. Of course, we have in mind that the clusters/label we have are not actual truthful data but a column that we created.

