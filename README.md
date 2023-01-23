# Zomato-Review-Sentniment-Analysis
This project aims to analyze customer sentiment towards restaurants using Zomato restaurant review data and segmenting them based on factors such as cost, cuisine, and other attributes.

Divided restaurants into segments using clustering techniques(PCA for visualtisations also) and also did sentiment analysis for the reviews.

First, I used KMeans clustering to cluster the restaurant based on cost, different cuisines, locality and meta-tags. Once a similar clusters of restaurants formed, I tried to analyse various clusters and look for some trend in the data.

Also this project focussed on Customers and Company, I analyzed the sentiments of the reviews given by the customer in the data and made some useful conclusion in the form of Visualizations. Basically the Sentiment analysis

![image](https://user-images.githubusercontent.com/33064867/208748670-56be71d4-a50e-4474-82f9-55c3a1efe1c0.png)

-- Project Status: [Completed]

# Objective
India is quite famous for its diverse multi cuisine available in a large number of restaurants and hotel resorts, which is reminiscent of unity in diversity. The Project focuses on Customers and Company, we have to analysed the sentiments of the reviews given by the customer and made some useful conclusion in the form of Visualizations. Also, clustered the zomato restaurants into different segments.

# Methods Used
Descriptive Statistics
Data Visualization
Machine Learning -Supervised-Learning -Unsupervised-Learning

# Technologies
Python
Pandas
Numpy
Matplotlib
Seaborn
Scikit-learn
NLP

# Data
The Zomato-Restaurants dataset comprises of 2 files. "Zomato restaurant reviews" has Name of the Restaurant, Name of the customer, their review, rating, follower details, Time of Review and number of photos uploaded along with the review. This data has been mainly used for Sentiment analysis. "Zomato Restaurant names and Metadata" has the details of Name of the Restaurant, Link to order on their restaurant on zomato, Average cost, Tags for the restaurants, Cuisines and timings. This data has been mainly used for clustering.

# Project Description
EDA - Performed exploratory data analysis and text preprocessing
Data Cleaning - We have to drop the entire feature as there are 50% null values.
Feature Selection - For sentiment analysis, we have used rating and reviews features. - For clustering we got cost, cuisine and timing of the restaurant as the features to build the model.
Model development - For sentiment analysis, developed different models like:- Multinomial NB, Logistic regression, Random forest classifier - For clustering the restaurants we have used the k-means and hirerchical clustering

# Feature Engineering
1. Exploratory data analysis and text preprocessing were performed.
2. Data cleaning was performed by dropping entire features that had 50% null values.
3. Feature selection was used for sentiment analysis, using the rating and reviews features.
4. For clustering, the cost, cuisine and timing of the restaurant were used as features to build the model.

# Algorithms Used
1. KMeans Clustering
2. Hierarchical Clustering
3. Multinomial Naive Bayes
4. Logistic Regression
5. Random Forest Classifier

# Needs of this project
data exploration
data processing/cleaning
text preprocessing/cleaning
sentiment analysis on reviews
cluster the restaurant into different segments.

# Sentiment Analysis
1. Plotted the distribution of ratings to have an understanding of the proportion of good and bad reviews.
2. Created visualizations including top 10/bottom 10 restaurants in terms of average rating.
3. Pre-processing was done such as removing emojis, punctuations and only used Adjectives and verbs to reduce dimensionality.
4. TF-IDF vectorizer was used to transform the dataset.
5. TextBlob() was used to do sentiment analysis.
6. A rmse score of 0.88 was achieved.
7. Created a word cloud for positive as well as negative phrases.

# Clustering
1. Calculated the time each Restaurant was opened weekly.
2. Pre-processing was done by clubbing some cuisines together so that one-hot encoding would be possible, removed unwanted variables and normalized the data.
3. Clustered the data using K-means and Hierarchical clustering.

# Future Scope
1. Incorporating more data from other sources such as social media to get a more complete picture of customer sentiment towards restaurants.
2. Enhancing the model using more advanced techniques such as deep learning and neural networks.
3. Predictive modeling using the segmented and analyzed data to predict customer sentiment or restaurant success in the future.
4. Building a recommendation system that can recommend restaurants to customers based on their preferences and past experiences.
5. Integrating with other systems such as a restaurant's website or mobile app, to provide real-time recommendations and feedback to customers.
6. Improving the visualizations by creating more interactive and dynamic visualizations that allow the user to explore the data in more detail.

# Conclusions
The end output of the project is segmented restaurants and useful conclusions from the customer reviews which can be used by the restaurant industry to improve their offerings, pricing strategies, and target marketing efforts. The project also provides a way to improve customer service and make sure that customer complaints are addressed in a timely manner.
