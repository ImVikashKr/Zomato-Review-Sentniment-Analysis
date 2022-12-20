# Zomato-Review-Sentniment-Analysis
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

# Needs of this project
data exploration
data processing/cleaning
text preprocessing/cleaning
sentiment analysis on reviews
cluster the restaurant into different segments.

# Sentiment analysis
-Plotted the distribution of ratings to have an understanding of the proportion of good and bad reviews. -Made many visualizations which include, top 10/bottom 10 restaurants in terms of average rating. -Had a clear understanding of the cost summary of the restaurants. -Done pre-processing such as removing emojis, punctuations and only used Adjectives and verbs to reduce dimensionality. -TF-IDF vectorizer was used to transform the dataset. -Used TextBlob() to do sentiment analysis. Got a rmse score of 0.88. -Prepared a word cloud for positive as well as negative phrases.

# Clustering
-Calculated the time each Restaurant was opened weekly. -Did the pre-processing, Clubbed some cuisines together so that one-hot encoding would be possible. -Removed the unwanted variables and Normalized the data. -Clustered the data using K-means as well as Hierarchical clustering.

# Conclusions
Divided restaurants into segments using clustering techniques and also did sentiment analysis for the reviews.
