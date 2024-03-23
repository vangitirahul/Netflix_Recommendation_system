#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# Load the dataset
df = pd.read_csv(r"C:\Users\vangi\OneDrive\Desktop\RAHUL\DATA SETS\Netflix_movies_and_tv_shows_clustering.csv")

# Display the first few rows of the dataframe
df.head()


# In[8]:


# Check for missing values and get a summary of the dataset
df_info = df.info()
df_describe = df.describe(include='all')

print('Missing values and data types:')
print(df_info)
print('\
Summary statistics:')
print(df_describe)


# In[9]:


# Handle missing values, check for duplicates, and ensure correct data types

# Fill missing values for 'director', 'cast', 'country' with 'Unknown'
df[['director', 'cast', 'country']] = df[['director', 'cast', 'country']].fillna('Unknown')

# For 'date_added', we'll remove rows with missing values as they are a small portion of the dataset
df = df.dropna(subset=['date_added'])

# Check for duplicates
duplicates = df.duplicated().sum()

# Ensure data types
# Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')

df['date_added'] = pd.to_datetime(df['date_added'])

# Convert 'release_year' to integer (should already be int but just to confirm)
df['release_year'] = df['release_year'].astype(int)

print('Missing values handled, duplicates checked, and data types ensured.')
print('Number of duplicate rows:', duplicates)


# In[11]:


import pandas as pd


tqdm.pandas()

# Load the dataset
df = pd.read_csv(r"C:\Users\vangi\OneDrive\Desktop\RAHUL\DATA SETS\Netflix_movies_and_tv_shows_clustering.csv")

# Distribution of Movies vs. TV Shows
content_type_distribution = df['type'].value_counts()

# Plotting the distribution
plt.figure(figsize=(10, 6), facecolor='white')
content_type_distribution.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title('Distribution of Netflix Content Types')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.show()

print('Distribution of content types plotted.')


# In[12]:


# Exploring trends over the years in terms of content added
df['year_added'] = df['date_added'].str[-4:]
yearly_content_count = df.groupby('year_added')['type'].value_counts().unstack().fillna(0)

# Plotting the trends
plt.figure(figsize=(12, 6), facecolor='white')
yearly_content_count.plot(kind='bar', stacked=True, color=['skyblue', 'lightgreen'])
plt.title('Trends of Netflix Content Added Over the Years')
plt.xlabel('Year Added')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Content Type', labels=['Movie', 'TV Show'])
plt.grid(axis='y', linestyle='--')
plt.show()

print('Trends of content added over the years plotted.')


# In[13]:


#creating a recommendation system
# Combine relevant features into a single string for each movie/show
df['combined_features'] = df['director'].fillna('') + ' ' + df['cast'].fillna('') + ' ' + df['listed_in'].fillna('') + ' ' + df['description'].fillna('')

# Display the head of the dataframe to check the combined features
df[['title', 'combined_features']].head()


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features to a matrix of TF-IDF features
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Display the shape of the TF-IDF matrix to understand its dimensions
print('Shape of TF-IDF Matrix:', tfidf_matrix.shape)


# In[15]:


from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display the shape of the cosine similarity matrix to understand its dimensions
print('Shape of Cosine Similarity Matrix:', cosine_sim.shape)


# In[16]:


def recommend_titles(title, cosine_sim=cosine_sim, df=df, top_n=5):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top 10 most similar movies
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

# Test the recommendation function with a sample title
sample_title = 'Inception'
recommended_titles = recommend_titles(sample_title)
print('Recommendations for "' + sample_title + '":')
print(recommended_titles)


# In[ ]:




