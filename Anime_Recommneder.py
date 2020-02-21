import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

anime = pd.read_csv("C:\\Users\\Excalibur\\Desktop\\Recommender_System\\anime.csv")
anime.dtypes
anime['anime_id'] = anime['anime_id'].astype(np.int32)
anime['members'] = anime['members'].astype(np.int32)
anime['rating'] = anime['rating'].astype(np.float16)

ratings = pd.read_csv("C:\\Users\\Excalibur\\Desktop\\Recommender_System\\ratings\\rating.csv")

mem = anime.memory_usage(index=True).sum()
mem/(1024**2)

mem2 = ratings.memory_usage(index=True).sum()
mem2/(1024**2)

# Need to optimize the memory usage
# Checking datatype of each column
ratings.dtypes

# Determine max value to get right byte to assign
# int8	Byte (-128 to 127)
# int16	Integer (-32768 to 32767)
# int32	Integer (-2147483648 to 2147483647)
np.max(ratings['user_id'])
np.min(ratings['user_id'])

np.max(ratings['anime_id'])
np.min(ratings['anime_id'])

np.max(ratings['rating'])
np.min(ratings['rating'])

# Reduce the memory by changing the datatype
ratings['user_id'] = ratings['user_id'].astype(np.int32)
ratings['anime_id'] = ratings['anime_id'].astype(np.int32)
ratings['rating'] = ratings['rating'].astype(np.int8)
mem2 = ratings.memory_usage(index=True).sum()
mem2/(1024**2)
# Reduced memory from 178 MB to 67 MB

ratings['rating'] = ratings['rating'].replace({-1: np.nan})
ratings.head(10)

# Considering only type movie
anime_movies = anime[anime['type'] == 'Movie']

# Merging the user ratings with movie data
merge_anime = ratings.merge(anime_movies, on='anime_id')

merge_anime.rename(columns={'rating_x': 'rating_user'}, inplace=True)

# Generating Pivot table
pivot_table = amovie_data.pivot_table(index=['user_id'], columns=['anime_id'], values='rating_user')

# Normalize data using min-max scaller
normalized_amovie = pivot_table.apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)), axis=1)

# Remove all those users who have not rated any movies
normalized_amovie.fillna(0, inplace=True)
normalized_amovie = normalized_amovie.T
normalized_amovie = normalized_amovie.loc[:, (normalized_amovie != 0).any(axis=0)]

# Convert normalized values into array
x = normalized_amovie.values

# Applying cosine similarity for the normalized data
cosine_sim = cosine_similarity(x)
item_sim = pd.DataFrame(cosine_sim, index=normalized_amovie.index, columns=normalized_amovie.index)

# Recommends anime movies based on highest similarity


def recommend(id, user_id, a):
    try:
        sorted_user_predictions = item_sim[id].sort_values(ascending=False).reset_index()
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['anime_id'].isin(
            a)].sort_values(by=id, ascending=False).head(4)
        movies_df = recommendations_df.merge(
            anime_movies, on='anime_id').rename(columns={id: 'similarity'})
        watched_anime = anime_movies[anime_movies['anime_id'] == id].name.values
        movie_name = movies_df.name.values
        similarity = movies_df.similarity.values
        azip = zip(movie_name, similarity)
        print('Since user {0} has watched anime {1}'.format(user_id, watched_anime))
        for anime, sim in azip:
            print('{0} and Similarity: {1:.2f}'.format(anime, sim))
    except KeyError:
        pass


# Provides a list of already watched movies of that user
def user_recommend(user_id):
    a = []
    user_rated = ratings[ratings['user_id'] == user_id]
    user_movie_aw = user_rated.merge(anime_movies, on='anime_id')
    a = user_movie_aw.anime_id.values
    for i in range(len(a)):
        recommend(a[i], user_id, a)
        #watched_list = user_movie_aw.name.values
    #print('User {0} has already rated {1} anime movies'.format(user_id,len(a)))
    #print(*watched_list, sep="\n")

# user_recommend(5)


''''
anime.info()
ratings['rating'].describe(include='all')
ratings.groupby('rating')['anime_id'].nunique()
ratings.hist(column='rating', figsize=(10, 10), bins=5, grid=False)


type_count = anime_df['type'].value_counts()
type_count = type_count.reset_index()
type_count.columns = ['type', 'counts']
type_count
plt.bar(type_count['type'], type_count['counts'], align='center', alpha=1)'''
