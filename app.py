from flask import Flask, render_template,request,redirect,url_for,session

#DBMS connection
import pymysql as ps 
import pandas as pd
import numpy as np

db = ps.connect(
  host="localhost",
  user="root",
  password="password",
  database="users"
)
cur=db.cursor()

#Reading the DataFrames from CSV
df1= pd.read_csv('./tmdb_5000_credits.csv')
df2 = pd.read_csv('./tmdb_5000_movies.csv')

#Merging both the DataFrames
df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

#Calculating the weighted average as some movies have less ratings
C = df2['vote_average'].mean()

m = df2['vote_count'].quantile(0.9)

#Count of movies that qualify for the required criteria
q_movies = df2.copy().loc[df2['vote_count'] >= m]

#Calculating the weighted rating of the movies
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

pop= df2.sort_values('popularity', ascending=False)

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

app = Flask(__name__)
app.secret_key = "abcd"

auth="Please log in to continue"

@app.route("/")
def initial():
    session["status"] = False
    return render_template("login.html", auth=auth)

@app.route("/login",methods = ["POST","GET"])
def index():
    if request.method == "POST" and request.form['submit'] == 'Login':
        username = request.form['username']
        password = request.form['password']
        cur.execute(f"SELECT username FROM users WHERE username = '" + f"{username}" + "' AND password = '" f"{password}" + "'")
        Id_s = cur.fetchall()
        print(Id_s)
        if len(Id_s) == 0:
            return render_template("login.html", auth="Wrong Login Credentials")
        else:
            return redirect(url_for("movies"))
    else:
        return render_template("login.html", auth=auth)


@app.route("/movies", methods= ["POST", "GET"])
def movies():
    if request.method == "POST" and request.form['submit'] == 'Recommend':
        movie = request.form['movie']
        recommendations = get_recommendations(movie)
        recommendations = recommendations.to_list()
        return render_template("recommendations.html", recommendations = recommendations, movie = movie)
    else:
        movies_list = df2['title'].to_list()
        return render_template('movies.html', movies_list = movies_list)


if __name__=='__main__':
    app.run(host='localhost',port=5000)
