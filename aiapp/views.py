from django.shortcuts import render
from django.http import HttpResponse
import os
import pickle
import numpy as np
import pandas as pd

FILENAME = 'movie_metadata_filtered_aftercsv.csv'
def _make_in_format(test):
    FILENAME = "movie_metadata_filtered_aftercsv.csv"
    dir_path = os.path.dirname(os.path.realpath(FILENAME))
    FILENAME = dir_path + "/aiapp/movie_metadata_filtered_aftercsv.csv" 
    datadf = pd.read_csv(FILENAME)
    #separate classes and stuffs
    datadf = datadf.drop(datadf.columns[[0,9]],axis=1)
    # print(datadf.mean())
    # test = test.drop(test.columns[[0,9]],axis = 1)

    #normalize
    test = test.mean()
    dfmean = datadf.mean()
    dfmax = datadf.max()
    dfmin = datadf.min()
    # print(dfmean)
    # print(test)
    datadf = (test-dfmean)/(dfmax-dfmin)
    X = np.array(datadf)
    X = X.reshape(1,9)
    return X


# Create your views here.
def predict_rating(request):
    if request.method == 'POST':
        actor_1_facebook_likes = request.POST.get('actor_1_facebook_likes')
        actor_2_facebook_likes = request.POST.get('actor_2_facebook_likes')
        director_facebook_likes = request.POST.get('director_facebook_likes')
        movie_facebook_likes = request.POST.get('movie_facebook_likes')
        cast_facebook_likes = request.POST.get('cast_facebook_likes')
        movie_budget = request.POST.get('movie_budget')
        num_critic_for_reviews = request.POST.get('num_critic_for_reviews')
        num_user_reviews = request.POST.get('num_user_reviews')
        num_voted_users = request.POST.get('num_voted_users')

        ##logRegression Model
        Pkl_Filename = "logreg_model.pkl"  
        dir_path = os.path.dirname(os.path.realpath(Pkl_Filename))
        FILE2 = dir_path + "/aiapp/test.csv" 
        dir_path += "/aiapp/logreg_model.pkl"
        print(dir_path)
        with open(dir_path, 'rb') as file:  
            Pickled_Model = pickle.load(file)
        test = pd.DataFrame({
                    'num_critic_for_reviews': [num_critic_for_reviews],
                    'director_facebook_likes': [director_facebook_likes],
                    'actor_1_facebook_likes': [actor_1_facebook_likes],
                    'num_voted_users': [num_voted_users],
                    'cast_total_facebook_likes': [cast_facebook_likes],
                    'num_user_for_reviews': [num_user_reviews],
                    'budget': [movie_budget],
                    'actor_2_facebook_likes': [actor_2_facebook_likes],
                    # 'imdb_score': [8],
                    'movie_facebook_likes': [movie_facebook_likes]
                })
        
        p = _make_in_format(test)
        
        print("Normalized Data that you have sent is ",p)
        
        prediction = Pickled_Model.predict(p)

        print("The predicted IMDB score is ", prediction)
        if prediction<3:
            return HttpResponse("Prediction of Movie is <b>Disaster</b> with IMDB rating " + str(prediction))
        elif prediction>=3 and prediction<5:
            return HttpResponse("Prediction of Movie is <b>Flop</b> with IMDB rating " + str(prediction))
        elif prediction>=5 and prediction<7:
            return HttpResponse("Prediction of Movie is <b>Average</b> with IMDB rating " + str(prediction))
        elif prediction>=7 and prediction<9:
            return HttpResponse("Prediction of Movie is <b>Hit</b> with IMDB rating " + str(prediction))
        else:
            return HttpResponse("Prediction of Movie is <b>Blockbuster</b> with IMDB rating " + str(prediction))
        return HttpResponse("Error")
    else:
        return render(request, 'index.html', {})