from django.shortcuts import render
from django.http import HttpResponse
# ML libraries start
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)

FILENAME = 'movie_metadata_filtered_aftercsv.csv'
FILE2 = 'test.csv'
THRESHOLD_PREDICTION = 1
# end

def _make_in_format(filename):
    datadf = pd.read_csv(filename)
    #separate classes and stuffs
    y = np.array(datadf['imdb_score'])
    datadf = datadf.drop(datadf.columns[[0,9]],axis=1)
    #normalize
    datadf = (datadf-datadf.mean())/(datadf.max()-datadf.min())
    X = np.array(datadf)

    return X,y

def accuracy_score(y_test,predictions):
        correct = []
        for i in range(len(y_test)):
            if y_test[i]>=predictions[i]-THRESHOLD_PREDICTION and y_test[i]<=predictions[i]+THRESHOLD_PREDICTION:
                correct.append(1)
            else:
                correct.append(0)

        accuracy = sum(map(int,correct))*1.0/len(correct)
        return accuracy


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

        ##logRegression
        X,y = _make_in_format(FILENAME)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
        model = LogisticRegression(solver='newton-cg',multi_class='ovr',max_iter=200,penalty='l2')
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)
        
        logReg_accuracy = accuracy_score(y_test,predictions)*100
        


        return HttpResponse(actor_1_facebook_likes)
    else:
        return render(request, 'index.html', {})