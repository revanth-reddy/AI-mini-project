# ML libraries start
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle 
import os

np.random.seed(0)

FILENAME = 'movie_metadata_filtered_aftercsv.csv'
FILE2 = 'test.csv'
THRESHOLD_PREDICTION = 1
# end

def _make_in_format(filename):
    dir_path = os.path.dirname(os.path.realpath(filename))
    filename = dir_path +"/aiapp/"+ filename
    datadf = pd.read_csv(filename)
    #separate classes and stuffs
    y = np.array(datadf['imdb_score'])
    datadf = datadf.drop(datadf.columns[[0,9]],axis=1)
    # print("*******")
    # print(datadf)
    # print(datadf.mean())
    # print(datadf.max())
    # print(datadf.min())
    # print("*******")
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


def LogRegression():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    model = LogisticRegression(solver='newton-cg',multi_class='ovr',max_iter=200,penalty='l2')
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    print("LogRegression ",accuracy_score(y_test,predictions)*100)

    Pkl_Filename = "logreg_model.pkl"  

    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(model, file)
    
    with open(Pkl_Filename, 'rb') as file:  
        Pickled_Model = pickle.load(file)
    
    p,q = _make_in_format(FILE2)
    print(p)
    
    print(Pickled_Model.predict(p))
    print("Actual imdb",q)

def Knn():
    X,y = _make_in_format(FILENAME)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    # print(X_train)
    model = KNeighborsClassifier(algorithm='ball_tree')
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    # ,num_critic_for_reviews,director_facebook_likes,actor_1_facebook_likes,num_voted_users,cast_total_facebook_likes,num_user_for_reviews,budget,actor_2_facebook_likes,imdb_score,movie_facebook_likes
    # 3,813,22000,27000,1144337,106759,2701,250000000,23000,8,164000
    # p,q = _make_in_format(FILE2)
    # print(p)
    # print(model.predict(p))
    # print("Actual imdb",q)
    Pkl_Filename = "knn_model.pkl"  

    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(model, file)
    
    print("knn score ",accuracy_score(y_test,predictions)*100)
    
    with open(Pkl_Filename, 'rb') as file:  
        Pickled_Model = pickle.load(file)
    
    p,q = _make_in_format(FILE2)
    print(p)
    
    print(Pickled_Model.predict(p))
    print("Actual imdb",q)



def main():
    Knn()
    LogRegression()

if __name__ == '__main__':
    main()
