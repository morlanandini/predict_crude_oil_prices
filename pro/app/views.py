import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split




from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def fun(request):


    return render(request,'index.html')

def show(request):
    test_dtm={}
    x=int(request.GET['date'])
    y=int(request.GET['mon'])
    z=int(request.GET['year'])
    test_dtm['date']=[int(request.GET['date'])]
    test_dtm['Month']=[int(request.GET['mon'])]
    test_dtm['Year']=[int(request.GET['year'])]
    test_dtm = pd.DataFrame.from_dict(test_dtm)

    


    df=pd.read_csv('/Users/nandinim/Desktop/A3/major_project/front_end/oil.csv', engine='python')
    df.dropna(inplace = True)
    # new data frame with split value columns
    new = df["Date"].str.split("/", n = 2, expand = True)
  
    # making separate first name column from new data frame
    df["date"]= new[0]
  
    # making separate last name column from new data frame
    df["Month"]= new[1]
    df["Year"]= new[2]
    # Dropping old Name columns
    df.drop(columns =["Date"], inplace = True)


    X=df.iloc[:,1:].values
    Y=df.iloc[:,0].values
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1)


    rand_forest_model =  RandomForestRegressor(n_estimators = 1000 , max_features = 2, oob_score = True ,  random_state = 115)
    rand_forest_model.fit(x_train,y_train)

    y_pred2=rand_forest_model.predict(x_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))
    pred = rand_forest_model.predict(test_dtm)
    test_dtm['Predicted Price'] = pred

    print(test_dtm['Predicted Price'])

    



    
    


   
    return render(request,'output.html',{'date':x,'month':y,'year':z,'price':test_dtm.iloc[0,3]})