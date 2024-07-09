import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template
app=Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
     return render_template("gui.html")
@app.route('/pred', methods =["POST", "GET"])
def pred():
    gen=int(request.form.get("GEN"))
    dis=int(request.form.get("DIS"))
    c1=float(request.form.get("SEM1"))
    c2=float(request.form.get("SEM2"))
    c3=int(request.form.get("prev"))
    c4=int(request.form.get("enroll"))
    c5=int(request.form.get("enroll1"))
    df=pd.read_csv("dataset.csv")
    features = df[['Gender','Displaced','Curricular units 1st sem (grade)','Curricular units 2nd sem (grade)',
                   'Previous qualification','Curricular units 1st sem (enrolled)','Curricular units 2nd sem (enrolled)']]
    target = df['Target']
    X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=1,random_state = 2)
    dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    dt.fit(X_train, y_train)
    
    ip=[]
    ip.append(gen)
    ip.append(dis)
    ip.append(c1)
    ip.append(c2)
    ip.append(c3)
    ip.append(c4)
    ip.append(c5)
    #knn_ip=pd.core.frame.DataFrame(ip)
    Result=dt.predict([ip])
    return render_template('gui.html',Result=Result)
        

if __name__=='__main__':
    app.run(debug=True,port=4123) 
