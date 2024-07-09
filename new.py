#====================import libraries=================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import StandardScaler

#==================data selection===================================
print("Data Selection")
print()
dataframe=pd.read_csv("twodata.csv")
print(dataframe.head(10))
print()


#==================preprocessing===============================
#checking missing values
print("Checking Missing Values")
print()
print(dataframe.isnull().sum())
print()

#drop unneccesary columns because its not required
print()
print(dataframe.head(10))
columns=["Mother's qualification","Father's qualification",
         "Mother's occupation","Father's occupation","Inflation rate",
         "Marital status","Application mode",
         "Daytime/evening attendance","Curricular units 1st sem (credited)",
         "Curricular units 2nd sem (without evaluations)",
         "Curricular units 1st sem (without evaluations)",
         "Curricular units 2nd sem (credited)","Application order"]
dataframe.drop(columns, inplace=True, axis=1)
print(dataframe.head(10))
print() 


#====================Data splitting==============================

#split the data into test and train 
X =dataframe.drop(["Target"],axis=1)
Y = dataframe['Target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=2)
print("Data splitting")
print()
print("Training data shape",X_train.shape)
print("Testing data shape",X_test.shape) 
print()


#=================Clustering=====================================
student_df=pd.read_csv("twodata.csv")
student_df.head()
relevant_cols = ["Course", 
                 "Curricular units 1st sem (credited)"]
student_df = student_df[relevant_cols]
scaler = StandardScaler()
scaler.fit(student_df)
scaled_data = scaler.transform(student_df)  
clusters_centers = []
k_values = []
    
for k in range(1, 12):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(scaled_data)   
    clusters_centers.append(kmeans.inertia_)
    k_values.append(k)
figure = plt.subplots(figsize = (8, 4))
kmeans = KMeans(n_clusters = 2)
kmeans.fit(scaled_data)
student_df["clusters"] = kmeans.labels_
student_df.head()
plt.scatter(student_df["Course"], 
            student_df['Curricular units 1st sem (credited)'], 
            c = student_df["clusters"])
kmeans.fit(X)
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='r')
plt.show()
#=======================classification===========================

#========================logistic regression=======================

#initialize the model
LR = LogisticRegression(C=0.01, solver='liblinear')

LR.fit(X_train,y_train)

lr_prediction=LR.predict(X_train)

#confusion matrix
print("-----------------------------------------------------")
print("Result for logistic regression")
cm=confusion_matrix(y_train, lr_prediction)
print()
print("1.Confusion Matrix",cm)
print()

TP1 = cm[0][0]
FP1 = cm[0][1]
FN1 = cm[1][0]
TN1 = cm[1][1]

#Total TP,TN,FP,FN
Total=TP1+FP1+FN1+TN1

#Accuracy Calculation
Accuracy_lr=((TP1+TN1)/Total)*100
print("2.Accuracy",Accuracy_lr,'%')
print()

#===========================decision tree=======================

#initialize the model
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)

#fitting the model
dt.fit(X_train, y_train)

#predicting the model
dt_prediction=dt.predict(X_train)

#confusion matrix
print("-----------------------------------------------------")
print("Result for Decision tree")
cm1=confusion_matrix(y_train, dt_prediction)

print("1.Confusion Matrix",cm1)
print()

TP = cm1[0][0]
FP = cm1[0][1]
FN = cm1[1][0]
TN = cm1[1][1]

#Total TP,TN,FP,FN
Total=TP+FP+FN+TN

#Accuracy Calculation
Accuracy_dt=((TP+TN)/Total)*100
print("2.Accuracy",Accuracy_dt,'%')
print()