import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
class Classifer:
    def __init__(self,features_1,features_2,features_3):
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        np.random.seed(0)
        self.df = df = pd.DataFrame({
      'product_id': np.arange(1,21),
      'category': np.random.choice(['A','B','C'],20),
      'price': np.random.randint(100,1000,20),
      'discount': np.random.randint(0,50,20),
      'units_sold': np.random.randint(1,100,20)
       })
        df['revenue'] = df['price'] * df['units_sold'] * (1 - df['discount']/100)

  
    def Logistic_Regression(self):
           df2 = pd.DataFrame({
          'product_id': np.arange(1,21),
          'category': np.random.choice(['A','B','C'],20),
          'price': np.random.randint(100,1000,20),
          'discount': np.random.randint(0,50,20),
          'units_sold': np.random.randint(1,100,20) 
           })
           df2['revenue'] = df2['price'] * df2['units_sold'] * (1 - df2['discount']/100)
           df2['high_revenue'] = (df2['revenue'] > 5000).astype(int)
           x = df2[['price','discount','units_sold']]
           y = df2['high_revenue']
           x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
           model = LogisticRegression()
           model.fit(x_train,y_train)
           new_df = pd.DataFrame([[self.features_1,self.features_2,self.features_3]],columns=['price','discount','units_sold'])
           prediction = model.predict(new_df)
           global message
           if prediction[0] == 0:
               message = "Revenue is so low"
           else:
               message =  "Revenue is so high"
           return message

    def Random_Forest_Classifier(self):
           df2 = pd.DataFrame({
          'product_id': np.arange(1,21),
          'category': np.random.choice(['A','B','C'],20),
          'price': np.random.randint(100,1000,20),
          'discount': np.random.randint(0,50,20),
          'units_sold': np.random.randint(1,100,20) 
           })
           df2['revenue'] = df2['price'] * df2['units_sold'] * (1 - df2['discount']/100)
           df2['high_revenue'] = (df2['revenue'] > 5000 ).astype(int)
           x = df2[['price','discount','units_sold']]
           y = df2['high_revenue']
           x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
           model = RandomForestClassifier(n_estimators=1000,random_state=42)
           model.fit(x_train,y_train)
           new_df = pd.DataFrame([[self.features_1,self.features_2,self.features_3]],columns=['price','discount','units_sold'])
           prediction = model.predict(new_df)
           global  message 
           if prediction[0] == 0:
                message = "Revenue is so Low"
           else:
                message = "Revenue is so High"
           return message

    def KNN_Classifier(self):
           df2 = pd.DataFrame({
          'product_id': np.arange(1,21),
          'category': np.random.choice(['A','B','C'],20),
          'price': np.random.randint(100,1000,20),
          'discount': np.random.randint(0,50,20),
          'units_sold': np.random.randint(1,100,20) 
           })
           df2['revenue'] = df2['price'] * df2['units_sold'] * (1 - df2['discount']/100)
           df2['high_revenue'] = (df2['revenue'] > 5000).astype(int)
           x = df2[['price','discount','units_sold']]
           y = df2['high_revenue']
           x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
           model = KNeighborsClassifier(n_neighbors=4)
           model.fit(x_train,y_train)
           new_df = pd.DataFrame([[self.features_1,self.features_2,self.features_3]],columns=['price','discount','units_sold'])
           prediction = model.predict(new_df)
           global message 
           if prediction[0] == 0:
                message = "Revenue is so Low"
           else:
                message = "Revenue is so High"
           return message






