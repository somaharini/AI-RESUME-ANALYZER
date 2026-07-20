import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
#read the Csv file
df=pd.read_csv("churn.csv")

#cleaning the data: droping the customerID column and converting Total Charges to numeric
df.drop("customerID",axis=1,inplace=True)
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce").fillna(0)

#encoding the categorical columns
le=LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col]=le.fit_transform(df[col])

#splitting the data into train and test sets\
x=df.drop("Churn",axis=1)
y=df["Churn"]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
model=xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss')
model.fit(x_train,y_train)

#save the model to pickle file
with open("model.pkl","wb") as f:
    pickle.dump(model,f)