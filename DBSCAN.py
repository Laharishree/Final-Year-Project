from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn import metrics
features=["HOSTEL_ID","HOSTEL_NAME","ADDRESS","CITY","COUNTRY","PINCODE","TYPE","RATING","LATITUDE","LONGITUDE","URL","PRICE","FACILITY","GENDER"]

    #reading csv file
data=pd.read_csv("Dataset.csv",names=features)
db=DBSCAN(eps=0.3,min_samples=10,metric='euclidean').fit(data[[]])