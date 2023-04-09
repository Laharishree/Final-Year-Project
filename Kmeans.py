from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
import gmplot


#using geopy to access the latitude and longitude of the given address
from geopy.geocoders import Nominatim

#establishing connection
location_Connection=Nominatim(user_agent="geoapiExercise")
address=input("Enter the location\n")

#get the latitude and longitude 
try:
    
    grid=location_Connection.geocode(address)
    features=["HOSTEL_ID","HOSTEL_NAME","ADDRESS","CITY","COUNTRY","PINCODE","TYPE","RATING","LATITUDE","LONGITUDE","URL","PRICE","FACILITY","GENDER"]

        #reading csv file
    df=pd.read_csv("Dataset.csv",names=features)

    #     #data cleaning using pandas
    df.drop(["FACILITY"],axis=1)

        #converting series to numpy array
    locations=df[["LATITUDE","LONGITUDE","RATING"]].to_numpy()

        #print location latitude longitude
    print(grid.latitude,grid.longitude)

        #implementing Kmeans
    kmeans=KMeans(n_clusters=10,random_state=0).fit(locations)
    

    nearest_location=kmeans.predict([[grid.latitude,grid.longitude,4.0]])

    print(df.loc[nearest_location[0]])

    

    my_map2 = folium.Map(location = [df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE],
                                         zoom_start = 20)
 
# CircleMarker with radius
    folium.Marker([df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE],
               popup = df.loc[nearest_location[0]].HOSTEL_NAME).add_to(my_map2)
    
    my_map2.save(r"GeoLocationDataAnalysis/map.html")

except:
        print("Couldn't find the location \ntry another Location")    

