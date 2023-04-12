from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt



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
    locations=df[["LATITUDE","LONGITUDE"]].to_numpy()

        #print location latitude longitude
    print(grid.latitude,grid.longitude)

        #implementing Kmeans
    kmeans=KMeans(n_clusters=15,random_state=42).fit(locations)
    

    nearest_location=kmeans.predict([[grid.latitude,grid.longitude]])

    print(df.loc[nearest_location[0]])

    

    my_map2 = folium.Map(location = [df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE],
                                         zoom_start = 20, control_scale=True)
 
# CircleMarker with radius
    folium.Marker([df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE],
               popup = df.loc[nearest_location[0]].HOSTEL_NAME).add_to(my_map2)
    folium.Marker([grid.latitude,grid.longitude],
               popup = "My Location").add_to(my_map2)    
    folium.PolyLine(locations=[(grid.latitude,grid.longitude),(df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE)],line_opacity=0.5).add_to(my_map2)
    my_map2.save("map.html")
    # map = folium.Map(location=[df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE],
    #                                      zoom_start = 20, control_scale=True)

    # folium.Marker(location=[df.loc[nearest_location[0]].LATITUDE,df.loc[nearest_location[0]].LONGITUDE], popup = df.loc[nearest_location[0]].HOSTEL_NAME,
    #           icon=folium.Icon(color='red', icon='pushpin')).add_to(map)
              
    # map.save("map.html")


except:
         print("")    

