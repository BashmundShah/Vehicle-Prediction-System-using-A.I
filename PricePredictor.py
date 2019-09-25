import pickle
import pandas as pd
import numpy as np
import sys

def mapping(Make,Model,Year,Mileage, EngineType, City, Color, EngineCapacity):
    df=pd.read_csv("cars/preprocessed.csv")
    df = df.iloc[0:0]
    df=df.drop(['price','id','Unnamed: 0'],axis=1)
    df.loc[0, 'year'] = int(Year)
    df.loc[0, 'mileage'] = int(Mileage)
    df.loc[0, 'enginecapacity'] = int(EngineCapacity)
    df[df.columns[pd.Series(df.columns).str.startswith('make_')]]=0
    df[df.columns[pd.Series(df.columns).str.startswith('model_')]]=0
    df[df.columns[pd.Series(df.columns).str.startswith('color_')]]=0
    df[df.columns[pd.Series(df.columns).str.startswith('city')]]=0
    df.loc[0,df.columns[pd.Series(df.columns).str.startswith('make_'+str.lower(Make))]]=1
    df.loc[0,df.columns[pd.Series(df.columns).str.startswith('model_'+str.lower(Model))]]=1
    df.loc[0,df.columns[pd.Series(df.columns).str.startswith('color_'+str.lower(Color))]]=1
    df.loc[0,df.columns[pd.Series(df.columns).str.startswith('city_'+str.lower(City))]]=1
    if(str.lower(EngineType)=='automatic'):
        df.loc[0, 'enginetype'] = 0
    elif(str.lower(EngineType)=='manual'):
        df.loc[0, 'enginetype'] = 1
    df=df.astype('int64')
    return df
if(len(sys.argv)==9):
	df=mapping(Make=sys.argv[1],Model=sys.argv[2],Year=sys.argv[3],Mileage=sys.argv[4], EngineType=sys.argv[5], City=sys.argv[6], Color=sys.argv[7], EngineCapacity=sys.argv[8])

lr = pickle.load(open('Models/linear_model.sav', 'rb'))
gbr = pickle.load(open('Models/gradientbooster_model.sav', 'rb'))
rfr = pickle.load(open('Models/randomforest_model.sav', 'rb'))

prices=np.array(np.round(lr.predict(df),-3))
prices=np.append(prices,np.round(gbr.predict(df),-3))
prices=np.append(prices,np.round(rfr.predict(df),-3))

print(prices)