from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling 
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
import lazypredict 
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import seaborn as sns
import base64
import io
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pyexpat


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("automl app karthik")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download","lazypredict"])
    st.info("This project application helps you build and explore your data.")

with st.sidebar.header('2.set parameters'):
    split_size=st.sidebar.slider('data split ratio(% for training set)',10,90,80,5)
    seed_number=st.sidebar.slider('set the random state number',1,100,42,1)    

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
if choice =="lazypredict":
    def build_model(df):
        x=df.iloc[:,:-1]
        y=df.iloc[:,-1]

        st.markdown('**1.2. dataset dimension**')
        st.write('x')
        st.info(x.shape)
        st.write('y')
        st.info(y.shape)
        
        st.markdown('**1.3. variable details***')
        st.write('x variable (first 20 are shown)')
        st.write(list(x.columns[:20]))
        st.write('y variable')
        st.info(y.name)

        #build lazy model
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
        reg=LazyRegressor(verbose=0,ignore_warnings=False,custom_metric=None)
        models_train,prediction_train=reg.fit(x_train,x_train,y_train,y_train)
        models_test,prediction_test=reg.fit(x_train,x_test,y_train,y_test)
        st.write(prediction_train)
        st.write(prediction_test)
        regressor = RandomForestRegressor(n_estimators=100)
        regressor.fit(x_train,y_train)
        st.write('training set')
        st.write(prediction_train)
        st.markdown(filedownload(prediction_train,'training.csv'),unsafe_allow_html=True)

        
        st.write('test set')
        st.write(prediction_train)
        st.markdown(filedownload(prediction_test,'test.csv'),unsafe_allow_html=True)
       
    def filedownload(df,filename):
        csv=df.to_csv(index=False)
        b64=base64.b64encode(csv.encode()).decode()
        href=f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
        return href    
  
    build_model(df)