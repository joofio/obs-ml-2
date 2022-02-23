import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as pl

import shap
from shap import Explanation
from help_functions import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn import preprocessing
from lightgbm import LGBMClassifier
import json
import joblib

st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

reversed_label_encoder={1: 'Vaginal', 0: 'Cesariana'}
with open('051_prod_feature_data.json') as json_file:
    columns = json.load(json_file)


row={}
pred_cols=[]
total_cols=len(columns)

print(total_cols//6)
print(total_cols%6)
cols_dicts={}
#print(columns)
keys=list(columns.keys())
COL_VALUE=6
replace_col=["apres.feto.34"]
for i in range(0,total_cols,COL_VALUE):
    cols_dicts["col"+str(i)+"0"],cols_dicts["col"+str(i)+"1"],cols_dicts["col"+str(i)+"2"],cols_dicts["col"+str(i)+"3"],cols_dicts["col"+str(i)+"4"],cols_dicts["col"+str(i)+"5"]=st.columns(COL_VALUE)
    for j in range(0,COL_VALUE):
       # print(i,j)
        if (i+j)>=total_cols:
            break
        col=keys[i+j]
        value_col=columns[col]
       # print(value_col)
        ncol=" ".join([c.capitalize() if c!="IMC" else c for c in col.split("_") ])

        options=[str(cols).replace("nan","Desconhecido") for cols in value_col[1]]

        if value_col[0] in["cat","ord"]:
            if options==["0","1"]:
                print(options)
                options=["Não","Sim"]
            row[col]=[cols_dicts["col"+str(i)+str(j)].selectbox(ncol, options,key=col)]
        #      
        if value_col[0] in["int"]:
            max_val=value_col[1][1]
            step=1.0

            if max_val>1000:
                step=100.0
            row[col]=[cols_dicts["col"+str(i)+str(j)].number_input(ncol,min_value=value_col[1][0],max_value=max_val,value=value_col[1][2],step=step,key=col)]



#for col,values in columns.items():
 #   pred_cols.append(col)
  #  ncol=" ".join(col.split("_"))
   # options=[str(cols).replace("nan","Unknown") for cols in values[1]]
    #options=[str(cols) for cols in values[1]]

  #  if values[0] in["cat","ord"]:
  #      row[col]=[st.sidebar.selectbox(ncol, options,key=col)]
  #      
 #   if values[0] in["int"]:
  #      row[col]=[st.sidebar.number_input(ncol,min_value=values[1][0],max_value=values[1][1],value=values[1][2],step=0.5,key=col)]


st.markdown("""Please select the options on the sidebar for the model to predict the delivery type. Click the button in the end of the sidebar to start prediction""")


filename = '051_prod_lgbm.sav'
loaded_model = joblib.load(filename)
filename = '051_prod_pipeline.sav'
pipeline = joblib.load(filename)

filename = '051_prod_explainer.sav'
explainer = joblib.load(filename)


def create_outcome(le,arr):
    outcome_dict={}
    for idx,class_ in reversed_label_encoder.items():
        print(idx,class_,arr)
        print(arr[0][idx])
        outcome_dict[class_]=[str(round(arr[0][idx]*100,2)) +" %"]
    return pd.DataFrame.from_dict(outcome_dict)



make_prediction=st.sidebar.button('Make Prediction')
explaining=st.sidebar.button('Make Prediction with Shap Values')

def streamlit_predict(row):
    df=pd.DataFrame.from_dict(row)
    st.write('Predicting for')
   # st.write(row)
    st.dataframe(df)

    X=pipeline.transform(df.replace("Desconhecido","nan").replace("Não","0").replace("Sim","1"))
    #X=pipeline.transform(df.replace("Desconhecido","nan"))
   # st.write("ipeline")
    df1=transfrom_array_to_df_onehot(pipeline,X,onehot=False,overal_imp=True)
   # st.dataframe(df1)
   
    pred=loaded_model.predict(X)
    pred_proba=loaded_model.predict_proba(X)
    st.markdown("### The prediction is:  ")
    print(pred)
    st.write(reversed_label_encoder[pred[0]])
    st.dataframe(create_outcome(reversed_label_encoder,pred_proba))
    return df1,X,pred,pred_proba


if make_prediction:
    streamlit_predict(row)

if explaining:
    df1,X,pred,pred_proba=streamlit_predict(row)
    st.write('Explaining using SHAP values...')
    shap_values = explainer.shap_values(X,check_additivity=False)
    #Now we can plot relevant plots that will help us analyze the model.
    st.subheader("Summary Plot")
    shap.summary_plot(shap_values, X, plot_type="bar", class_names= ["Cesariana","Vaginal"], feature_names = df1.columns)
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
    st.subheader("Force Plot")
    shap.force_plot(explainer.expected_value[pred[0]], shap_values[pred[0]],df1,matplotlib=True,show=False,figsize=(30,10))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()

#https://github.com/sgoede/streamlit-boston-app/blob/master/boston_xgb_app.py

