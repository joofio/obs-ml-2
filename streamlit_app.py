import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as pl
import collections
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
from sklearn.metrics import (
plot_confusion_matrix,roc_curve, auc,
roc_auc_score,
)

st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

reversed_label_encoder={1: 'Vaginal', 0: 'Cesariana'}
od = collections.OrderedDict(sorted(reversed_label_encoder.items()))
with open('051_prod_feature_data.json') as json_file:
    columns = json.load(json_file)

THRESHOLD_DEF=0.70
prc=pd.read_csv("prc.csv")
tpr=pd.read_csv("tpr.csv",index_col=[0])
auc_data=pd.read_csv("auc.csv")
#print(od)
row={}
pred_cols=[]
total_cols=len(columns)

#print(total_cols//6)
#print(total_cols%6)
cols_dicts={}
#print(columns)
keys=list(columns.keys())
COL_VALUE=6

st.header("Delivery Type Prediction")
st.markdown("""Please select the options for the model to predict the delivery type. Click the button in the sidebar to start prediction""")

for i in range(0,total_cols,COL_VALUE):
    cols_dicts["col"+str(i)+"0"],cols_dicts["col"+str(i)+"1"],cols_dicts["col"+str(i)+"2"],cols_dicts["col"+str(i)+"3"],cols_dicts["col"+str(i)+"4"],cols_dicts["col"+str(i)+"5"]=st.columns(COL_VALUE)
    for j in range(0,COL_VALUE):
       # print(i,j)
        if (i+j)>=total_cols:
            break
        col=keys[i+j]
        label=columns[col][2]
      #  print(col)
        value_col=columns[col]
       # print(value_col)
        ncol=" ".join([c.capitalize() if c!="IMC" else c for c in label.split("_") ])

        options=[str(cols).replace("nan","Desconhecido") for cols in value_col[1]]

        if value_col[0] in["cat","ord"]:
            if options==["0","1"]:
             #   print(options)
                options=["Não","Sim"]
            row[col]=[cols_dicts["col"+str(i)+str(j)].selectbox(ncol, options,key=col)]
        #      
        if value_col[0] in["int"]:
            max_val=value_col[1][1]
            step=1.0

            if max_val>1000:
                step=100.0
            row[col]=[cols_dicts["col"+str(i)+str(j)].number_input(ncol,min_value=value_col[1][0],max_value=max_val,value=value_col[1][2],step=step,key=col)]




filename = '051_prod_lgbm.sav'
loaded_model = joblib.load(filename)
filename = '051_prod_pipeline.sav'
pipeline = joblib.load(filename)

filename = '051_prod_explainer.sav'
explainer = joblib.load(filename)

def predict_with_threshold(x,threshold=0.9):
    print(x)
    if x>=threshold:
        return 1
    else:
        return 0

def create_outcome(le,arr):
    outcome_dict={}
    for idx,class_ in le.items():
      #  print(idx,class_,arr)
      #  print(arr[0][idx])
        outcome_dict[class_]=[str(round(arr[0][idx]*100,2)) +" %"]
    outcome_dict["Threshold"]=THRESHOLD
    return pd.DataFrame.from_dict(outcome_dict)



make_prediction=st.sidebar.button('Make Prediction')
explaining=st.sidebar.button('Make Prediction with Shap Values')
THRESHOLD=st.sidebar.slider("Threshold for prediction", min_value=0.0, max_value=1.0, value=THRESHOLD_DEF, step=0.01)


with st.expander("Thresholds Definition"):
    vcol1, vcol2,vcol3= st.columns(3)

   # print(prc.head())
    fig = px.area(prc,
    x="recall", y="precision",
    title=f'Precision-Recall Curve',
    labels=dict(x='Recall', y='Precision'),hover_data=["thresholds"],
    width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    #fig.show()
    vcol1.plotly_chart(fig, use_container_width=True)
    fig_thresh = px.line(
    tpr, title='TPR and FPR at every threshold',
    width=700, height=500
    )

    fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    vcol2.plotly_chart(fig_thresh, use_container_width=True)
    fpr=auc_data["fpr"].values
    tpr=auc_data["tpr"].values
    fig_auc= px.area(auc_data,
        x="fpr", y="tpr",
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),hover_data=["thresholds"],
        width=700, height=500,#hover_data=thresholds
    )
    fig_auc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig_auc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc.update_xaxes(constrain='domain')
    vcol3.plotly_chart(fig_auc, use_container_width=True)

def streamlit_predict(row):
    df=pd.DataFrame.from_dict(row)
    st.write('Predicting for')
    st.dataframe(df)
    X=pipeline.transform(df.replace("Desconhecido","nan").replace("Não","0").replace("Sim","1"))
    df1=transfrom_array_to_df_onehot(pipeline,X,onehot=False,overal_imp=True)
    pred_proba=loaded_model.predict_proba(X)
    pred=predict_with_threshold(pred_proba[0][1],THRESHOLD)
    st.markdown("### The prediction is:  ")
    st.write(reversed_label_encoder[pred])
    st.dataframe(create_outcome(od,pred_proba))
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

