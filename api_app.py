from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

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
import json
import joblib
from help_functions import *
from pydantic import BaseModel, Field
import datetime
from enum import Enum

ENC={0:"Cesariana",1:"Vaginal"}

THRESHOLD=0.66667

class BinX(str,Enum):
    x = "X"


class BishopScore(int, Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5

class BishopDilatacao(int, Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5

class GrupoRobson(str, Enum):
    one = "1.0"
    two = "2"
    three = "3"
    four = "4"
    five = "5"
    six = "6"
    seven = "7"
    eigth= "8"
    nine = "9"
    ten = "10"

class BishopExtincao(int, Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5

class Bacia(str, Enum):
    l = "L"
    a = "ADEQUADA"
    i = "INADEQUADA"

class FeatureImportance(BaseModel):
    IDADE_MATERNA: float
    PESO_INICIAL: float
    IMC: float
    NUMERO_CONSULTAS_PRE_NATAL: float
    PESO_ADMISSAO_INTERNAMENTO: float
    ESTIMATIVA_PESO_ECO_30: float
    ESTIMATIVA_PESO_ECO_32: float
    ESTIMATIVA_PESO_ECO_33: float
    ESTIMATIVA_PESO_ECO_34: float
    ESTIMATIVA_PESO_ECO_35: float
    ESTIMATIVA_PESO_ECO_36: float
    ESTIMATIVA_PESO_ECO_38: float
    ESTIMATIVA_PESO_ECO_39: float
    ESTIMATIVA_PESO_ECO_40: float
    ESTIMATIVA_PESO_ECO_41: float
    EUTOCITO_ANTERIOR: float
    CESARIANAS_ANTERIOR: float
    BISHOP_SCORE: float
    BISHOP_DILATACAO: float
    BISHOP_EXTINCAO: float
    APRESENTACAO_NO_PARTO: float
    HIPERTENSAO_CRONICA: float
    BACIA: float
    GRUPO_ROBSON: float
    APRESENTACAO_ADMISSAO: float
    TRAB_PARTO_ENTRADA_ESPONTANEO: float
    HIPERTENSAO_PRE_ECLAMPSIA: float
    TRAB_PARTO_NO_PARTO: float

class Prediction(BaseModel):
    model: str
    result: str
    feature_importance: FeatureImportance = {}
    timestamp: datetime.date

class Proba(BaseModel):
    Cesariana: str
    Vaginal: str

class PredictionProba(BaseModel):
    model: str
    result: str
    outcome: Proba
    feature_importance: FeatureImportance = {}
    timestamp: datetime.date

class DecisionType(str,Enum):
    green="Ok"
    yellow="Possible Problem"
    red="Warning"

class Decision(BaseModel):
    model: str
    result_reality: str
    feature_importance: FeatureImportance = {}
    timestamp: datetime.date
    decision: DecisionType


class TrabalhoPartoNoParto(str,Enum):
    one='Ausência de trabalho de parto' 
    two='Espontâneo' 
    three='Indução com misoprostol'
    four='Indução ocitócica' 
    seven='Indução com dinoprostona' 
    nine='Indução'
    five='Indução com misoprostol+ocitocina' 
    ten='Indução mecânica'
    six='Indução mecânica+ocitocina' 
    eigth='Indução com dinoprostona+ocitocina' 

class ApresentacaoAdmissao(str,Enum):
    one='-1'
    two='apr.cefala.3' 
    three='apr.pelv.1' 
    four='apr.trans.3' 
    five='apr.esp.1'
    six='apr.desc.1' 
    seven='apr.face.2'


class ApresentacaoParto(str,Enum):
    one='Cefálica de vértice' 
    two='Pélvica' 
    three='Desconhecida' 
    four='Face' 
    five='Espádua'

class BinSim(str,Enum):
    yes="S"

class TipoParto(str,Enum):
    ces="Cesariana"
    vag="Vaginal"

class Features(BaseModel):
    IDADE_MATERNA: int
    PESO_INICIAL: float
    IMC: float
    NUMERO_CONSULTAS_PRE_NATAL: int
    PESO_ADMISSAO_INTERNAMENTO: float
    ESTIMATIVA_PESO_ECO_30: float
    ESTIMATIVA_PESO_ECO_32: float
    ESTIMATIVA_PESO_ECO_33: float
    ESTIMATIVA_PESO_ECO_34: float
    ESTIMATIVA_PESO_ECO_35: float
    ESTIMATIVA_PESO_ECO_36: float
    ESTIMATIVA_PESO_ECO_38: float
    ESTIMATIVA_PESO_ECO_39: float
    ESTIMATIVA_PESO_ECO_40: float
    ESTIMATIVA_PESO_ECO_41: float
    EUTOCITO_ANTERIOR: int
    CESARIANAS_ANTERIOR: int
    BISHOP_SCORE: BishopScore = Field(
        0, title="The description of the item", 
    )
    BISHOP_DILATACAO: BishopDilatacao  = Field(
        0, title="The description of the item", 
    )
    BISHOP_EXTINCAO: BishopExtincao  = Field(
        0, title="The description of the item", 
    )
    APRESENTACAO_NO_PARTO: ApresentacaoParto  = Field(
        "Pélvica", title="The description of the item", 
    )
    HIPERTENSAO_CRONICA: BinX  = Field(
        "X", title="The description of the item", 
    )
    BACIA: Bacia  = Field(
        "L", title="The description of the item", 
    )
    GRUPO_ROBSON: GrupoRobson  = Field(
        "1.0", title="The description of the item",
    )
    APRESENTACAO_ADMISSAO: ApresentacaoAdmissao  = Field(
        "apr.cefala.3", title="The description of the item", 
    )
    TRAB_PARTO_ENTRADA_ESPONTANEO: BinSim  = Field(
        "S", title="The description of the item", 
    )
    HIPERTENSAO_PRE_ECLAMPSIA: BinX  = Field(
        "X", title="The description of the item", 
    )
    TRAB_PARTO_NO_PARTO: TrabalhoPartoNoParto  = Field(
        "Espontâneo", title="The description of the item"
    )

class FeaturesWithDelivery(BaseModel):
    IDADE_MATERNA: int
    PESO_INICIAL: float
    IMC: float
    NUMERO_CONSULTAS_PRE_NATAL: int
    PESO_ADMISSAO_INTERNAMENTO: float
    ESTIMATIVA_PESO_ECO_30: float
    ESTIMATIVA_PESO_ECO_32: float
    ESTIMATIVA_PESO_ECO_33: float
    ESTIMATIVA_PESO_ECO_34: float
    ESTIMATIVA_PESO_ECO_35: float
    ESTIMATIVA_PESO_ECO_36: float
    ESTIMATIVA_PESO_ECO_38: float
    ESTIMATIVA_PESO_ECO_39: float
    ESTIMATIVA_PESO_ECO_40: float
    ESTIMATIVA_PESO_ECO_41: float
    EUTOCITO_ANTERIOR: int
    CESARIANAS_ANTERIOR: int
    BISHOP_SCORE: BishopScore = Field(
        0, title="The description of the item", 
    )
    BISHOP_DILATACAO: BishopDilatacao  = Field(
        0, title="The description of the item", 
    )
    BISHOP_EXTINCAO: BishopExtincao  = Field(
        0, title="The description of the item", 
    )
    APRESENTACAO_NO_PARTO: ApresentacaoParto  = Field(
        "Pélvica", title="The description of the item", 
    )
    HIPERTENSAO_CRONICA: BinX  = Field(
        "X", title="The description of the item", 
    )
    BACIA: Bacia  = Field(
        "L", title="The description of the item", 
    )
    GRUPO_ROBSON: GrupoRobson  = Field(
        "1.0", title="The description of the item",
    )
    APRESENTACAO_ADMISSAO: ApresentacaoAdmissao  = Field(
        "apr.cefala.3", title="The description of the item", 
    )
    TRAB_PARTO_ENTRADA_ESPONTANEO: BinSim  = Field(
        "S", title="The description of the item", 
    )
    HIPERTENSAO_PRE_ECLAMPSIA: BinX  = Field(
        "X", title="The description of the item", 
    )
    TRAB_PARTO_NO_PARTO: TrabalhoPartoNoParto  = Field(
        "Espontâneo", title="The description of the item"
    )
    TIPO_PARTO: TipoParto  = Field(
        "Cesariana", title="The description of the item"
    )

app = FastAPI()





def create_outcome(arr,enc):
    outcome_dict={}
    for idx,class_ in enumerate(enc.values()):
        outcome_dict[class_]=str(round(arr[0][idx]*100,2)) +" %"
    return outcome_dict

def create_result(y_pred_proba,threshold,enc):
    print(y_pred_proba)
    if y_pred_proba[0,1]>=threshold:
        return enc[1]
    else:
        return enc[0]



filename = '051_prod_lgbm.sav'
loaded_model = joblib.load(filename)
filename = '051_prod_pipeline.sav'
pipeline = joblib.load(filename)

filename = '051_prod_explainer.sav'
explainer = joblib.load(filename)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict",response_model=Prediction)
async def get_predict(row: Features):

    X_new=preprocess_row(row,pipeline)
 
    testing=loaded_model.predict(X_new.values)
    pred_proba=loaded_model.predict_proba(X_new.values)
    result=create_result(pred_proba,THRESHOLD,ENC)
   # print(testing)
    if ENC[testing[0]]!=result:
        print("EEEERRRRORRR")
    resp={"model":"DecisionTree","result":result,"timestamp":datetime.datetime.now()}
    return resp

@app.post("/predict_proba",response_model=PredictionProba)
async def get_predict_proba(row: Features):

    X_new=preprocess_row(row,pipeline)
 
    
    pred_proba=loaded_model.predict_proba(X_new.values)
    result=create_result(pred_proba,THRESHOLD,ENC)
    outcome=create_outcome(pred_proba,ENC)
    print(outcome)
    resp={"model":"lightGBM","result":result,"outcome":outcome,"timestamp":datetime.datetime.now()}
    return resp

@app.post("/decision",response_model=Decision)
async def get_decision(row: FeaturesWithDelivery):
    tipo_parto=row.TIPO_PARTO
    delattr(row,"TIPO_PARTO")
    X_new=preprocess_row(row,pipeline)
    level="Ok"
    testing=loaded_model.predict(X_new.values)
    pred_proba=loaded_model.predict_proba(X_new.values)
    result=create_result(pred_proba,THRESHOLD,ENC)
   # print(testing)

    if ENC[testing[0]]!=result:
        print("EEEERRRRORRR")
    if tipo_parto=="Cesariana" and result=="Vaginal":
        if pred_proba[0][0]<0.2:
            level="Warning"
        elif pred_proba[0][0]>=0.2 and pred_proba[0][0]<THRESHOLD:
            level="Possible Problem"
  

    resp={"model":"DecisionTree","result_reality":tipo_parto,"timestamp":datetime.datetime.now(),"decision":level}
    return resp