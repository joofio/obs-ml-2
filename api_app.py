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
from fastapi.logger import logger as fastapi_logger
from logging.handlers import RotatingFileHandler
import logging

formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler('logfile.log', backupCount=0)
logging.getLogger("fastapi")
fastapi_logger.addHandler(handler)
handler.setFormatter(formatter)

fastapi_logger.info('****************** Starting Server *****************')


ENC={0:"Cesariana",1:"Vaginal"}

THRESHOLD=0.70000

class BishopScore(int, Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eigth= 8
    nine = 9
    ten = 10
    eleven =11
    twelve = 12

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



class TipoParto(str,Enum):
    ces="Cesariana"
    vag="Vaginal"

class Apresentacao32(str,Enum):
    cefálica="cefálica"
    nan="nan"
    cefálica_dorso_à_esquerda="cefálica dorso à esquerda"
    pélvica="pélvica"
    cefálica_dorso_posterior="cefálica dorso posterior"
    cefálica_dorso_à_direita="cefálica dorso à direita"
    pélvica_dorso_à_direita="pélvica dorso à direita"
    pélvica_dorso_posterior="pélvica dorso-posterior"
    situação_transversa="situação transversa"
    situação_transversa_polo_cefálico_à_esquerda="situação transversa, polo cefálico à esquerda"
    cefálica_muito_insinuada="cefálica muito insinuada"
    pélvica_dorso_anterior="pélvica dorso-anterior"
    cefálica_dorso_anterior="cefálica dorso anterior"
    situação_transversa_polo_cefálico_à_direita="situação transversa, polo cefálico à direita"
    pelve_desdobrada="pelve desdobrada"
    pelve_franca_com_dorso_anterior_esquerdo="pelve franca com dorso anterior esquerdo"
    pelve_franca_dorso_à_esquerda="pelve franca, dorso à esquerda"
    cefálica_insinuada="cefálica insinuada"
    posterior_alta="posterior alta"
    pélvica_dorso_à_esquerda="pélvica dorso à esquerda"
    cefálico_dorso_à_esquerda="cefálico dorso à esquerda"
    instável="instável"
    transversa_dorso_superior="transversa, dorso superior"
    cefálica_muito_inusitada="cefálica muito-inusitada"
    pélvica_modo_pés="pélvica modo pés"
    espadua="espadua" 

class BinSim(str,Enum):
    yes="Sim"

class ApresentacaoParto(str,Enum):
    one='Cefálica de vértice' 
    two='Pélvica' 
    three='Desconhecida' 
    four='Face' 
    five='Espádua'
    
class BinNumber(str,Enum):
    zero="0"
    one="1"
class Features(BaseModel):
    apresentacao_32: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    estimativa_peso_eco_32: float
    estimativa_peso_eco_37: float
    apresentacao_35:  Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    cesarianas_anterior: int
    partclpe1: BinNumber = Field(
        "1", title="Correção laceração perineal de 1º grau", 
    )
    apresentacao_30: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    bishop_extincao: BishopExtincao  = Field(
        0, title="The description of the item", 
    )
    apresentacao_27: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    apresentacao_28: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    estimativa_peso_eco_41: float
    apresfeto34: BinNumber = Field(
        "1", title="Apresentação/situação fetal anómala", 
    )
    apresentacao_34: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    apresentacao_29: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    estimativa_peso_eco_40: float
    partepisepir:  BinNumber = Field(
        "1", title="Episiotomia e Episiorrafia", 
    )
    apresentacao_36: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    apresfeto34pelve:  BinNumber = Field(
        "1", title="Apresentação  pélvica > 34 sem.", 
    )
    apresentacao_38: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    peso_inicial: float
    partclpe2:  BinNumber = Field(
        "1", title="Correção laceração perineal de 2º grau", 
    )
    parto23p:  BinNumber = Field(
        "1", title="Outras ocorrências do 2º e 3º período do parto", 
    )
    gravpartlp1:  BinNumber = Field(
        "1", title="Laceração perineal do 1º grau", 
    )
    apresentacao_31: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    grupo_robson: GrupoRobson  = Field(
        "1.0", title="The description of the item",
    )
    estimativa_peso_eco_39: float
    tpartoesp:  BinNumber = Field(
        "1", title="Trabalho de parto", 
    )
    bishop_score: BishopScore = Field(
        0, title="The description of the item", 
    )
    apresentacao_no_parto: ApresentacaoParto  = Field(
        "Pélvica", title="The description of the item", 
    )
    trab_parto_entrada_espontaneo: BinSim  = Field(
        "Sim", title="The description of the item", 
    )
    IDADE_MATERNA: int
    partaep:  BinNumber = Field(
        "1", title="Técnicas do neuroeixo", 
    )
    cirulaqt:  BinNumber = Field(
        "1", title="Laqueação tubária", 
    )
    gravpartlp:  BinNumber = Field(
        "1", title="Laceração perineal", 
    )
    apresentacao_37: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    apresentacao_33: Apresentacao32 = Field(
        "cefálica", title="The description of the item", 
    )
    estimativa_peso_eco_38: float
    gravpartlp2:  BinNumber = Field(
        "1", title="Laceração perineal do 2º grau", 
    )
    partocomp:  BinNumber = Field(
        "1", title="Complicações do parto", 
    )
   

class FeaturesWithDelivery(BaseModel):
    IDADE_MATERNA: int
    PESO_INICIAL: float
    IMC: float
    NUMERO_CONSULTAS_PRE_NATAL: int
  

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
    df=pd.DataFrame.from_dict(row)
    print(df)
    X=pipeline.transform(df.replace("Desconhecido","nan"))
    pred_proba=loaded_model.predict_proba(X)
    pred=predict_with_threshold(pred_proba[0][1],THRESHOLD)
 
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