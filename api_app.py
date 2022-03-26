from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

import json
import joblib
from help_functions import *
from pydantic import BaseModel, Field
import datetime
from enum import Enum
from fastapi.logger import logger as fastapi_logger
from logging.handlers import RotatingFileHandler
import logging
from fastapi.encoders import jsonable_encoder
from fhir.resources.patient import Patient
from fhir.resources.communicationrequest import CommunicationRequest
from fhir.resources.fhirtypes import CommunicationRequestType
from fhir.resources.fhirtypes import CommunicationType

from fhir.resources.communication import Communication


formatter = logging.Formatter(
    "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s",
    "%Y-%m-%d %H:%M:%S",
)
handler = RotatingFileHandler("logfile.log", backupCount=0)
logging.getLogger("fastapi")
fastapi_logger.addHandler(handler)
handler.setFormatter(formatter)
fastapi_logger.setLevel(logging.INFO)

fastapi_logger.info("****************** Starting Server *****************")


ENC = {0: "Cesariana", 1: "Vaginal"}

THRESHOLD = 0.70000


class BishopScore(int, Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eigth = 8
    nine = 9
    ten = 10
    eleven = 11
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
    eigth = "8"
    nine = "9"
    ten = "10"


class ApresentacaoAdmissao(str, Enum):
    cefálica = "Cefálica"
    pelv = "Pélvica"
    Transversa = "Transversa"
    Face = "Face"
    Espadua = "Espádua"


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
    timestamp: datetime.datetime


class Proba(BaseModel):
    Cesariana: str
    Vaginal: str


class PredictionProba(BaseModel):
    model: str
    result: str
    outcome: Proba
    feature_importance: FeatureImportance = {}
    timestamp: datetime.datetime


class DecisionType(str, Enum):
    green = "Ok"
    yellow = "Possible Problem"
    red = "Warning"


class Decision(BaseModel):
    model: str
    result_reality: str
    feature_importance: FeatureImportance = {}
    timestamp: datetime.date
    decision: DecisionType


class TipoParto(str, Enum):
    ces = "Cesariana"
    vag = "Vaginal"


class Apresentacao32(str, Enum):
    cefálica = "cefálica"
    nan = "nan"
    cefálica_dorso_à_esquerda = "cefálica dorso à esquerda"
    pélvica = "pélvica"
    cefálica_dorso_posterior = "cefálica dorso posterior"
    cefálica_dorso_à_direita = "cefálica dorso à direita"
    pélvica_dorso_à_direita = "pélvica dorso à direita"
    pélvica_dorso_posterior = "pélvica dorso-posterior"
    situação_transversa = "situação transversa"
    situação_transversa_polo_cefálico_à_esquerda = (
        "situação transversa, polo cefálico à esquerda"
    )
    cefálica_muito_insinuada = "cefálica muito insinuada"
    pélvica_dorso_anterior = "pélvica dorso-anterior"
    cefálica_dorso_anterior = "cefálica dorso anterior"
    situação_transversa_polo_cefálico_à_direita = (
        "situação transversa, polo cefálico à direita"
    )
    pelve_desdobrada = "pelve desdobrada"
    pelve_franca_com_dorso_anterior_esquerdo = (
        "pelve franca com dorso anterior esquerdo"
    )
    pelve_franca_dorso_à_esquerda = "pelve franca, dorso à esquerda"
    cefálica_insinuada = "cefálica insinuada"
    posterior_alta = "posterior alta"
    pélvica_dorso_à_esquerda = "pélvica dorso à esquerda"
    cefálico_dorso_à_esquerda = "cefálico dorso à esquerda"
    instável = "instável"
    transversa_dorso_superior = "transversa, dorso superior"
    cefálica_muito_inusitada = "cefálica muito-inusitada"
    pélvica_modo_pés = "pélvica modo pés"
    espadua = "espadua"


class BinSim(str, Enum):
    yes = "Sim"


class ApresentacaoParto(str, Enum):
    one = "Cefálica de vértice"
    two = "Pélvica"
    three = "Desconhecida"
    four = "Face"
    five = "Espádua"


class BinNumber(str, Enum):
    zero = "0"
    one = "1"


class Features(BaseModel):
    APRESENTACAO_31: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    APRESENTACAO_28: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )

    ESTIMATIVA_PESO_ECO_39: int
    APRESENTACAO_30: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    APRESENTACAO_37: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    GRUPO_ROBSON: GrupoRobson = Field(
        "1.0",
        title="The description of the item",
    )
    BISHOP_SCORE: BishopScore = Field(
        0,
        title="The description of the item",
    )
    PESO_INICIAL: float

    ESTIMATIVA_PESO_ECO_32: float
    IDADE_MATERNA: int
    IMC: float
    APRESENTACAO_35: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    APRESENTACAO_32: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    APRESENTACAO_26: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    ESTIMATIVA_PESO_ECO_24: float
    APRESENTACAO_34: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    ESTIMATIVA_PESO_ECO_40: float

    APRESENTACAO_33: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    APRESENTACAO_NO_PARTO: ApresentacaoParto = Field(
        "Pélvica",
        title="The description of the item",
    )
    NUMERO_CONSULTAS_PRE_NATAL: int
    APRESENTACAO_38: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    APRESENTACAO_29: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    ESTIMATIVA_PESO_ECO_41: float
    BISHOP_EXTINCAO: BishopExtincao = Field(
        0,
        title="The description of the item",
    )
    APRESENTACAO_ADMISSAO: ApresentacaoAdmissao = Field(
        "Cefálica",
        title="The description of the item",
    )
    APRESENTACAO_27: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    ESTIMATIVA_PESO_ECO_33: int
    CESARIANAS_ANTERIOR: int

    APRESENTACAO_36: Apresentacao32 = Field(
        "cefálica",
        title="The description of the item",
    )
    ESTIMATIVA_PESO_ECO_37: float
    TRAB_PARTO_ENTRADA_ESPONTANEO: BinSim = Field(
        "Sim",
        title="The description of the item",
    )
    ESTIMATIVA_PESO_ECO_38: float

    partmcdtctgs: BinNumber = Field("1", title="Cardiotocografia em trabalho de parto")
    apresfeto34: BinNumber = Field("1", title="Apresentação/situação fetal anómala")
    tpartorpm: BinNumber = Field(
        "1", title="Rotura de membranas no trabalho de parto em:"
    )
    rnucin: BinNumber = Field("1", title="Internamento em UCIN")
    partvig: BinNumber = Field("1", title="Vigilância do trabalho de parto")
    gravfetoaltcf: BinNumber = Field("1", title="Alteração do crescimento fetal")
    tpartoesp: BinNumber = Field("1", title="Trabalho de parto")
    apresfeto34pelve: BinNumber = Field("1", title="Apresentação  pélvica > 34 sem.")
    partocomp: BinNumber = Field("1", title="Complicações do parto")
    cirulaqt: BinNumber = Field("1", title="Laqueação tubária")
    puercompcica: BinNumber = Field(
        "1", title="Complicações da cicatriz da parede abdominal"
    )
    tpartorpmespo: BinNumber = Field(
        "1", title="Rotura de membranas no trabalho de parto (espontânea)"
    )
    gravplac: BinNumber = Field("1", title="Placenta prévia")
    parto23P: BinNumber = Field(
        "1", title="Outras ocorrências do 2º e 3º período do parto"
    )
    carddhta: BinNumber = Field("1", title="Doença hipertensiva")
    partocompcervical: BinNumber = Field("1", title="Circular cervical apertada")
    partaep: BinNumber = Field("1", title="Técnicas do neuroeixo")


class FeaturesWithDelivery(BaseModel):
    IDADE_MATERNA: int
    PESO_INICIAL: float
    IMC: float
    NUMERO_CONSULTAS_PRE_NATAL: int


app = FastAPI()


def create_outcome(arr, enc):
    outcome_dict = {}
    for idx, class_ in enumerate(enc.values()):
        outcome_dict[class_] = str(round(arr[0][idx] * 100, 2)) + " %"
    return outcome_dict


def create_result(y_pred_proba, threshold, enc):
    print(y_pred_proba)
    if y_pred_proba[0, 1] >= threshold:
        return enc[1]
    else:
        return enc[0]


def predict_with_threshold(x, threshold=0.9):
    if x >= threshold:
        return 1
    else:
        return 0


def treat_row_data(row):
    df = pd.DataFrame([jsonable_encoder(row)])
    df.to_csv("test.csv")
    df.rename(
        columns={
            "tpartoesp": "tparto.esp",
            "parto23P": "parto.23P",
            "gravpartlp2": "grav.part.lp2",
            "apresfeto34": "apres.feto.34",
            "gravpartlp": "grav.part.lp",
            "partclpe1": "part.clpe1",
            "gravpartlp1": "grav.part.lp1",
            "partclpe2": "part.clpe2",
            "apresfeto34pelve": "apres.feto.34.pelve",
            "partocomp": "parto.comp",
            "partepisepir": "part.epis.epir",
            "cirulaqt": "ciru.laqt",
            "partaep": "part.aep",
            "partmcdtctgs": "part.mcdt.ctgs",
            "carddhta": "card.dhta",
            "partvig": "part.vig",
            "rnucin": "rn.ucin",
            "tpartorpmespo": "tparto.rpm.espo",
            "tpartorpm": "tparto.rpm",
            "puercompcica": "puer.comp.cica",
            "gravfetoaltcf": "grav.feto.altcf",
            "gravplac": "grav.plac",
            "partocompcervical": "parto.comp.cervical",
        },
        inplace=True,
    )
    if any(pd.isna(df)):
        X = get_transformers_nan(df, pipeline)
    else:
        X = pipeline.transform(df)
    return X


filename = "051_prod_lgbm.sav"
loaded_model = joblib.load(filename)
filename = "051_prod_pipeline.sav"
pipeline = joblib.load(filename)

filename = "051_prod_explainer.sav"
explainer = joblib.load(filename)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", response_model=Prediction)
async def get_predict(row: Features):
    fastapi_logger.info("called predict")

    X = treat_row_data(row)
    pred_proba = loaded_model.predict_proba([X])
    pred = predict_with_threshold(pred_proba[0][1], THRESHOLD)

    testing = loaded_model.predict([X])
    pred_proba = loaded_model.predict_proba([X])
    result = create_result(pred_proba, THRESHOLD, ENC)

    # current date and time
    now = datetime.datetime.now()

    # print(testing)

    if ENC[testing[0]] != result:
        print("EEEERRRRORRR")
    resp = {
        "model": "LgbmV1",
        "result": result,
        "timestamp": now,
    }
    fastapi_logger.info("called predict" + "|||" + str(df) + "|||" + str(resp))
    return resp


@app.post("/predict_proba", response_model=PredictionProba)
async def get_predict_proba(row: Features):

    X = treat_row_data(row)

    pred_proba = loaded_model.predict_proba(X_new.values)
    result = create_result(pred_proba, THRESHOLD, ENC)
    outcome = create_outcome(pred_proba, ENC)
    print(outcome)
    resp = {
        "model": "LgbmV1",
        "result": result,
        "outcome": outcome,
        "timestamp": datetime.datetime.now(),
    }
    return resp


@app.post("/decision", response_model=Decision)
async def get_decision(row: FeaturesWithDelivery):
    tipo_parto = row.TIPO_PARTO
    delattr(row, "TIPO_PARTO")
    X = treat_row_data(row)
    level = "Ok"
    testing = loaded_model.predict([X])
    pred_proba = loaded_model.predict_proba([X])
    result = create_result(pred_proba, THRESHOLD, ENC)
    # print(testing)

    if ENC[testing[0]] != result:
        print("EEEERRRRORRR")
    if tipo_parto == "Cesariana" and result == "Vaginal":
        if pred_proba[0][0] < 0.2:
            level = "Warning"
        elif pred_proba[0][0] >= 0.2 and pred_proba[0][0] < THRESHOLD:
            level = "Possible Problem"

    resp = {
        "model": "LgbmV1",
        "result_reality": tipo_parto,
        "timestamp": datetime.datetime.now(),
        "decision": level,
    }
    return resp


@app.post("/fhir/predict", response_model=CommunicationType)
async def get_predict(row: CommunicationRequestType):
    fastapi_logger.info("called predict")

    return Communication()


# name: HumanName = Field( None, alias="name", title="A name associated with the patient", description="A name associated with the individual.", # if property is element of this resource. element_property=True, )
