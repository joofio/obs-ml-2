import re
import scipy.stats as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
import scipy


def to_object(x):
    return pd.DataFrame(x).astype(str)


def to_number(x):
    return pd.DataFrame(x).astype(float)


def get_ci_model_from_clf(clf):

    params = []
    for k, v in clf.cv_results_.items():

        if k == "params" and type(v) == list:
            #  print(k,v)
            for p in v:
                # print(p)
                z = []
                for d, e in p.items():
                    z.append(str(d) + "=" + str(e))
                #    print(d,e)
                params.append("|".join(z))
    # print(params)

    param_train_score = {str(d): [] for d in params}
    pattern = "split\d{1,2}_\S+"
    for k, v in clf.cv_results_.items():
        if re.match(pattern, k):
            for idx, para in enumerate(param_train_score):
                param_train_score[para].append(v[idx])
    train_score_ci = {
        k: st.norm.interval(alpha=0.95, loc=np.mean(v), scale=scipy.stats.sem(v))
        for k, v in param_train_score.items()
    }
    return train_score_ci


def plot_error_bar(ci_rf):
    def color_statistical_sig(x, max_val):
        plus = x["plus"]
        # print(minus,plus,max_val)
        if plus >= max_val:
            # print("---",plus,max_val)
            return "not sig"
        return "sig"

    ff = pd.DataFrame(ci_rf)
    fft = ff.transpose()
    fft.columns = ["minus", "plus"]
    fft["mean"] = fft.apply(np.mean, axis=1)
    fft["e_plus"] = fft["plus"] - fft["mean"]
    fft["e_minus"] = fft["mean"] - fft["minus"]
    max_val = fft["plus"].max()
    # print(max_val)
    min_val = fft[fft["minus"] > 0]["minus"].min()

    min_plus_idx = fft[fft["plus"] > 0]["plus"].idxmax()
    min_plus = fft.loc[min_plus_idx, "minus"]
    # tt.loc['criterion=gini|min_samples_split=20']["minus"]
    #  print(min_plus)
    fft["max"] = fft["plus"].apply(lambda x: "max" if x == max_val else "not max")
    fft["significant"] = fft.apply(
        lambda x: color_statistical_sig(x, max_val=min_plus), axis=1
    )
    # print(fft)
    fft["hover_data"] = (
        round(fft["minus"], 4).astype(str) + " +- " + round(fft["plus"], 4).astype(str)
    )
    # print(fft["hover_data"])
    fig = px.scatter(
        fft,
        x=fft.index,
        y="mean",
        error_y="e_plus",
        error_y_minus="e_minus",
        color="significant",
        symbol="max",
        hover_data=["hover_data"],
    )
    fig.update(layout_yaxis_range=[min_val - 0.1, max_val + 0.1])

    fig.show()
    return fft


def transfrom_array_to_df_onehot(pl, nparray, onehot=True, overal_imp=False):
    col_list = []
    col_list_int = pl["preprocessor"].transformers_[0][2]  # changes col location
    # print(col_list_int)
    ordinal_col = pl["preprocessor"].transformers[1][2]
    original_col = pl["preprocessor"].transformers[2][2]
    col_list = col_list_int + ordinal_col
    if onehot:
        encoded_col = (
            pl["preprocessor"]
            .transformers_[2][1]
            .named_steps["OneHotEnconding"]
            .get_feature_names_out()
        )

        # print(len(encoded_col))
        new_enconded_list = []
        for idx, col in enumerate(original_col):
            for n_col in encoded_col:
                # print(idx,col)
                # print("x"+str(idx))
                if "x" + str(idx) + "_" in n_col:
                    #   print(col,n_col)
                    new_enconded_list.append(col + "_" + n_col.split("_")[-1])

        col_list = col_list + new_enconded_list
        print(col_list)
        # print(len(col_list))
    else:
        col_list = col_list + original_col
    if overal_imp == True:
        imputed_cols_idx = pl["imputer"].indicator_.features_
        imputed_indicator = [col_list[i] for i in imputed_cols_idx]
        # print(imputed_indicator)
        # print(len(imputed_indicator))
        for imp_col in imputed_indicator:
            col_list.append(imp_col + "_imput_indicator")
    print(col_list)
    df1 = pd.DataFrame(nparray, columns=col_list)
    return df1


def get_transformers_nan(df, pl):
    to_list = []
    col_list = []
    col_list_int = pl["preprocessor"].transformers_[0][2]  # changes col location
    # print(col_list_int)
    ordinal_col = pl["preprocessor"].transformers[1][2]
    original_col = pl["preprocessor"].transformers[2][2]
    col_list = col_list_int + ordinal_col
    tt = pl.transform(df)
    imputed_cols_idx = pl["imputer"].indicator_.features_
    imputed_indicator = [col_list[i] for i in imputed_cols_idx]
    # print(imputed_indicator)
    # print(len(imputed_indicator))
    for imp_col in imputed_indicator:
        to_list.append(imp_col + "_imput_indicator")
    # print(to_list)
    missing_imp = np.zeros((1, len(to_list)))
    # print(missing_imp)
    # print(tt)
    final_np = np.append(tt, missing_imp)

    return final_np
