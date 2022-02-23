import re
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
    
    
def transfrom_array_to_df_onehot(pl,nparray,onehot=True,overal_imp=False):
    col_list=[]
    col_list_int = pl["preprocessor"].transformers_[0][2] #changes col location
    #print(col_list_int)
    ordinal_col=pl["preprocessor"].transformers[1][2]
    original_col=pl["preprocessor"].transformers[2][2]
    col_list=col_list_int+ordinal_col
    if onehot:
        encoded_col=pl["preprocessor"].transformers_[2][1].named_steps["OneHotEnconding"].get_feature_names_out()
    
        #print(len(encoded_col))
        new_enconded_list=[]
        for idx,col in enumerate(original_col):
            for n_col in encoded_col:
            #print(idx,col)
           # print("x"+str(idx))
                if "x"+str(idx)+"_" in n_col:
                 #   print(col,n_col)
                    new_enconded_list.append(col+"_"+n_col.split("_")[-1])
        
        col_list=col_list+new_enconded_list
        print(col_list)
        #print(len(col_list))
    else:
        col_list=col_list+original_col
    if overal_imp==True:
        imputed_cols_idx=pl["imputer"].indicator_.features_
        imputed_indicator=[col_list[i] for i in imputed_cols_idx]
       # print(imputed_indicator)
       # print(len(imputed_indicator))
        for imp_col in imputed_indicator:
            col_list.append(imp_col+"_imput_indicator")
    df1 = pd.DataFrame(nparray, columns=col_list)
    return df1