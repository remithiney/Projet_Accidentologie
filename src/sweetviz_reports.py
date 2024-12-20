# %%
import os
import pandas as pd
import sweetviz as sv

# %%
#df_caract= pd.read_csv('../data/processed/caracteristiques_2019_2022_process.csv')
#df_lieux= pd.read_csv('../data/processed/lieux_2019_2022_process.csv', low_memory= False)
#df_usagers= pd.read_csv('../data/processed/usagers_2019_2022_process.csv', low_memory= False)
#df_vehicules= pd.read_csv('../data/processed/vehicules_2019_2022_process.csv')
df_merged= pd.read_csv('../data/processed/merged_data_2019_2022.csv', low_memory= False)
#df_encoded= pd.read_csv('../data/processed/encoded_data_2019_2022.csv', low_memory= False)
dic= {'merged': df_merged}

# %%
for cle, df in dic.items():
    rapport= sv.analyze(df)
    rapport.show_html(f'{cle}.html')


