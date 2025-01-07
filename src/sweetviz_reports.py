# %%
import os
import pandas as pd
import sweetviz as sv

# %%
#df_caract= pd.read_csv('../data/processed/caracteristiques_2019_2022_process.csv')
#df_lieux= pd.read_csv('../data/processed/lieux_2019_2022_process.csv', low_memory= False)
#df_usagers= pd.read_csv('../data/processed/usagers_2019_2022_process.csv', low_memory= False)
#df_vehicules= pd.read_csv('../data/processed/vehicules_2019_2022_process.csv')
#df_merged_2020= pd.read_csv('../data/processed/merged_data_2020_2020.csv', low_memory= False)
df_merged_2019_2022= pd.read_csv('../data/processed/merged_data_2019_2022.csv', low_memory= False)
#df_encoded= pd.read_csv('../data/processed/encoded_data_2019_2022.csv', low_memory= False)
dic= {'merged2019_2022':df_merged_2019_2022}

# %%
for cle, df in dic.items():
    rapport= sv.analyze(df)
    rapport.show_html(f'{cle}.html')

#report = sv.compare([df_merged_2020, "2020"],[df_merged_2019_2022, "2019_2022"])
#report.show_html("comparison_report.html")
