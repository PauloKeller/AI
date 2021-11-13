# import all libraries needed
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class servico_militar_model():
    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files wich were saved
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
    
    def load_and_clean_data(self, data_file, number_of_samples):
        df = pd.read_csv(data_file,delimiter=',')
        
        self.df_with_predictions = df.copy()
        
        df['PESO'] = df['PESO'].fillna(round(df['PESO'].mean(),2))
        df['ALTURA'] = df['ALTURA'].fillna(round(df['ALTURA'].mean(),2))
        df['CABECA'] = df['CABECA'].fillna(round(df['CABECA'].mean(),2))
        df['CALCADO'] = df['CALCADO'].fillna(round(df['CALCADO'].mean(),2))
        df['ALTURA'] = df['ALTURA'].fillna(round(df['ALTURA'].mean(),2))
        df['CINTURA'] = df['CINTURA'].fillna(round(df['CINTURA'].mean(),2))
        
        fill_religion = 'Sem Religião'
        df['RELIGIAO'] = df['RELIGIAO'].fillna(fill_religion)
        df['PAIS_NASCIMENTO'] = np.where(df['PAIS_NASCIMENTO'] != 'BRASIL', 1, 0)
        df['PAIS_RESIDENCIA'] = np.where(df['PAIS_RESIDENCIA'] != 'BRASIL', 1, 0)
        df['ZONA_RESIDENCIAL'] = np.where(df['ZONA_RESIDENCIAL'] != 'Urbana', 1, 0)
        df['CONVOCADO'] = np.where(df['DISPENSA'] == 'Sem dispensa', 1, 0)
        
        df['ESTADO_CIVIL'] = df['ESTADO_CIVIL'].map({
            'Solteiro':0, 
            'Casado':1, 
            'Outros':1, 
            'Viúvo':1, 
            'Separado Judicialmente':1, 
            'Desquitado':1, 
            'Divorciado':1
        })
        
        uf_by_region_id = {
            'RO':1, 
            'AC':1, 
            'AM':1, 
            'RR':1, 
            'PA':1, 
            'AP':1, 
            'TO':1,
            'MA':2,
            'PI':2, 
            'CE':2, 
            'RN':2, 
            'PB':2, 
            'PE':2, 
            'AL':2,
            'SE':2, 
            'BA':2, 
            'MG':3, 
            'ES':3,
            'RJ':3, 
            'SP':3, 
            'PR':4,
            'SC':4, 
            'RS':4,
            'MS':5, 
            'MT':5,
            'GO':5, 
            'DF':5,
            'KK':6,
            'RR':6,
        }
        df['UF_JSM'] = df['UF_JSM'].map(uf_by_region_id)
        
        def calculate_out_of_year(born_year, enlistment_year):
            age = enlistment_year - born_year
            return age >= 19
        
        out_of_year = calculate_out_of_year(df['ANO_NASCIMENTO'], df['VINCULACAO_ANO'])
        df['OUT_OF_YEAR'] = out_of_year
        df['OUT_OF_YEAR'] = np.where(df['OUT_OF_YEAR'], 1, 0)
        
        religion_id = {
            'Católica':1, 
            'Congregacional':1, 
            'Presbiteriana':1, 
            'Pentecostal':1, 
            'Ecumênico':1, 
            'Batista':1, 
            'Para-Protestante':1,
            'Metodista':1,
            'Adventista':1, 
            'Anabatistas':1, 
            'Anglicanos':1, 
            'Luterana':1, 
            'Campbelita':1, 
            'Espírita':1,
            'Esotérica':1, 
            'Afro-Brasileira':1, 
            'Tradições Indígenas':1, 
            'Budismo':1,
            'Ateu':0, 
            'Sem Declaração':0,
            'Sem Religião': 0,
        }
        df['RELIGIAO'] = df['RELIGIAO'].map(religion_id)
        
        education_id = {
            'Analfabeto': 0,
            'Alfabetizado': 0,
            '1° Ano - Ensino Fundamental':1,
            '2° Ano - Ensino Fundamental':1,
            '3° Ano - Ensino Fundamental':1,
            '4° Ano - Ensino Fundamental':1,
            '5° Ano - Ensino Fundamental':1,
            '6° Ano - Ensino Fundamental':1,
            '7° Ano - Ensino Fundamental':1,
            '8° Ano - Ensino Fundamental':1,
            '9° Ano - Ensino Fundamental':1,
            'Ensino Fundamental Completo':2,
            '1° Ano - Ensino Médio':2,
            '2° Ano - Ensino Médio':2,
            '3° Ano - Ensino Médio':2,
            '4° Ano - Ensino Médio (Profissionalizante)':2,
            'Ensino Médio Completo':2,
            '1° Semestre - Ensino Superior':3,
            '2° Semestre - Ensino Superior':3,
            '3° Semestre - Ensino Superior':3,
            '4° Semestre - Ensino Superior':3,
            '5° Semestre - Ensino Superior':3,
            '6° Semestre - Ensino Superior':3,
            '7° Semestre - Ensino Superior':3,
            '8° Semestre - Ensino Superior':3,
            '9° Semestre - Ensino Superior':3,
            '10° Semestre - Ensino Superior':3,
            'Ensino Superior Completo':3,
            'Mestrado':3,
            'Pós-Graduação':3,
            'Doutorado':3,
            'Pós-Doutorado':3,
        }
        df['ESCOLARIDADE'] = df['ESCOLARIDADE'].map(education_id)
        
        education_columns = pd.get_dummies(df['ESCOLARIDADE'])
        education_type_0 = education_columns.loc[:, 0]
        education_type_1 = education_columns.loc[:, 1]
        education_type_2 = education_columns.loc[:, 2]
        education_type_3 = education_columns.loc[:, 3]
        
        df = pd.concat([df, education_type_0, education_type_1, education_type_2, education_type_3], axis=1)
        df.head()
        
        column_names = ['ANO_NASCIMENTO', 'PESO', 'ALTURA', 'CABECA', 'CALCADO', 'CINTURA',
       'RELIGIAO', 'MUN_NASCIMENTO', 'UF_NASCIMENTO', 'PAIS_NASCIMENTO',
       'ESTADO_CIVIL', 'SEXO', 'ESCOLARIDADE', 'VINCULACAO_ANO',
       'DISPENSA', 'ZONA_RESIDENCIAL', 'MUN_RESIDENCIA', 'UF_RESIDENCIA',
       'PAIS_RESIDENCIA', 'JSM', 'MUN_JSM', 'UF_JSM', 'OUT_OF_YEAR',
       'CONVOCADO', 'EDUC_0', 'EDUC_1', 'EDUC_2', 'EDUC_3']
        pd.columns = column_names
        
        uf_columns = pd.get_dummies(df['UF_JSM'])
        jsm_uf_type_1 = uf_columns.loc[:, 1]
        jsm_uf_type_2 = uf_columns.loc[:, 2]
        jsm_uf_type_3 = uf_columns.loc[:, 3]
        jsm_uf_type_4 = uf_columns.loc[:, 4]
        jsm_uf_type_5 = uf_columns.loc[:, 5]
        jsm_uf_type_6 = uf_columns.loc[:, 6]
        
        df = pd.concat([
            df, 
            jsm_uf_type_1, 
            jsm_uf_type_2, 
            jsm_uf_type_3, 
            jsm_uf_type_4, 
            jsm_uf_type_5,
            jsm_uf_type_6
        ], axis=1)
        
        column_names = ['ANO_NASCIMENTO', 'PESO', 'ALTURA', 'CABECA', 'CALCADO', 'CINTURA',
       'RELIGIAO', 'MUN_NASCIMENTO', 'UF_NASCIMENTO', 'PAIS_NASCIMENTO',
       'ESTADO_CIVIL', 'SEXO', 'ESCOLARIDADE', 'VINCULACAO_ANO',
       'DISPENSA', 'ZONA_RESIDENCIAL', 'MUN_RESIDENCIA', 'UF_RESIDENCIA',
       'PAIS_RESIDENCIA', 'JSM', 'MUN_JSM', 'UF_JSM', 'OUT_OF_YEAR',
       'CONVOCADO', 'EDUC_0', 'EDUC_1', 'EDUC_2', 'EDUC_3', 'JSM_UF_1', 'JSM_UF_2', 'JSM_UF_3', 'JSM_UF_4', 'JSM_UF_5', 'JSM_UF_6']
        df.columns = column_names
        
        targets_0 = df.copy()
        targets_0 = targets_0.loc[targets_0['CONVOCADO'] == 0].sample(frac=1)
        targets_0 = targets_0[:int(number_of_samples/2)]
        targets_0['CONVOCADO'].value_counts()
        
        targets_1 = df.copy()
        targets_1 = targets_1.loc[targets_1['CONVOCADO'] == 1].sample(frac=1)
        targets_1 = targets_1[:int(number_of_samples/2)]
        targets_1['CONVOCADO'].value_counts()
        
        df = pd.concat([targets_0, targets_1])
        df.sample(frac=1)
        
        df = df.drop([
            'ANO_NASCIMENTO', 
            'DISPENSA',
            'CONVOCADO',
            'MUN_NASCIMENTO', 
            'ESCOLARIDADE', 
            'UF_NASCIMENTO', 
            'SEXO', 'VINCULACAO_ANO', 
            'MUN_RESIDENCIA', 
            'UF_RESIDENCIA', 
            'JSM', 
            'MUN_JSM', 
            'UF_JSM'],axis=1)

        self.preprocessed_data = df.copy()
        
        self.data = self.scaler.transform(df)
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
    def predicted_outupt_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data