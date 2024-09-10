
# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename)).set_index('Loan_ID')
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    # Eliminanos na's
    df=df.dropna()
    # Recodificación de variables
    lb=LabelEncoder()
    df['Gender']=lb.fit_transform(df['Gender'])
    df['Married']=lb.fit_transform(df['Married'])
    df['Education']=lb.fit_transform(df['Education'])
    df['Self_Employed']=lb.fit_transform(df['Self_Employed'])
    df['Property_Area']=lb.fit_transform(df['Property_Area'])
    # Cambiamos el tipo de datos
    df['LoanAmount']=df['LoanAmount'].apply(np.int64)
    df['CoapplicantIncome']=df['CoapplicantIncome'].apply(np.int64)
    df['Loan_Amount_Term']=df['Loan_Amount_Term'].apply(np.int64)
    df['Credit_History']=df['Credit_History'].apply(np.int64)
    # Eliminación de columna / ruido
    df=df.drop(['Dependents'], axis=1)
    # Eliminación columnas por redundancia información
    df=df.drop(['Loan_Amount_Term','Gender','Education'], axis=1)
    df_f=df.drop(['Married'], axis=1)
    print('Transformación de datos completa')
    return df_f

def data_preparation_target(df):
    # Recodificación de variables
    lb=LabelEncoder()
    df['Loan_Status']=lb.fit_transform(df['Loan_Status'])
    print('Transformación de variable objetivo completa')
    return df



# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('defaultloan.csv')
    tdf1 = data_preparation(df1)
    tdf1_t = data_preparation_target(tdf1)
    data_exporting(tdf1,'loan_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('defaultloan_new.csv')
    tdf2 = data_preparation(df2)
    tdf2_t = data_preparation_target(tdf2)
    data_exporting(tdf2,'loan_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('defaultloan_score.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3,'loan_score.csv')
    
if __name__ == "__main__":
    main()
