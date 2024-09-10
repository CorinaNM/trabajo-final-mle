# Código de Entrenamiento - Modelo de Predicción de elegibilidad para préstamos
############################################################################


import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('Loan_ID')
    X_train = df.drop(['Loan_Status'],axis=1)
    y_train = df[['Loan_Status']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    lr_mod = LogisticRegression()
    lr_mod.fit(X_train,y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(lr_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('loan_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
