# Código de Scoring - Modelo de Predicción de elegibilidad para préstamos
############################################################################

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('Loan_ID')
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('../data/scores/', scores))
    print(scores, 'exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('loan_score.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()