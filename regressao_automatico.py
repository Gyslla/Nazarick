# Script feito para gerar um csv automatizado de regressão referente aos testes dos datasets nas ferramentas de mineração de processos
# Feito por Gyslla em 16/08/2024

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import csv

data = pd.DataFrame({
    'experimento': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'tanalise': [81.3, 85.6, 81.3, 84.1, 90.4, 82.1, 95.8, 93.5, 85.5],
    'teventos': [390, 390, 390, 3117, 2965, 3117, 2517, 2511, 2511],
    'qtdcolunas': [15, 15, 15, 17, 17, 17, 6, 6, 6]
})

X = data[['teventos', 'qtdcolunas']]
y = data['tanalise']
modelo = LinearRegression()
modelo.fit(X, y)

csv_file = "previsoes.csv"
file = open(csv_file, 'w', newline='')
writer = csv.writer(file)
writer.writerow(["timestamp", "total_eventos", "total_colunas", "previsao"])
arquivo = open("previsoes.txt", "a")

for eventos in range(3000, 100001, 500):
    colunas = 20
    previsao = modelo.predict([[eventos, colunas]])
    data_hora_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    arquivo.write(f"{data_hora_atual} >>> Previsão do tempo total de análise para {eventos} eventos e {colunas} colunas: {previsao[0]} minutos\n")
    writer.writerow([data_hora_atual, eventos, colunas, previsao[0]])
    print(f"Previsão gravada para {eventos} eventos e {colunas} colunas.")

arquivo.close()
file.close()
