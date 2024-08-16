# Script feito para gerar um grafico de regressao referente aos testes dos datasets nas ferramentas de mineração de processos
# É solicitado uma quantidade de eventos e colunas estimados para gerar o gráfico
# Feito por Gyslla em 28/04/2024

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'experimento': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'total_analise': [81.3, 85.6, 81.3, 84.1, 90.4, 82.1, 95.8, 93.5, 85.5],
    'total_eventos': [390, 390, 390, 3117, 2965, 3117, 2517, 2511, 2511],
    'colunas': [15, 15, 15, 17, 17, 17, 6, 6, 6]
})

X = data[['total_eventos', 'colunas']]
y = data['total_analise']
modelo = LinearRegression()
modelo.fit(X, y)

eventos = int(input("Digite o total de eventos para fazer a previsão: "))
colunas = int(input("Digite a quantidade de colunas para fazer a previsão: "))
previsao = modelo.predict([[eventos, colunas]])

print(f"\nPrevisão para {eventos} eventos e {colunas} colunas:")
print(f"Tempo total de análise previsto: {previsao[0]} minutos")

plt.figure(figsize=(8, 6))
plt.scatter(data['total_eventos'], data['total_analise'], color='blue', label='Dados')
plt.plot([0, eventos], [modelo.predict([[0, colunas]])[0], previsao[0]], color='red', label='Modelo')
plt.axvline(x=eventos, color='green', linestyle='--', label=f'{eventos} eventos')
plt.title('Tempo total de análise em função do número de eventos')
plt.xlabel('Número de eventos')
plt.ylabel('Tempo total de análise (min)')
plt.legend()
plt.grid(True)
plt.show()
