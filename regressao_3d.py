# Script feito para gerar um grafico de regressao referente aos testes dos datasets nas ferramentas de mineração de processos
# É solicitado uma quantidade de eventos e colunas estimados para gerar o gráfico
# Feito por Gyslla em 28/04/2024

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

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

print("\n")
eventos = int(input("Quantidade de eventos: "))
colunas = int(input("Quantidade de colunas: "))
previsao = modelo.predict([[eventos, colunas]])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['teventos'], data['qtdcolunas'], data['tanalise'], color='blue', label='Dados')
x_surf, y_surf = np.meshgrid(np.linspace(data['teventos'].min(), data['teventos'].max(), 10), np.linspace(data['qtdcolunas'].min(), data['qtdcolunas'].max(), 10))
z_surf = modelo.predict(np.array([x_surf.ravel(), y_surf.ravel()]).T).reshape(x_surf.shape)
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3, label='Modelo')
ax.scatter(eventos, colunas, previsao, color='green', s=100, label='Previsão')
ax.set_title('Tempo total de análise em função dos eventos e colunas')
ax.set_xlabel('Total de eventos')
ax.set_ylabel('Quantidade de colunas')
ax.set_zlabel('Tempo total de análise (min)')
ax.legend()
plt.show()

print(f"\n\n>>> Previsão do tempo total de análise para {eventos} eventos e {colunas} colunas: {previsao[0]} minutos\n")
#print(f"Tempo total de análise previsto: {previsao[0]} minutos")
