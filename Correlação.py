import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
df = pd.read_csv(r'D:\Documentos\Mestrado\2025.1\Redes Neurais\Trabalho Final\Itabuna_Vendas_Sorvete.csv')

# Exibir as primeiras linhas do DataFrame e as informações das colunas
print(df.head())
print(df.info())

# Certifique-se de que os tipos de dados das colunas numéricas estejam corretos
df['Vendas'] = pd.to_numeric(df['Vendas'], errors='coerce')
df['Temperatura'] = pd.to_numeric(df['Temperatura'], errors='coerce')
df['Chuva'] = pd.to_numeric(df['Chuva'], errors='coerce')

# Remover linhas com valores nulos que podem ter sido introduzidos pelo 'coerce'
df.dropna(inplace=True)

# Calcular a matriz de correlação
correlation_matrix = df[['Vendas', 'Temperatura', 'Chuva']].corr()

# Exibir a matriz de correlação
print("\nMatriz de Correlação:")
print(correlation_matrix)

# Visualizar a matriz de correlação usando um heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação: Venda de Sorvete vs. Temperatura vs. Chuva')
plt.show()