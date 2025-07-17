import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\Documentos\Mestrado\2025.1\Redes Neurais\Trabalho Final\Itabuna_Vendas_Sorvete.csv')


X = df[['Temperatura', 'DiaSemana', 'Chuva']].values
y = df['Vendas'].values.reshape(-1, 1)

# Normalização
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Separação treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, shuffle=False)

# Criação MLP
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='tanh', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='mse')

# Treinamento
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[early_stop])

# Previsões
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_real = scaler_y.inverse_transform(y_test)

# Calcular resíduos (erros) entre previsões e valores reais
residuos = (y_real - y_pred).flatten()

# Desvio padrão dos resíduos
std_residuos = np.std(residuos)

# Limites superior e inferior: previsão ± 1 desvio padrão
y_upper = y_pred.flatten() + std_residuos
y_lower = y_pred.flatten() - std_residuos

# Gráfico vendas
plt.figure(figsize=(12,6))
plt.plot(y_real, label='Vendas Reais')
plt.plot(y_pred, label='Vendas Previstas', linestyle = '--')

plt.fill_between(
    range(len(y_pred)),
    y_lower,
    y_upper,
    color='green',
    alpha=0.3,
    label='±1 Desvio Padrão'
)

plt.title('Previsão de Vendas de Sorvete - Itabuna/BA')
plt.xlabel('Dias')
plt.ylabel('Quantidade de Sorvetes')
plt.legend()
plt.show()

# Gráfico MSE
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='MSE Treino')
plt.plot(history.history['val_loss'], label='MSE Validação')
plt.title('MSE do Modelo durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend()
plt.show()

# Próxima venda
nova_entrada = X_scaled[-1].reshape(1, -1)
proxima_venda_scaled = model.predict(nova_entrada)
proxima_venda = scaler_y.inverse_transform(proxima_venda_scaled)
print(f'Previsão da próxima venda: {proxima_venda[0][0]:.0f} sorvetes')