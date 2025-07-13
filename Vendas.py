import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# Carregar o arquivo CSV
df = pd.read_csv(r'D:\Documentos\Mestrado\2025.1\Redes Neurais\Trabalho Final\Itabuna_Vendas_Sorvete.csv')

X = df[['Temperatura', 'DiaSemana', 'Chuva']].values
y = df['Vendas'].values.reshape(-1, 1)

# Normalização
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Separação treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Modificar a arquitetura do modelo para ter duas saídas (média e desvio padrão)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(2) # Duas saídas: média e desvio padrão (log-variance para garantir positividade)
])

# Definir uma função de perda customizada (Negative Log-Likelihood para distribuição Normal)
def normal_dist_loss(y_true, y_pred):
    mu, log_var = tf.split(y_pred, num_or_size_splits=2, axis=1)
    var = K.exp(log_var)
    precision = 1.0 / var
    loss = 0.5 * K.mean(precision * (y_true - mu)**2 + log_var, axis=-1)
    return loss

model.compile(optimizer='adam', loss=normal_dist_loss)

# Treinamento
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[early_stop])

# Previsões (média e desvio padrão)
y_pred_params = model.predict(X_test)
print(f"Shape of y_pred_params after model.predict(X_test): {y_pred_params.shape}")

y_pred_params_np = np.array(y_pred_params)
if y_pred_params_np.shape[-1] != 2:
    raise ValueError(f"Expected model output shape with last dimension 2, but got {y_pred_params_np.shape}")

y_pred_mu_scaled = y_pred_params_np[:, 0:1]
y_pred_log_var_scaled = y_pred_params_np[:, 1:2]

y_pred_var_scaled = np.exp(y_pred_log_var_scaled)
y_pred_std_scaled = np.sqrt(y_pred_var_scaled)

# Inverter a normalização
y_pred_mu = scaler_y.inverse_transform(y_pred_mu_scaled)

y_pred_std = y_pred_std_scaled * (scaler_y.data_range_[0]) # data_range_[0] é o range (max-min)

y_real = scaler_y.inverse_transform(y_test)

# Gráfico vendas com incerteza
plt.figure(figsize=(12,6))
plt.plot(y_real, label='Vendas Reais')
plt.plot(y_pred_mu, label='Média Prevista')
plt.fill_between(range(len(y_pred_mu)),
                 (y_pred_mu - 1.96 * y_pred_std).flatten(),
                 (y_pred_mu + 1.96 * y_pred_std).flatten(),
                 color='green', alpha=0.2, label='Intervalo de Confiança (95%)')
plt.title('Previsão de Vendas de Sorvete com Incerteza')
plt.xlabel('Dias')
plt.ylabel('Quantidade de Sorvetes')
plt.legend()
plt.show()

# Gráfico MSE (agora Loss - Negative Log-Likelihood)
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss Treino')
plt.plot(history.history['val_loss'], label='Loss Validação')
plt.title('Loss do Modelo durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Negative Log-Likelihood')
plt.legend()
plt.show()

# Próxima venda com incerteza
nova_entrada = X_scaled[-1].reshape(1, -1)
proxima_venda_params_scaled = model.predict(nova_entrada)
print(f"Shape of proxima_venda_params_scaled after model.predict(nova_entrada): {proxima_venda_params_scaled.shape}")

proxima_venda_params_np = np.array(proxima_venda_params_scaled)
if proxima_venda_params_np.shape[-1] != 2:
    raise ValueError(f"Expected model output shape with last dimension 2 for next prediction, but got {proxima_venda_params_np.shape}")

proxima_venda_mu_scaled = proxima_venda_params_np[:, 0:1]
proxima_venda_log_var_scaled = proxima_venda_params_np[:, 1:2]

proxima_venda_var_scaled = np.exp(proxima_venda_log_var_scaled)
proxima_venda_std_scaled = np.sqrt(proxima_venda_var_scaled)

proxima_venda_mu = scaler_y.inverse_transform(proxima_venda_mu_scaled)
proxima_venda_std = proxima_venda_std_scaled * (scaler_y.data_range_[0])

print(f'Previsão da próxima venda (Média): {proxima_venda_mu[0][0]:.0f} sorvetes')
print(f'Estimativa de Incerteza (Desvio Padrão): {proxima_venda_std[0][0]:.0f} sorvetes')
print(f'Intervalo de Confiança 95% (aprox): [{proxima_venda_mu[0][0] - 1.96 * proxima_venda_std[0][0]:.0f}, {proxima_venda_mu[0][0] + 1.96 * proxima_venda_std[0][0]:.0f}]')