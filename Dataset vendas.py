import numpy as np
import pandas as pd

# Gerar datas
datas = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

# Simular temperatura para Itabuna/BA (clima tropical quente e úmido)
# Média anual ~25 a 28°C, com variação suave
np.random.seed(42)
dias_do_ano = datas.dayofyear
temperaturas = 26 + 3 * np.sin(2 * np.pi * dias_do_ano / 365) + np.random.normal(0, 1.5, len(datas))

# Dia da semana (0=segunda, 6=domingo)
dias_semana = datas.dayofweek

# Simular chuvas chance maior no verão (novembro a março)
chuva = []
for data in datas:
    if data.month in [11, 12, 1, 2, 3]:
        chance = 0.4
    else:
        chance = 0.2
    alpha = chance * 10
    beta = (1 - chance) * 10
    chuva.append(np.random.beta(alpha, beta))
chuva = np.array(chuva)

# Vendas
# Base: 50 sorvetes
# + 5 sorvetes por °C acima de 20°C
# + 15 se for sábado/domingo
# - 20% se chover
vendas = []
for temp, dia_semana, chuvoso in zip(temperaturas, dias_semana, chuva):
    base = 50
    temp_extra = 5 * (temp - 20)
    fim_de_semana = 15 if dia_semana >= 5 else 0
    chuva_factor = 0.8 if chuvoso else 1.0
    venda = (base + temp_extra + fim_de_semana) * chuva_factor + np.random.normal(0, 5)
    vendas.append(max(0, venda))

# DataFrame final
df_sorvete = pd.DataFrame({
    'Data': datas,
    'Temperatura': np.round(temperaturas, 2),
    'DiaSemana': dias_semana,
    'Chuva': chuva,
    'Vendas': np.round(vendas, 0).astype(int)
})

# Salvar CSV
csv_path = "Trabalho Final/Itabuna_Vendas_Sorvete.csv"
df_sorvete.to_csv(csv_path, index=False)

df_sorvete.head(), csv_path