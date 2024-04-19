import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
try:
    dados = pd.read_csv('Treinar.csv', delimiter=';', encoding='utf-8')
except Exception as e:
    print("Erro ao carregar o arquivo CSV:", e)

# Substituir valores vazios por 0
dados.fillna(0, inplace=True)

# Converter colunas para valores numéricos, lidando com exceções
for column in dados.columns:
    try:
        dados[column] = pd.to_numeric(dados[column])
    except ValueError:
        dados[column] = 0  # Substituir valores não numéricos por 0


# Separar os recursos (X) e o alvo (y)
X = dados.drop(columns=['Outcome'])
y = dados['Outcome']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Inicializar e treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print('Erro quadrático médio:', mse)

###################################################################################
#dados a serem analizados 

# Dados de entrada do usuário (substitua os valores com os dados reais do usuário)
entrada_usuario = {
    'Glucose': 143,
    'BloodPressure': 75,
    'BMI': 211,
    'Age': 52,
    # Adicione mais features conforme necessário
}

# Criar um DataFrame a partir dos dados de entrada do usuário
entrada_df = pd.DataFrame([entrada_usuario])

# Realizar a previsão com o modelo treinado
resultado = modelo.predict(entrada_df)

# Definir um limiar de decisão
limiar = 0.56

# Verificar o resultado e imprimir a mensagem apropriada
if resultado > limiar:
    print("Reação a Diabetes Positivo")
else:
    print("Reação a Diabetes Negativo")
    