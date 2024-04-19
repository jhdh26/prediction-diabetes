from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Carregar os dados
dados = pd.read_csv('Treinar.csv', delimiter=';', encoding='utf-8')
dados.fillna(0, inplace=True)
for column in dados.columns:
    try:
        dados[column] = pd.to_numeric(dados[column])
    except ValueError:
        dados[column] = 0

# Separar os recursos (X) e o alvo (y)
X = dados.drop(columns=['Outcome'])
y = dados['Outcome']

# Inicializar e treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    entrada_usuario = request.form.to_dict()
    entrada_df = pd.DataFrame([entrada_usuario])
    resultado = modelo.predict(entrada_df)
    limiar = 0.56
    if resultado > limiar:
        mensagem = "Reação a Diabetes Positivo"
    else:
        mensagem = "Reação a Diabetes Negativo"
    return jsonify({'result': mensagem})

if __name__ == '__main__':
    app.run(debug=True)
