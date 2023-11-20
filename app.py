from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

def carregar_dataset():
    dataset = pd.read_csv("dataset-hemograma.csv")
    return dataset

dataset_hemograma = carregar_dataset()

y = dataset_hemograma["diagnostico"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset_hemograma.drop(["id", "diagnostico"], axis=1))

model = LogisticRegression()
model.fit(X_scaled, y)

@app.route("/diagnostico", methods=["POST"])
def sugerir_diagnostico():
    data = request.get_json(force=True)

    required_fields = [
        "eritrocitos",
        "hemoglobina",
        "hematocrito",
        "hcm",
        "vgm",
        "chgm",
        "metarrubricitos",
        "proteina_plasmatica",
        "leucocitos",
        "leucograma",
        "segmentados",
        "bastonetes",
        "blastos",
        "metamielocitos",
        "mielocitos",
        "linfocitos",
        "monocitos",
        "eosinofilos",
        "basofilos",
        "plaquetas",
    ]

    for field in required_fields:
        if field not in data:
            logging.error(f"Campo obrigatório ausente: {field}")
            return jsonify({"erro": f"O campo {field} é obrigatório."}), 400

    features = [data[field] for field in required_fields]
    features_scaled = scaler.transform([features])

    diagnostico_predito = model.predict(features_scaled)[0]

    logging.debug(f"Diagnóstico previsto: {diagnostico_predito}")
    
    id_correspondente = dataset_hemograma.loc[dataset_hemograma['diagnostico'] == diagnostico_predito, 'id'].values[0]
    print(id_correspondente)
    return jsonify({"id":"{}".format(id_correspondente),"diagnostico_predito": diagnostico_predito})

if __name__ == "__main__":
    app.run(debug=True)
