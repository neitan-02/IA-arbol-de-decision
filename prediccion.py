import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
import joblib
import os

# 📌 Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["RetoMate"]
progreso_col = db["progresos"]
tareas_col = db["tareas"]

# 🧩 Extraer y combinar datos
progresos = list(progreso_col.find())
tareas = {str(t["_id"]): t for t in tareas_col.find()}

# 🔄 Preparar dataset con inferencia de dificultad_objetivo
data = []
# 🎯 USAR "dificil" SIN ACENTO
dificultad_map = {"facil": 0, "media": 1, "dificil": 2}
inv_map = {v: k for k, v in dificultad_map.items()}

for p in progresos:
    tarea = tareas.get(str(p["id_tarea"]))
    if tarea:
        dificultad_actual = tarea.get("dificultad", "media")
        # 🎯 Normalizar a minúsculas sin acentos
        dificultad_actual = dificultad_actual.lower().strip()
        if dificultad_actual == "fácil":
            dificultad_actual = "facil"
        elif dificultad_actual == "difícil":
            dificultad_actual = "dificil"
            
        dificultad_num = dificultad_map.get(dificultad_actual, 1)

        if "puntaje" not in p or "correcto" not in p:
            continue

        # 🧠 inferencia de siguiente dificultad
        if p["correcto"]:
            dificultad_objetivo = min(dificultad_num + 1, 2)
        else:
            dificultad_objetivo = max(dificultad_num - 1, 0)

        data.append({
            "puntaje": p["puntaje"],
            "correcto": int(p["correcto"]),
            "dificultad_actual": dificultad_num,
            "dificultad_objetivo": dificultad_objetivo
        })

df = pd.DataFrame(data)
df.dropna(inplace=True)

# Si no hay datos suficientes, crear un modelo por defecto simple
modelo_path = "modelo_dificultad.pkl"
if df.shape[0] >= 10:
    X = df[["puntaje", "correcto", "dificultad_actual"]]
    y = df["dificultad_objetivo"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    joblib.dump(clf, modelo_path)
else:
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0,0,1]], [1])
    joblib.dump(clf, modelo_path)

# 🌐 Servidor Flask
app = Flask(__name__)
modelo = joblib.load(modelo_path)

@app.route("/predecir", methods=["POST"])
def predecir_api():
    data = request.get_json()
    if not data or not all(k in data for k in ("puntaje", "correcto", "dificultad_actual")):
        return jsonify({"error": "Faltan campos requeridos (puntaje, correcto, dificultad_actual)"}), 400

    
    dificultad_map = {"facil": 0, "media": 1, "dificil": 2}
    
    dificultad_actual_raw = data["dificultad_actual"]
    if isinstance(dificultad_actual_raw, int):
        dificultad_actual_num = dificultad_actual_raw
    else:
        # 🎯 Normalizar entrada
        dificultad_str = str(dificultad_actual_raw).strip().lower()
        if dificultad_str in ["fácil", "facil"]:
            dificultad_str = "facil"
        elif dificultad_str in ["difícil", "dificil"]:
            dificultad_str = "dificil"
            
        dificultad_actual_num = dificultad_map.get(dificultad_str, 1)

    entrada = [[
        float(data.get("puntaje", 0)),
        1 if data.get("correcto") else 0,
        int(dificultad_actual_num)
    ]]
    
    try:
        pred = modelo.predict(entrada)[0]
    except Exception as e:
        # fallback simple
        if data.get("correcto"):
            pred = min(dificultad_actual_num + 1, 2)
        else:
            pred = max(dificultad_actual_num - 1, 0)

 
    etiquetas = {0: "facil", 1: "media", 2: "dificil"}
    return jsonify({"dificultad_sugerida": etiquetas.get(int(pred), "media")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)