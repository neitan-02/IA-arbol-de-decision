# prediccion.py
import os
from flask import Flask, request, jsonify
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import joblib

MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB", "RetoMate")
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_dificultad.pkl")
PORT = int(os.environ.get("PORT", 3000))

if not MONGO_URI:
    raise ValueError("❌ Debes definir la variable MONGO_URI en tu entorno")

# ------------------------------
# 2️⃣ Conexión a MongoDB
# ------------------------------
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
progreso_col = db["progresos"]
tareas_col = db["tareas"]

# ------------------------------
# 3️⃣ Preparar datos y entrenar modelo (si no existe)
# ------------------------------
def entrenar_modelo():
    progresos = list(progreso_col.find())
    tareas = {str(t["_id"]): t for t in tareas_col.find()}

    data = []
    dificultad_map = {"facil": 0, "media": 1, "dificil": 2}

    for p in progresos:
        tarea = tareas.get(str(p["id_tarea"]))
        if tarea:
            dificultad_actual = tarea.get("dificultad", "media").lower().strip()
            if dificultad_actual == "fácil": dificultad_actual = "facil"
            elif dificultad_actual == "difícil": dificultad_actual = "dificil"
            dificultad_num = dificultad_map.get(dificultad_actual, 1)

            if "puntaje" not in p or "correcto" not in p:
                continue

            dificultad_objetivo = min(dificultad_num + 1, 2) if p["correcto"] else max(dificultad_num - 1, 0)

            data.append({
                "puntaje": p["puntaje"],
                "correcto": int(p["correcto"]),
                "dificultad_actual": dificultad_num,
                "dificultad_objetivo": dificultad_objetivo
            })

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    if df.shape[0] >= 10:
        X = df[["puntaje", "correcto", "dificultad_actual"]]
        y = df["dificultad_objetivo"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
    else:
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit([[0,0,1]], [1])

    joblib.dump(clf, MODEL_PATH)
    return clf

# Cargar modelo existente o entrenar
if os.path.exists(MODEL_PATH):
    modelo = joblib.load(MODEL_PATH)
else:
    modelo = entrenar_modelo()

# ------------------------------
# 4️⃣ Servidor Flask
# ------------------------------
app = Flask(__name__)

@app.route("/predecir", methods=["POST"])
def predecir_api():
    data = request.get_json()
    required_fields = ("puntaje", "correcto", "dificultad_actual")
    if not data or not all(k in data for k in required_fields):
        return jsonify({"error": f"Faltan campos requeridos {required_fields}"}), 400

    dificultad_map = {"facil": 0, "media": 1, "dificil": 2}
    dificultad_actual_raw = data["dificultad_actual"]

    if isinstance(dificultad_actual_raw, int):
        dificultad_actual_num = dificultad_actual_raw
    else:
        d_str = str(dificultad_actual_raw).lower().strip()
        if d_str in ["fácil","facil"]: d_str="facil"
        elif d_str in ["difícil","dificil"]: d_str="dificil"
        dificultad_actual_num = dificultad_map.get(d_str,1)

    entrada = [[float(data.get("puntaje",0)), 1 if data.get("correcto") else 0, int(dificultad_actual_num)]]

    try:
        pred = modelo.predict(entrada)[0]
    except:
        pred = min(dificultad_actual_num+1,2) if data.get("correcto") else max(dificultad_actual_num-1,0)

    etiquetas = {0: "facil", 1: "media", 2: "dificil"}
    return jsonify({"dificultad_sugerida": etiquetas.get(int(pred), "media")})

# ------------------------------
# 5️⃣ Ejecutar servidor
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
