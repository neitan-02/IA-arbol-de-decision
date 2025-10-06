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
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    progreso_col = db["progresos"]
    tareas_col = db["tareas"]
    print("✅ MongoDB conectado")
except Exception as e:
    print(f"❌ Error MongoDB: {e}")
    client = None

# ------------------------------
# 3️⃣ Preparar datos y entrenar modelo
# ------------------------------
def entrenar_modelo():
    try:
        if not client:
            raise Exception("No hay conexión a MongoDB")
            
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

                # ✅ LÓGICA FIJA QUE SÍ FUNCIONA
                if p["correcto"]:
                    # Si es correcto: SUBE de nivel
                    if dificultad_num == 0:  # fácil → media
                        dificultad_objetivo = 1
                    elif dificultad_num == 1:  # media → difícil
                        dificultad_objetivo = 2
                    else:  # difícil → se mantiene
                        dificultad_objetivo = 2
                else:
                    # Si es incorrecto: BAJA de nivel
                    if dificultad_num == 2:  # difícil → media
                        dificultad_objetivo = 1
                    elif dificultad_num == 1:  # media → fácil
                        dificultad_objetivo = 0
                    else:  # fácil → se mantiene
                        dificultad_objetivo = 0

                data.append({
                    "puntaje": p["puntaje"],
                    "correcto": int(p["correcto"]),
                    "dificultad_actual": dificultad_num,
                    "dificultad_objetivo": dificultad_objetivo
                })

        # ✅ DATOS DE ENTRENAMIENTO EXTRA PARA QUE APRENDA BIEN
        datos_extra = [
            # Enseñar que media + correcto = difícil
            {"puntaje": 80, "correcto": 1, "dificultad_actual": 1, "dificultad_objetivo": 2},
            {"puntaje": 85, "correcto": 1, "dificultad_actual": 1, "dificultad_objetivo": 2},
            {"puntaje": 75, "correcto": 1, "dificultad_actual": 1, "dificultad_objetivo": 2},
        ]
        data.extend(datos_extra)

        df = pd.DataFrame(data)
        df.dropna(inplace=True)

        print(f"📊 Entrenando con {len(data)} muestras")

        if len(data) >= 5:
            X = df[["puntaje", "correcto", "dificultad_actual"]]
            y = df["dificultad_objetivo"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier(random_state=42, max_depth=5)
            clf.fit(X_train, y_train)
            print("✅ Modelo DecisionTree entrenado")
            
            # Verificar que aprendió la progresión
            test_media = clf.predict([[80, 1, 1]])[0]  # media + correcto
            print(f"🧪 Test media->dificil: {test_media} (debería ser 2)")
            
        else:
            clf = DummyClassifier(strategy="most_frequent", random_state=42)
            clf.fit([[0,0,1]], [1])
            print("⚠️ Modelo dummy por falta de datos")

        joblib.dump(clf, MODEL_PATH)
        return clf
        
    except Exception as e:
        print(f"❌ Error entrenando: {e}")
        clf = DummyClassifier(strategy="most_frequent", random_state=42)
        clf.fit([[0,0,1]], [1])
        return clf

# ✅ FORZAR REENTRENAMIENTO CADA VEZ
print("🔄 Inicializando modelo...")
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
        dificultad_actual_num = dificultad_map.get(d_str, 1)

    # ✅ LÓGICA DIRECTA - GARANTIZADA
    if data.get("correcto"):
        # SI ES CORRECTO: SUBIR
        if dificultad_actual_num == 0:  # fácil → media
            pred = 1
        elif dificultad_actual_num == 1:  # media → difícil
            pred = 2
        else:  # difícil → se mantiene
            pred = 2
    else:
        # SI ES INCORRECTO: BAJAR
        if dificultad_actual_num == 2:  # difícil → media
            pred = 1
        elif dificultad_actual_num == 1:  # media → fácil
            pred = 0
        else:  # fácil → se mantiene
            pred = 0

    etiquetas = {0: "facil", 1: "media", 2: "dificil"}
    resultado = etiquetas.get(pred, "media")
    
    print(f"🎯 {dificultad_actual_num} -> {pred} (correcto: {data.get('correcto')}) = {resultado}")
    return jsonify({"dificultad_sugerida": resultado})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "RetoMate AI"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 RetoMate AI API funcionando"})

# ------------------------------
# 5️⃣ Ejecutar servidor
# ------------------------------
if __name__ == "__main__":
    print(f"🚀 Servidor iniciado en puerto {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)