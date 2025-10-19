# IA - Árbol de decisión (prediccion.py)

Este directorio contiene el script `prediccion.py`, una API ligera con Flask que carga/entrena un modelo (Decision Tree) para predecir la dificultad sugerida de la siguiente tarea en la app [RetoMate](https://github.com/luisillo2048/RetoMate).

## Qué hace `prediccion.py`

- Se conecta a una base de datos MongoDB (colecciones `progresos` y `tareas`).
- Extrae registros de progreso para construir un dataset simple con las columnas: `puntaje`, `correcto`, `dificultad_actual` y la etiqueta objetivo `dificultad_objetivo`.
- Entrena un `DecisionTreeClassifier` (o un `DummyClassifier` si no hay suficientes datos) y guarda el modelo en disco (`modelo_dificultad.pkl` por defecto).
- Expone una pequeña API Flask con los endpoints:
  - `POST /predecir` — recibe `puntaje`, `correcto`, `dificultad_actual` y devuelve `dificultad_sugerida` (`facil`, `media` o `dificil`).
  - `GET /health` — estado del servicio.
  - `GET /` — mensaje de bienvenida.

> Nota: además del modelo ML, el endpoint `/predecir` implementa una lógica determinista (si la respuesta fue correcta sube dificultad; si fue incorrecta baja), garantizando un comportamiento sensato incluso sin modelo.

## Variables de entorno

Las siguientes variables influyen en el comportamiento del script:

- `MONGO_URI` (requerida): URI de conexión a MongoDB. Si no está definida el script falla.
- `MONGO_DB` (opcional): Nombre de la base de datos. Por defecto `RetoMate`.
- `MODEL_PATH` (opcional): Ruta donde se guarda/carga el modelo. Por defecto `modelo_dificultad.pkl`.
- `PORT` (opcional): Puerto para Flask. Por defecto `3000`.

Ejemplo (macOS / zsh):

```bash
export MONGO_URI="mongodb+srv://usuario:pass@cluster0.mongodb.net"
export MONGO_DB="RetoMate"
export MODEL_PATH="./modelo_dificultad.pkl"
export PORT=3000
```

## Dependencias

El script usa las siguientes librerías Python:

- flask
- pymongo
- pandas
- scikit-learn
- joblib

Puedes instalarlas con pip:

```bash
pip install flask pymongo pandas scikit-learn joblib
```

Si quieres reproducir el entorno exactamente, añade las versiones que uses a un `requirements.txt` o crea un entorno virtual.

## Cómo ejecutar

1. Exporta las variables de entorno necesarias (mínimo `MONGO_URI`).
2. Ejecuta:

```bash
python prediccion.py
```

El script intentará conectarse a MongoDB, entrenar (o crear un modelo dummy) y arrancará un servidor Flask en `0.0.0.0:${PORT}`.

## Ejemplos de uso

Request al endpoint `/predecir` (JSON):

```json
{
  "puntaje": 85,
  "correcto": true,
  "dificultad_actual": "media"
}
```

Respuesta ejemplo:

```json
{
  "dificultad_sugerida": "dificil"
}
```

También acepta `dificultad_actual` como número (0=facil, 1=media, 2=dificil).