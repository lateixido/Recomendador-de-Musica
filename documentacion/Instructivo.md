# Instructivo Recomendador de Música

## 1) Estructura de Carpetas

```
Recomendador-de-Musica/
│
├─ front_end/
│   └─ index.html
│
├─ model-python/
│   ├─ train_model.py
│   ├─ model_api.py
│   └─ music_recommender_numeric_small.joblib
      (aquí se guardarán los artefactos del modelo entrenado)
│
├─ datasets/
│   ├─ light_spotify_dataset.csv
│
├─ documentacion/
│   ├─ Instructivo.md
│   ├─ FrontEnd.md
│   ├─ Readme.md
```

---

## 2) Entrenar el modelo

Desde el directorio `model-python` ejecutar:

```
python train_model.py
```

Esto generará los artefactos de entrenamiento (`*.joblib`) para no tener que correr el entrenador cada vez que se ejecute el `frontend/`.

---

## 3) Levantar el Frontend (HTML)

Desde el directorio `frontend/` ejecutar:

```
python -m http.server [puerto_frontend]
```

Correrá en:  
[http://localhost:[puerto_frontend]](http://localhost:[puerto_frontend])

---

## 4) Levantar el Backend (FastAPI)

Desde el directorio `model-python` ejecutar:

```
python -m uvicorn model_api:app --host 0.0.0.0 --port [puerto_backend] --reload
```

Correrá en:  
[http://localhost:[puerto_backend]](http://localhost:[puerto_backend])