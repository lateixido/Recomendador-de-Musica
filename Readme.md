# Trabajo Práctico Machine Learning - Recomendador Musical
# Grupo 1 - 29/09/2025

# Integrantes
- Macías, Juliana
- Cortés Cid, Francisco
- Moreno, Nahuel
- Teixido, Leonardo

# Entrenador

Este programa entrena un **sistema de recomendación musical** basado en características numéricas de canciones utilizando **KNN (Nearest Neighbors)** con métrica de coseno. A continuación se explica paso por paso.

---

## 1. Importación de librerías

- `pandas`: manejo de datasets tabulares.
- `numpy`: operaciones numéricas.
- `joblib`: guardar y cargar modelos entrenados.
- `sklearn.preprocessing`: escalado de variables (`StandardScaler`) y binarización (aunque aquí solo se usa el escalado).
- `sklearn.neighbors`: algoritmo `NearestNeighbors` (KNN).
- `scipy.sparse`: manejo de matrices dispersas para optimizar memoria.

---

## 2. Carga y limpieza del dataset

- Se carga el dataset `light_spotify_dataset.csv`.
- Se definen columnas necesarias: `song`, `artist`, `Danceability`, `Energy`, `Positiveness`, `Loudness`.
- Se valida que esas columnas existan.
- Se eliminan filas con valores nulos en dichas columnas.

---

## 3. Preparación de features

- Se extraen las features numéricas relevantes (`Danceability`, `Energy`, `Positiveness`, `Loudness`).
- Se escalan usando `StandardScaler` para normalizar los valores (media 0, varianza 1).
- Se convierten a una **matriz dispersa** (`csr_matrix`) para ahorrar memoria.

---

## 4. Entrenamiento del modelo KNN

- Se define `n_neighbors = 10`, `metric = cosine`, `algorithm = brute`.
- El modelo se ajusta con la matriz escalada `X`.

### Justificación de hiperparámetros:
- `n_neighbors=10`: cantidad de recomendaciones a devolver.
- `metric=cosine`: mide la **similaridad angular** entre vectores, útil en datos normalizados.
- `algorithm=brute`: recomendado en datasets pequeños con pocas features.

---

## 5. Construcción de vectores de canciones

Función `build_feature_vector(row_idx)`:
- Toma el índice de una canción en el dataframe.
- Escala sus features igual que el dataset original.
- Devuelve un vector disperso para poder compararlo con el modelo KNN.

---

## 6. Búsqueda del índice de una canción

Función `get_track_index(track, artist=None)`:
- Busca la canción por nombre (ignorando mayúsculas/minúsculas y espacios).
- Si hay varios artistas con el mismo título, pide especificar el artista.
- Lanza errores claros si no encuentra coincidencias.

---

## 7. Función principal de recomendación

Función `recommend_by_track_name(track, top_k=10, artist=None)`:
- Obtiene el índice de la canción de referencia.
- Construye el vector de features de esa canción.
- Consulta el modelo KNN para encontrar las `top_k` canciones más similares.
- Devuelve una lista de diccionarios con:
  - `song`, `artist`
  - Features (`Danceability`, `Energy`, `Positiveness`, `Loudness`)
  - Distancia coseno al track semilla.

---

## 8. Guardado de artefactos

Se guardan en `music_recommender_numeric_small.joblib` los siguientes objetos:
- Modelo KNN entrenado.
- Escalador.
- Hiperparámetros.
- Forma de la matriz de features.
- Subconjunto del dataframe con columnas clave.

---

## 9. Prueba rápida (smoke test)

- Pide al usuario ingresar una canción.
- Si el título es ambiguo, solicita también el artista.
- Devuelve en consola una lista de recomendaciones con valores de features y distancia.

---

## 10. Ejemplo fijo

Ejemplo con la canción `"Dance"` de **Rick Astley**:
- Se buscan los 10 vecinos más cercanos.
- Se calcula la similaridad derivada de la distancia coseno.
- Se muestran en pantalla las canciones más similares con sus distancias y similitudes.

---

# API Recomendador Spotify KNN (numérico)

Se describe la API de recomendación musical basada en **FastAPI** y un modelo **KNN** entrenado con características numéricas de canciones.

---

## Características principales

- Permite generar recomendaciones musicales a partir de una canción semilla.
- Utiliza un modelo KNN entrenado sobre las siguientes características numéricas:
  - Danceability
  - Energy
  - Positiveness
  - Loudness
- Maneja ambigüedad de títulos y artistas.
- Permite filtrar recomendaciones y controlar la cantidad de resultados.
- Expuesto como API REST lista para integrarse con frontends.

---

## Tecnologías y librerías

- FastAPI creación de endpoints REST.
- Pandas y Numpy manejo y procesamiento de datos.
- Joblib carga de modelo y escalador previamente entrenados.
- Scipy (`csr_matrix`) manejo de matrices dispersas.
- CORS Middleware acceso seguro desde frontends locales.

---

## Rutas de la API

### Endpoints principales

- `/recommend/{song_name}`  
  Devuelve recomendaciones para una canción.  
  - Parámetros: `artist` (opcional), `n` cantidad de recomendaciones.  
  - Respuestas posibles:
    - `not_found` canción no encontrada.
    - `need_artist` título ambiguo, se requieren artistas.
    - `ok` recomendaciones generadas.

- `/search?song=nombre`  
  Búsqueda exacta de canciones. Retorna coincidencias y posibles artistas.

- `/`  
  Endpoint raíz. Verifica que la API está activa.

- `/health`  
  Chequeo de salud de la API. Indica si el CSV y el modelo fueron cargados.

---

## Flujo de funcionamiento

1. **Startup**: carga del CSV y artefactos (`knn_model` y `scaler`).
2. **Normalización**: todas las cadenas se convierten a minúsculas y se eliminan espacios extra.
3. **Resolución de ambigüedad**: se determina si la canción existe y si se necesita especificar artista.
4. **Vector de características**: se construye el vector normalizado para la canción semilla.
5. **Cálculo de similitud**: KNN calcula las distancias coseno y genera el ranking de recomendaciones.
6. **Respuesta**: devuelve la canción original y un listado de canciones recomendadas con características y similitud.

---

## Variables y configuraciones importantes

- `NUM_COLS`: columnas numéricas del dataset usadas para KNN.
- `ID_COLS`: columnas de identificación (`song` y `artist`).
- `CSV_PATH`: ruta al dataset CSV.
- `ARTIFACTS_PATH`: ruta al artefacto entrenado (modelo + scaler).
- CORS configurado para permitir acceso desde distintos puertos y frontends locales.

---

## Funcionalidades destacadas

- Manejo de duplicados y ambigüedad en títulos.
- Escalado de features usando `StandardScaler` para consistencia con entrenamiento.
- KNN con matriz dispersa para eficiencia en memoria.
- Control de cantidad de recomendaciones y validaciones de entrada.
- Fácil integración con frontend o sistemas externos mediante endpoints REST.

---

# Tecnologías
- Python, scikit-learn, pandas
- HTML5, CSS3, JavaScript
- Git, GitHub