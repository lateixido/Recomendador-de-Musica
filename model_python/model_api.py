from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from fastapi.middleware.cors import CORSMiddleware

# Crear la aplicación FastAPI
app = FastAPI(title="API Recomendador Spotify KNN (numérico)")

# ---------------------------------------
# Configuración de CORS (para permitir que el frontend acceda al backend)
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",     # Puerto común para servidores estáticos locales
        "http://127.0.0.1:5500",     # Variante con 127.0.0.1
        "null",                      # Necesario si se abre el frontend con file://
        # Agregar otros puertos si se usan frameworks modernos como Vite/Next (ej: 5173, 3000)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Variables globales --------
df = None
knn_model = None
scaler = None

# Columnas esperadas en el dataset
NUM_COLS = ["Danceability", "Energy", "Positiveness", "Loudness"]
ID_COLS = ["song", "artist"]

# Rutas de los archivos
CSV_PATH = "light_spotify_dataset.csv"                 
ARTIFACTS_PATH = "recomendador_musical.joblib"   # Ajustado al nombre real del artefacto

# ---------------------------
# Startup: carga de datos y modelos
# ---------------------------
@app.on_event("startup")
def load_data():
    """
    Cargar dataset y artefactos del modelo (knn_model + scaler).
    """
    global df, knn_model, scaler

    df_local = pd.read_csv(CSV_PATH)

    # Validar columnas obligatorias
    missing = [c for c in (ID_COLS + NUM_COLS) if c not in df_local.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas en el CSV: {missing}")

    # Eliminar filas con valores nulos en columnas importantes
    df_local = df_local.dropna(subset=ID_COLS + NUM_COLS).reset_index(drop=True)

    # Cargar artefactos del entrenamiento
    artifacts = joblib.load(ARTIFACTS_PATH)
    if "knn_model" not in artifacts or "scaler" not in artifacts:
        raise RuntimeError("El archivo de artefactos no contiene 'knn_model' y/o 'scaler'.")

    # Asignar a variables globales
    df = df_local
    knn_model = artifacts["knn_model"]
    scaler = artifacts["scaler"]

    print("✅ Datos y modelos cargados correctamente.")


# ---------------------------------------
# Funciones auxiliares
# ---------------------------------------
def _norm(s: str) -> str:
    """Normalizar cadenas (minúsculas, sin espacios extras)."""
    return s.casefold().strip()


def resolve_song(song_name: str):
    """
    Resolver una canción solo por nombre.
    Devuelve un diccionario con:
      - status: 'not_found' | 'need_artist' | 'single'
      - indices (si es 'single') o lista de artistas (si es 'need_artist')
    """
    name_mask = df["song"].astype(str).str.casefold().str.strip() == _norm(song_name)
    if not name_mask.any():
        return {
            "status": "not_found",
            "message": "Canción no encontrada, por favor escriba otra:"
        }

    idxs = df.index[name_mask].tolist()
    if len(idxs) > 1:
        # Si hay más de un artista con la misma canción
        artists = (
            df.loc[idxs, "artist"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        return {
            "status": "need_artist",
            "message": "Por favor seleccione un artista",
            "options": {"artists": artists}
        }

    return {"status": "single", "index": idxs[0]}


def resolve_song_with_artist(song_name: str, artist: str):
    """
    Resolver una canción teniendo en cuenta también el artista.
    Si el artista no coincide, se ofrecen opciones.
    """
    name_mask = df["song"].astype(str).str.casefold().str.strip() == _norm(song_name)
    if not name_mask.any():
        return {
            "status": "not_found",
            "message": "Canción no encontrada, por favor escriba otra:"
        }

    artist_mask = df["artist"].astype(str).str.casefold().str.strip() == _norm(artist)
    both_mask = name_mask & artist_mask

    if not both_mask.any():
        # El título existe pero el artista no coincide -> ofrecer lista de artistas posibles
        idxs = df.index[name_mask].tolist()
        artists = (
            df.loc[idxs, "artist"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        return {
            "status": "need_artist",
            "message": "Por favor seleccione un artista",
            "options": {"artists": artists}
        }

    idxs = df.index[both_mask].tolist()
    return {"status": "single", "index": idxs[0]}


def build_feature_vector(row_idx: int) -> csr_matrix:
    """Construir vector de características normalizado para una canción."""
    row = df.loc[row_idx, NUM_COLS].to_numpy(dtype=float).reshape(1, -1)
    row_scaled = scaler.transform(row)
    return csr_matrix(row_scaled)


def recommend_by_index(row_idx: int, top_k: int = 10):
    """
    Generar recomendaciones a partir del índice de una canción.
    Devuelve la canción original + un listado de canciones similares.
    """
    q_vec = build_feature_vector(row_idx)
    # Se piden más vecinos de los necesarios para evitar duplicados y excluir la canción base
    distances, indices = knn_model.kneighbors(q_vec, n_neighbors=top_k + 1)

    recs = []
    seen = set()

    for d, i in zip(distances[0], indices[0]):
        if i == row_idx:
            continue  # saltar la canción original
        key = (df.at[i, "song"], df.at[i, "artist"])
        if key in seen:
            continue
        seen.add(key)

        rec = df.iloc[i]
        recs.append({
            "song": rec["song"],
            "artist": rec["artist"],
            "Danceability": float(rec["Danceability"]),
            "Energy": float(rec["Energy"]),
            "Positiveness": float(rec["Positiveness"]),
            "Loudness": float(rec["Loudness"]),
            "similarity": float(max(0.0, min(1.0, 1.0 - float(d))))  # 1 - distancia coseno
        })
        if len(recs) == top_k:
            break

    seed = df.iloc[row_idx]
    return {
        "original_song": {
            "song": seed["song"],
            "artist": seed["artist"],
            "features": {
                "Danceability": float(seed["Danceability"]),
                "Energy": float(seed["Energy"]),
                "Positiveness": float(seed["Positiveness"]),
                "Loudness": float(seed["Loudness"]),
            }
        },
        "recommendations": recs
    }


# ---------------------------------------
# Endpoints REST
# ---------------------------------------
@app.get("/recommend/{song_name}")
def recommend(song_name: str,
              artist: str | None = Query(default=None),
              n: int = Query(default=10, ge=1, le=50)):
    """
    Lógica de recomendación:
    - Si la canción no existe -> retorna {status:'not_found'}
    - Si existe con varios artistas y no se especifica -> retorna {status:'need_artist'}
    - Si existe de forma única o con artista válido -> retorna {status:'ok', recomendaciones}
    """
    if artist is None:
        res = resolve_song(song_name)
    else:
        res = resolve_song_with_artist(song_name, artist)

    if res["status"] in ["not_found", "need_artist"]:
        return res

    # single index -> generar recomendaciones
    payload = recommend_by_index(res["index"], top_k=n)
    return {"status": "ok", **payload}


@app.get("/search")
def search(song: str):
    """Buscar una canción exacta y devolver posibles artistas si hay ambigüedad."""
    name_norm = _norm(song)
    mask = df["song"].astype(str).str.casefold().str.strip() == name_norm
    matches = df.loc[mask, ["song", "artist"]].drop_duplicates().to_dict(orient="records")
    if not len(matches):
        return {"status": "not_found", "message": "Canción no encontrada, por favor escriba otra:", "query": song}
    artists = sorted({m["artist"] for m in matches if "artist" in m and pd.notna(m["artist"])})
    return {"status": "need_artist" if len(artists) > 1 else "single",
            "query": song,
            "options": {"artists": artists} if len(artists) > 1 else None,
            "matches": matches,
            "count": len(matches)}


@app.get("/")
def root():
    """Endpoint raíz: confirma que la API está activa."""
    return {"message": "API de Recomendación de Música (numérica)", "status": "active"}


@app.get("/health")
def health_check():
    """Chequeo de salud: confirma que el dataset y modelos fueron cargados."""
    return {
        "status": "healthy",
        "csv_loaded": df is not None,
        "model_loaded": knn_model is not None,
        "scaler_loaded": scaler is not None,
        "feature_columns": NUM_COLS,
        "rows": int(df.shape[0]) if df is not None else 0
    }