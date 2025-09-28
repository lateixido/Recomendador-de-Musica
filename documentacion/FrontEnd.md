# 🎵 Recomendador de Discos

Una aplicación web que recomienda música similar basándose en características musicales.

## 🚀 Características

- **Búsqueda inteligente**: Encuentra canciones por título
- **Desambiguación automática**: Selecciona entre artistas cuando hay coincidencias
- **Recomendaciones precisas**: Basadas en similitud musical
- **Métricas detalladas**: Danceability, Energy, Positiveness y Loudness
- **Interfaz intuitiva**: Diseño limpio y fácil de usar

## 🛠️ Tecnologías

- HTML5
- CSS3 (Grid, Flexbox, Variables CSS)
- JavaScript ES6+
- API REST

## 📦 Instalación

1. Clona o descarga el proyecto
2. Asegúrate de tener el backend ejecutándose en `http://localhost:8080`
3. Abre `index.html` en tu navegador o sirve los archivos estáticos

## 🎯 Uso

1. **Ingresa una canción**: Escribe el nombre de una canción en el campo de búsqueda
2. **Selecciona artista** (si es necesario): Si hay múltiples opciones, elige el artista correcto
3. **Obtén recomendaciones**: La aplicación mostrará canciones similares con sus métricas
4. **Explora características**: Ve los detalles de danceability, energy, positiveness y loudness

## 📊 Características musicales

- **Danceability**: Qué tan bailable es la canción
- **Energy**: Intensidad y actividad percibida
- **Positiveness**: Tonalidad emocional positiva
- **Loudness**: Volumen percibido

## 🌐 API

La aplicación espera un backend en `http://localhost:8080` con los siguientes endpoints:

- `GET /recommend/{song}?artist={artist}`
- Respuestas esperadas:
  - `status: "ok"` - Recomendaciones exitosas
  - `status: "need_artist"` - Requiere selección de artista
  - `status: "not_found"` - Canción no encontrada

## 🎨 Personalización

Se pueden modificar los colores y estilos editando las variables CSS en `:root`:

```css
:root {
  --bg: #f4f4f4;        /* Color de fondo */
  --card: #fff;         /* Color de tarjetas */
  --primary: #007bff;   /* Color principal */
  /* ... más variables */
}