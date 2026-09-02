# AI Club Condesa — Laboratorio de Gen AI

Material de las sesiones del **AI Club Condesa**: los notebooks del bootcamp de
IA generativa más una app de Streamlit que los convierte en un laboratorio
interactivo para trabajar en clase.

## Contenido

| Archivo | Qué es |
|---|---|
| `app.py` | Laboratorio interactivo (Streamlit): evaluación, RAG, clasificación y function calling con SQLite |
| `day-1-evaluation-and-structured-output.ipynb` | Día 1: evaluación de modelos y salidas estructuradas |
| `day-2-document-q-a-with-rag.ipynb` | Día 2: preguntas sobre documentos con RAG |
| `day-3-building-an-agent-with-langgraph.ipynb` | Día 3: construcción de un agente con LangGraph |
| `day-3-function-calling-with-the-gemini-api.ipynb` | Día 3: function calling con Gemini sobre una base SQLite |

## Cómo correr la app

1. Instala las dependencias:

   ```
   pip install -r requirements.txt
   ```

2. Consigue una API key gratuita en [Google AI Studio](https://aistudio.google.com/apikey).

3. Arranca la app (Windows: doble clic en `run_app.bat`):

   ```
   streamlit run app.py
   ```

4. Pega tu API key en el menú lateral. También puedes dejarla en la variable de
   entorno `GEMINI_API_KEY` para no escribirla cada vez.

> La API key **nunca** se guarda en el repositorio.

## Modelos

- Texto y RAG: `gemini-3.6-flash` (la app detecta los disponibles con tu API key)
- Embeddings: `gemini-embedding-001`

La pestaña de function calling usa una base SQLite en memoria y de solo lectura.
Permite ver cada función y consulta SQL que Gemini utiliza para responder.

Google retira nombres de modelo con el tiempo: `embedding-001`, `text-embedding-004`
y `gemini-2.5-flash` ya devuelven error 404. Si algo deja de funcionar, el primer sospechoso es
un nombre de modelo caducado.
