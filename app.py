import os
import re
import enum
import functools
import sqlite3
import tempfile
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ChromaDB arrastra binarios (gRPC) que algunos equipos bloquean por política de
# seguridad de Windows. Se importa únicamente cuando el usuario decide indexar;
# así un fallo de Chroma no retrasa ni tumba las demás pestañas.
Documents = list
Embeddings = list


class EmbeddingFunction:
    """Interfaz mínima compatible con las colecciones de ChromaDB."""


def obtener_chroma_client():
    if "chroma_client" in st.session_state:
        return st.session_state.chroma_client
    if error := st.session_state.get("chroma_error"):
        raise RuntimeError(error)

    try:
        import chromadb
        st.session_state.chroma_client = chromadb.Client()
        return st.session_state.chroma_client
    except Exception as e:
        st.session_state.chroma_error = str(e)
        raise RuntimeError(
            "ChromaDB no pudo cargarse. En Windows suele ocurrir cuando una "
            "política de seguridad bloquea los binarios de gRPC."
        ) from e

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Club Condesa | Gemini & RAG Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .score-badge {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & Client Init
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_available_embed_models(key: str):
    # Modelos de embeddings vigentes. Los antiguos (embedding-001, text-embedding-004,
    # text-embedding-001, gemini-embedding-exp-*) fueron retirados y devuelven 404.
    fallback_models = ["gemini-embedding-001"]
    if not key:
        return fallback_models
    try:
        c = genai.Client(api_key=key)
        # Le preguntamos a la API qué soporta realmente esta key.
        models = []
        for m in c.models.list():
            if m.supported_actions and "embedContent" in m.supported_actions:
                name = m.name.replace("models/", "") if m.name.startswith("models/") else m.name
                models.append(name)
        if not models:
            return fallback_models

        # gemini-embedding-001 primero; experimentales al final.
        def _rank(n: str):
            return (0 if n == "gemini-embedding-001" else 1, "exp" in n, n)

        return sorted(set(models), key=_rank)
    except Exception:
        return fallback_models


@st.cache_data(ttl=3600)
def get_available_text_models(key: str):
    # Los nombres de modelo caducan: preguntamos a la API en vez de fijarlos a mano.
    fallback_models = ["gemini-3.6-flash"]
    if not key:
        return fallback_models
    try:
        c = genai.Client(api_key=key)
        excluir = ("embedding", "imagen", "veo", "tts", "image", "aqa", "live", "audio")
        models = []
        for m in c.models.list():
            if not (m.supported_actions and "generateContent" in m.supported_actions):
                continue
            name = m.name.replace("models/", "") if m.name.startswith("models/") else m.name
            if not name.startswith("gemini-") or any(x in name for x in excluir):
                continue
            models.append(name)
        if not models:
            return fallback_models

        def _rank(n: str):
            v = re.search(r"gemini-(\d+)\.(\d+)", n)
            version = (int(v.group(1)), int(v.group(2))) if v else (0, 0)
            raro = any(x in n for x in ("preview", "exp", "thinking", "lite", "latest"))
            tipo = 0 if "flash" in n else (1 if "pro" in n else 2)
            return (-version[0], -version[1], raro, tipo, n)

        return sorted(set(models), key=_rank)
    except Exception:
        return fallback_models


@st.cache_resource
def get_gemini_client(key: str):
    if not key:
        return None
    return genai.Client(api_key=key)


def extract_text_from_pdf(file_bytes) -> str:
    reader = PdfReader(file_bytes)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


# ---------------------------------------------------------
# Clasificación con embeddings (Día 2b)
#
# El notebook original entrena la capa densa con Keras. Aquí la misma
# capa (softmax de una sola capa) va en NumPy: son ~20 líneas, entrena
# en milisegundos con estos volúmenes y evita arrastrar TensorFlow.
# ---------------------------------------------------------
def embed_para_clasificar(key: str, embed_model: str, textos: list) -> np.ndarray:
    """Vectoriza textos con task_type CLASSIFICATION y los normaliza a norma 1."""
    if not key:
        raise ValueError("No se ha configurado la API Key de Gemini. Ingrésala en el menú lateral izquierdo.")

    c = genai.Client(api_key=key)
    BATCH_SIZE = 100
    vectores = []

    for i in range(0, len(textos), BATCH_SIZE):
        lote = [str(t).strip() for t in textos[i:i + BATCH_SIZE]]
        try:
            respuesta = c.models.embed_content(
                model=embed_model,
                contents=lote,
                config=types.EmbedContentConfig(task_type="CLASSIFICATION"),
            )
        except Exception:
            # Algunos modelos/endpoints rechazan task_type.
            respuesta = c.models.embed_content(model=embed_model, contents=lote)

        if not respuesta.embeddings:
            raise ValueError("La API de Gemini no devolvió embeddings para los textos proporcionados.")

        vectores.extend([e.values for e in respuesta.embeddings])

    X = np.array(vectores, dtype=np.float32)
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(normas, 1e-9)


def ajustar_proyeccion_2d(X: np.ndarray):
    """PCA por SVD: devuelve (centro, ejes) para mapear 3072 dimensiones a 2."""
    centro = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - centro, full_matrices=False)
    return centro, Vt[:2]


def proyectar_2d(X: np.ndarray, centro: np.ndarray, ejes: np.ndarray) -> np.ndarray:
    return (X - centro) @ ejes.T


def entrenar_softmax(X, y, num_clases, epocas=150, lr=0.5, l2=1e-3, semilla=0):
    """Equivalente NumPy de la capa densa + softmax que el notebook arma en Keras."""
    generador = np.random.default_rng(semilla)
    n, d = X.shape
    W = generador.normal(0, 0.01, (d, num_clases)).astype(np.float32)
    b = np.zeros(num_clases, dtype=np.float32)

    Y = np.zeros((n, num_clases), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    historial = []
    for epoca in range(epocas):
        probs = _softmax(X @ W + b)
        perdida = float(-np.log(probs[np.arange(n), y] + 1e-9).mean())
        precision = float((probs.argmax(axis=1) == y).mean())
        historial.append({"Época": epoca + 1, "Pérdida": perdida, "Precisión": precision})

        grad = (probs - Y) / n
        W -= lr * (X.T @ grad + 2 * l2 * W)
        b -= lr * grad.sum(axis=0)

    return W, b, pd.DataFrame(historial)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predecir_probabilidades(X, W, b) -> np.ndarray:
    return _softmax(X @ W + b)


EJEMPLOS_CLASIFICACION = [
    ("Cocina", "Sofríe la cebolla a fuego bajo hasta que quede transparente antes de agregar el jitomate."),
    ("Cocina", "La masa necesita reposar al menos una hora para que el gluten se relaje y estire mejor."),
    ("Cocina", "Si el caldo queda salado, agrega una papa cruda y déjala hervir unos minutos."),
    ("Cocina", "Precalienta el horno a 180 grados antes de meter el molde con la mezcla."),
    ("Cocina", "El aguacate se oxida rápido, ponle unas gotas de limón para conservarlo verde."),
    ("Cocina", "Para que el arroz quede suelto, enjuágalo hasta que el agua salga clara."),
    ("Deportes", "El equipo cerró la primera mitad con dos goles de ventaja y el estadio se vino abajo."),
    ("Deportes", "El maratón exige entrenar series largas por lo menos cuatro veces por semana."),
    ("Deportes", "El árbitro marcó falta al borde del área y le sacó la segunda tarjeta amarilla."),
    ("Deportes", "El nadador rompió su marca personal en los doscientos metros estilo libre."),
    ("Deportes", "Ficharon a un defensa central para reforzar la zaga en el mercado de invierno."),
    ("Deportes", "El tenista salvó tres puntos de partido antes de llevarse el set en muerte súbita."),
    ("Tecnología", "El modelo se entrena en la nube y luego se despliega detrás de una API con autenticación."),
    ("Tecnología", "La actualización trae cifrado de extremo a extremo para los respaldos en la nube."),
    ("Tecnología", "El procesador de la nueva laptop mejora el rendimiento sin castigar la batería."),
    ("Tecnología", "Hay que migrar la base de datos antes de que la versión anterior deje de recibir parches."),
    ("Tecnología", "El código no compila porque falta declarar la dependencia en el archivo de configuración."),
    ("Tecnología", "El servidor respondió con error 500 y los registros apuntan a una fuga de memoria."),
]


# ---------------------------------------------------------
# Function calling sobre SQLite (Día 3)
# ---------------------------------------------------------
def crear_bd_tienda() -> sqlite3.Connection:
    """Crea una base de ejemplo aislada y de solo lectura para una sesión."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            price REAL NOT NULL
        );
        CREATE TABLE staff (
            staff_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            staff_id INTEGER NOT NULL REFERENCES staff(staff_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id)
        );

        INSERT INTO products VALUES
            (1, 'Laptop', 799.99),
            (2, 'Keyboard', 129.99),
            (3, 'Mouse', 29.99);
        INSERT INTO staff VALUES
            (1, 'Alice', 'Smith'),
            (2, 'Bob', 'Johnson'),
            (3, 'Charlie', 'Williams');
        INSERT INTO orders VALUES
            (1, 'David Lee', 1, 1),
            (2, 'Emily Chen', 2, 2),
            (3, 'Frank Brown', 1, 3);
    """)
    conn.execute("PRAGMA query_only = ON")
    return conn


def crear_herramientas_bd(conn: sqlite3.Connection, registro: list[dict]):
    """Construye las herramientas que Gemini puede invocar y registra cada llamada."""

    def list_tables() -> list[str]:
        """Devuelve los nombres de las tablas disponibles en la base de datos."""
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        resultado = [row[0] for row in rows]
        registro.append({"función": "list_tables", "argumentos": {}, "resultado": resultado})
        return resultado

    def describe_table(table_name: str) -> list[str]:
        """Describe las columnas de una tabla existente.

        Args:
            table_name: Nombre exacto de una tabla devuelta por list_tables.
        """
        tablas = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if table_name not in tablas:
            raise ValueError(f"Tabla no permitida: {table_name!r}")

        identificador = table_name.replace('"', '""')
        rows = conn.execute(f'PRAGMA table_info("{identificador}")').fetchall()
        resultado = [f"{row[1]} ({row[2]})" for row in rows]
        registro.append({
            "función": "describe_table",
            "argumentos": {"table_name": table_name},
            "resultado": resultado,
        })
        return resultado

    def execute_query(sql: str) -> list[list[str]]:
        """Ejecuta una consulta SQL de solo lectura y devuelve hasta 100 filas.

        Args:
            sql: Una sola sentencia SQLite SELECT o WITH de solo lectura.
        """
        statement = sql.strip().rstrip(";").strip()
        primera_palabra = statement.split(None, 1)[0].upper() if statement else ""
        if primera_palabra not in {"SELECT", "WITH"} or ";" in statement:
            raise ValueError("Solo se permite una consulta SELECT o WITH de solo lectura.")

        cursor = conn.execute(statement)
        columnas = [col[0] for col in cursor.description or []]
        rows = cursor.fetchmany(101)
        truncado = len(rows) > 100
        rows = rows[:100]
        resultado = [columnas] + [[str(value) for value in row] for row in rows]
        registro.append({
            "función": "execute_query",
            "argumentos": {"sql": statement},
            "resultado": resultado,
            "truncado": truncado,
        })
        return resultado

    return [list_tables, describe_table, execute_query]


INSTRUCCION_FUNCTION_CALLING = """Eres un asistente que responde preguntas sobre
la base de datos SQLite de una tienda de computación. Usa siempre las herramientas
para consultar los datos: primero descubre las tablas y sus columnas cuando sea
necesario y luego ejecuta una consulta SELECT. No inventes datos. Resume el resultado
en español claro y menciona cuando la base no contenga información suficiente.
La base es estrictamente de solo lectura: nunca solicites INSERT, UPDATE, DELETE,
DROP, ALTER, PRAGMA ni más de una sentencia SQL."""

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", width=50)
    st.title("Configuración")
    
    # API Key Handling
    default_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    api_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        help="Ingresa tu clave de Google AI Studio (aistudio.google.com)"
    )
    
    if not api_key:
        st.warning("⚠️ Ingresa tu Gemini API Key para habilitar la generación, evaluación y RAG.")
    else:
        st.success("🔑 API Key configurada")
        
    client = get_gemini_client(api_key) if api_key else None
    
    st.divider()
    
    dynamic_text_options = get_available_text_models(api_key)
    model_choice = st.selectbox(
        "Modelo de Gemini (Texto/RAG)",
        options=dynamic_text_options,
        index=0,
        help="Detectados desde tu API Key. Los modelos flash son los mas rapidos para evaluacion y RAG."
    )
    
    dynamic_embed_options = get_available_embed_models(api_key)
    embed_model_choice = st.selectbox(
        "Modelo de Embeddings",
        options=dynamic_embed_options,
        index=0,
        help="Detectados desde tu API Key. gemini-embedding-001 es el modelo vigente."
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 para respuestas deterministas y consistentes."
    )
    
    st.divider()
    st.markdown("### Acerca de")
    st.markdown("""
    **AI Club Condesa**
    * 📊 **Día 1**: Evaluación LLM & Salidas Estructuradas
    * 🔍 **Día 2**: RAG con ChromaDB & Embeddings
    * 🧭 **Día 2b**: Clasificación con Embeddings
    * 🛠️ **Día 3**: Function Calling con SQLite
    """)

# Custom ChromaDB Embedding Function that securely uses the current API key
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, key: str, embed_model: str = "gemini-embedding-001", document_mode: bool = True):
        self.api_key = key
        self.embed_model = embed_model
        self.document_mode = document_mode

    def __call__(self, input: Documents) -> Embeddings:
        if not self.api_key:
            raise ValueError("No se ha configurado la API Key de Gemini. Ingrésala en el menú lateral izquierdo.")
        
        c = genai.Client(api_key=self.api_key)
        task = "RETRIEVAL_DOCUMENT" if self.document_mode else "RETRIEVAL_QUERY"

        clean_input = [str(doc).strip() for doc in input if str(doc).strip()]
        if not clean_input:
            raise ValueError("Los documentos a indexar están vacíos.")

        # La API limita cuántos textos acepta por petición: enviamos por lotes.
        BATCH_SIZE = 100
        vectors: Embeddings = []

        for i in range(0, len(clean_input), BATCH_SIZE):
            batch = clean_input[i:i + BATCH_SIZE]
            try:
                # Primero con task_type
                response = c.models.embed_content(
                    model=self.embed_model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task),
                )
            except Exception:
                # Fallback sin config (algunos modelos/endpoints rechazan task_type)
                try:
                    response = c.models.embed_content(
                        model=self.embed_model,
                        contents=batch,
                    )
                except Exception as e2:
                    raise ValueError(f"Error al llamar a embed_content con modelo '{self.embed_model}': {str(e2)}\nPrueba seleccionando otro Modelo de Embeddings en el menú lateral.")

            if not response.embeddings:
                raise ValueError("La API de Gemini no devolvió embeddings para los textos proporcionados.")

            vectors.extend([e.values for e in response.embeddings])

        return vectors

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🤖 AI Club Condesa Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Laboratorio interactivo de Evaluación con LLM-as-a-Judge y RAG con Gemini y ChromaDB</div>', unsafe_allow_html=True)

tab_eval, tab_rag, tab_clf, tab_tools = st.tabs([
    "📊 Día 1: Evaluación & Salidas Estructuradas",
    "🔍 Día 2: Document Q&A con RAG (ChromaDB)",
    "🧭 Día 2b: Clasificación con Embeddings",
    "🛠️ Día 3: Function Calling con SQLite",
])

# =========================================================
# TAB 1: EVALUATION & STRUCTURED OUTPUT (DAY 1)
# =========================================================
with tab_eval:
    st.header("Evaluación de Modelos y Salidas Estructuradas")
    st.markdown("""
    Aprende a evaluar respuestas de IA utilizando técnicas de **LLM-as-a-Judge**, 
    rúbricas de evaluación paso a paso y coerción a **Enums tipados de Python**.
    """)
    
    eval_mode = st.radio(
        "Modo de Evaluación:",
        ["1. Evaluación Puntual (Pointwise)", "2. Evaluación por Pares (Pairwise)", "3. Ranking y Torneo de Prompts"],
        horizontal=True
    )
    
    class SummaryRating(enum.Enum):
        VERY_GOOD = '5'
        GOOD = '4'
        OK = '3'
        BAD = '2'
        VERY_BAD = '1'

    class AnswerComparison(enum.Enum):
        A = 'A'
        SAME = 'SAME'
        B = 'B'

    # Rúbricas
    SUMMARY_PROMPT = """\
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models.
We will provide you with the user input and an AI-generated response.
Evaluate the quality based on the Criteria provided below.
Assign a rating from 5, 4, 3, 2, 1 following the Rating Rubric and Evaluation Steps. Give step-by-step explanations.

# Criteria
- Instruction following: The response satisfies all requirements.
- Groundedness: The response contains information included only in the context.
- Conciseness: The response is direct without losing key info.
- Fluency: Well-organized and easy to read.

# Rating Rubric
5: (Very good). Follows instructions, grounded, concise, fluent.
4: (Good). Follows instructions, grounded, concise, fluent.
3: (Ok). Mostly follows instructions, grounded, but not very concise/fluent.
2: (Bad). Grounded, but does not follow instructions.
1: (Very bad). Not grounded.

# Inputs
## Prompt:
{prompt}

## AI Response:
{response}
"""

    QA_PAIRWISE_PROMPT = """\
# Instruction
You are an expert evaluator. Evaluate the quality of two AI responses (Response A and Response B) for the given prompt and context.
Compare them and declare the winner: "A", "SAME", or "B".

# User Input
{prompt}

# Response A
{baseline_model_response}

# Response B
{response}
"""

    if eval_mode == "1. Evaluación Puntual (Pointwise)":
        st.subheader("Evaluación Puntual (Pointwise LLM-as-a-Judge)")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            doc_context = st.text_area(
                "Texto o Documento de Contexto:",
                value="Gemini 2.5 Flash is Google's next-generation fast, lightweight multimodal model designed for high-frequency, low-latency reasoning, coding, and document analysis tasks. It features long context capabilities and natively supports structured outputs.",
                height=150
            )
            user_prompt = st.text_input("Instrucción / Prompt del Usuario:", value="Explica qué es Gemini 2.5 Flash en una sola oración.")
            
            gen_btn = st.button("🚀 Generar y Evaluar", width="stretch")
            
        with col2:
            if gen_btn:
                if not api_key:
                    st.error("⚠️ Por favor ingresa tu Gemini API Key en el menú lateral izquierdo.")
                else:
                    try:
                        with st.spinner(f"1. Generando respuesta con {model_choice}..."):
                            gen_resp = client.models.generate_content(
                                model=model_choice,
                                contents=[f"Contexto: {doc_context}\n\nPregunta: {user_prompt}"],
                                config=types.GenerateContentConfig(temperature=temperature)
                            )
                            ai_answer = gen_resp.text
                            
                        st.markdown("### 📝 Respuesta Generada:")
                        st.info(ai_answer)
                        
                        with st.spinner("2. Evaluando con Rúbrica y Structured Output..."):
                            chat = client.chats.create(model=model_choice)
                            eval_msg = SUMMARY_PROMPT.format(prompt=f"{user_prompt}\nContexto: {doc_context}", response=ai_answer)
                            raw_eval = chat.send_message(message=eval_msg)
                            
                            # Coerción a Enum estructurado
                            struct_config = types.GenerateContentConfig(
                                response_mime_type="text/x.enum",
                                response_schema=SummaryRating,
                            )
                            struct_resp = chat.send_message(
                                message="Convert the final score to the enum.",
                                config=struct_config
                            )
                            final_rating = struct_resp.parsed
                            
                        st.markdown("### 🎯 Veredicto y Calificación Estructurada:")
                        rating_colors = {
                            "VERY_GOOD": "🟢 Excelente (5/5)",
                            "GOOD": "🔵 Bueno (4/5)",
                            "OK": "🟡 Aceptable (3/5)",
                            "BAD": "🟠 Regular (2/5)",
                            "VERY_BAD": "🔴 Deficiente (1/5)"
                        }
                        rating_name = final_rating.name if hasattr(final_rating, 'name') else str(final_rating)
                        st.success(f"**Calificación:** {rating_colors.get(rating_name, rating_name)}")
                        
                        with st.expander("🔍 Ver Análisis Detallado del Evaluador", expanded=True):
                            st.markdown(raw_eval.text)
                    except Exception as e:
                        st.error(f"Error durante la evaluación: {str(e)}")

    elif eval_mode == "2. Evaluación por Pares (Pairwise)":
        st.subheader("Evaluación por Pares (A vs B)")
        st.markdown("Compara dos respuestas distintas para determinar cuál cumple mejor con la rúbrica.")
        
        c_prompt = st.text_input("Pregunta / Tarea:", value="¿Cómo funciona la transmisión automática en el Googlecar?")
        col_a, col_b = st.columns(2)
        with col_a:
            resp_a = st.text_area("Respuesta A:", value="Tiene Park, Reverse, Neutral, Drive y Low. Para cambiar de marcha solo mueves la palanca a la posición deseada.", height=120)
        with col_b:
            resp_b = st.text_area("Respuesta B:", value="El vehículo cuenta con una transmisión totalmente automática donde el usuario puede seleccionar entre múltiples modalidades dependiendo del clima.", height=120)
            
        if st.button("⚖️ Comparar Respuestas", width="stretch"):
            if not api_key:
                st.error("⚠️ Ingresa tu Gemini API Key en el menú lateral izquierdo.")
            else:
                try:
                    with st.spinner(f"Juzgando respuestas con {model_choice}..."):
                        chat = client.chats.create(model=model_choice)
                        judge_prompt = QA_PAIRWISE_PROMPT.format(
                            prompt=c_prompt,
                            baseline_model_response=resp_a,
                            response=resp_b
                        )
                        eval_res = chat.send_message(message=judge_prompt)
                        
                        struct_config = types.GenerateContentConfig(
                            response_mime_type="text/x.enum",
                            response_schema=AnswerComparison,
                        )
                        struct_res = chat.send_message(
                            message="Convert the final judgment to enum choice (A, SAME, or B).",
                            config=struct_config
                        )
                        winner = struct_res.parsed
                        
                    st.markdown("### 🏆 Resultado del Juez:")
                    winner_name = winner.name if hasattr(winner, 'name') else str(winner)
                    if winner_name == "A":
                        st.success("🥇 **Ganador: Respuesta A**")
                    elif winner_name == "B":
                        st.success("🥇 **Ganador: Respuesta B**")
                    else:
                        st.info("🤝 **Empate (Ambas son igualmente válidas)**")
                        
                    with st.expander("Detalle del Juicio", expanded=True):
                        st.markdown(eval_res.text)
                except Exception as e:
                    st.error(f"Error durante la comparación: {str(e)}")

    elif eval_mode == "3. Ranking y Torneo de Prompts":
        st.subheader("Ranking Automático de Prompts (Total Ordering)")
        st.markdown("Ejecuta un torneo comparativo para ordenar diferentes estrategias de prompt por efectividad.")
        
        st.code("""
1. Terse Guidance: 'Answer in a single sentence.'
2. Moderate Guidance: 'Provide a brief answer with minimal citations.'
3. Cited Guidance: 'Provide a thorough, detailed answer citing background.'
        """, language="yaml")
        
        st.info("Este método utiliza `functools.total_ordering` en Python para permitir que `sorted([prompt_a, prompt_b, ...])` funcione automáticamente usando Gemini como evaluador de comparación.")

# =========================================================
# TAB 2: DOCUMENT Q&A WITH RAG (DAY 2)
# =========================================================
with tab_rag:
    st.header("Document Q&A con RAG y ChromaDB")
    st.markdown("""
    Indexa documentos en **ChromaDB** usando el modelo oficial de embeddings **`gemini-embedding-001`** 
    y genera respuestas aumentadas con el modelo de Gemini que elijas en el menu lateral.
    """)

    # El cliente se crea de forma diferida al pulsar Indexar. Esto mantiene
    # disponibles las demás pestañas aunque ChromaDB no funcione en el equipo.
    chroma_client = st.session_state.get("chroma_client")
    if chroma_error := st.session_state.get("chroma_error"):
        st.error(
            "⚠️ ChromaDB no pudo cargarse en esta computadora, así que esta pestaña "
            "está deshabilitada. **Las demás pestañas funcionan con normalidad.**"
        )
        st.caption(f"Detalle técnico: {chroma_error}")
        st.info(
            "Causa más común en Windows: una política de seguridad del sistema "
            "bloquea los componentes de gRPC que ChromaDB necesita. "
            "En Kaggle o Colab esta pestaña funciona sin cambios."
        )

    rag_col1, rag_col2 = st.columns([1, 1])

    with rag_col1:
        st.subheader("1. Base de Conocimiento (Indexación)")
        
        doc_source = st.radio("Fuente de Documentos:", ["Dataset Googlecar (Predeterminado)", "Cargar Archivo Propio (TXT/PDF)"], horizontal=True)
        
        default_docs = [
            "Operating the Climate Control System: Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console. Temperature: Turn clockwise to increase, counterclockwise to decrease. Mode: Auto adjusts temperature automatically. Cool blows cool air. Heat blows warm air. Defrost blows warm air onto windshield.",
            "Touchscreen Display: Your Googlecar has a large touchscreen display that provides access to navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon. For example, touch Navigation to get directions or Music to play your favorite songs.",
            "Shifting Gears: Your Googlecar has an automatic transmission. Park locks the wheels. Reverse is used to back up. Neutral is for stopped in traffic. Drive is used to drive forward. Low is used for snow or slippery conditions."
        ]
        
        if doc_source == "Dataset Googlecar (Predeterminado)":
            if st.button("📥 Indexar Documentos de Googlecar en ChromaDB", width="stretch"):
                if not api_key:
                    st.error("⚠️ Por favor ingresa tu Gemini API Key en el menú lateral izquierdo antes de indexar.")
                else:
                    try:
                        with st.spinner(f"Generando embeddings con {embed_model_choice} e indexando..."):
                            chroma_client = obtener_chroma_client()
                            embed_fn = GeminiEmbeddingFunction(key=api_key, embed_model=embed_model_choice, document_mode=True)

                            
                            # Clean/recreate collection
                            try:
                                chroma_client.delete_collection("googlecar_kb")
                            except Exception:
                                pass
                            
                            collection = chroma_client.create_collection(name="googlecar_kb", embedding_function=embed_fn)
                            collection.add(
                                documents=default_docs,
                                ids=[f"doc_{i}" for i in range(len(default_docs))]
                            )
                        st.success(f"✅ ¡{len(default_docs)} documentos indexados en ChromaDB!")
                    except Exception as e:
                        st.error(f"Error al indexar documentos: {str(e)}")
        else:
            uploaded_file = st.file_uploader("Sube un archivo TXT o PDF", type=["txt", "pdf"])
            if uploaded_file and st.button("📥 Indexar Documento Subido", width="stretch"):
                if not api_key:
                    st.error("⚠️ Por favor ingresa tu Gemini API Key en el menú lateral izquierdo antes de indexar.")
                else:
                    try:
                        with st.spinner("Extrayendo texto y generando embeddings..."):
                            chroma_client = obtener_chroma_client()
                            if uploaded_file.name.lower().endswith(".pdf"):
                                text = extract_text_from_pdf(uploaded_file)
                            else:
                                text = uploaded_file.read().decode("utf-8", errors="ignore")
                            
                            if not text.strip():
                                st.error("No se pudo extraer texto del archivo subido. Asegúrate de que no esté vacío.")
                            else:
                                # Dividir en fragmentos (chunks) por párrafos
                                raw_paragraphs = text.split("\n\n")
                                paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) > 20]
                                if not paragraphs:
                                    paragraphs = [text.strip()]
                                    
                                embed_fn = GeminiEmbeddingFunction(key=api_key, embed_model=embed_model_choice, document_mode=True)
                                
                                try:
                                    chroma_client.delete_collection("googlecar_kb")
                                except Exception:
                                    pass
                                
                                collection = chroma_client.create_collection(name="googlecar_kb", embedding_function=embed_fn)
                                collection.add(
                                    documents=paragraphs,
                                    ids=[f"chunk_{i}" for i in range(len(paragraphs))]
                                )
                                st.success(f"✅ ¡{len(paragraphs)} fragmentos indexados con éxito!")
                    except Exception as e:
                        st.error(f"Error al procesar el archivo: {str(e)}")

        # Count documents in active collection
        active_count = 0
        try:
            temp_embed_fn = GeminiEmbeddingFunction(key=api_key if api_key else "dummy", embed_model=embed_model_choice, document_mode=True)
            active_col = chroma_client.get_or_create_collection(name="googlecar_kb", embedding_function=temp_embed_fn)
            active_count = active_col.count()
        except Exception:
            active_count = 0
            
        st.metric("Documentos en Vector DB", active_count)
        
        with st.expander("👀 Ver documentos almacenados"):
            if active_count > 0:
                try:
                    peek_data = active_col.peek(5)
                    for idx, doc in enumerate(peek_data["documents"]):
                        st.markdown(f"**Doc {idx+1}:** {doc[:160]}...")
                except Exception:
                    st.caption("No se pudieron cargar las previsualizaciones.")
            else:
                st.caption("Aún no hay documentos indexados. Haz clic en 'Indexar' arriba.")

    with rag_col2:
        st.subheader("2. Consulta Semántica y RAG Q&A")
        
        query_text = st.text_input(
            "Pregunta sobre los documentos:",
            value="How do you use the touchscreen to play music?"
        )
        
        top_k = st.slider("Número de fragmentos a recuperar (Top K):", min_value=1, max_value=5, value=2)
        
        if st.button("🔍 Buscar y Responder con RAG", width="stretch"):
            if not api_key:
                st.error("⚠️ Ingresa tu Gemini API Key en el menú lateral izquierdo.")
            elif active_count == 0:
                st.warning("⚠️ Primero debes indexar documentos en la columna izquierda (Paso 1).")
            else:
                try:
                    with st.spinner("1. Buscando fragmentos semánticamente relevantes..."):
                        query_embed_fn = GeminiEmbeddingFunction(key=api_key, embed_model=embed_model_choice, document_mode=False)
                        q_col = chroma_client.get_collection(name="googlecar_kb", embedding_function=query_embed_fn)
                        
                        query_res = q_col.query(
                            query_texts=[query_text],
                            n_results=min(top_k, active_count)
                        )
                        retrieved_docs = query_res["documents"][0] if query_res.get("documents") else []
                        
                    st.markdown("### 📚 Fragmentos Recuperados (Contexto):")
                    if not retrieved_docs:
                        st.warning("No se encontraron fragmentos relevantes.")
                    else:
                        for i, doc in enumerate(retrieved_docs):
                            st.info(f"**Fragmento {i+1}:**\n\n{doc}")
                        
                        with st.spinner(f"2. Generando respuesta con {model_choice}..."):
                            rag_prompt = f"""You are a helpful and informative assistant that answers questions using the reference passages provided below.
Break down complicated concepts, be comprehensive, and maintain a friendly and conversational tone.

QUESTION: {query_text}

REFERENCE CONTEXT:
"""
                            for i, doc in enumerate(retrieved_docs):
                                rag_prompt += f"\n[Passage {i+1}]: {doc}\n"
                                
                            rag_response = client.models.generate_content(
                                model=model_choice,
                                contents=rag_prompt,
                                config=types.GenerateContentConfig(temperature=temperature)
                            )
                            
                        st.markdown("### 💡 Respuesta Generada (RAG):")
                        st.success(rag_response.text)
                except Exception as e:
                    st.error(f"Error en la consulta RAG: {str(e)}")

# =========================================================
# TAB 3: CLASIFICACIÓN CON EMBEDDINGS (DÍA 2B)
# =========================================================
with tab_clf:
    st.header("Clasificación con Embeddings")
    st.markdown("""
    Los embeddings convierten cada texto en una lista de números. Textos que
    significan cosas parecidas caen cerca unos de otros. Aquí lo vas a **ver**:
    primero el mapa de significados, luego un clasificador entrenado encima.
    """)

    if "clf_datos" not in st.session_state:
        st.session_state.clf_datos = pd.DataFrame(
            EJEMPLOS_CLASIFICACION, columns=["Categoría", "Texto"]
        )

    # ---------- Paso 1: los ejemplos ----------
    st.subheader("1. Ejemplos de entrenamiento")
    st.caption(
        "Edita las celdas, agrega renglones o cambia las categorías. "
        "Con 4 o 5 ejemplos por categoría ya alcanza."
    )

    datos = st.data_editor(
        st.session_state.clf_datos,
        num_rows="dynamic",
        width="stretch",
        height=280,
        column_config={
            "Categoría": st.column_config.TextColumn("Categoría", width="small"),
            "Texto": st.column_config.TextColumn("Texto", width="large"),
        },
        key="clf_editor",
    )

    datos = datos.dropna(subset=["Categoría", "Texto"])
    datos = datos[
        datos["Categoría"].astype(str).str.strip().ne("")
        & datos["Texto"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)

    categorias = sorted(datos["Categoría"].astype(str).str.strip().unique().tolist())

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Ejemplos", len(datos))
    col_m2.metric("Categorías", len(categorias))
    col_m3.metric(
        "Vectorizados",
        len(st.session_state.get("clf_X", [])) if "clf_X" in st.session_state else 0,
    )

    # ---------- Paso 2: vectorizar ----------
    st.subheader("2. Convertir los textos en vectores")

    if st.button("🧮 Generar embeddings y dibujar el mapa", width="stretch"):
        if not api_key:
            st.error("⚠️ Ingresa tu Gemini API Key en el menú lateral izquierdo.")
        elif len(categorias) < 2:
            st.error("⚠️ Necesitas al menos 2 categorías distintas.")
        elif len(datos) < len(categorias) * 2:
            st.error("⚠️ Necesitas al menos 2 ejemplos por categoría.")
        else:
            try:
                textos = datos["Texto"].astype(str).tolist()
                with st.spinner(f"Vectorizando {len(textos)} textos con {embed_model_choice}..."):
                    X = embed_para_clasificar(api_key, embed_model_choice, textos)
                    centro, ejes = ajustar_proyeccion_2d(X)

                st.session_state.clf_X = X
                st.session_state.clf_y = np.array(
                    [categorias.index(str(c).strip()) for c in datos["Categoría"]]
                )
                st.session_state.clf_categorias = categorias
                st.session_state.clf_centro = centro
                st.session_state.clf_ejes = ejes
                st.session_state.clf_textos = textos
                # El clasificador anterior ya no corresponde a estos vectores.
                st.session_state.pop("clf_W", None)
                st.session_state.pop("clf_historial", None)
                st.session_state.pop("clf_ultima_prediccion", None)

                st.success(
                    f"✅ {len(textos)} textos convertidos en vectores de "
                    f"{X.shape[1]:,} dimensiones cada uno."
                )
            except Exception as e:
                st.error(f"Error al generar embeddings: {str(e)}")

    if "clf_X" in st.session_state:
        st.markdown("#### 🗺️ Mapa de significados")
        st.caption(
            "Cada punto es un texto. De las miles de dimensiones del embedding, "
            "aquí se ven las 2 que más separan a los ejemplos. Nadie le dijo al modelo "
            "cuáles eran las categorías: se agrupan solos por significado."
        )

        puntos = proyectar_2d(
            st.session_state.clf_X, st.session_state.clf_centro, st.session_state.clf_ejes
        )
        mapa = pd.DataFrame({
            "x": puntos[:, 0],
            "y": puntos[:, 1],
            "Categoría": [st.session_state.clf_categorias[i] for i in st.session_state.clf_y],
            "Texto": st.session_state.clf_textos,
        })

        prediccion = st.session_state.get("clf_ultima_prediccion")
        if prediccion:
            mapa = pd.concat([mapa, pd.DataFrame([{
                "x": prediccion["punto"][0],
                "y": prediccion["punto"][1],
                "Categoría": "★ Tu texto",
                "Texto": prediccion["texto"],
            }])], ignore_index=True)

        st.scatter_chart(mapa, x="x", y="y", color="Categoría", size=90, height=420)

        # ---------- Paso 3: entrenar ----------
        st.subheader("3. Entrenar el clasificador")
        st.caption(
            "El mapa ya separa los temas, pero para decidir por sí sola la máquina "
            "necesita trazar las fronteras. Eso es lo que aprende aquí."
        )

        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            epocas = st.slider("Vueltas de entrenamiento (épocas):", 20, 400, 150, step=10)
            tasa = st.select_slider(
                "Qué tan grandes son los pasos de ajuste:",
                options=[0.1, 0.25, 0.5, 1.0, 2.0],
                value=0.5,
            )
            entrenar_btn = st.button("🎯 Entrenar", width="stretch")

        if entrenar_btn:
            with st.spinner("Entrenando..."):
                W, b, historial = entrenar_softmax(
                    st.session_state.clf_X,
                    st.session_state.clf_y,
                    len(st.session_state.clf_categorias),
                    epocas=epocas,
                    lr=tasa,
                )
            st.session_state.clf_W = W
            st.session_state.clf_b = b
            st.session_state.clf_historial = historial

        if "clf_historial" in st.session_state:
            historial = st.session_state.clf_historial
            with col_e2:
                final = historial.iloc[-1]
                st.metric("Precisión sobre los ejemplos", f"{final['Precisión'] * 100:.0f}%")
                st.metric("Pérdida final", f"{final['Pérdida']:.4f}")
                st.caption(
                    "La precisión se mide sobre los mismos ejemplos con los que entrenó, "
                    "así que es optimista. Para saber si de verdad aprendió, "
                    "pruébalo abajo con texto que no esté en la tabla."
                )

            graf1, graf2 = st.columns(2)
            with graf1:
                st.markdown("**Pérdida (qué tan equivocado está)**")
                st.line_chart(historial.set_index("Época")["Pérdida"], height=220)
            with graf2:
                st.markdown("**Precisión (qué tanto acierta)**")
                st.line_chart(historial.set_index("Época")["Precisión"], height=220)

            # ---------- Paso 4: probar ----------
            st.subheader("4. Probarlo con texto nuevo")
            st.caption("Escribe algo que no esté en la tabla y mira cómo lo clasifica.")

            texto_nuevo = st.text_area(
                "Texto a clasificar:",
                value="Dejé reposando la masa mientras revisaba el marcador del partido.",
                height=90,
            )

            if st.button("🔮 Clasificar", width="stretch"):
                if not texto_nuevo.strip():
                    st.warning("Escribe algo primero.")
                else:
                    try:
                        with st.spinner("Vectorizando y clasificando..."):
                            X_nuevo = embed_para_clasificar(
                                api_key, embed_model_choice, [texto_nuevo]
                            )
                            probs = predecir_probabilidades(
                                X_nuevo, st.session_state.clf_W, st.session_state.clf_b
                            )[0]
                            punto = proyectar_2d(
                                X_nuevo,
                                st.session_state.clf_centro,
                                st.session_state.clf_ejes,
                            )[0]

                        cats = st.session_state.clf_categorias
                        ganadora = cats[int(probs.argmax())]
                        confianza = float(probs.max())

                        st.session_state.clf_ultima_prediccion = {
                            "texto": texto_nuevo,
                            "punto": punto,
                        }

                        if confianza >= 0.6:
                            st.success(f"### {ganadora}  ·  {confianza * 100:.1f}% de confianza")
                        else:
                            st.warning(
                                f"### {ganadora}  ·  {confianza * 100:.1f}% de confianza\n\n"
                                "Confianza baja: el texto mezcla temas o no se parece a "
                                "ninguna categoría."
                            )

                        st.bar_chart(
                            pd.DataFrame({"Probabilidad": probs}, index=cats),
                            height=240,
                        )
                        st.caption("Vuelve a mirar el mapa de arriba: tu texto aparece como ★.")
                    except Exception as e:
                        st.error(f"Error al clasificar: {str(e)}")
        else:
            st.info("👆 Entrena el clasificador para poder probarlo con texto nuevo.")
    else:
        st.info("👆 Genera los embeddings para ver el mapa y entrenar el clasificador.")

# =========================================================
# TAB 4: FUNCTION CALLING CON SQLITE (DÍA 3)
# =========================================================
with tab_tools:
    st.header("Function Calling con Gemini y SQLite")
    st.markdown("""
    Haz preguntas en lenguaje natural sobre una tienda. Gemini no recibe las tablas
    completas: descubre el esquema, construye SQL y llama funciones Python para
    consultar una base local de **solo lectura**.
    """)

    if "fc_conn" not in st.session_state:
        st.session_state.fc_conn = crear_bd_tienda()
    if "fc_mensajes" not in st.session_state:
        st.session_state.fc_mensajes = []
    if "fc_history" not in st.session_state:
        st.session_state.fc_history = []

    col_datos, col_chat = st.columns([0.85, 1.5], gap="large")

    with col_datos:
        st.subheader("Base de datos local")
        st.caption("Los datos viven en memoria y se reinician al cerrar la sesión.")

        tabla_visible = st.selectbox(
            "Inspeccionar tabla",
            ["products", "staff", "orders"],
            key="fc_tabla_visible",
        )
        df_tabla = pd.read_sql_query(f'SELECT * FROM "{tabla_visible}"', st.session_state.fc_conn)
        st.dataframe(df_tabla, width="stretch", hide_index=True)

        if st.button("🧹 Reiniciar conversación y datos", width="stretch"):
            try:
                st.session_state.fc_conn.close()
            except Exception:
                pass
            st.session_state.fc_conn = crear_bd_tienda()
            st.session_state.fc_mensajes = []
            st.session_state.fc_history = []
            st.rerun()

        st.markdown("#### Prueba estas preguntas")
        st.markdown("""
        - ¿Cuál es el producto más barato?
        - ¿Cuántas órdenes atendió cada vendedor?
        - ¿Qué productos ha vendido Alice?
        - ¿Quién todavía no tiene ventas?
        """)

    with col_chat:
        st.subheader("Chat con herramientas")

        if not st.session_state.fc_mensajes:
            st.info("Escribe una pregunta para ver cómo Gemini decide qué funciones llamar.")

        for mensaje in st.session_state.fc_mensajes:
            with st.chat_message(mensaje["role"]):
                st.markdown(mensaje["content"])
                llamadas = mensaje.get("llamadas", [])
                if llamadas:
                    with st.expander(f"🔧 {len(llamadas)} llamada(s) a herramientas"):
                        for numero, llamada in enumerate(llamadas, start=1):
                            st.markdown(f"**{numero}. `{llamada['función']}`**")
                            if llamada["argumentos"]:
                                st.json(llamada["argumentos"])
                            st.caption("Resultado devuelto a Gemini")
                            st.json(llamada["resultado"])
                            if llamada.get("truncado"):
                                st.warning("El resultado se limitó a 100 filas.")

        prompt_bd = st.chat_input(
            "Pregunta algo sobre productos, vendedores u órdenes…",
            key="fc_prompt",
        )

        if prompt_bd:
            st.session_state.fc_mensajes.append({"role": "user", "content": prompt_bd})
            with st.chat_message("user"):
                st.markdown(prompt_bd)

            if not api_key:
                respuesta_texto = "⚠️ Ingresa tu Gemini API Key en el menú lateral para consultar la base."
                st.session_state.fc_mensajes.append({
                    "role": "assistant",
                    "content": respuesta_texto,
                    "llamadas": [],
                })
                with st.chat_message("assistant"):
                    st.warning(respuesta_texto)
            else:
                llamadas_actuales = []
                herramientas = crear_herramientas_bd(
                    st.session_state.fc_conn, llamadas_actuales
                )
                try:
                    with st.chat_message("assistant"):
                        with st.spinner("Gemini está consultando la base…"):
                            chat_bd = client.chats.create(
                                model=model_choice,
                                history=st.session_state.fc_history,
                                config=types.GenerateContentConfig(
                                    system_instruction=INSTRUCCION_FUNCTION_CALLING,
                                    tools=herramientas,
                                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                        maximum_remote_calls=8
                                    ),
                                    temperature=temperature,
                                ),
                            )
                            respuesta = chat_bd.send_message(prompt_bd)
                            respuesta_texto = respuesta.text or "Gemini no devolvió una respuesta textual."
                            st.session_state.fc_history = chat_bd.get_history()

                        st.markdown(respuesta_texto)
                        if llamadas_actuales:
                            with st.expander(
                                f"🔧 {len(llamadas_actuales)} llamada(s) a herramientas",
                                expanded=True,
                            ):
                                for numero, llamada in enumerate(llamadas_actuales, start=1):
                                    st.markdown(f"**{numero}. `{llamada['función']}`**")
                                    if llamada["argumentos"]:
                                        st.json(llamada["argumentos"])
                                    st.caption("Resultado devuelto a Gemini")
                                    st.json(llamada["resultado"])

                    st.session_state.fc_mensajes.append({
                        "role": "assistant",
                        "content": respuesta_texto,
                        "llamadas": llamadas_actuales,
                    })
                except Exception as e:
                    mensaje_error = f"No se pudo completar la consulta: {e}"
                    st.session_state.fc_mensajes.append({
                        "role": "assistant",
                        "content": mensaje_error,
                        "llamadas": llamadas_actuales,
                    })
                    with st.chat_message("assistant"):
                        st.error(mensaje_error)
