import os
import enum
import functools
import tempfile
import urllib.request
import streamlit as st
from google import genai
from google.genai import types
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Club Condesa | Gemini 2.5 & RAG Studio",
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
    
    model_choice = st.selectbox(
        "Modelo de Gemini (Texto/RAG)",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="gemini-2.5-flash es rápido y altamente eficiente para evaluación y RAG."
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
st.markdown('<div class="sub-header">Laboratorio interactivo de Evaluación con LLM-as-a-Judge y RAG con Gemini 2.5 y ChromaDB</div>', unsafe_allow_html=True)

tab_eval, tab_rag = st.tabs(["📊 Día 1: Evaluación & Salidas Estructuradas", "🔍 Día 2: Document Q&A con RAG (ChromaDB)"])

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
            
            gen_btn = st.button("🚀 Generar y Evaluar", use_container_width=True)
            
        with col2:
            if gen_btn:
                if not api_key:
                    st.error("⚠️ Por favor ingresa tu Gemini API Key en el menú lateral izquierdo.")
                else:
                    try:
                        with st.spinner("1. Generando respuesta con Gemini 2.5 Flash..."):
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
            
        if st.button("⚖️ Comparar Respuestas", use_container_width=True):
            if not api_key:
                st.error("⚠️ Ingresa tu Gemini API Key en el menú lateral izquierdo.")
            else:
                try:
                    with st.spinner("Juzgando respuestas con Gemini 2.5..."):
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
    y genera respuestas aumentadas con **`gemini-2.5-flash`**.
    """)

    # Persistent in-memory ChromaDB Client
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.Client()
    
    chroma_client = st.session_state.chroma_client

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
            if st.button("📥 Indexar Documentos de Googlecar en ChromaDB", use_container_width=True):
                if not api_key:
                    st.error("⚠️ Por favor ingresa tu Gemini API Key en el menú lateral izquierdo antes de indexar.")
                else:
                    try:
                        with st.spinner(f"Generando embeddings con {embed_model_choice} e indexando..."):
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
            if uploaded_file and st.button("📥 Indexar Documento Subido", use_container_width=True):
                if not api_key:
                    st.error("⚠️ Por favor ingresa tu Gemini API Key en el menú lateral izquierdo antes de indexar.")
                else:
                    try:
                        with st.spinner("Extrayendo texto y generando embeddings..."):
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
        
        if st.button("🔍 Buscar y Responder con RAG", use_container_width=True):
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
