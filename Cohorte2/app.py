"""Visor didáctico de libretas — Cohorte 2 del AI Club Condesa.

Muestra cada libreta en tres vistas:
  - Crudo: tal cual llegó de Kaggle (nunca se modifica).
  - Rayos X: solo las celdas que hubo que corregir, con el error real, el diff y el porqué.
  - Corregido: la versión que sí corre en local.

El "qué cambió" se calcula aquí comparando crudo vs corregido; las fichas en criticas/
solo aportan el porqué. La app no llama a ningún modelo.
"""

import base64
import difflib
import json
import re
from pathlib import Path

import nbformat
import streamlit as st

RAIZ = Path(__file__).parent
CRUDO = RAIZ / "crudo"
CORREGIDO = RAIZ / "corregido"
CRITICAS = RAIZ / "criticas"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

CATEGORIAS = {
    "entorno-kaggle": "🏠 Solo existe en Kaggle",
    "dependencia": "📦 Falta una librería",
    "secreto": "🔑 Manejo de la llave",
    "proceso-bloqueante": "⏳ Bloquea la libreta",
    "modelo": "🤖 Modelo no disponible",
    "lógica": "🧠 Error de lógica",
}


@st.cache_data
def cargar_libreta(ruta: str, _mtime: float):
    return nbformat.read(ruta, as_version=4)


def leer(ruta: Path):
    if not ruta.exists():
        return None
    return cargar_libreta(str(ruta), ruta.stat().st_mtime)


def leer_criticas(nombre: str) -> dict:
    ruta = CRITICAS / (Path(nombre).stem + ".json")
    if not ruta.exists():
        return {"resumen": "", "criticas": []}
    return json.loads(ruta.read_text(encoding="utf-8"))


def fuente(celda) -> str:
    return celda.source if isinstance(celda.source, str) else "".join(celda.source)


def texto(valor) -> str:
    return ANSI.sub("", valor if isinstance(valor, str) else "".join(valor))


def celdas_cambiadas(crudo, corregido) -> list[int]:
    return [
        i
        for i, (a, b) in enumerate(zip(crudo.cells, corregido.cells))
        if fuente(a) != fuente(b)
    ]


# ---------- Render de celdas ----------


def mostrar_salidas(celda):
    for salida in celda.get("outputs", []):
        tipo = salida.get("output_type")
        if tipo == "stream":
            contenido = texto(salida.get("text", "")).strip("\n")
            if not contenido:
                continue
            if salida.get("name") == "stderr":
                st.caption("Salida de advertencias (stderr)")
            st.code(contenido, language=None, wrap_lines=True)
        elif tipo == "error":
            st.error(f"{salida.get('ename')}: {texto(salida.get('evalue', ''))}")
            with st.expander("Ver traceback completo"):
                st.code(texto("\n".join(salida.get("traceback", []))), language=None)
        elif tipo in ("display_data", "execute_result"):
            datos = salida.get("data", {})
            if "image/png" in datos:
                st.image(base64.b64decode(datos["image/png"]))
            elif "text/html" in datos:
                st.html(texto(datos["text/html"]))
            elif "text/markdown" in datos:
                st.markdown(texto(datos["text/markdown"]))
            elif "text/plain" in datos:
                st.code(texto(datos["text/plain"]), language=None, wrap_lines=True)


def mostrar_celda(i: int, celda, marcada: bool = False):
    if celda.cell_type == "markdown":
        st.markdown(fuente(celda))
        return
    if celda.cell_type != "code":
        return
    contador = celda.get("execution_count")
    etiqueta = f"Celda {i}" + (f" · ejecución [{contador}]" if contador else "")
    with st.container(border=True):
        if marcada:
            st.markdown(f"**🔧 {etiqueta} — corregida** (ver Rayos X)")
        else:
            st.caption(etiqueta)
        st.code(fuente(celda), language="python", line_numbers=True)
        if celda.get("outputs"):
            with st.expander("Salida guardada", expanded=True):
                mostrar_salidas(celda)


def mostrar_libreta(nb, marcadas: set[int] = frozenset()):
    for i, celda in enumerate(nb.cells):
        mostrar_celda(i, celda, i in marcadas)


# ---------- Rayos X ----------


def diff_unificado(antes: str, despues: str) -> str:
    lineas = difflib.unified_diff(
        antes.splitlines(), despues.splitlines(), "crudo", "corregido", lineterm="", n=2
    )
    return "\n".join(lineas)


def mostrar_rayos_x(crudo, corregido, fichas: dict):
    cambiadas = celdas_cambiadas(crudo, corregido)
    criticas = {c["celda"]: c for c in fichas.get("criticas", [])}

    col1, col2, col3 = st.columns(3)
    col1.metric("Celdas en la libreta", len(crudo.cells))
    col2.metric("Celdas corregidas", len(cambiadas))
    col3.metric("Verificadas corriendo", sum(1 for c in criticas.values() if c.get("verificado")))
    if fichas.get("resumen"):
        st.info(fichas["resumen"])

    sin_ficha = [i for i in cambiadas if i not in criticas]
    if sin_ficha:
        st.warning(f"Celdas cambiadas sin explicación: {sin_ficha}")

    vista = st.radio(
        "Vista del cambio",
        ["Unificada", "Lado a lado"],
        horizontal=True,
        help="Unificada se lee mejor en celular.",
    )

    for i in cambiadas:
        ficha = criticas.get(i, {})
        antes, despues = fuente(crudo.cells[i]), fuente(corregido.cells[i])
        with st.container(border=True):
            st.subheader(f"Celda {i} · {ficha.get('titulo', 'Sin ficha')}")
            categoria = ficha.get("categoria", "")
            estado = "✅ verificada corriendo" if ficha.get("verificado") else "⚠️ sin verificar"
            st.caption(f"{CATEGORIAS.get(categoria, categoria)} · {estado}")

            if ficha.get("error_real"):
                st.markdown("**Qué pasó al correr el crudo en local**")
                st.code(ficha["error_real"], language=None, wrap_lines=True)

            st.markdown("**Qué cambió**")
            if vista == "Unificada":
                st.code(diff_unificado(antes, despues), language="diff", wrap_lines=True)
            else:
                izq, der = st.columns(2)
                izq.caption("Crudo")
                izq.code(antes, language="python")
                der.caption("Corregido")
                der.code(despues, language="python")

            if ficha.get("por_que"):
                st.markdown("**Por qué truena**")
                st.markdown(ficha["por_que"])
            if ficha.get("como_se_arreglo"):
                st.markdown("**Cómo se arregló**")
                st.markdown(ficha["como_se_arreglo"])

    notas = fichas.get("notas", [])
    if notas:
        st.divider()
        st.subheader("Notas que no requirieron cambio")
        for nota in notas:
            with st.expander(nota["titulo"]):
                st.markdown(nota["texto"])


# ---------- Página ----------


def main():
    st.set_page_config(page_title="Visor de libretas · Cohorte 2", page_icon="🔬", layout="wide")

    libretas = sorted(p.name for p in CRUDO.glob("*.ipynb"))
    if not libretas:
        st.error("No hay libretas en la carpeta crudo/.")
        return

    with st.sidebar:
        st.title("🔬 Visor de libretas")
        st.caption("AI Club Condesa · Cohorte 2")
        nombre = st.selectbox("Libreta", libretas)
        st.markdown(
            "**Crudo**: tal cual llegó de Kaggle, nunca se toca.\n\n"
            "**Rayos X**: solo lo que hubo que arreglar para que corra en local, y por qué.\n\n"
            "**Corregido**: la versión que sí corre."
        )

    crudo = leer(CRUDO / nombre)
    corregido = leer(CORREGIDO / nombre)
    fichas = leer_criticas(nombre)

    st.title(nombre.removesuffix(".ipynb"))
    tab_crudo, tab_rayos, tab_corr = st.tabs(["📄 Crudo", "🩻 Rayos X", "✅ Corregido"])

    with tab_crudo:
        mostrar_libreta(crudo)

    if corregido is None:
        for tab in (tab_rayos, tab_corr):
            with tab:
                st.info("Esta libreta todavía no tiene versión corregida.")
        return

    if len(corregido.cells) != len(crudo.cells):
        st.error("El corregido no tiene las mismas celdas que el crudo; corre herramientas/verificar.py.")
        return

    with tab_rayos:
        mostrar_rayos_x(crudo, corregido, fichas)
    with tab_corr:
        mostrar_libreta(corregido, set(celdas_cambiadas(crudo, corregido)))


main()
