"""Del enlace de Kaggle a los errores reales en los dos entornos, sin pasos a mano.

  python herramientas/preparar.py bajar  <enlace-o-slug>   baja el crudo y registra su huella
  python herramientas/preparar.py local  <libreta>         corre el crudo y el corregido aqui
  python herramientas/preparar.py kaggle <libreta>         corre el crudo y el corregido en Kaggle
  python herramientas/preparar.py todo   <enlace-o-slug>   todo, con los dos entornos a la vez
  python herramientas/preparar.py fichas <libreta>         re-emite los borradores sin volver a correr

<libreta> es el nombre del archivo sin .ipynb (por ejemplo day-2a-agent-tools).

No hace falta tocar Kaggle a mano: si la libreta no existe en la cuenta, se crea al subirla.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import nbformat
from nbclient import NotebookClient

if hasattr(sys.stdout, "reconfigure"):          # la consola de Windows es cp1252 y los
    sys.stdout.reconfigure(encoding="utf-8")    # titulos del curso vienen con emojis

RAIZ = Path(__file__).resolve().parent.parent
USUARIO_KAGGLE = "ricardosauza"
DUENIO_CURSO = "kaggle5daysofai"
# Kaggle devuelve 404 a la cadena genérica "Mozilla/5.0": la trata como robot. Hay que
# identificarse como un navegador completo para que la página del curso se deje leer.
AGENTE = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
ESTADOS_FINALES = ("COMPLETE", "ERROR", "CANCEL")
TOPE_SESION = 1800          # segundos: si la libreta se cuelga, que muera del lado de Kaggle
GRACIA_TRAS_SUBIR = 25      # Kaggle sigue reportando la corrida ANTERIOR unos segundos
BORRADORES = RAIZ / ".borradores"
# Frases con que las libretas del curso avisan un fallo que atraparon sin tronar.
AVISOS_DE_FALLO = ("❌", "Failed to resolve remote A2A agent", "Server may not be ready")

# Una corrida lanzada por API no alcanza el almacen de secretos de Kaggle: aunque la casilla
# de GOOGLE_API_KEY este marcada, la libreta recibe "Connection error trying to communicate
# with service". Por eso a la COPIA QUE SE SUBE (nunca al archivo de corregido/) se le pone un
# respaldo que lee la llave de un dataset privado. Lo que se publica sigue siendo fiel al curso.
DATASET_LLAVE = f"{USUARIO_KAGGLE}/claves-privadas-cohorte-2"
ARCHIVO_LLAVE = "google_api_key.txt"
CARPETA_LLAVE = DATASET_LLAVE.split("/")[-1]

RESPALDO_LLAVE = f'''
# --- Solo para la verificacion automatica; no forma parte de la libreta del curso. ---
import os as _os, pathlib as _pl
if not _os.environ.get("GOOGLE_API_KEY"):
    # Kaggle no siempre monta el dataset en /kaggle/input/<slug>: a veces lo anida.
    # Buscarlo en vez de suponer la ruta.
    _raiz = _pl.Path("/kaggle/input")
    _hallados = sorted(_raiz.rglob("{ARCHIVO_LLAVE}")) if _raiz.exists() else []
    if _hallados:
        _os.environ["GOOGLE_API_KEY"] = _hallados[0].read_text().strip()
        print(f"\\u2705 Llave tomada de {{_hallados[0]}} (verificacion automatica).")
    else:
        # Un fallo mudo aqui reaparece 8 celdas despues como un error del modelo.
        _hay = sorted(str(p.relative_to(_raiz)) for p in _raiz.rglob("*") if p.is_file())[:20] \\
            if _raiz.exists() else []
        print(f"\\u274c RESPALDO SIN LLAVE: no hay ningun {ARCHIVO_LLAVE} bajo /kaggle/input. Hay: {{_hay}}")
'''

_candado_aviso = threading.Lock()


def aviso(texto=""):
    with _candado_aviso:          # los dos entornos escriben a la vez
        print(texto, flush=True)


def fuente(celda) -> str:
    return celda.source if isinstance(celda.source, str) else "".join(celda.source)


def celdas_ejecutables(nb):
    """Celdas que el contador In[N] cuenta: las de codigo con contenido. Las vacias se saltan."""
    return [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and fuente(c).strip()]


def kaggle(*args, timeout=None):
    """Llama al cliente de Kaggle del propio entorno, siempre en UTF-8."""
    entorno = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=entorno, cwd=RAIZ, timeout=timeout)
    salida = (r.stdout or "") + (r.stderr or "")
    if "Authentication required" in salida:
        raise SystemExit(
            "\nFalta la credencial de Kaggle, o la de siempre dejo de servir.\n"
            "  Genera un token en https://www.kaggle.com/settings/api (Generate New Token)\n"
            f"  y guardalo tal cual en {Path.home() / '.kaggle' / 'access_token'}\n")
    return salida


# --------------------------------------------------------------------------
# Arreglos conocidos. Se buscan POR CONTENIDO, nunca por numero de celda, y cada
# uno trae la plantilla de su ficha: el texto es siempre el mismo, lo que cambia
# es la celda y el error medido.
# --------------------------------------------------------------------------

LLAVE_LOCAL = '''import os
from dotenv import find_dotenv, load_dotenv

# En local la llave vive en un archivo .env (copia .env.example y pon tu llave).
load_dotenv(find_dotenv(usecwd=True))
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    print("✅ Setup and authentication complete.")
else:
    print("\U0001f511 Authentication Error: crea un archivo .env con GOOGLE_API_KEY=tu-llave (ver .env.example).")'''

COMENTARIO_REINTENTOS = ("exp_base=2,  # Delay multiplier "
                         "(con 7 las esperas eran 1, 7, 49 y 60 s (tope): ~2 min colgada)")

ARREGLOS = [
    {
        "id": "llave-local",
        "entornos": {"local"},
        "detecta": lambda s: "kaggle_secrets" in s,
        "aplica": lambda s: LLAVE_LOCAL,
        "ficha": {
            "titulo": "La llave se leía de un servicio que solo existe en Kaggle",
            "categoria": "entorno-kaggle",
            "por_que": "`kaggle_secrets` es la caja fuerte de llaves de Kaggle. Fuera de Kaggle esa "
                       "librería no existe, así que la celda truena en el `import`, antes siquiera de "
                       "llegar al `try` que iba a atrapar el error. La llave nunca se carga y cada "
                       "celda que llama al modelo falla después.",
            "como_se_arreglo": "La llave se guarda en un archivo `.env` en tu computadora (nunca se "
                               "sube a GitHub) y se carga con `python-dotenv`. Si no la encuentra, "
                               "avisa cómo crearla en lugar de tronar.",
        },
    },
    {
        "id": "reintentos",
        "entornos": {"local", "kaggle"},
        "detecta": lambda s: re.search(r"exp_base\s*=\s*7", s) is not None,
        "aplica": lambda s: re.sub(r"exp_base\s*=\s*7,(?:[ \t]*#[^\n]*)?", COMENTARIO_REINTENTOS, s),
        "ficha": {
            "titulo": "Los reintentos cuelgan la celda casi 2 minutos",
            "categoria": "proceso-bloqueante",
            "por_que": "Con `exp_base=7` cada reintento espera 7 veces más que el anterior: 1, 7, 49 "
                       "y luego 60 segundos (la librería pone un tope de 60 por espera). Cuando se "
                       "acaba la cuota, la celda se queda «corriendo» unos 2 minutos sin decir nada "
                       "y parece que todo se trabó.",
            "como_se_arreglo": "`exp_base=2`, con esperas de 1, 2, 4 y 8 segundos. Sigue reintentando "
                               "los errores pasajeros, pero si la cuota está agotada lo dice en unos "
                               "17 segundos.",
            "error_si_no_aparecio": "En esta corrida no se agotó la cuota, así que el problema no "
                                    "apareció solo. Medido aparte contra un servidor falso que siempre "
                                    "contesta 429 RESOURCE_EXHAUSTED (no gasta cuota): con `exp_base=7` "
                                    "la llamada se quedó colgada 118.8 segundos; con `exp_base=2`, 16.8.",
        },
    },
    {
        "id": "servidor-local",
        "entornos": {"local"},
        "detecta": lambda s: "subprocess.Popen(" in s and re.search(r'^\s*"uvicorn",', s, re.M) is not None,
        "aplica": lambda s: arregla_servidor_local(s),
        "ficha": {
            "titulo": "El servidor del segundo agente arrancaba con otro Python y moría en silencio",
            "categoria": "dependencia",
            "por_que": "La celda lanza el servidor escribiendo solo `uvicorn`, y Windows usa el primer "
                       "`uvicorn` que encuentra en el PATH, que puede ser el de otro Python de tu "
                       "computadora, sin las librerías del curso. Ese servidor muere al arrancar, pero "
                       "su error va a un tubo que nadie lee: la celda espera 150 segundos, dice "
                       "«Server may not be ready» y sigue. Después el agente de soporte contesta "
                       "«no pude conectar con el catálogo» y ninguna celda marca error. Además usaba "
                       "`/tmp`, que en Windows solo existe si alguien lo creó, y nunca apagaba el "
                       "servidor, que se quedaba ocupando el puerto 8001. Y aun con el Python "
                       "correcto, la bitácora del servidor iba a un tubo (`PIPE`) que nadie leía: en "
                       "Windows ese tubo es chico, se llena a la primera pregunta y el servidor se "
                       "congela a media respuesta. Cada pregunta quedaba colgada hasta el tope de "
                       "10 minutos. En Kaggle (Linux) el tubo es más grande y no se nota.",
            "como_se_arreglo": "Se lanza con `sys.executable -m uvicorn`, el mismo Python que corre la "
                               "libreta. El archivo va a la carpeta temporal del sistema "
                               "(`tempfile.gettempdir()`) y la bitácora del servidor a un archivo "
                               "`product_catalog_server.log` en vez de al tubo. Si el servidor muere, "
                               "la espera se corta y se imprime su error real. Y el servidor se apaga "
                               "solo al cerrar la libreta.",
            "error_si_no_aparecio": "La celda no marca error: imprime «⚠️ Server may not be ready yet» "
                                    "tras 150 s. Arrancado a mano, el servidor muere con "
                                    "«ModuleNotFoundError: No module named 'google.adk'» porque "
                                    "`uvicorn` resolvía al Python global. Luego las celdas de prueba "
                                    "responden «Failed to resolve remote A2A agent product_catalog_agent».",
        },
    },
]


def arregla_servidor_local(s: str) -> str:
    """Hace que el servidor auxiliar corra con el Python de la libreta y no muera en silencio."""
    s = "import sys, tempfile, atexit\nTMP = tempfile.gettempdir()  # /tmp no existe en Windows\n\n" + s
    s = re.sub(r'open\("/tmp/([^"]+)"', r'open(os.path.join(TMP, "\1")', s)
    s = re.sub(r'print\("([^"\n]*)/tmp/([^"\n\s]+)"\)', r"""print(f"\1{os.path.join(TMP, '\2')}")""", s)
    s = re.sub(r'cwd="/tmp",[^\n]*', 'cwd=TMP,  # la carpeta temporal del sistema', s)
    s = re.sub(r'^(\s*)"uvicorn",', r'\1sys.executable, "-m", "uvicorn",  # el MISMO Python de la libreta',
               s, count=1, flags=re.M)
    # stdout/stderr=PIPE sin que nadie lea: en Windows el tubo es chico, se llena con la
    # bitacora del servidor y este se congela a media respuesta (cada pregunta colgada 10 min).
    s = re.sub(r'(server_process = subprocess\.Popen\()',
               r'registro_servidor = open(os.path.join(TMP, "product_catalog_server.log"), "w")\n\1', s, count=1)
    s = re.sub(r'stdout=subprocess\.PIPE,[^\n]*', 'stdout=registro_servidor,  # a un archivo: un tubo sin leer se llena', s)
    s = re.sub(r'stderr=subprocess\.PIPE,[^\n]*', 'stderr=subprocess.STDOUT,', s)
    s = re.sub(r'^(\s*)time\.sleep\(5\)',
               r'\1if server_process.poll() is not None:  # el servidor ya murio: no esperar mas\n'
               r'\1    break\n\1time.sleep(5)', s, count=1, flags=re.M)
    s += ('\n\nif server_process.poll() is not None:\n'
          '    print("\\n\u274c El servidor se cerró al arrancar. Su error:")\n'
          '    registro_servidor.flush()\n'
          '    with open(registro_servidor.name, encoding="utf-8", errors="replace") as r:\n'
          '        print(r.read()[-1500:])\n\n'
          '# Apagar el servidor al cerrar la libreta, para no dejar ocupado el puerto 8001.\n'
          'atexit.register(server_process.terminate)\n')
    return s


def arreglo_por_id(cid):
    return next(a for a in ARREGLOS if a["id"] == cid)


def construir_corregido(libreta: str, entorno: str):
    """Copia el crudo a corregido/<entorno>/ con los arreglos que apliquen.

    Devuelve (celdas_tocadas, se_escribio). No depende de haber corrido nada.
    """
    nb = nbformat.read(RAIZ / "crudo" / f"{libreta}.ipynb", as_version=4)
    tocadas = []
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        for arreglo in ARREGLOS:
            if entorno in arreglo["entornos"] and arreglo["detecta"](fuente(celda)):
                celda.source = arreglo["aplica"](fuente(celda))
                tocadas.append((i, arreglo["id"]))
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None

    destino = RAIZ / "corregido" / entorno / f"{libreta}.ipynb"
    destino.parent.mkdir(parents=True, exist_ok=True)

    # Nunca pisar correcciones hechas a mano. El archivo de comparacion va a .borradores/,
    # fuera de corregido/, porque verificar.py exige un crudo por cada .ipynb que vive ahi.
    if destino.exists():
        previo = nbformat.read(destino, as_version=4)
        if [fuente(c) for c in previo.cells] != [fuente(c) for c in nb.cells]:
            BORRADORES.mkdir(exist_ok=True)
            al_lado = BORRADORES / f"{libreta}.{entorno}.generado.ipynb"
            nbformat.write(nb, al_lado)
            aviso(f"    OJO: corregido/{entorno}/{libreta}.ipynb ya trae arreglos que yo no sé hacer.\n"
                  f"         No lo toqué. Dejé mi versión en {al_lado.relative_to(RAIZ)} para comparar.")
            return tocadas, False

    nbformat.write(nb, destino)
    return tocadas, True


# --------------------------------------------------------------------- bajar

def bajar(referencia: str) -> str:
    """Trae el .ipynb publico tal cual y registra su huella. Sin sesion ni navegador."""
    partes = [p for p in referencia.rstrip("/").split("/") if p]
    duenio, slug = DUENIO_CURSO, partes[-1]
    if "code" in partes:
        i = partes.index("code")
        duenio, slug = partes[i + 1], partes[i + 2]

    def pide(url):
        req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    pagina = f"https://www.kaggle.com/code/{duenio}/{slug}"
    html = pide(pagina).decode("utf-8", "replace")
    m = re.search(r"scriptVersionId%3D(\d+)", html) or re.search(r'"scriptVersionId":\s*(\d+)', html)
    if not m:
        raise SystemExit(f"No encontré la versión del guion en {pagina}. "
                         "Revisa que la libreta sea pública y que el enlace esté bien.")
    datos = pide(f"https://www.kaggle.com/kernels/scriptcontent/{m.group(1)}/download")

    destino = RAIZ / "crudo" / f"{slug}.ipynb"
    huella = hashlib.sha256(datos).hexdigest()
    if destino.exists():
        if hashlib.sha256(destino.read_bytes()).hexdigest() == huella:
            aviso(f"  El crudo de {slug} ya estaba y es idéntico. No lo toco.")
            return slug
        raise SystemExit(f"Ya existe {destino} con otro contenido. Si de verdad quieres "
                         "reemplazar el crudo, bórralo a mano primero.")

    nb = nbformat.reads(datos.decode("utf-8"), as_version=4)
    nbformat.validate(nb)
    destino.write_bytes(datos)

    lock_path = RAIZ / "crudo.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[destino.name] = huella
    lock_path.write_text(json.dumps(dict(sorted(lock.items())), indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    codigo = sum(1 for c in nb.cells if c.cell_type == "code")
    aviso(f"  Crudo guardado: {destino.name} ({len(nb.cells)} celdas, {codigo} de código). Huella registrada.")
    return slug


# --------------------------------------------------------------------- local

def corre_local(ruta: Path, timeout: int = 600):
    """Ejecuta sin detenerse en los errores. Devuelve [(celda, nombre, mensaje)].

    Una corrida que se muere NO puede salir verde: si el nucleo se cae, el cuaderno queda
    sin salidas y la busqueda de errores no encuentra ninguno. Eso se veia igual que
    "corrio limpio" y daba por buena una libreta que nunca se ejecuto. Por eso al final se
    comprueba que cada celda de codigo traiga su contador In[N]; la que no corrio, se reporta.
    """
    nb = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                             interrupt_on_timeout=True,   # una celda colgada no mata la corrida
                             resources={"metadata": {"path": str(ruta.parent)}})

    # Cada ejecucion va en su propio hilo, con su propio bucle de eventos. En el hilo principal
    # la primera corrida deja el bucle cerrado y la segunda muere con RuntimeError antes de
    # ejecutar una sola celda: asi se perdia la corrida del corregido.
    corte = {}

    def ejecuta():
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            cliente.execute()
        except Exception as e:                            # p. ej. una celda que nunca termina
            corte["error"] = f"{type(e).__name__}: {e}".strip()

    hilo = threading.Thread(target=ejecuta)
    hilo.start()
    hilo.join()

    fallos = []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                mensaje = (o.get("evalue") or "").strip().splitlines()
                fallos.append((i, o.get("ename"), mensaje[0] if mensaje else ""))
                break
        else:
            # Muchas libretas atrapan el error y solo lo IMPRIMEN: la celda no marca error y la
            # corrida saldria "sin errores" aunque nada funcionara (asi paso en day-5a).
            texto = "".join(o.get("text", "") for o in c.get("outputs", []) if o.get("output_type") == "stream")
            for linea in texto.splitlines():
                if any(m in linea for m in AVISOS_DE_FALLO):
                    fallos.append((i, "AvisoImpreso", linea.strip()[:200]))
                    break

    sin_correr = [i for i in celdas_ejecutables(nb) if nb.cells[i].get("execution_count") is None]
    if corte or sin_correr:
        if corte:
            aviso(f"    (la corrida se cortó: {corte['error'][:120]})")
        if sin_correr:
            aviso(f"    (ATENCIÓN: {len(sin_correr)} celda(s) nunca se ejecutaron: "
                  f"{', '.join(str(i) for i in sin_correr[:12])}"
                  f"{' ...' if len(sin_correr) > 12 else ''})")
        detalle = corte.get("error", "")
        fallos.append((-1, "CorridaIncompleta",
                       f"{len(sin_correr)} celda(s) nunca se ejecutaron"
                       + (f"; la corrida se cortó con {detalle}" if detalle else "")
                       + ". Este resultado NO sirve como prueba de que la libreta corre."))
    return fallos


def informe(titulo: str, fallos):
    aviso(f"\n  {titulo}")
    if not fallos:
        aviso("    sin errores")
        return
    for celda, nombre, mensaje in fallos:
        donde = f"celda {celda}" if celda >= 0 else "sin celda"
        aviso(f"    {donde}: {nombre}: {mensaje[:150]}")


# -------------------------------------------------------------------- kaggle

_imagenes = {}
_candado_imagen = threading.Lock()


def imagen_del_curso(libreta: str):
    """La imagen exacta que usa la libreta del curso.

    Sin esto, una libreta nueva arranca con la imagen MAS RECIENTE de Kaggle (hoy Python 3.13)
    en vez de la del curso (3.11), y entonces el corregido y el crudo corren en entornos
    distintos: la comparacion deja de valer y aparecen fallos que el alumno nunca vera.
    """
    with _candado_imagen:
        if libreta in _imagenes:
            return _imagenes[libreta]
        destino = BORRADORES / "curso" / libreta
        destino.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "pull", f"{DUENIO_CURSO}/{libreta}", "-p", str(destino), "-m")
        meta = destino / "kernel-metadata.json"
        imagen = None
        if meta.exists():
            imagen = json.loads(meta.read_text(encoding="utf-8")).get("docker_image") or None
        _imagenes[libreta] = imagen
        return imagen


def corre_kaggle(libreta: str, ipynb: Path, slug_destino: str, etiqueta: str):
    """Sube la libreta a la cuenta propia (creandola si no existe), la corre y lee sus errores."""
    slug = f"{USUARIO_KAGGLE}/{slug_destino}"
    trabajo = BORRADORES / "kaggle" / slug_destino
    trabajo.mkdir(parents=True, exist_ok=True)

    # Se escribe desde cero. Nunca se reusa un kernel-metadata.json bajado de Kaggle: trae un
    # campo id_no que el cliente prefiere sobre el slug, y la subida se iria a la libreta vieja.
    metadatos = {
        "id": slug,
        "title": slug_destino.replace("-", " ").title()[:50],
        "code_file": ipynb.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET_LLAVE],
        "competition_sources": [],
        "kernel_sources": [],
    }
    imagen = imagen_del_curso(libreta)
    if imagen:
        metadatos["docker_image"] = imagen
    else:
        aviso(f"    OJO: no pude leer la imagen del curso; {etiqueta} correrá con la más "
              "reciente de Kaggle, que puede no ser la que usa la clase.")
    (trabajo / "kernel-metadata.json").write_text(json.dumps(metadatos, indent=2) + "\n",
                                                 encoding="utf-8")

    # La copia que sube lleva el respaldo de la llave; el archivo de corregido/ no se toca.
    nb = nbformat.read(ipynb, as_version=4)
    puesto = False
    for celda in nb.cells:
        if celda.cell_type == "code" and "kaggle_secrets" in fuente(celda):
            celda.source = fuente(celda) + RESPALDO_LLAVE
            puesto = True
            break
    if not puesto:                      # ninguna celda pide la llave a Kaggle: va al principio
        for celda in nb.cells:
            if celda.cell_type == "code":
                celda.source = RESPALDO_LLAVE.strip() + "\n\n" + fuente(celda)
                break
    nbformat.write(nb, trabajo / ipynb.name)

    aviso(f"    subiendo {etiqueta} a {slug} ...")
    salida = kaggle("kernels", "push", "-p", str(trabajo), "-t", str(TOPE_SESION))
    if "successfully pushed" not in salida:
        raise SystemExit(f"No pude subir {etiqueta}: {salida.strip()[:400]}")

    # Kaggle sigue reportando la corrida ANTERIOR unos segundos: sin esta gracia, el estado
    # que se lee es el COMPLETE de la vez pasada y el informe sale verde en falso.
    time.sleep(GRACIA_TRAS_SUBIR)

    inicio = time.monotonic()
    # "logs --follow" se queda escuchando y vuelve cuando la corrida termina: una sola llamada
    # en vez de sondear el estado cada 25 segundos (el cliente no tiene defensa contra el 429).
    kaggle("kernels", "logs", slug, "-f", timeout=TOPE_SESION + 300)   # bloquea hasta que termina

    # El estado tarda unos segundos en cuajar despues de que el registro cierra.
    estado, veredicto = "", "?"
    for _ in range(6):
        estado = kaggle("kernels", "status", slug)
        veredicto = next((e for e in ESTADOS_FINALES if e in estado), "?")
        if veredicto != "?":
            break
        time.sleep(10)
    aviso(f"    {etiqueta}: terminó en {int(time.monotonic() - inicio) + GRACIA_TRAS_SUBIR} s ({veredicto})")

    # El registro completo se pide aparte: lo que llega por el flujo en vivo puede venir cortado.
    fallos = errores_del_registro(ipynb, kaggle("kernels", "logs", slug))
    # Si Kaggle dice ERROR, nunca reportar "sin errores": eso fue justo lo que paso en day-5a.
    if veredicto != "COMPLETE" and not fallos:
        fallos.append((-1, "CorridaFallida", f"Kaggle reporta {veredicto} pero el registro no dijo en que celda"))
    return fallos


def errores_del_registro(ipynb: Path, registro: str):
    """Traduce el registro de Kaggle a [(celda, nombre, mensaje)] usando los In[N].

    El mapa de celdas se saca del archivo QUE CORRIO, no del crudo: un arreglo que agregue
    o quite celdas desplazaria todos los numeros.
    """
    # El cliente puede anteponer avisos (p. ej. "outdated kaggle version") al JSON: si se
    # intenta leer todo junto falla, el texto queda con las comillas escapadas, ninguna marca
    # "In [N]" coincide y el informe sale "sin errores" en falso. Leer desde el primer "[{".
    inicio_json = registro.find("[{")
    try:
        texto = "".join(e["data"] for e in json.loads(registro[inicio_json:]))
    except Exception:
        texto = registro.replace('\\"', '"').replace("\\n", "\n")
    if texto.strip() and "Exception encountered" not in texto and "Traceback" in texto:
        aviso("    ⚠️  el registro trae un Traceback pero ninguna marca In [N] legible: revisalo a mano")
    codigo = celdas_ejecutables(nbformat.read(ipynb, as_version=4))

    marcas = list(re.finditer(r'Exception encountered at "In \[(\d+)\]"', texto))
    fallos = []
    for k, m in enumerate(marcas):
        n = int(m.group(1))
        celda = codigo[n - 1] if 0 < n <= len(codigo) else -1
        fin = marcas[k + 1].start() if k + 1 < len(marcas) else len(texto)
        trozo = texto[m.end():fin]
        # El traceback empieza con una linea de guiones; lo de antes es salida de celdas que
        # corrieron bien y puede contener la palabra "Error". Solo miramos el traceback.
        corte = trozo.find("-" * 20)
        lineas = re.findall(r"^(\w*(?:Error|Exception)): (.*)$", trozo[corte:] if corte >= 0 else trozo, re.M)
        nombre, mensaje = lineas[-1] if lineas else ("?", "")
        fallos.append((celda, nombre, mensaje.strip()))

    # El aviso de los Secrets sale SIEMPRE en una corrida por API y no es un problema:
    # solo importa si el respaldo del dataset tampoco encontro la llave.
    if "RESPALDO SIN LLAVE" in texto:
        detalle = re.search(r"RESPALDO SIN LLAVE: (.*)", texto)
        fallos.append((-1, "SinLlave", detalle.group(1)[:200] if detalle else "el respaldo no halló la llave"))
    return fallos


# ------------------------------------------------------------------- fichas

def plantilla_critica(celda: int, cid: str, fallos, entorno: str):
    """Rellena lo que se sabe solo; deja marcado con <<...>> lo que hay que escribir."""
    a = arreglo_por_id(cid)["ficha"]
    propios = [f for f in fallos if f[0] == celda]
    if propios:
        error = "\n".join(f"{n}: {m}" for _, n, m in propios)
    else:
        error = a.get("error_si_no_aparecio", "<<qué error salió de verdad en esta corrida>>")
    return {
        "celda": celda,
        "titulo": a["titulo"],
        "categoria": a["categoria"],
        "error_real": error,
        "por_que": a["por_que"],
        "como_se_arreglo": a["como_se_arreglo"],
        "verificado": True,
    }


def guarda_evidencia(libreta: str, entorno: str, tocadas, fallos_crudo, fallos_corregido):
    """Deja en disco todo lo medido, para re-emitir las fichas sin volver a correr nada."""
    BORRADORES.mkdir(exist_ok=True)
    (BORRADORES / f"{libreta}.{entorno}.evidencia.json").write_text(json.dumps({
        "libreta": libreta, "entorno": entorno,
        "tocadas": [[i, cid] for i, cid in tocadas],
        "crudo": [[c, n, m] for c, n, m in fallos_crudo],
        "corregido": [[c, n, m] for c, n, m in fallos_corregido],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emite_borrador(libreta: str, entorno: str):
    """Escribe criticas/<entorno>/<libreta>.json si no existe, a partir de la evidencia."""
    ev_path = BORRADORES / f"{libreta}.{entorno}.evidencia.json"
    if not ev_path.exists():
        aviso(f"  No hay evidencia de {libreta} en {entorno}. Corre antes: preparar.py {entorno} {libreta}")
        return
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    destino = RAIZ / "criticas" / entorno / f"{libreta}.json"
    if destino.exists():
        aviso(f"  criticas/{entorno}/{libreta}.json ya existe. No lo toco.")
        return

    fallos_crudo = [tuple(x) for x in ev["crudo"]]
    criticas = [plantilla_critica(i, cid, fallos_crudo, entorno) for i, cid in ev["tocadas"]]
    limpio = not ev["corregido"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({
        "resumen": f"<<una frase sobre qué le pasó a esta libreta en {entorno}>>. "
                   f"Cambiaron {len(criticas)} celda(s). "
                   + ("Con esos cambios la libreta corre completa."
                      if limpio else "OJO: el corregido todavía falla."),
        "criticas": criticas,
        "notas": [{"titulo": "<<título de una nota>>", "texto": "<<algo que no requirió cambiar código>>"}],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pendientes = destino.read_text(encoding="utf-8").count("<<")
    aviso(f"  Borrador en criticas/{entorno}/{libreta}.json  ({pendientes} huecos por llenar)")


# --------------------------------------------------------------------- pasos

def paso_local(libreta: str, resultados: dict):
    aviso("\n[LOCAL] corriendo el crudo y preparando el corregido ...")
    tocadas, _ = construir_corregido(libreta, "local")
    fallos_crudo = corre_local(RAIZ / "crudo" / f"{libreta}.ipynb")
    fallos_corr = corre_local(RAIZ / "corregido" / "local" / f"{libreta}.ipynb")
    guarda_evidencia(libreta, "local", tocadas, fallos_crudo, fallos_corr)
    resultados["local"] = (tocadas, fallos_crudo, fallos_corr)


def paso_kaggle(libreta: str, resultados: dict):
    aviso("\n[KAGGLE] subiendo el crudo y el corregido a la vez ...")
    tocadas, _ = construir_corregido(libreta, "kaggle")
    salida = {}

    def corre(clave, ruta, slug, etiqueta):
        # Un hilo que revienta se llevaria su mitad del informe en silencio.
        try:
            salida[clave] = corre_kaggle(libreta, ruta, slug, etiqueta)
        except Exception as e:
            aviso(f"    {etiqueta} no llegó a buen puerto: {type(e).__name__}: {str(e)[:200]}")
            salida[clave] = [(-1, "NoCorrio", f"{type(e).__name__}: {str(e)[:120]}")]

    hilos = [
        threading.Thread(target=corre, args=("crudo", RAIZ / "crudo" / f"{libreta}.ipynb",
                                             libreta, "el crudo")),
        threading.Thread(target=corre, args=("corregido", RAIZ / "corregido" / "kaggle" / f"{libreta}.ipynb",
                                             f"{libreta}-corregido", "el corregido")),
    ]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()

    guarda_evidencia(libreta, "kaggle", tocadas, salida.get("crudo", []), salida.get("corregido", []))
    resultados["kaggle"] = (tocadas, salida.get("crudo", []), salida.get("corregido", []))


def resume(entorno: str, datos):
    tocadas, crudo, corregido = datos
    aviso(f"\n=== {entorno.upper()} ===")
    aviso("  arreglos aplicados: " + (", ".join(f"celda {i} ({cid})" for i, cid in tocadas) or "ninguno"))
    informe("el crudo:", crudo)
    informe("el corregido:", corregido)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("comando", choices=["bajar", "local", "kaggle", "todo", "fichas"])
    p.add_argument("referencia", help="enlace de Kaggle o nombre de la libreta")
    a = p.parse_args()
    arranque = time.monotonic()

    if a.comando == "bajar":
        bajar(a.referencia)
        return
    if a.comando == "fichas":
        libreta = a.referencia.removesuffix(".ipynb")
        for entorno in ("local", "kaggle"):
            emite_borrador(libreta, entorno)
        return

    resultados = {}
    if a.comando == "todo":
        aviso("\n[BAJAR]")
        libreta = bajar(a.referencia)
        entornos = ["local", "kaggle"]
    else:
        libreta = a.referencia.removesuffix(".ipynb")
        entornos = [a.comando]

    # Los dos entornos no dependen entre si: el de Kaggle es casi toda espera de red.
    pasos = {"local": paso_local, "kaggle": paso_kaggle}
    hilos = [threading.Thread(target=pasos[e], args=(libreta, resultados)) for e in entornos]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()

    for entorno in entornos:
        if entorno in resultados:
            resume(entorno, resultados[entorno])
            emite_borrador(libreta, entorno)

    aviso(f"\nListo en {int(time.monotonic() - arranque)} s. "
          "Llena los huecos <<...>> de las fichas y corre herramientas/verificar.py")


if __name__ == "__main__":
    main()
