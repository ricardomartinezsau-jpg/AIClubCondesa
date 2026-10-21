"""Del enlace de Kaggle a los errores reales en los dos entornos, sin pasos a mano.

  python herramientas/preparar.py bajar  <enlace-o-slug>   baja el crudo y registra su huella
  python herramientas/preparar.py local  <libreta>         corre el crudo y el corregido en esta computadora
  python herramientas/preparar.py kaggle <libreta>         corre el crudo y el corregido en Kaggle
  python herramientas/preparar.py todo   <enlace-o-slug>   los tres, en orden

<libreta> es el nombre del archivo sin .ipynb (por ejemplo day-2a-agent-tools).

Lo unico que no hace solo: marcar la casilla del secreto GOOGLE_API_KEY en una libreta
nueva de Kaggle (Add-ons > Secrets). Kaggle no lo expone por API. El comando avisa cuando
toca y el resto sigue solo.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent.parent
USUARIO_KAGGLE = "ricardosauza"
DUENIO_CURSO = "kaggle5daysofai"
AGENTE = "Mozilla/5.0"
ESTADOS_FINALES = ("COMPLETE", "ERROR", "CANCEL")

# Una corrida lanzada por API no alcanza el almacen de secretos de Kaggle: aunque la
# casilla de GOOGLE_API_KEY este marcada, la libreta recibe "Connection error trying to
# communicate with service". Para poder verificarla sin que nadie pinche nada, a la COPIA
# QUE SE SUBE (nunca al archivo que se guarda en corregido/) se le pone un respaldo que
# lee la llave de un dataset privado. Lo que se commitea sigue siendo fiel al curso.
DATASET_LLAVE = f"{USUARIO_KAGGLE}/claves-privadas-cohorte-2"
ARCHIVO_LLAVE = "google_api_key.txt"

RESPALDO_LLAVE = f'''
# --- Solo para la verificacion automatica; no forma parte de la libreta del curso. ---
# Las corridas lanzadas por API no ven los Kaggle Secrets, asi que si la llave no llego,
# se lee del dataset privado {DATASET_LLAVE}.
if not os.environ.get("GOOGLE_API_KEY"):
    import pathlib as _pl
    _llave = _pl.Path("/kaggle/input/{DATASET_LLAVE.split('/')[-1]}/{ARCHIVO_LLAVE}")
    if _llave.exists():
        os.environ["GOOGLE_API_KEY"] = _llave.read_text().strip()
        print("\\u2705 Llave tomada del dataset privado (verificacion automatica).")
'''


def aviso(texto=""):
    print(texto, flush=True)


def fuente(celda) -> str:
    return celda.source if isinstance(celda.source, str) else "".join(celda.source)


def indices_de_codigo(nb):
    """Posicion real de cada celda de codigo, para traducir los In[N] del registro."""
    return [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]


def kaggle(*args):
    """Llama al cliente de Kaggle del propio entorno, siempre en UTF-8."""
    entorno = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=entorno, cwd=RAIZ)
    salida = (r.stdout or "") + (r.stderr or "")
    if "Authentication required" in salida:
        raise SystemExit(
            "\nFalta la credencial de Kaggle, o la de siempre dejo de servir.\n"
            "  Genera un token en https://www.kaggle.com/settings/api (Generate New Token)\n"
            f"  y guardalo tal cual en {Path.home() / '.kaggle' / 'access_token'}\n"
        )
    return salida


# --------------------------------------------------------------------------
# Arreglos conocidos. Se buscan POR CONTENIDO, nunca por numero de celda, para
# que sirvan en cualquier libreta del curso.
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
    },
    {
        "id": "reintentos",
        "entornos": {"local", "kaggle"},
        "detecta": lambda s: re.search(r"exp_base\s*=\s*7", s) is not None,
        "aplica": lambda s: re.sub(r"exp_base\s*=\s*7,(?:[ \t]*#[^\n]*)?",
                                   COMENTARIO_REINTENTOS, s),
    },
]


def construir_corregido(libreta: str, entorno: str):
    """Copia el crudo a corregido/<entorno>/ con los arreglos que apliquen."""
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

    # Nunca pisar correcciones hechas a mano: si ya hay un corregido distinto del
    # que generariamos, dejamos el nuestro al lado y avisamos.
    if destino.exists():
        previo = nbformat.read(destino, as_version=4)
        if [fuente(c) for c in previo.cells] != [fuente(c) for c in nb.cells]:
            al_lado = destino.with_name(f"{libreta}.generado.ipynb")
            nbformat.write(nb, al_lado)
            aviso(f"    OJO: {destino.relative_to(RAIZ)} ya tiene arreglos que yo no se hacer.\n"
                  f"         No lo toque. Dejé mi version en {al_lado.relative_to(RAIZ)} para comparar.")
            return tocadas

    nbformat.write(nb, destino)
    return tocadas


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
        with urllib.request.urlopen(req) as r:
            return r.read()

    pagina = f"https://www.kaggle.com/code/{duenio}/{slug}"
    html = pide(pagina).decode("utf-8", "replace")
    m = re.search(r"scriptVersionId%3D(\d+)", html) or re.search(r'"scriptVersionId":\s*(\d+)', html)
    if not m:
        raise SystemExit(f"No encontre la version del guion en {pagina}. "
                         "Revisa que la libreta sea publica y que el enlace este bien.")
    datos = pide(f"https://www.kaggle.com/kernels/scriptcontent/{m.group(1)}/download")

    destino = RAIZ / "crudo" / f"{slug}.ipynb"
    huella = hashlib.sha256(datos).hexdigest()
    if destino.exists():
        if hashlib.sha256(destino.read_bytes()).hexdigest() == huella:
            aviso(f"  El crudo de {slug} ya estaba y es identico. No lo toco.")
            return slug
        raise SystemExit(f"Ya existe {destino} con otro contenido. Si de verdad quieres "
                         "reemplazar el crudo, borralo a mano primero.")

    nb = nbformat.reads(datos.decode("utf-8"), as_version=4)
    nbformat.validate(nb)
    destino.write_bytes(datos)

    lock_path = RAIZ / "crudo.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock[destino.name] = huella
    lock_path.write_text(json.dumps(dict(sorted(lock.items())), indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    codigo = sum(1 for c in nb.cells if c.cell_type == "code")
    aviso(f"  Crudo guardado: {destino.name} ({len(nb.cells)} celdas, {codigo} de codigo). Huella registrada.")
    return slug


# --------------------------------------------------------------------- local

def corre_local(ruta: Path, timeout: int = 900):
    """Ejecuta sin detenerse en los errores. Devuelve [(celda, nombre, mensaje)]."""
    nb = nbformat.read(ruta, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                   resources={"metadata": {"path": str(ruta.parent)}}).execute()
    fallos = []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                mensaje = (o.get("evalue") or "").strip().splitlines()
                fallos.append((i, o.get("ename"), mensaje[0] if mensaje else ""))
                break
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

def corre_kaggle(libreta: str, ipynb: Path, etiqueta: str, espera_max=2400):
    """Sube la libreta a la cuenta propia, la corre y lee los errores del registro."""
    slug = f"{USUARIO_KAGGLE}/{libreta}"
    trabajo = RAIZ / ".kaggle-trabajo" / libreta
    trabajo.mkdir(parents=True, exist_ok=True)
    meta = trabajo / "kernel-metadata.json"

    if not meta.exists():
        salida = kaggle("kernels", "pull", slug, "-p", str(trabajo), "-m")
        if not meta.exists():
            raise SystemExit(
                f"\nNo existe la libreta {slug} en tu cuenta.\n"
                f"  Abre https://www.kaggle.com/code/{DUENIO_CURSO}/{libreta} y dale a Copy & Edit una vez.\n"
                f"  Despues, en Add-ons > Secrets, marca la casilla de GOOGLE_API_KEY.\n"
                f"  (Kaggle dijo: {salida.strip()[:200]})\n"
            )
    fuentes = sorted({*json.loads(meta.read_text(encoding="utf-8")).get("dataset_sources", []),
                      DATASET_LLAVE})
    datos = json.loads(meta.read_text(encoding="utf-8"))
    datos.update({"code_file": ipynb.name, "enable_internet": True, "is_private": True,
                  "dataset_sources": fuentes})
    meta.write_text(json.dumps(datos, indent=2) + "\n", encoding="utf-8")

    # La copia que sube lleva el respaldo de la llave; el archivo de corregido/ no se toca.
    nb = nbformat.read(ipynb, as_version=4)
    for celda in nb.cells:
        if celda.cell_type == "code" and "kaggle_secrets" in fuente(celda):
            celda.source = fuente(celda) + RESPALDO_LLAVE
            break
    nbformat.write(nb, trabajo / ipynb.name)

    aviso(f"    subiendo {etiqueta} a {slug} ...")
    salida = kaggle("kernels", "push", "-p", str(trabajo))
    if "successfully pushed" not in salida:
        raise SystemExit(f"No pude subirla: {salida.strip()[:400]}")

    inicio = time.monotonic()
    while True:
        estado = kaggle("kernels", "status", slug)
        if any(e in estado for e in ESTADOS_FINALES):
            break
        if time.monotonic() - inicio > espera_max:
            raise SystemExit(f"La corrida de {slug} lleva mas de {espera_max // 60} minutos. "
                             "La dejo corriendo alla y salgo.")
        time.sleep(25)
    aviso(f"    termino en {int(time.monotonic() - inicio)} s")

    return errores_del_registro(libreta, kaggle("kernels", "logs", slug))


def errores_del_registro(libreta: str, registro: str):
    """Traduce el registro de Kaggle a [(celda, nombre, mensaje)] usando los In[N]."""
    try:
        texto = "".join(e["data"] for e in json.loads(registro))
    except Exception:
        texto = registro
    nb = nbformat.read(RAIZ / "crudo" / f"{libreta}.ipynb", as_version=4)
    codigo = indices_de_codigo(nb)
    marcas = list(re.finditer(r'Exception encountered at "In \[(\d+)\]"', texto))
    fallos = []
    for k, m in enumerate(marcas):
        n = int(m.group(1))
        celda = codigo[n - 1] if 0 < n <= len(codigo) else -1
        fin = marcas[k + 1].start() if k + 1 < len(marcas) else len(texto)
        # El traceback repite el nombre del error arriba (sin dos puntos) y abajo
        # con el mensaje real. Nos quedamos con el ultimo, que es el que explica.
        lineas = re.findall(r"^(\w*(?:Error|Exception)): (.*)$", texto[m.end():fin], re.M)
        nombre, mensaje = lineas[-1] if lineas else ("?", "")
        fallos.append((celda, nombre, mensaje.strip()))
    # El aviso de la llave solo importa si el respaldo del dataset tampoco la encontro:
    # en una corrida por API el mensaje de los Secrets sale siempre, y no es un problema.
    if "Authentication Error" in texto and "Llave tomada del dataset privado" not in texto:
        fallos.append((-1, "SinLlave",
                       f"ni los Secrets ni el dataset {DATASET_LLAVE} dieron la llave"))
    return fallos


# --------------------------------------------------------------------- pasos

def paso_local(libreta: str):
    aviso("\n[LOCAL]")
    aviso("  corriendo el crudo tal cual ...")
    informe("el crudo en esta computadora:", corre_local(RAIZ / "crudo" / f"{libreta}.ipynb"))
    tocadas = construir_corregido(libreta, "local")
    aviso("\n  arreglos aplicados: " +
          (", ".join(f"celda {i} ({cid})" for i, cid in tocadas) or "ninguno"))
    aviso("  corriendo el corregido ...")
    informe("el corregido en esta computadora:",
            corre_local(RAIZ / "corregido" / "local" / f"{libreta}.ipynb"))


def paso_kaggle(libreta: str):
    aviso("\n[KAGGLE]")
    informe("el crudo en Kaggle:",
            corre_kaggle(libreta, RAIZ / "crudo" / f"{libreta}.ipynb", "el crudo"))
    tocadas = construir_corregido(libreta, "kaggle")
    aviso("\n  arreglos aplicados: " +
          (", ".join(f"celda {i} ({cid})" for i, cid in tocadas) or "ninguno"))
    informe("el corregido en Kaggle:",
            corre_kaggle(libreta, RAIZ / "corregido" / "kaggle" / f"{libreta}.ipynb", "el corregido"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("comando", choices=["bajar", "local", "kaggle", "todo"])
    p.add_argument("referencia", help="enlace de Kaggle o nombre de la libreta")
    a = p.parse_args()

    if a.comando == "bajar":
        bajar(a.referencia)
        return

    if a.comando == "todo":
        aviso("\n[BAJAR]")
        libreta = bajar(a.referencia)
        paso_local(libreta)
        paso_kaggle(libreta)
    else:
        libreta = a.referencia.removesuffix(".ipynb")
        (paso_local if a.comando == "local" else paso_kaggle)(libreta)

    aviso("\nFalta escribir las fichas en criticas/<entorno>/ y correr herramientas/verificar.py")


if __name__ == "__main__":
    main()
