"""Comprueba que el visor no mienta y que no se suba ninguna llave.

Falla (código 1) si:
  - un crudo cambió respecto a su huella en crudo.lock.json, o no tiene huella;
  - crudo y corregido no tienen el mismo número de celdas;
  - una celda corregida no tiene ficha en criticas/, o una ficha apunta a una celda sin cambios;
  - aparece una llave de API en los archivos que se van a commitear.

Uso: python herramientas/verificar.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parent.parent
PATRON_LLAVE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
IGNORADOS = {".venv", ".env", "__pycache__", ".ipynb_checkpoints", "sample-agent"}


def fuente(celda) -> str:
    return celda.source if isinstance(celda.source, str) else "".join(celda.source)


def llave_local() -> str | None:
    env = RAIZ / ".env"
    if not env.exists():
        return None
    for linea in env.read_text(encoding="utf-8").splitlines():
        if linea.startswith("GOOGLE_API_KEY="):
            valor = linea.split("=", 1)[1].strip().strip('"').strip("'")
            return valor if len(valor) > 20 else None
    return None


def revisar_crudos(errores: list[str]):
    lock = json.loads((RAIZ / "crudo.lock.json").read_text(encoding="utf-8"))
    for ruta in sorted((RAIZ / "crudo").glob("*.ipynb")):
        huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
        if ruta.name not in lock:
            errores.append(f"{ruta.name}: crudo sin huella en crudo.lock.json")
        elif lock[ruta.name] != huella:
            errores.append(f"{ruta.name}: el crudo fue modificado (la huella no coincide)")


def revisar_correcciones(errores: list[str]):
    sueltos = [p.name for p in (RAIZ / "corregido").glob("*.ipynb")]
    if sueltos:
        errores.append(f"corregido/: libretas fuera de una carpeta de entorno (kaggle/ o local/): {sueltos}")
    for ruta in sorted((RAIZ / "corregido").glob("*/*.ipynb")):
        entorno = ruta.parent.name
        etiqueta = f"{entorno}/{ruta.name}"
        crudo_ruta = RAIZ / "crudo" / ruta.name
        if not crudo_ruta.exists():
            errores.append(f"{etiqueta}: hay corregido pero no crudo")
            continue
        crudo = nbformat.read(crudo_ruta, as_version=4)
        corregido = nbformat.read(ruta, as_version=4)
        if len(crudo.cells) != len(corregido.cells):
            errores.append(
                f"{etiqueta}: crudo tiene {len(crudo.cells)} celdas y corregido {len(corregido.cells)}"
            )
            continue

        cambiadas = {
            i for i, (a, b) in enumerate(zip(crudo.cells, corregido.cells)) if fuente(a) != fuente(b)
        }
        fichas_ruta = RAIZ / "criticas" / entorno / (ruta.stem + ".json")
        fichas = json.loads(fichas_ruta.read_text(encoding="utf-8")) if fichas_ruta.exists() else {}
        con_ficha = {c["celda"] for c in fichas.get("criticas", [])}

        for i in sorted(cambiadas - con_ficha):
            errores.append(f"{etiqueta}: la celda {i} cambió y no tiene ficha")
        for i in sorted(con_ficha - cambiadas):
            errores.append(f"{etiqueta}: la ficha de la celda {i} apunta a una celda sin cambios")


def revisar_llaves(errores: list[str]):
    propia = llave_local()
    for ruta in RAIZ.rglob("*"):
        if not ruta.is_file() or IGNORADOS & set(ruta.relative_to(RAIZ).parts):
            continue
        try:
            contenido = ruta.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if PATRON_LLAVE.search(contenido) or (propia and propia in contenido):
            errores.append(f"{ruta.relative_to(RAIZ)}: contiene una llave de API")


def main() -> int:
    errores: list[str] = []
    revisar_crudos(errores)
    revisar_correcciones(errores)
    revisar_llaves(errores)
    if errores:
        print("❌ Verificación fallida:")
        for e in errores:
            print("  -", e)
        return 1
    print("✅ Crudos intactos, cada cambio tiene su ficha y no hay llaves expuestas.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
