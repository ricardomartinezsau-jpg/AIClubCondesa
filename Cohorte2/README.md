# Cohorte 2 — Visor didáctico de libretas

Cada libreta del curso de Kaggle se ve en tres vistas:

- **📄 Crudo**: tal cual llegó de Kaggle. Nunca se toca; su huella está en `crudo.lock.json`.
- **🩻 Rayos X**: solo las celdas que hubo que corregir para que corra en local, con el error real que salió, qué cambió (calculado, no escrito a mano) y por qué.
- **✅ Corregido**: la versión que sí corre en local.

## Correr el visor

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

El visor no llama a ningún modelo: no necesita llave ni cuesta por uso.

## Correr las libretas corregidas

1. Copia `.env.example` como `.env` y pon tu llave de Gemini (`.env` nunca se sube).
2. Instala lo que pide la libreta (para ADK: `pip install google-adk python-dotenv jupyter`).
3. Abre `corregido/<libreta>.ipynb` en Jupyter.

## Sumar una libreta nueva

1. Suelta el `.ipynb` de Kaggle en `crudo/` y agrega su huella:
   `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" crudo/<libreta>.ipynb`
   → pégala en `crudo.lock.json`.
2. Córrela tal cual en local y anota cada error real.
3. Copia el crudo a `corregido/` y arregla **solo lo crítico** (lo que truena o engaña), sin agregar ni borrar celdas.
4. Escribe la ficha de cada celda cambiada en `criticas/<libreta>.json`.
5. Corre `python herramientas/verificar.py` antes de cada commit: falla si el crudo cambió, si una celda cambiada no tiene ficha o si hay una llave expuesta.

## Formato de las fichas

```json
{
  "resumen": "Una frase sobre qué le pasó a esta libreta en local.",
  "criticas": [
    {
      "celda": 6,
      "titulo": "Nombre corto del problema",
      "categoria": "entorno-kaggle | dependencia | secreto | proceso-bloqueante | modelo | lógica",
      "error_real": "Mensaje que salió al correr el crudo en local",
      "por_que": "Explicación para estudiante",
      "como_se_arreglo": "Qué se hizo",
      "verificado": true
    }
  ],
  "notas": [{"titulo": "...", "texto": "Observaciones que no requirieron cambiar código"}]
}
```

## Publicar en línea (gratis)

En [share.streamlit.io](https://share.streamlit.io): **Create app** → repo `ricardomartinezsau-jpg/AIClubCondesa`, rama `main`, archivo `Cohorte2/app.py` → **Deploy**. No hace falta configurar secretos.
