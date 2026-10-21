# Cohorte 2 — Visor didáctico de libretas

Cada libreta del curso de Kaggle se ve en tres vistas, para dos entornos: **Kaggle** (donde trabaja la clase) y **Local** (tu computadora).

- **📄 Crudo**: tal cual llegó de Kaggle. Nunca se toca; su huella está en `crudo.lock.json`.
- **🩻 Rayos X**: solo las celdas que hubo que corregir para que corra en ese entorno, con el error real que salió, qué cambió (calculado, no escrito a mano) y por qué.
- **✅ Corregido**: la versión que sí corre en ese entorno.

## Correr el visor

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

El visor no llama a ningún modelo: no necesita llave ni cuesta por uso.

## Correr las libretas corregidas

1. Copia `.env.example` como `.env` y pon tu llave de Gemini (`.env` nunca se sube).
2. Instala lo que pide la libreta (para ADK: `pip install google-adk python-dotenv jupyter`).
3. Abre `corregido/local/<libreta>.ipynb` en Jupyter. En Kaggle, importa `corregido/kaggle/<libreta>.ipynb` (File → Import Notebook) y conecta `GOOGLE_API_KEY` en Add-ons → Secrets.

## Sumar una libreta nueva

Un comando hace todo lo mecánico:

```powershell
python herramientas/preparar.py todo https://www.kaggle.com/code/kaggle5daysofai/<libreta>
```

Baja el crudo y registra su huella, lo corre tal cual en los dos entornos, aplica los arreglos que ya conocemos, vuelve a correr para comprobar que quedaron limpios, y al final imprime **los errores reales de cada entorno con su número de celda**. Los pasos sueltos son `bajar`, `local` y `kaggle`.

Busca los arreglos por contenido, no por número de celda, así que sirve en cualquier libreta. Y si un corregido ya trae arreglos hechos a mano que la herramienta no sabe hacer, **no lo pisa**: deja su versión al lado como `<libreta>.generado.ipynb` y avisa.

Después quedan dos cosas, que sí requieren criterio:

1. Escribe la ficha de cada celda cambiada en `criticas/<entorno>/<libreta>.json`, con el error real que imprimió la herramienta. Si en un entorno no hubo que cambiar nada, deja el corregido igual al crudo: el visor lo dice.
2. Corre `python herramientas/verificar.py` antes de cada commit: falla si el crudo cambió, si una celda cambiada no tiene ficha o si hay una llave expuesta.

### Lo que hay que preparar una sola vez

- **Token de Kaggle**: genéralo en [kaggle.com/settings/api](https://www.kaggle.com/settings/api) (*Generate New Token*) y guárdalo tal cual en `%USERPROFILE%\.kaggle\access_token`. Ojo: ese botón **muestra** el token, no descarga ningún archivo.
- **Copia de la libreta en tu cuenta**: entra a la libreta del curso y dale a *Copy & Edit* una vez. La herramienta no puede crearla.
- **La llave para las corridas automáticas**: una corrida lanzada por API **no alcanza los Kaggle Secrets**, aunque la casilla esté marcada (devuelve `Connection error trying to communicate with service`). Por eso la copia que se sube lleva un respaldo que lee la llave del dataset privado `ricardosauza/claves-privadas-cohorte2`. Ese respaldo **no se guarda** en `corregido/`: lo que se publica sigue siendo fiel a lo que hace un alumno, que sí usa los Secrets.

## Formato de las fichas

```json
{
  "resumen": "Una frase sobre qué le pasó a esta libreta en este entorno.",
  "criticas": [
    {
      "celda": 6,
      "titulo": "Nombre corto del problema",
      "categoria": "entorno-kaggle | dependencia | secreto | proceso-bloqueante | modelo | lógica",
      "error_real": "Mensaje que salió al correr el crudo en este entorno",
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
