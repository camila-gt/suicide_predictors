# Conductas autolíticas en adolescentes de México — ENSANUT 2024

Análisis descriptivo y clasificación supervisada del **antecedente autorreportado de intento de suicidio alguna vez en la vida**, con KDD como marco metodológico y SHAP para interpretar el modelo. Incluye adolescentes de 10–19 años, con y sin antecedente de abuso sexual autorreportado (ASI).

El modelo no predice intentos futuros, no establece causalidad y no es una herramienta clínica validada.

## Ejecutar

Requisitos: Python **3.11 o posterior**, PowerShell y la base original en `data/adolescentes_ensanut2024_w.csv`. Ejecuta desde la raíz del proyecto:

```powershell
# Crea .venv si hace falta, instala dependencias, ejecuta pruebas y análisis.
.\run_pipeline.ps1

# Repetir sin reinstalar dependencias.
.\run_pipeline.ps1 -NoInstalar

# Abrir únicamente el dashboard: no ejecuta pruebas ni análisis.
.\run_pipeline.ps1 -SoloDashboard -NoInstalar
```

Si PowerShell bloquea el script, habilita su ejecución solo para esa sesión:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Opciones adicionales:

| Opción | Efecto |
|---|---|
| `-NoDashboard` | No ofrece abrir Streamlit al terminar. |
| `-OmitirShap` | Omite el cálculo y las dos figuras SHAP. |
| `-Hilos 4` | Limita el paralelismo; por defecto, `-1`. |
| `-Bootstrap 500` | Número de remuestreos por UPM; `0` los desactiva. |

También puede ejecutarse el análisis con `python -m suicide_predictors run` desde el entorno instalado. El código está en `suicide_predictors/`; no existe una entrada `src/`.

## Dónde están los resultados

Cada ejecución completa tiene una carpeta propia:

```text
output/runs/<identificador>/
├── data/          Bases descriptiva y ML; calidad de predictores
├── output/        Evaluación, modelos, selección y explicaciones
├── figuras/       PNG a 300 dpi y PDF vectoriales
└── manifest.json  Estado, configuración, entorno y hashes
```

`output/current.json` indica qué ejecución consulta el dashboard. Solo se actualiza tras completar correctamente el análisis; una ejecución fallida no sustituye la anterior. Por defecto se generan **20 figuras**: 13 exploratorias, una de preprocesamiento, una de selección, tres de clasificación y dos SHAP.

Las carpetas de ejecuciones y el puntero están ignorados por Git. Permanecen accesibles en el disco para el usuario y para una IA con acceso local, pero no se incluyen automáticamente en el repositorio remoto. Evita publicar microdatos individuales sin revisar las condiciones de uso.

## Procedimiento

1. Selección y preparación de la muestra con ASI y desenlace válidos.
2. Construcción de 21 predictores, separando desconocimiento (`NaN`) de no aplicabilidad (`-1`).
3. Partición agrupada por UPM y comparación de cuatro clasificadores mediante validación cruzada anidada.
4. RFECV e imputación ajustadas dentro de los entrenamientos correspondientes; selección por MCC no ponderado.
5. Evaluación interna, métricas ponderadas adicionales, intervalos por bootstrap de UPM e interpretación SHAP de la probabilidad de la clase positiva.

La partición de prueba es interna y fue inspeccionada previamente; no constituye validación externa ni una muestra históricamente intacta. Los factores de expansión y el agrupamiento por UPM no equivalen por sí solos a incorporar íntegramente el diseño de encuesta.

## Modelos guardados

- `modelo_evaluado.joblib`: modelo ajustado en desarrollo y evaluado en prueba.
- `mejor_modelo.joblib`: modelo de demostración ajustado con la muestra analítica completa, utilizado por el simulador y SHAP.

Ambos incluyen selección de columnas y preprocesamiento. Las métricas de prueba corresponden al primero, no al modelo de demostración.

## Documentación local

- `Docs/md/pipeline.md`: arquitectura y metodología implementada.
- `Docs/md/variables.md`: reactivos, codificación y aplicabilidad.
- `Docs/md/dashboard.md`: navegación y presentación de resultados.
- `Docs/md/resultados.md`: resultados vinculados a una ejecución identificada.
- `Docs/md/figuras_apa.md`: inventario y notas de las figuras.
- `Docs/md/guia.md`: recorrido técnico de lectura.

`Docs/` está ignorado por Git; estos documentos son locales. La redacción de la tesis se mantiene separada en `tesis/`.

## Desarrollo

Las dependencias de ejecución están fijadas en `requirements.txt`. `requirements-dev.txt` añade Ruff y es opcional para análisis y dashboard.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check suicide_predictors dashboard tests
```
