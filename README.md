# Impacto del abuso sexual infantil en las conductas autolíticas de adolescentes en México

Este proyecto utiliza una metodología de **Minería de Datos (KDD)** y **Machine Learning supervisado explicable (SHAP)** para predecir el riesgo de intento de suicidio en adolescentes mexicanos de 10 a 19 años a partir de los datos crudos de la **ENSANUT Continua 2024**, analizando el abuso sexual infantil (ASI) como un predictor de riesgo central.

🌐 **Demo en vivo (Streamlit Cloud):** [https://suicidepredictors.streamlit.app/](https://suicidepredictors.streamlit.app/)

---

## Requisitos previos

| Herramienta | Versión mínima | Verificar con |
| :--- | :--- | :--- |
| Python | 3.10+ | `python --version` |

> **Aviso:** Ejecutar todos los comandos desde la **raíz del proyecto** (`seminario_investigacion/`). Los scripts usan rutas relativas que no funcionan desde otras carpetas.

---

## Ejecución del pipeline

El proyecto cuenta con un orquestador DevOps en PowerShell que automatiza todo el proceso de ciencia de datos:

### Opción A — Ejecución automática (Recomendada)

```powershell
# Pipeline completo + opción de lanzar el dashboard al final:
.\run_pipeline.ps1

# Solo el dashboard (si el pipeline ya fue ejecutado antes):
.\run_pipeline.ps1 -SoloDashboard
```

El script verificará el entorno virtual `.venv`, instalará todas las dependencias necesarias de Python y ejecutará los scripts del pipeline en orden secuencial.

### Opción B — Ejecución manual paso a paso

Si prefieres ejecutar el flujo KDD paso a paso, sigue estas instrucciones:

**1. Activar el entorno virtual e instalar dependencias**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Ejecutar el pipeline de Minería de Datos (KDD) en orden**
```powershell
# 1. Preparacion descriptiva
python src/01_an_descriptivo.py

# 2. Preprocesamiento, recodificación de variables y escala de depresión CES-D 7
python src/02_preprocesamiento.py

# 3. Generación automatizada de las figuras descriptivas de la muestra
python src/03_exploracion.py

# 4. Entrenamiento y validación cruzada (Nested CV) de los 4 clasificadores
python src/04_clasificacion.py

# 5. Cálculo y visualización de valores SHAP (Inteligencia Artificial Explicable)
python src/05_shap.py
```

**3. Lanzar el dashboard interactivo de Streamlit**
```powershell
streamlit run dashboard/app.py
```

El dashboard abrirá automáticamente en tu navegador web en `http://localhost:8501`.