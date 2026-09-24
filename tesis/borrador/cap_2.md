# Capítulo 2: Estado del arte y marco teórico

## 2.1 Estado del arte y trabajos relacionados

### Estrategia de búsqueda y selección
Para identificar las soluciones tecnológicas y modelos clínicos que han abordado la predicción del riesgo autolítico en adolescentes con antecedentes de trauma, se llevó a cabo una búsqueda de literatura utilizando Google Scholar. Debido a la naturaleza interdisciplinaria del problema, la estrategia de búsqueda se construyó combinando términos del ámbito clínico y computacional, tales como: *machine learning*, *suicide prediction*, *child sexual abuse*, *explainable AI* y *SHAP values*. La selección priorizó estudios empíricos publicados recientemente que implementaran algoritmos predictivos aplicados a la salud mental poblacional, excluyendo trabajos puramente teóricos o modelos de "caja negra" sin métricas de validación claras.

### Análisis de investigaciones relacionadas
En la última década, la psiquiatría computacional ha migrado de la estadística tradicional a los algoritmos de aprendizaje automático (*Machine Learning*) para superar el rendimiento azaroso de las escalas de cribado clínico vigentes \cite{linthicumMachineLearningSuicide2019,tangAnalysisEvaluationExplainable2024}. 

En el ámbito de la predicción, Walsh et al. \cite{walshPredictingSuicideAttempts2018} marcaron un hito al procesar expedientes clínicos electrónicos (EHR) de adolescentes con algoritmos de *Random Forest*, logrando discriminar el riesgo autolítico frente a casos de depresión activa con un área bajo la curva (AUC-ROC) de 0.87 a 0.90. De manera complementaria, Su et al. \cite{suMachineLearningSuicide2020} demostraron, sobre una cohorte longitudinal de 41,721 pacientes pediátricos, que es viable alcanzar exactitudes discriminativas estables (AUC-ROC de 0.81 a 0.86) empleando únicamente covariables de rutina (demográficas, diagnósticos previos y fármacos), lo que valida el potencial del aprendizaje automático como herramienta de tamizaje de bajo costo.

No obstante, la exactitud no garantiza aplicabilidad clínica. A pesar de los altos valores de AUC-ROC, la adopción de estos sistemas enfrenta la barrera de la "caja negra". Como lo plantean Joyce et al. \cite{joyceExplainableArtificialIntelligence2023} en su marco TIFU, la toma de decisiones en salud mental exige interpretabilidad. Para resolver esto, Nordin et al. \cite{nordinExplainableMachineLearning2022} aplicaron metodologías de Inteligencia Artificial Explicable (XAI) basadas en valores SHAP, demostrando que calcular la contribución marginal de los factores de riesgo permite generar explicaciones visuales que el personal de salud puede comprender.

Desde una perspectiva algorítmica aplicada al trauma, trabajos recientes como el de Niu et al. \cite{niuDecodingVitalVariables2025} utilizaron modelos de *Machine Learning* para decodificar las fases del suicidio en víctimas de abuso, demostrando que los algoritmos no lineales son capaces de aislar los mediadores clínicos (como la pérdida de autocompasión) que determinan el paso de la ideación a la acción, alcanzando valores de AUC-ROC superiores a 0.80. Sin embargo, estos enfoques aún no logran integrar una explicabilidad clínica amigable para el usuario final en entornos de atención primaria.

### Matriz comparativa de trabajos relacionados

| Trabajo | Contexto y objetivo | Método o propuesta | Evaluación (Métricas) | Limitación relevante |
| :--- | :--- | :--- | :--- | :--- |
| Walsh et al. (2018) \cite{walshPredictingSuicideAttempts2018} | Predecir intentos de suicidio en adolescentes usando datos clínicos | Algoritmo *Random Forest* sobre registros EHR longitudinales | AUC-ROC: 0.83 - 0.90 | Falta de interpretabilidad clínica (caja negra). |
| Su et al. (2020) \cite{suMachineLearningSuicide2020} | Riesgo autolítico con covariables de rutina en población pediátrica | Selección de características y clasificación ML | AUC-ROC: 0.81 - 0.86 | No se integra XAI para visualizar predictores a nivel individual. |
| Nordin et al. (2022) \cite{nordinExplainableMachineLearning2022} | Transparencia algorítmica en predicción de comportamiento suicida | Uso de SHAP para proveer Inteligencia Artificial Explicable | Explicabilidad global y local demostrada | No se enfoca específicamente en adolescentes con trauma (ASI). |
| Niu et al. (2025) \cite{niuDecodingVitalVariables2025} | Decodificar fases suicidas en víctimas de abuso sexual | *Machine Learning* para identificar mediadores vitales | AUC-ROC superior a 0.80 | Modelado sobre muestras hospitalarias o reducidas geográficamente. |

*Tabla 2.1: Matriz comparativa de investigaciones relevantes.*

### Discusión de la brecha de investigación
El cruce de esta evidencia revela un avance bifurcado: por un lado, existen modelos clínicos precisos pero opacos \cite{suMachineLearningSuicide2020,walshPredictingSuicideAttempts2018}; por otro lado, existen implementaciones de XAI que aún no se han aplicado al perfil específico de la vulnerabilidad por trauma temprano \cite{nordinExplainableMachineLearning2022}. A diferencia de los trabajos previos basados en muestras hospitalarias limitadas \cite{hebertUnveilingSuicidalRisk2025}, existe un vacío tecnológico respecto a la creación de un modelo que simultáneamente (1) prediga el intento de suicidio en adolescentes con ASI a escala poblacional y (2) integre la explicabilidad SHAP desde el diseño algorítmico. Esta investigación atiende precisamente dicha brecha, proveyendo un motor predictivo interpretable validado en una base de datos representativa a nivel nacional (ENSANUT).

---

## 2.2 Fundamentos teóricos y conceptuales

### Fundamentos clínicos y epidemiológicos del trauma
El sustrato conceptual de esta investigación se rige bajo el **Modelo de Vulnerabilidad Acumulada**, el cual postula que el trauma del abuso sexual interactúa con las desigualdades estructurales y la acumulación de desventajas socioambientales, o Determinantes Sociales de la Salud (SDoH). El estudio de cohorte longitudinal de Colburn et al. \cite{colburnCumulativeDeterminantsAdolescent2025} evidenció que el riesgo de intento de suicidio aumenta drásticamente con cada déficit adicional en dimensiones como estabilidad económica o vivienda. Cuando la vulnerabilidad estructural se superpone a la victimización por ASI, la probabilidad basal de intento de suicidio antes de los 18 años puede superar el 0.50 en los adolescentes que enfrentan ambos traumas \cite{colburnCumulativeDeterminantsAdolescent2025}.

Dentro de esta cascada, el Abuso Sexual Infantil (ASI) opera como un estresor primario que cataliza mediadores psicopatológicos concurrentes (como sintomatología depresiva, desregulación conductual limitante y conductas alimentarias de riesgo) que materializan la transición de la ideación al acto suicida. En una muestra clínica infantil, Hébert et al. \cite{hebertUnveilingSuicidalRisk2025} identificaron mediante árboles de decisión que el subgrupo con mayor riesgo autolítico inminente es aquel donde coinciden niveles clínicos de depresión y una desregulación emocional severa, la cual incapacita al menor para modular estados afectivos dolorosos.

A su vez, Brokke et al. \cite{brokkeEffectSexualAbuse2022} demostraron mediante ecuaciones estructurales que los síntomas disociativos median directamente el 68\% de la relación entre el antecedente de abuso sexual y los intentos de suicidio. La disociación sostenida funciona como un mecanismo de desconexión somática que reduce el miedo al dolor corporal, confiriendo al adolescente la "capacidad adquirida" para ejecutar autolesiones letales. La extrema complejidad de esta cascada de riesgos (SDoH, depresión, desregulación y disociación) justifica la necesidad ineludible de aplicar algoritmos computacionales no lineales capaces de capturar todas estas interacciones multidimensionales.

### Aprendizaje automático supervisado en salud mental
La **clasificación supervisada** es una rama de la Inteligencia Artificial donde un algoritmo aprende a mapear variables independientes (como determinantes biopsicosociales) hacia una etiqueta conocida (intento vs. no intento). Sin embargo, en epidemiología psiquiátrica, el evento autolítico representa una fracción mínima de la muestra, generando un **desbalance extremo de clases**.

Para abordar esta asimetría y evitar sesgos de sobreajuste (\*overfitting\*), la evaluación requiere metodologías robustas como la validación cruzada estratificada por conglomerados, así como el balanceo de pesos \cite{pedregosaScikitlearnMachineLearning2011}. Adicionalmente, se emplea la **Eliminación Recursiva de Características con Validación Cruzada (RFECV)**, un método envolvente (\*wrapper\*) que suprime iterativamente variables redundantes o ruidosas \cite{guyonGeneSelectionCancer2002,guyonIntroductionVariableFeature2003}. Esto asegura modelos parsimoniosos, esenciales para que la herramienta final sea ágil y aplicable en la consulta de atención primaria sin recolectar cientos de variables innecesarias.

### Evaluación explicable e interpretabilidad algorítmica (XAI)
El alto rendimiento discriminativo no basta si el modelo se comporta como una "caja negra" inescrutable para el médico tratante. Por ende, la **Inteligencia Artificial Explicable (XAI)** es un requisito ineludible. 

Para explicar la predicción algorítmica, este trabajo adopta los **Valores SHAP** (\*SHapley Additive exPlanations\*) \cite{lundbergUnifiedApproachInterpreting2017}. Fundamentado en la teoría de juegos cooperativos, SHAP distribuye equitativamente la responsabilidad de la predicción entre todos los factores de riesgo. La ventaja crítica de SHAP radica en su dualidad:
- **Explicabilidad global:** Identifica qué factores tienen mayor peso general en toda la población.
- **Explicabilidad local:** Permite desglosar la predicción de un paciente individual (ej. visualizar si la depresión grave empujó la probabilidad de riesgo, contrarrestando el efecto protector del apoyo familiar).

Finalmente, el desempeño de estos modelos debe evaluarse con métricas resilientes al desbalance de clases, priorizando el **Coeficiente de Correlación de Matthews (MCC)** y el **Área Bajo la Curva ROC (AUC-ROC)** por encima de la simple exactitud general (\*accuracy\*). Matemáticamente, el MCC es superior en datos biomédicos asimétricos porque evalúa la matriz de confusión en su totalidad (involucrando verdaderos y falsos positivos y negativos), de modo que solo arroja un puntaje alto si el algoritmo predice correctamente tanto a la clase minoritaria (intento suicida) como a la mayoritaria \cite{chiccoAdvantagesMatthewsCorrelation2020}.

### Conclusión y transición metodológica
En conjunto, estos tres pilares teóricos —el Modelo de Vulnerabilidad Acumulada, la optimización predictiva mediante RFECV y MCC, y la interpretabilidad algorítmica de SHAP— no son conceptualizaciones abstractas, sino que constituyen el andamiaje directo de esta investigación. Estos fundamentos se emplearán operativamente para precisar la hipótesis de trabajo, definir y seleccionar las variables biopsicosociales extraídas de la ENSANUT, orientar el *pipeline* experimental que se detallará en el Capítulo 3 (Metodología), y finalmente, servirán como marco interpretativo para discutir los hallazgos clínicos y tecnológicos en los Capítulos 4 y 5.
