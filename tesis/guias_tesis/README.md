# Plantilla de tesis — Ingeniería de Software UAZ

Propuesta de plantilla institucional en LaTeX para el Programa Académico de
Ingeniería de Software de la Unidad Académica de Ingeniería Eléctrica de la
Universidad Autónoma de Zacatecas.

## Antes de comenzar

La plantilla separa el contenido académico, los datos de portada y la
configuración visual para reducir cambios accidentales. No es necesario conocer
LaTeX a profundidad: empiece modificando `configuracion/datos_tesis.tex` y los
archivos dentro de `capitulos/`. Evite editar
`configuracion/estilo_tesis.tex` salvo que el programa académico autorice un
cambio institucional.

Las instrucciones en gris son recomendaciones de redacción, no texto de la
tesis. Los ejemplos muestran la sintaxis de LaTeX y contienen datos ficticios.
Ambos pueden mantenerse visibles durante el trabajo y ocultarse al final.

## Inicio rápido en Overleaf

1. Descargue este proyecto como ZIP y no lo descomprima.
2. En Overleaf seleccione **New Project > Upload Project** y cargue el ZIP.
3. Verifique en **Menu > Main document** que el archivo principal sea
   `main.tex`.
4. Abra `configuracion/datos_tesis.tex` y sustituya título, autoría, dirección,
   codirección y sinodales.
5. Presione **Recompile**. La fecha, numeración, tabla de contenido, índices y
   referencias cruzadas se actualizan durante la compilación.
6. Redacte cada capítulo en su carpeta y cargue sus imágenes en la subcarpeta
   `imagenes/` del mismo capítulo.
7. Agregue las fuentes a `bibliografia/referencias.bib` y cítelas con
   `\cite{clave}`. El documento utiliza referencias numéricas estilo IEEE.

## Qué se actualiza automáticamente

- El mes y el año de la portada interior se obtienen de la fecha del sistema
  que compila el proyecto. En Overleaf se usa la fecha de sus servidores; no se
  necesita una consulta web ni escribir la fecha manualmente.
- La numeración de capítulos, secciones, figuras, tablas y ecuaciones se genera
  a partir de los comandos de LaTeX.
- La tabla de contenido comienza en una página nueva y cada índice posterior
  también inicia en una página nueva.
- Las citas numéricas y la bibliografía se ordenan con `ieeetr`.
- Las referencias creadas con `\label` y `\cref` se actualizan al recompilar.

Si la fecha oficial debe ser distinta de la fecha de compilación, cambie sólo
esta línea en `configuracion/datos_tesis.tex`:

```latex
\newcommand{\mesAnioPublicacion}{septiembre de 2026}
```

No conviene obtener la fecha mediante una conexión a Internet desde LaTeX:
Overleaf y muchas instalaciones bloquean esas consultas, y el documento dejaría
de ser reproducible. La fecha del compilador ofrece el mismo resultado sin esa
dependencia.

## Compilación

La opción recomendada es:

```bash
latexmk -pdf main.tex
```

El flujo equivalente es ejecutar `pdflatex`, `bibtex` y dos veces `pdflatex`.
En Overleaf, cargue el contenido completo del proyecto y seleccione `main.tex`
como documento principal.

Es normal que una cita, el número de una figura o una entrada del índice muestre
`?` después de la primera compilación. Compile nuevamente; `latexmk` lo hace de
forma automática. No edite a mano los archivos auxiliares (`.aux`, `.toc`,
`.lof`, `.lot`, `.loe` o `.bbl`).

## Mapa del proyecto

- `main.tex`: ordena las partes del documento; no contiene los datos personales.
- `configuracion/datos_tesis.tex`: único punto para editar portada, responsables,
  fecha e interruptores de ayuda.
- `configuracion/estilo_tesis.tex`: diseño institucional y comandos automáticos.
- `secciones_preliminares/`: acta, declaración, dedicatoria, agradecimientos,
  resumen, abstract y acrónimos.
- `capitulos/`: un directorio por capítulo y una subcarpeta `imagenes/` dentro de
  cada uno.
- `bibliografia/referencias.bib`: base bibliográfica del proyecto.
- `apendices/`: instrumentos, evidencia extensa y materiales complementarios.
- `guia_de_uso/`: ejemplos copiables que no se agregan al PDF por sí solos.
- `checklist_entrega.md`: revisión académica y técnica antes de entregar.

## Instrucciones y ejemplos removibles

Durante la redacción conserve estas líneas en
`configuracion/datos_tesis.tex`:

```latex
\mostrarInstruccionestrue
\mostrarEjemplostrue
```

Para la versión limpia cambie `true` por `false`. También puede retirar una sola
recomendación eliminando el comando `\instruccion{...}` correspondiente. No
borre llaves aisladas: seleccione desde `\instruccion{` hasta su llave final.

## Convenciones del proyecto

- Los nombres de archivos y carpetas usan minúsculas, guion bajo y términos
  descriptivos.
- Cada etiqueta debe tener un prefijo: `cap:`, `sec:`, `fig:`, `tab:`, `eq:` o
  `ap:`.
- Toda figura y tabla debe citarse y explicarse en el texto antes o
  inmediatamente después de aparecer.
- Para registrar una ecuación en su índice, agregue después del entorno
  `equation` la instrucción
  `\agregarEcuacionAlIndice{Descripción breve}`.
- Los datos, código, instrumentos y materiales publicados deben indicar versión,
  licencia y restricciones de acceso.
- El archivo `guia_de_uso/ejemplos_latex.tex` contiene ejemplos listos para
  copiar de citas, figuras, tablas, ecuaciones numeradas y referencias cruzadas.

## Estructura académica base

La plantilla propone seis capítulos: introducción; estado del arte y marco
teórico; metodología; resultados; discusión; y conclusiones y trabajo futuro. Esta
secuencia puede adaptarse con autorización de la persona directora, pero cada
función académica debe permanecer identificable.

El capítulo 2 sigue la lógica de desarrollo de la perspectiva teórica descrita
por Hernández-Sampieri y Mendoza, pero utiliza una estructura sencilla. Primero
presenta el estado del arte y los trabajos relacionados; después desarrolla los
fundamentos teóricos y conceptuales. Los antecedentes propios del problema se
conservan en la Introducción para evitar duplicarlos. Dentro de los fundamentos,
cada tesista agrega únicamente los subtemas que necesita para explicar su objeto
de estudio, sustentar decisiones o interpretar los resultados.

El capítulo 3 reúne el diseño de investigación, la metodología de desarrollo y
la descripción de la propuesta de ingeniería de software. La integración permite
explicar la solución dentro del proceso con el que fue construida, revisada y
validada. Aun dentro del mismo capítulo deben distinguirse tres funciones:

1. El método de investigación establece cómo se obtiene y analiza evidencia.
2. La metodología de desarrollo organiza la construcción de la solución.
3. La verificación, validación y aprobación comprueban aspectos diferentes de
   su calidad, pertinencia y aceptación.

Si una tesis no construye un artefacto y sólo compara métodos, analiza datos o
estudia un fenómeno, puede eliminar las subsecciones de descripción de la
propuesta que no correspondan sin crear un capítulo vacío.

### Cómo elegir la estructura de metodología y resultados

La metodología debe documentar tanto el diseño de investigación como el proceso
de desarrollo. En `Metodología de desarrollo de software` agregue las
subsecciones que requiera el enfoque real: UCD, Scrum, CRISP-DM, KDD,
prototipado, experimento controlado, estudio de caso u otro. No es obligatorio
conservar una lista fija. Scrum describe cómo se organiza el desarrollo, pero
no reemplaza la forma de obtener y analizar evidencia.

Use la ruta de evaluación que corresponda al objeto de estudio:

| Tipo de tesis o artefacto | Evidencia mínima esperada |
|---|---|
| Software interactivo / UCD | Usuarios representativos, tareas, instrumento o escala justificados, resultados y dispersión |
| Algoritmos / CRISP-DM / KDD | Datos y particiones, líneas base, métricas, repeticiones, incertidumbre y errores |
| Rendimiento, arquitectura o seguridad | Escenarios o criterios medibles, configuración reproducible y comparación pertinente |
| Estudio de caso o proceso | Selección del caso, fuentes, protocolo, indicadores y triangulación cuando corresponda |

El capítulo de resultados sigue esa estrategia de evaluación. Debe producir la
evidencia para contrastar la hipótesis y comprobar el cumplimiento de los
objetivos, pero no necesita repetir las preguntas de investigación como
subtítulos. Cuando el software será utilizado por personas, una demostración del
equipo no sustituye la validación con usuarios finales o representativos. De la
misma manera, la aprobación del cliente acredita el cumplimiento de acuerdos del
proyecto, pero no reemplaza la evidencia necesaria para sostener la hipótesis.

## Operaciones frecuentes

### Agregar una sección o capítulo

Use `\section{Nombre}` o `\subsection{Nombre}` dentro del capítulo. Para agregar
un capítulo completo, cree su carpeta y archivo `.tex`, incluya una subcarpeta
`imagenes/` y agréguelo en `main.tex` con `\include{ruta/sin_extension}`.

### Agregar y citar una imagen

Guarde el archivo en la carpeta `imagenes/` del capítulo y use el bloque de
`guia_de_uso/ejemplos_latex.tex`. La ruta distingue mayúsculas de minúsculas.
Evite espacios, acentos y nombres como `imagen1_final_final.png`; prefiera
`flujo_validacion_usuarios.png`.

### Agregar referencias IEEE

Copie la entrada BibTeX de la fuente en `bibliografia/referencias.bib`, asigne
una clave estable y cite con `\cite{clave}`. No escriba manualmente `[1]`: el
número puede cambiar cuando se agregan nuevas fuentes.

### Tablas, ecuaciones e índices

`guia_de_uso/ejemplos_latex.tex` contiene ejemplos de tabla, ecuación numerada,
figura y referencias cruzadas. Para incluir una ecuación en el índice agregue
después del entorno `equation`:

```latex
\agregarEcuacionAlIndice{Descripción breve de la ecuación}
```

## Problemas comunes

- **No aparece una cita:** confirme que la clave de `\cite{...}` coincida con
  `referencias.bib` y compile otra vez.
- **No aparece una imagen:** revise extensión, mayúsculas y ruta relativa al
  proyecto.
- **Una referencia muestra `??`:** la etiqueta debe existir y se requieren dos
  compilaciones.
- **La tabla se sale del margen:** reduzca texto, use `tabularx` o divídala; no
  reduzca toda la tesis de tamaño.
- **El guion bajo produce error:** dentro de una oración escríbalo como `\_`.
  En nombres de archivos sí puede utilizar guiones bajos, como lo hace esta
  plantilla; copie la ruta exacta en `\includegraphics`.
- **El índice no refleja cambios:** limpie los archivos auxiliares en Overleaf y
  vuelva a compilar desde cero.

## Antes de entregar

1. Resuelva todas las advertencias relevantes y referencias sin definir.
2. Sustituya o elimine datos y valores de ejemplo.
3. Oculte instrucciones y ejemplos.
4. Compruebe que cada objetivo, pregunta e hipótesis tenga evidencia y cierre.
5. Verifique tablas, figuras, ecuaciones, apéndices y fuentes citadas.
6. Ejecute `checklist_entrega.md` con la persona directora.
