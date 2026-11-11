# CHANGELOG: Auditoría y Respuesta a la Revisión Metodológica
**Proyecto:** Adversidad a lo largo del curso de la vida, macroestructura cerebral (sMRI) y dolor crónico  
**Repositorio:** `chronic-pain-aces-smri`  
**Destinatarios:** Dra. Loreto Muñoz y equipo de investigación     
**Fecha:** Noviembre de 2026  
**Documento base:** *Revisión del repositorio chronic-pain-aces-smri (29 de octubre de 2026)*

---

## 3. Puntos Críticos y Condiciones de Interpretación

### 3.1: Control del efecto de centro y escáner (site)
* **Requerimiento:** Derivar `site` a partir del prefijo del ID (centros de reclutamiento), agregarlo como covariable para controlar el efecto de escáner en las ROIs y excluir/fusionar el centro con $n = 1$ (MA).
* **Corrección realizada:** 
  1. En `src/preprocessing.py` se implementó la extracción formal del centro: `df["site"] = df["ID"].str.extract(r"^([A-Za-z]+)")`.
  2. Se excluyó deliberadamente el sujeto perteneciente al centro `MA` ($n = 1$) para evitar singularidades matemáticas.
  3. **Modelos de mediación OLS:** Se incorporó `site` como covariable categórica obligatoria en todos los modelos lineales de la vía $a$ ($\text{Adversidad} \to \text{Cerebro}$) y vías directas.
  4. **Modelos logísticos ($N = 405$):** Se excluyó `site` de la regresión logística binaria debido a separación completa (*complete separation*) en centros que presentaban 100% de prevalencia de dolor, lo cual generaba inestabilidad numérica extrema en el estimador de máxima verosimilitud (documentado explícitamente en la Sección 2 del reporte LaTeX).
  5. **Modelo SEM unificado ($N = 141$):** El ajuste neuroanatómico se controló mediante volumen intracraneal total ($\text{TIV}_{\text{cm}^3}$), edad, sexo y diagnóstico clínico. Se omitieron los dummies de centro para evitar sobreparametrización ($k = 70$ coeficientes sobre $N = 141$), protegiendo la convergencia y la condición positiva definida de la matriz hessiana.

---

### 3.2: Transparencia muestral (N analizada), atrición y congenilidad en la imputación múltiple
* **Requerimiento:** Corregir las $N$ reportadas en el README (reemplazar $726$ y $242$ por $405/406$ y $141$), explicitar el diagrama de flujo de atrición y garantizar congenilidad en la imputación (restringir a la cohorte con resonancia antes de imputar para no asumir independencia espuria entre cerebro y adversidad).
* **Corrección realizada:**
  1. **Atrición y $N$ formalizada:** Se formalizó la cohorte analítica primaria en $N = 405$ participantes (criterios: CN o AD confirmado, dato válido en `CP_1` y TIV válido). De ellos, se identificó la submuestra clínica fenotípica de $N = 141$ participantes con dolor crónico activo ($\text{CP\_1} = 1$) que completaron la batería multidimensional.
  2. **Congenilidad en MICE:** La imputación múltiple por ecuaciones encadenadas (`src/preprocessing.py`) se restringió a la cohorte que cuenta con neuroimagen válida ($N = 405$), incorporando los mediadores volumétricos cerebrales, TIV y covariables analíticas dentro del pool de predictores (`BayesianRidge`, $m = 10$). Esto previene el sesgo de atenuación hacia cero en el sendero $a$.
  3. **Actualización documental:** Se eliminaron las cifras manuales obsoletas ($726$ y $242$) del `README.md`, de las tablas de salida y de la Sección 2.1 del reporte LaTeX.

---

### 3.3: Ajuste de potencia estadística y jerarquía de desenlaces
* **Requerimiento:** Declarar el efecto mínimo detectable a la potencia disponible ($N = 405$ vs. $N = 141$), establecer la presencia de dolor (`CP_1`) o extensión espacial como desenlace primario y tratar la intensidad como exploratoria.
* **Corrección realizada:**
  1. **Jerarquización en dos etapas:** 
     * *Etapa 1 (Detección de riesgo poblacional):* Presencia clínica de dolor crónico (`CP_1`, $N = 405$) evaluada mediante regresión logística multivariada (potencia suficiente para detectar $\text{OR} \ge 1.22$).
     * *Etapa 2 (Severidad multidimensional):* Submuestra con dolor ($N = 141$), modelada mediante la variable latente unificada `Pain_Latent_Z` para minimizar el error de medición.
  2. **Sensibilidad:** Los desenlaces univariados de intensidad (`CP_Intensity_Z`) y extensión de regiones corporales (`n_regions_Z`) se reclasificaron formalmente como análisis exploratorios de sensibilidad y se trasladaron al Anexo Suplementario (Figura S2).
  3. Se incluyó en la Discusión/Conclusiones el reconocimiento explícito del umbral mínimo detectable para los efectos indirectos mediados.

---

## 4. Análisis de Robustez y Extensión Metodológica

### 4.1: Hipótesis primaria y delimitación estricta de familias FDR
* **Requerimiento:** Definir formalmente familias FDR independientes con justificación teórica y biológica sustantiva, evitando particiones arbitrarias post-hoc.
* **Corrección realizada:** En la Sección 2.4 de `reports/report.tex` y en `src/utils.py` se establecieron tres familias confirmatorias independientes bajo corrección Benjamini-Hochberg ($q < 0.05$):
  * **Familia de detección de riesgo (Logística, $N=405$):** $m = 5$ pruebas para los predictores de adversidad a lo largo del curso de vida y $m = 7$ pruebas para las estructuras cerebrales.
  * **Familia de búsqueda espacial (Mediación OLS, $N=141$):** $m = 7$ ROIs candidatas evaluadas simultáneamente por cada sendero de adversidad hacia el dolor latente.
  * **Familia de red estructural (SEM, $N=141$):** El conjunto unificado de los $k = 70$ coeficientes direccionales de regresión ($\beta$), excluyendo covarianzas residuales de error y cargas factoriales fijadas como marcadores métricos.

---

### 4.2: Inclusión de mediador de control negativo (Corteza Calcarina)
* **Requerimiento:** Incorporar una región cerebral sin relación teórica con dolor ni con adversidad (e.g., corteza calcarina) para contrastar atrofia global o confusión residual.
* **Corrección realizada:** 
  1. Se incorporó la Corteza Calcarina Bilateral (`Calcarine_bilateral_Z`) en `src/config.py`, extrayendo sus volúmenes en `src/features.py`.
  2. Se testeó en todos los modelos logísticos, de mediación OLS, moderación y mediación inversa.
  3. **Resultado:** Resultó completamente nula en todos los modelos ($ab \approx 0$, todos los $p_{\text{FDR}} > 0.95$), confirmando la especificidad anatómica y descartando artefactos de atrofia generalizada no controlada.

---

### 4.3: Modelo de mediación inversa (Falsification Test: Dolor hacia Cerebro)
* **Requerimiento:** Evaluar la dirección causal inversa ($\text{Adversidad} \to \text{Dolor} \to \text{Cerebro}$) para confrontar la hipótesis de atrofia inducida por dolor (*wear-and-tear* neurotóxico).
* **Corrección realizada:** 
  1. Se implementó la rutina automatizada de mediación inversa en `src/modeling.py` (líneas 356–374), generando `outputs/tables/inverse_mediation_Pain_Latent_Z_results.csv`.
  2. **Resultado empírico:** Si bien la privación adulta predijo directamente el dolor latente ($a = +0.213, p_{\text{FDR}} = 0.028^*$), el dolor latente no predijo reducciones volumétricas en ninguna de las 7 ROIs ($b \in [-0.063, +0.050]$, todos $p_{\text{FDR}} > 0.93$), resultando en efectos indirectos completamente nulos ($ab \in [-0.012, +0.008]$, todos $p_{\text{FDR}} \ge 0.94$).
  3. Se incorporó este análisis como prueba de falsificación causal en la Sección 3.2 del reporte y se tabuló como Tabla suplementaria S4 en el Anexo.

---

### 4.4: Modelo de Ecuaciones Estructurales (SEM) multivariado en paralelo
* **Requerimiento:** Estimar un modelo estructural unificado que evalúe simultáneamente las ROIs controlando por sus covarianzas residuales y resuelva la cascada vital.
* **Corrección realizada:** 
  1. Se desarrolló el módulo `src/sem_modeling.py` empleando el paquete `semopy`.
  2. El modelo integró simultáneamente las 6 ROIs cerebrales con sus covarianzas de error correlacionadas (`~~`), la cascada temporal de adversidad (Infancia $\to$ Adultez), el constructo latente de dolor y el ajuste por covariables exógenas.
  3. Se computaron e informaron los índices globales de bondad de ajuste ($\chi^2/\text{df} = 2.44, \text{CFI} = 0.924, \text{TLI} = 0.887, \text{RMSEA} = 0.101$) y se generó el Forest Plot de arquitectura estructural (Figura 3).

---

### 4.5: Análisis de moderación por diagnóstico clínico (CN vs. AD)
* **Requerimiento:** Comprobar si las vías de adversidad y volumen cerebral interactúan con el deterioro cognitivo/Alzheimer.
* **Corrección realizada:** Se ejecutó el pipeline de moderación multivariada (`outputs/tables/moderation_Pain_Latent_Z_results.csv`). Ningún término de interacción ($\text{Adversidad} \times \text{Diagnóstico}$ o $\text{ROI} \times \text{Diagnóstico}$) sobrevivió al control FDR ($p_{\text{FDR}} > 0.45$), justificando formalmente el modelado conjunto de la cohorte con ajuste aditivo por diagnóstico.

---

### 4.6: Tratamiento psicométrico de los desenlaces de dolor
* **Requerimiento:** Evitar tratar la extensión corporal (`n_regions`, inflada en ceros en el 65% de la muestra) mediante OLS estándar sin justificación; diferenciar presencia de dolor de severidad continua.
* **Corrección realizada:** 
  1. La presencia binaria se modeló estrictamente mediante regresión logística en la muestra completa ($N = 405$).
  2. En la submuestra con dolor crónico ($N = 141$), se aplicó un Análisis Factorial Exploratorio univariado sobre las tres dimensiones escaladas (intensidad numérica $0-10$, frecuencia semanal y conteo de sitios anatómicos dolorosos $1-17$), derivando la variable tipificada continua **`Pain_Latent_Z`** (Latent Pain Severity), libre de inflación de ceros y con mayor consistencia psicométrica.

---

### 4.7: Depuración de ítems de adversidad y operacionalización simétrica de amenaza
* **Requerimiento:** Retirar ítems anacrónicos en la infancia (computador, internet, smartphone) que confundían privación con cohorte de nacimiento; resolver el efecto piso de amenaza en la adultez para no descartarla del manuscrito.
* **Corrección realizada:** 
  1. Se auditó y depuró `src/adversity_domains.json`, asegurando que la privación infantil refleje carencias materiales universales a su cohorte generacional (necesidades básicas insatisfechas y privación de activos del hogar tradicionales).
  2. Para evitar descartar la dimensión de amenaza fuera de la infancia por efecto piso en escalas continuas, se crearon dos métricas:
     * `Threat_adulthood_any_Z`: indicador dicotómico de exposición a violencia adulta (utilizado en regresión logística, $N = 405$) y en los modelos de mediación OLS, $N = 141$, tras superar el filtro de asimetría.
     * `Cumulative_threat_adulthood_Z`: carga acumulada continua utilizada en SEM ($N = 141$) para modelar la cascada vital continua y la relación dosis-respuesta con la severidad del dolor latente.
  3. Esto permitió preservar la simetría conceptual del marco DMAP (Amenaza vs. Privación) a lo largo de todo el curso de vida.

---

## 5. Auditoría del Repositorio y Calidad de Código

### 5.2.1: Robustez en la creación de directorios (`mkdir`)
* **Corrección:** Se incorporó `OUTPUTS_FIGS_DIR.mkdir(parents=True, exist_ok=True)` y `DATA_PROCESSED_DIR.mkdir(...)` al inicio de cada script ejecutor (`preprocessing.py`, `features.py`, `modeling.py`, `sem_modeling.py`, `visualization.py`), garantizando ejecución limpia e ininterrumpida desde un clon de Git en blanco.

### 5.2.2: Corrección gráfica de significancia (FDR vs. Nominal) y gramática visual unificada
* **Corrección:** Se corrigió el bug en `src/visualization.py` (`if is_sig_ab:` y `sig_p_roi`). Se unificó la gramática visual a través de todas las figuras (Figuras 1, 2 y 3) para discriminar inequívocamente tres niveles inferenciales:
  1. **Efectos confirmatorios post-FDR ($q < 0.05$):** Círculos azules (`#1f77b4`) con asterisco (`$*$`).
  2. **Efectos nominales no corregidos ($p < 0.05$ crudo y $q \ge 0.05$):** Diamantes naranjas (`#e66101`) sin asterisco.
  3. **Efectos nulos ($p \ge 0.05$ y $q \ge 0.05$):** Cuadrados grises (`#7f7f7f`).

### 5.2.3: Sintaxis y robustez en bucles de `modeling.py`
* **Corrección:** Se revisó la indentación y se introdujeron guardas estrictas de validación de columnas previo a la instanciación de fórmulas en `statsmodels`.

### 5.2.4: Recodificación consistente de `CP_Frequency`
* **Corrección:** Se armonizó la escala ordinal de frecuencia de dolor en `src/features.py` previo a la estandarización factorial, previniendo la generación espuria de valores `NaN`.

### 5.2.6: Combinación de parámetros bajo Reglas de Rubin (1987)
* **Corrección:** En `src/utils.py`, `pool_rubin_scalar` calcula la varianza intraconjunto ($W$) e interconjunto ($B$) a través de las $m = 10$ imputaciones, derivando estadísticos combinados con grados de libertad ajustados para la inferencia.

### 5.2.7: Reglas de exclusión por proporción de datos faltantes en escalas breves
* **Requerimiento:** Evaluar el impacto del umbral del 20% aplicado a nivel de subdominios (`missing_prop_domain > 0.20`), el cual en subescalas breves como `Physical_threat_adulthood` (3 ítems) penaliza asimétricamente a cualquier sujeto con una sola omisión ($1/3 = 33.3\% > 20\%$), afectando teóricamente a 34 sujetos (14 con resonancia). Se sugiere evaluar si conviene definir el umbral en número absoluto de ítems o aplicarlo exclusivamente a bloques temporales.
* **Auditoría empírica y justificación metodológica:**
  1. Se auditó la submuestra de participantes excluidos en `data/processed/excluded_subjects.csv` ($N = 14$ sujetos con resonancia).
  2. De los 14 sujetos excluidos, **13 participantes presentan un patrón de omisión masiva** (`item_missing_n = 42` de 104 ítems, correspondiente a una pérdida global del **$40.38\%$**), por lo que quedan descartados bajo el criterio global ($> 20\%$) y de bloques temporales independientemente del filtro por subdominio.
  3. El único participante restante (`AF094`) concentra 15 ítems omitidos en la etapa adulta ($15/33 = 45.4\%$), superando igualmente el umbral del bloque temporal.
  4. En consecuencia, el filtro no descartó participantes válidos con omisiones aisladas accidentales en la cohorte con resonancia. Las estimaciones de la cohorte primaria ($N = 405$) y de dolor ($N = 141$) son estadísticamente robustas. La flexibilización a nivel de bloques temporales se deja formalmente documentada como análisis de sensibilidad para la versión final del manuscrito.

### 5.2.9: Limpieza de nombres reservados de Python
* **Corrección:** Se eliminó el sombreo de la función incorporada `type` en los scripts de visualización y modelado.

---

## 6. Estado Actual de Entregables 

1. **Pipeline de Código:** Totalmente funcional, modular y determinista con semilla fija (`RANDOM_STATE = 42`).
2. **Resultados Empíricos Tabulados:** 19 archivos CSV consolidados en `outputs/tables/` cubriendo logística, mediación directa OLS, mediación inversa, moderación y parámetros SEM unificados.
3. **Figuras de Alta Resolución:** Figuras 1, 2 y 3 (principales) y Figuras suplementarias S2 y S3 en `outputs/figures/`.
4. **Manuscrito de Avance (`reports/report.tex`):** Compilado exitosamente en PDF de 4 secciones completas, incluyendo metodología rigurosa, resultados con figuras vectoriales, discusión ejecutiva orientada a la toma de decisiones con los PIs y Anexo con Tablas S1 a S4.