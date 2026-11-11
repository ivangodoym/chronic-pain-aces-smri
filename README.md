# Pipeline de Análisis: Adversidad, Neuroimagen Estructural (sMRI) y Dolor Crónico

Este repositorio contiene el pipeline computacional para el procesamiento de datos, control de calidad psicométrico y neuroanatómico, imputación múltiple por ecuaciones encadenadas (MICE) y modelado estadístico multivariado (Regresión Logística, Mediación OLS con remuestreo Monte Carlo y Modelos de Ecuaciones Estructurales SEM).

El estudio evalúa de forma dimensional la interrelación entre:
* **Adversidad a lo largo del curso de vida:** Dimensiones de **Amenaza** (*Threat*) y **Privación** (*Deprivation*) a través de tres etapas temporales (**Infancia**, **Adultez** y **Reciente**), siguiendo el modelo DMAP.
* **Morfometría cerebral macroestructural (sMRI):** Volúmenes bilaterales de sustancia gris en 6 regiones de interés (ROIs) de la Red de Saliencia y el sistema límbico (Tálamo, Ínsula Anterior, Ínsula Posterior, Corteza Cingulada Anterior [ACC], Hipocampo y Parahipocampo), junto con la Corteza Calcarina como **control negativo**, ajustados por volumen intracraneal total ($\text{TIV}$).
* **Fenotipos clínicos de dolor crónico:** Presencia clínica de dolor ($\text{CP\_1}$, $\ge 3$ meses), severidad multidimensional continua latente (**`Pain_Latent_Z`**), intensidad numérica ($0-10$) y extensión espacial corporal ($1-17$ regiones).
* **Covariables:** Edad, sexo, diagnóstico clínico (Cognitivamente Normal [CN] vs. Enfermedad de Alzheimer [AD]), TIV y centro/escáner de adquisición (`site`).

---

## Estructura del Repositorio

```text
chronic-pain-aces-smri/
│
├── data/
│   ├── raw/
│   │   └── ACEs_Neuroimages.xlsx      # Dataset clínico y estructural original
│   └── processed/                     # Salidas intermedias y procesadas
│       ├── excluded_subjects.csv      # Sujetos excluidos por exceso de missingness
│       ├── complete_cases.csv         # Submuestra de casos completos sin imputar
│       ├── imputed_datasets.csv       # Datasets imputados apilados (_imputation 1..m)
│       ├── qc_predictors_report.csv   # Control psicométrico (asimetría, suelo, outliers)
│       ├── analysis_ready.csv         # Datasets imputados con Z-scores y subdominios
│       └── analysis_ready_complete_cases.csv # Casos completos listos para sensibilidad
│
├── src/
│   ├── __init__.py                    # Inicializador de módulo
│   ├── config.py                      # Rutas globales, parámetros y constantes
│   ├── adversity_domains.json         # Mapeo de ítems a dominios DMAP
│   ├── utils.py                       # Reglas de Rubin, simulación Monte Carlo y FDR
│   ├── preprocessing.py               # QC, atrición muestral (N=405) y MICE (m=10)
│   ├── features.py                    # Ingeniería de variables, factor latente y Z-scores
│   ├── modeling.py                    # Regresión logística, mediación OLS e inversa
│   ├── sem_modeling.py                # Modelo SEM unificado (semopy, N=141)
│   └── visualization.py               # Generación de Forest Plots publicables (Figuras 1, 2, 3)
│
├── outputs/
│   ├── tables/                        # Resultados tabulados (FDR, CIs, parámetros pooled)
│   │   ├── logistic_CP_1_results.csv
│   │   ├── mediation_Pain_Latent_Z_results.csv
│   │   ├── inverse_mediation_Pain_Latent_Z_results.csv
│   │   ├── moderation_Pain_Latent_Z_results.csv
│   │   └── sem_pooled_parameters_Pain_Latent_Z.csv
│   └── figures/                       # Figuras vectoriales en alta resolución (300 DPI)
│       ├── figure_1_forest_plot_logistic_odds_ratios.png
│       ├── figure_2_forest_plot_mediation_Pain_Latent_Z.png
│       ├── figure_3_forest_plot_sem_network_Pain_Latent_Z.png
│       ├── figure_S2_forest_plot_mediation_CP_Intensity_Z.png
│       └── figure_S3_forest_plot_adversity_to_smri_Pain_Latent_Z.png
│
├── reports/
│   ├── report.tex                     # Manuscrito fuente en LaTeX
│   └── report.pdf                     # Reporte completo precompilado para lectura directa
│
├── run_all.py                         # Script maestro de ejecución completa del pipeline
├── requirements.txt                   # Dependencias mínimas del entorno
├── CHANGELOG.md                       # Informe de auditoría y respuesta a la revisión
└── README.md                          # Documentación del proyecto
```

---

## Flujo Metodológico y Tamaños Muestrales ($N$)

El pipeline ejecuta una secuencia estandarizada y reproducible:

### 1. Control de Calidad, Armonización de Centros y Atrición
* **Tratamiento de centinelas:** Reemplazo estandarizado de códigos centinela (`-99`) por valores nulos (`NaN`).
* **Criterios de Inclusión en la Cohorte Primaria ($N = 405$):**
  1. Diagnóstico confirmado de envejecimiento cognitivamente normal (`CN`) o deterioro por enfermedad de Alzheimer (`AD`).
  2. Registro completo en la variable de presencia de dolor crónico (`CP_1`).
  3. Resonancia magnética cerebral estructural válida (T1) con estimación de volumen intracraneal total (`TIV_cm3`) y segmentaciones FreeSurfer completas.
  4. Se derivó la variable de centro (`site`) a partir del prefijo de identificación (`CU`, `BN`, `AF`, `BE`, `LO`, `SL`), excluyendo el centro con bajo reclutamiento (`MA`, $n = 1$).
* **Submuestra Clínica de Dolor Crónico ($N = 141$):** Participantes que cumplen los criterios anteriores y presentan dolor crónico activo persistente ($\text{CP\_1} = 1$), completando la batería multidimensional de dolor.

---

### 2. Control de Datos Faltantes (Missingness) e Imputación Múltiple (MICE)
* **Regla de Exclusión:** Se excluyen registros con pérdida mayor al **20%** (`MAX_ITEM_MISSING = 0.20`) en los ítems de adversidad, evaluado de forma jerárquica por dominio y etapa temporal. La auditoría empírica (`data/processed/excluded_subjects.csv`) confirmó que los 14 participantes con resonancia descartados presentaban una omisión severa masiva ($>40\%$ global), manteniendo la flexibilización por bloques temporales como análisis de robustez metodológico documentado en el `CHANGELOG.md`.
* **Imputación Múltiple con Congenilidad:** Aplicada sobre la cohorte primaria ($N = 405$) utilizando `IterativeImputer` (`BayesianRidge`, semilla determinista `RANDOM_STATE = 42`) generando $m = 10$ conjuntos de datos independientes. Se incluyeron los volúmenes de las ROIs cerebrales y TIV en la imputación para evitar sesgos de atenuación en las relaciones cerebro-adversidad.
* **Consistencia Clínica:** Las variables continuas e imputadas se acotaron a sus rangos psicométricos válidos ($1-4$ para frecuencia, $0-10$ para intensidad, $[0, 1]$ para binarias).

---

### 3. Ingeniería de Características y Estandarización (`src/features.py`)
* **Fenotipificación Dimensional de Dolor:**
  * **Gravedad Latente del Dolor (`Pain_Latent_Z` - Desenlace Primario):** Factor univariado derivado mediante Análisis Factorial Exploratorio (*Factor Analysis*) sobre intensidad ($0-10$), frecuencia ($1-4$) y extensión anatómica ($1-17$), escaladas a $[0, 1]$ y estandarizadas en puntajes $Z$ sobre la submuestra de dolor ($N = 141$).
  * **Desenlaces de Sensibilidad:** Intensidad univariada (`CP_Intensity_Z`) y extensión de regiones corporales (`n_regions_Z`).
  * **Presencia de Dolor (`CP_1`):** Variable dicotómica modelada sobre la cohorte completa ($N = 405$).
* **Dominios de Adversidad (DMAP):** Agregación de ítems en dimensiones de **Amenaza** y **Privación** para cada bloque vital (**Infancia**, **Adultez** y **Reciente**), asegurando la simetría conceptual del marco mediante variables continuas acumulativas e indicadores dicotómicos de violencia adulta (`Threat_adulthood_any_Z`).
* **Estandarización Z:** Realizada estrictamente al interior de cada dataset imputado ($m = 1 \dots 10$) para evitar fugas de información (*data leakage*).

---

### 4. Modelado Estadístico e Inferencia

1. **Modelos Logísticos Multivariados ($N = 405$):**
   * Evaluación de Odds Ratios (OR) ajustando por edad, sexo y TIV. Exclusión de la covariable `site` para evitar inestabilidad por separación completa (*complete separation*).
2. **Modelos de Mediación OLS ($N = 141$):**
   * Estimación de vías directas ($c'$) e indirectas ($a \times b$) con pooling de Rubin y remuestreo de **Monte Carlo con 10.000 iteraciones** para los intervalos de confianza empíricos del efecto indirecto. Ajustado por covariables y centro de adquisición (`site`).
3. **Análisis de Falsificación Causal (Mediación Inversa):**
   * Evaluación empírica de la hipótesis de atrofia inducida por dolor ($\text{Adversidad} \to \text{Dolor} \to \text{ROIs sMRI}$).
4. **Modelo de Ecuaciones Estructurales SEM Multivariado ($N = 141$):**
   * Estimación simultánea en `semopy` de las 6 ROIs con covarianzas residuales de error (`~~`), cascada vital (Infancia $\to$ Adultez), vías directas hacia `Pain_Latent_Z` y reporte de índices de bondad de ajuste ($\chi^2/\text{df}$, CFI, TLI, RMSEA).
5. **Control de Error Tipo I (FDR de Benjamini-Hochberg):**
   * Delimitación estricta de tres familias biológicas independientes: familia de detección de riesgo ($m=5$ y $m=7$), familia de búsqueda espacial ($m=7$) y familia de red estructural ($k=70$).

---

### 5. Generación de Visualizaciones Publicables (`src/visualization.py`)

* **Figura 1:** Forest plot de Odds Ratios de regresión logística ($N = 405$), contrastando adversidad frente a ROIs cerebrales.
* **Figura 2:** Forest plot de disociación de vías directas ($c'$) vs. indirectas ($a \times b$) hacia la severidad del dolor latente (`Pain_Latent_Z`, $N = 141$).
* **Figura 3:** Forest plot de arquitectura estructural SEM ($N = 141$), ilustrando la cascada socioeconómica vital ($\beta = +0.822^*$) y la predicción directa de la amenaza adulta hacia el dolor ($\beta = +0.152^*$).
* **Figuras Suplementarias S2 y S3:** Sensibilidad sobre intensidad/extensión y vías aferentes de adversidad hacia el cerebro.

---

## Instrucciones de Ejecución

> [!NOTE]
> **Lectura directa vs. Reproducción del pipeline:**
> El reporte completo de avance con todas las figuras vectoriales integradas y tablas suplementarias se encuentra precompilado para lectura directa en [`reports/report.pdf`](reports/report.pdf).
> Debido a normativas éticas de confidencialidad y protección de datos clínicos de salud, el dataset original (`data/raw/ACEs_Neuroimages.xlsx`) y las carpetas de artefactos generados (`outputs/`) no se versionan en el repositorio. Para reproducir los análisis y regenerar todas las salidas, el investigador autorizado debe disponer del archivo de datos en `data/raw/` y ejecutar el pipeline maestro.

El pipeline analítico completo se ejecuta de forma secuencial, determinista y automatizada mediante el script maestro `run_all.py`. Ya no es necesario ejecutar cada módulo de forma individual:

```bash
python run_all.py
```

### Secuencia Automatizada del Pipeline:
1. **`src.preprocessing`**: Control de calidad, filtrado de casos analíticos válidos ($N = 405$), reporte de atrición e imputación múltiple congenil ($m = 10$).
2. **`src.features`**: Fenotipificación factorial de gravedad latente de dolor (`Pain_Latent_Z`, $N = 141$), agregación de dominios DMAP y estandarización $Z$ intrafold.
3. **`src.modeling`**: Modelos logísticos multivariados ($N = 405$), mediación directa OLS con remuestreo Monte Carlo ($10.000$ iteraciones), moderación clínica (CN vs. AD) y mediación inversa de falsificación causal.
4. **`src.sem_modeling`**: Estimación simultánea multivariada de ecuaciones estructurales (`semopy`, $N = 141$), ajuste de covarianzas de error y combinación bajo Reglas de Rubin.
5. **`src.visualization`**: Generación de las Figuras principales 1, 2 y 3 y suplementarias S2 y S3 en alta resolución (300 DPI) en `outputs/figures/`.
