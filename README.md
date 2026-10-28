## Pipeline de Análisis de Mediación: Adversidad, Neuroimagen y Dolor Crónico
Este repositorio contiene el pipeline de preparación de datos, control de calidad y tratamiento de datos faltantes mediante Imputación Múltiple por Ecuaciones Encadenadas (MICE). Está diseñado para armonizar e integrar datos de : 
- Adversidad a lo largo del ciclo vital: Amenaza y privación en infancia, adultez y período reciente. 
- Dolor Crónico: Presencia, frecuencia, intensidad y extensión espacial.
- Neuroimagen estructura (sMRI): Volumetría subcortical y cortical (TIV, tálamo, ínsula anterior y posterior, etc.).
- Covariables Clínicas y Demográficas: Edad, sexo, diagnóstico clínico (CN vs. AD) y volumen intracranial total (TIV)

### Estructura del Repositorio

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
│       ├── analysis_ready.csv         # Datasets imputados con Z-scores y sub-dominios
│       └── analysis_ready_complete_cases.csv # Casos completos listos para sensibilidad
│
├── src/
│   ├── __init__.py                    # Inicializador de módulo
│   ├── config.py                      # Rutas globales, parámetros y variables
│   ├── adversity_domains.json         # Mapeo de ítems a dominios de adversidad
│   ├── utils.py                       # Herramientas estadísticas (Rubin, Monte Carlo, FDR)
│   ├── preprocessing.py               # QC, filtrado de cohorte y MICE (m=10)
│   ├── features.py                    # Ingeniería de características y estandarización
│   ├── modeling.py                    # Mediación OLS, pooling de Rubin y Monte Carlo
│   └── visualization.py               # Generación de Heatmaps tri-panel y Forest Plots
│
├── outputs/
│   ├── tables/                        # Tablas de resultados estadísticos (FDR, CIs)
│   │   ├── mediation_CP_Intensity_Z_results.csv
│   │   └── mediation_n_regions_Z_results.csv
│   └── figures/                       # Figuras vectoriales/alta resolución
│       ├── heatmap_CP_Intensity_Z.png
│       ├── heatmap_n_regions_Z.png
│       ├── forest_plot_CP_Intensity_Z.png
│       └── forest_plot_n_regions_Z.png
│
├── requirements.txt                   # Dependencias mínimas del entorno
└── README.md                          # Documentación del proyecto
```

## Flujo Metodológico
El pipeline metodológico ejecuta una secuencia estandarizada a través de las siguientes fases:

---


### 1. Control de Calidad y Filtrado Clínico
* **Tratamiento de centinelas:** Reemplaza valores codificados como `-99` por valores nulos (`NaN`).
* **Criterio de Inclusión Primario:** Requiere dato válido en la variable de dolor crónico (`CP_1`). Los registros con `CP_1` ausente son descartados.
* **Filtro Diagnóstico:** Restringe la cohorte a sujetos con diagnóstico cognitivo normal (`CN` $\rightarrow 0$) o Enfermedad de Alzheimer (`AD` $\rightarrow 1$).

--- 

### 2. Criterios de Exclusión por Datos Faltantes (Missingness)
Para evitar imputaciones espurias en sujetos con escasez severa de información de adversidad, se aplica un umbral máximo de pérdida del **20%** (`MAX_ITEM_MISSING = 0.20`), evaluado de manera jerárquica:
1. **Por dominio específico** (según las dimensiones en `adversity_domains.json`).
2. **Por bloque temporal** (infancia, adultez y reciente).
3. **Global** sobre el total de ítems de adversidad.
Los participantes que superan este umbral son excluidos del análisis primario y registrados en `data/processed/excluded_subjects.csv`.

---

### 3. Imputación Múltiple por Ecuaciones Encadenadas (MICE)
Sobre la muestra filtrada, el pipeline aborda el missingness restante mediante `IterativeImputer` con estimador `BayesianRidge`:
* **Número de imputaciones:** Se generan $m = 10$ conjuntos de datos independientes con variabilidad estocástica muestreada a posteriori (`sample_posterior=True`).
* **Restricciones de dominio y consistencia clínica:**
  * Variables binarias (adversidades, regiones corporales de dolor, diagnóstico) son redondeadas y restringidas al rango $[0, 1]$.
  * Variable sexo (`demo_sex`) acotada a $[1, 2]$.
  * Coherencia diagnóstica de dolor: si `CP_1 == 0` (sin dolor crónico), frecuencia e intensidad se fijan en `NaN`; si `CP_1 == 1`, se acotan a sus rangos psicométricos válidos ($1\text{–}4$ para frecuencia, $0\text{–}10$ para intensidad).
* **Salidas:** Se genera tanto la submuestra de casos completos (`data/processed/complete_cases.csv`) como el archivo apilado consolidado (`data/processed/imputed_datasets.csv`, identificado por la columna `_imputation`).

---

### 4. Ingeniería de Características y Estandarización (`src/features.py`)
Sobre los $m = 10$ datasets imputados, el script genera las variables dimensionales para el modelado:
* **Desenlaces Clínicos de Dolor:**
  * **Intensidad de Dolor Crónico (`CP_Intensity_Z`):** Promedio de intensidad psicométrica (escala $0\text{–}10$) restringido estrictamente a la subcohorte que reporta dolor crónico activo (`CP_1 == 1`, $N = 242$), estandarizado en puntajes $Z$.
  * **Extensión Espacial del Dolor (`n_regions_Z`):** Suma discreta de regiones corporales afectadas (rango acotado $[0, 17]$), evaluada sobre la muestra analítica completa ($N = 726$), estandarizada en puntajes $Z$.
* **Dominios Dimensionales de Adversidad:**
  * Agregación jerárquica de ítems según `src/adversity_domains.json`: dimensiones de **Amenaza** (`Threat`) y **Privación** (`Deprivation`) calculadas para cada etapa vital (**Infancia**, **Adultez** y **Reciente**), junto con sus sub-dominios específicos (amenaza física/psicológica, privación de necesidades básicas/activos).
* **Estandarización Intra-Imputación (Sin Fuga de Información):**
  * Para evitar *data leakage*, el cálculo de medias y desviaciones estándar para la estandarización ($Z$) se realiza de forma estrictamente independiente dentro de cada dataset imputado ($m = 1 \dots 10$).
* **Salida Consolidada:** `data/processed/analysis_ready.csv`.

---

### 5. Modelado de Mediación Estadística Multivariada (`src/modeling.py`)
Evalúa de manera sistemática si las alteraciones volumétricas en regiones de interés (ROIs en Tálamo e Ínsula anterior/posterior) median la relación entre la exposición a adversidad en el ciclo vital y los desenlaces clínicos de dolor:
* **Ecuaciones de Regresión OLS:**
  * **Path $a$ (Adversidad $\rightarrow$ Cerebro):** $\text{ROI}_Z \sim \text{Adversidad}_Z + \text{Edad}_Z + \text{Sexo} + \text{TIV}_Z + \text{Diagnóstico}$
  * **Path $b$ y Path $c'$ (Cerebro + Adversidad $\rightarrow$ Dolor):** $\text{Dolor}_Z \sim \text{ROI}_Z + \text{Adversidad}_Z + \text{Edad}_Z + \text{Sexo} + \text{Diagnóstico}$
* **Combinación de Parámetros mediante Reglas de Rubin:**
  * Los estimadores $\beta$ y sus errores estándar ($SE$) se combinan a través de las 10 imputaciones cuantificando la varianza intra e inter-imputación para obtener estadísticos $z$ y valores $p$ insesgados.
* **Inferencia del Efecto Indirecto ($a \times b$) vía Simulación de Monte Carlo:**
  * Debido a la asimetría en la distribución del producto de dos variables normales, el efecto indirecto ($ab$), sus intervalos de confianza al 95% y sus valores $p$ empíricos se estiman mediante simulación de Monte Carlo con **10.000 extracciones estocásticas** por cada par mediador-desenlace.
* **Control de Falsos Descubrimientos (FDR):**
  * Corrección de Benjamini-Hochberg aplicada a nivel de bloque temporal (`childhood`, `adulthood`, `recent`) sobre los valores $p$ de los caminos $a$, $b$ y el indirecto $ab$ ($\alpha_{\text{FDR}} = 0.05$).
* **Diagnóstico de Multicolinealidad:**
  * Verificación automatizada del Factor de Inflación de la Varianza ($\text{VIF} < 5$) en cada regresión ajustada.

---

### 6. Visualización y Reporte Publicable (`src/visualization.py`)
Genera visualizaciones de alta resolución para la comunicación de resultados:
* **Heatmap Tri-Panel de Mediación:** Representación simultánea lado a lado de la matriz de efectos estandarizados:
  1. *Panel Izquierdo (Path $a$):* Adversidad $\rightarrow$ Volumetría cerebral.
  2. *Panel Central (Path $b$):* Volumetría cerebral $\rightarrow$ Dolor crónico.
  3. *Panel Derecho (Indirecto $ab$):* Magnitud del efecto mediado.
  * Generado independientemente para **Intensidad de Dolor** (`CP_Intensity_Z`) y **Extensión Espacial** (`n_regions_Z`).
* **Forest Plots:** Gráficos de bosque con estimaciones puntuales e intervalos de confianza al 95% para cada vía indirecta evaluada.

---

## Guía de Ejecución y Reproducibilidad

Para reproducir el flujo completo de análisis desde cero, asegúrese de contar con el entorno configurado (`pip install -r requirements.txt`) y ejecute la secuencia modular en orden:

```bash
# 1. Control de calidad e imputación múltiple (MICE m=10)
python -m src.preprocessing

# 2. Ingeniería de variables, QC de predictores y estandarización Z
python -m src.features

# 3. Modelado de mediación (Rubin pooling + Monte Carlo + FDR)
python -m src.modeling

# 4. Generación de tablas y figuras publicables
python -m src.visualization
```