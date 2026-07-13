# 🛡️ Denguard: Ecosistema Predictivo de Logística Farmacéutica de Última Milla
## Plataforma de Soporte a Decisiones Clínicas y Logísticas sobre el Ecosistema Colombiano de Datos Abiertos (Valle del Cauca • Colombia • 2026)

<p align="center">
  <img src="Denguardlogo.png" alt="Denguard Logo" width="550px">
</p>

<p align="center">
  <a href="https://observatory.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Desplegado en Streamlit"></a>
  <a href="https://colab.research.google.com/drive/1_ZHAxARnehdR7ifGCaTe-qMTHEg7ptrQ?usp=sharing"><img src="https://img.shields.io/badge/Google%20Colab-Training%20Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="Licencia: CC BY-NC-SA 4.0"></a>
  <img src="https://img.shields.io/badge/Audit-ALCOA%2B-blue?style=for-the-badge" alt="Estándar ALCOA+">
</p>

---

## 📌 1. Visión General y Arquitectura del Sistema

Denguard es una plataforma de software de grado de producción diseñada para resolver una de las fallas más críticas en la salud pública colombiana: el desabasto cíclico y reactivo de medicamentos e insumos esenciales durante los brotes de dengue. En lugar de operar bajo un esquema de reabastecimiento puramente empírico o reactivo (reaccionar cuando las urgencias hospitalarias ya están saturadas), Denguard fusiona la inteligencia epidemiológica cuantitativa con la ingeniería de la cadena de suministro de última milla, automatizando la toma de decisiones críticas para los 42 municipios del Valle del Cauca.

El sistema se fundamenta en un principio rector: el dato epidemiológico público debe transformarse inmediatamente en una orden de despacho farmacéutico parametrizada hacia cada punto de atención desde el centro de distribución centralizado SECCIONED.

---

## 🛑 2. Problema Abordado

En Colombia, las epidemias de dengue se abordan históricamente bajo esquemas de emergencia. Los hospitales y centros de atención primaria del Valle del Cauca reportan de forma obligatoria los casos al SIVIGILA (Sistema de Vigilancia en Salud Pública). Sin embargo, este flujo de datos padece de latencia burocrática y desarticulación operativa con los almacenes y operadores logísticos farmacéuticos.

El Patrón de Falla Estructural se compone de:
1. **Pico Epidemiológico:** Los casos aumentan de forma exponencial en municipios distantes o de alta endemicidad (ej. Buenaventura, Cartago, Tuluá).
2. **Saturación Hospitalaria:** El punto de atención agota su stock de seguridad de Acetaminofén 500mg (analgésico de primera línea libre de riesgo de sangrado) y Lactato de Ringer (solución cristaloide esencial para la reposición hídrica intravenosa y prevención del choque por dengue).
3. **Compra de Pánico Reactiva:** Al no prever la demanda de la semana epidemiológica entrante, las instituciones ejecutan compras directas bajo figuras de urgencia manifiesta, adquiriendo insumos a precios significativamente inflados respecto a los valores de referencia del SISMED, enfrentando sobrecostos logísticos de envío express y, en el peor de los escenarios, causando pérdidas humanas por retrasos en la fluidoterapia inicial.

---

## 🎯 3. Justificación (Valor Público o Empresarial)

### Valor Público:
* **Garantía del Derecho a la Salud:** Mitiga la mortalidad por dengue grave al asegurar que los insumos de primera línea para soporte hemodinámico (Lactato de Ringer) y manejo sintomático seguro (Acetaminofén) estén físicamente en el hospital *antes* de que ingresen los pacientes.
* **Transparencia en la Gestión:** Al utilizar el ecosistema de datos abiertos y auditoría ALCOA+, se elimina la discrecionalidad y se previene el riesgo de corrupción asociado a la contratación directa por urgencia manifiesta.

### Valor Empresarial/Operativo:
* **Eficiencia de la Cadena de Suministro:** Transforma un modelo *Push* (empujar inventario basado en promedios históricos obsoletos) en un modelo *Pull* dinámico guiado por la demanda epidemiológica proyectada. Esto reduce los costos de almacenamiento por exceso de stock en zonas de baja transmisión y optimiza los fletes de última milla desde el hub central SECCIONED.

---

## 📊 4. Ingesta de Datos y Datasets Utilizados

### 4.1. Dataset Utilizado (Mínimo uno de datos.gov.co)
* **Nombre de la Fuente:** SIVIGILA - Instituto Nacional de Salud (INS).
* **Identificador en el Portal de Datos Abiertos:** `resource/4hyg-wa9d` (Socrata API).
* **Descripción:** Microdatos históricos anonimizados de notificación obligatoria para Dengue en Colombia.

### 4.2. Datasets Utilizados Externos
* **IGAC / INVIAS:** Matrices oficiales de conectividad terrestre, distancias intermunicipales y estados de la red vial nacional para el Valle del Cauca.
* **Minsalud (Resolución 1403 de 2007):** Marco regulatorio que establece los lineamientos técnicos para el Modelo de Gestión del Servicio Farmacéutico.
* **SISMED (Sistema de Información de Precios de Medicamentos):** Base de datos consolidada con los precios de referencia institucionales y regulados para la compra de insumos farmacéuticos en Colombia.

### 4.3. Cantidad de Dataset Utilizado
* **Ventana Temporal:** 11 años de registros continuos (2007 - 2018).
* **Volumen de Datos:** Ingesta total de **+500,000 registros epidemiológicos de notificación individual**, consolidados y agregados en series de tiempo semanales para cada uno de los 42 municipios del departamento, lo que equivale a un total de 26,208 vectores de series temporales (`42 municipios x 52 semanas x 12 años`).

---

## 🛠️ 5. Componentes Técnicos y Modelado

### 5.1. Variables Seleccionadas
El modelo implementa ingeniería de características avanzadas basada en 8 variables (*features*) predictivas clave:
1. `inercia_t1`: Carga epidemiológica real registrada en la semana inmediata anterior ($t-1$).
2. `inercia_t2`: Carga epidemiológica real en la semana $t-2$.
3. `inercia_t3`: Carga epidemiológica real en la semana $t-3$.
4. `media_movil_4w`: Promedio móvil de las últimas 4 semanas para capturar tendencias macro y suavizar ruidos.
5. `sin_semana`: Transformación cíclica de la estacionalidad temporal mediante la función seno: $\sin(2\pi \cdot \text{semana} / 52)$.
6. `cos_semana`: Transformación cíclica de la estacionalidad temporal mediante la función coseno: $\cos(2\pi \cdot \text{semana} / 52)$.
7. `perfil_historico_mean`: Carga promedio histórica absoluta del municipio objetivo.
8. `perfil_historico_std`: Desviación estándar histórica del municipio, capturando la volatilidad endémica local.

### 5.2. Tipo de Análisis
Análisis **Predictivo** orientado al modelado numérico continuo de series de tiempo complejas independientes para múltiples locaciones geográficas de manera simultánea (*multi-site nowcasting*).

### 5.3. Modelo Utilizado
**Árbol de Decisión / Ensamble (Random Forest Regressor):** Se seleccionó este algoritmo sobre aproximaciones lineales tradicionales debido a su capacidad robusta para capturar relaciones no lineales y patrones estacionales fractales en los picos epidemiológicos, reduciendo drásticamente el riesgo de sobreajuste (*overfitting*) en datos con alta varianza local. El entrenamiento y validación se documentan en el [Pipeline de Google Colab](https://colab.research.google.com/drive/1_ZHAxARnehdR7ifGCaTe-qMTHEg7ptrQ?usp=sharing).

---

## 🚀 6. Resultados Clave e Interpretación

### Resultados Clave:
* **Coeficiente de Determinación ($R^2$):** **0.928** en un esquema de *holdout temporal estricto* utilizando el año 2018 completo como grupo de testeo independiente.
* **Error Absoluto Medio (MAE):** **0.54 casos por semana por municipio**, demostrando una precisión extrema para escenarios de atención primaria.

### Interpretación del Modelo:
El modelo demuestra que la **inercia epidemiológica de corto plazo (`inercia_t1` e `inercia_t2`)** aporta más del 65% de la importancia de las características (*feature importance*), lo que corrobora la naturaleza biológica de transmisión del virus: la densidad actual de vectores infectados determina directamente el volumen de contagios de la semana siguiente. Las variables cíclicas (`sin_semana`/`cos_semana`) corrigen los incrementos estructurales durante las temporadas de lluvias locales.

---

## 📐 7. Motor Logístico de Última Milla (Operations Research)

Cada predicción semanal se procesa síncronamente mediante lógica determinística de optimización de inventarios (Chopra & Meindl, 2016):

1. **Stock de Seguridad Dinámico ($SS$):**
   $$SS = Z \times \sigma_D \times \sqrt{LT}$$
   Donde $Z = 1.96$ (Nivel de Servicio del 95%), $\sigma_D$ es la variabilidad del modelo predictivo y $LT$ es el *Lead Time* derivado de INVIAS.
2. **Punto de Reorden ($ROP$):**
   $$ROP = (D_{promedio} \times LT) + SS$$
3. **Triaje Logístico:** Clasifica de manera automática los municipios en **CRÍTICO** (inventario por debajo del SS), **ALERTA** (inventario en zona de reorden) y **NORMAL**.

---

## 🤖 8. Agente Conversacional Multi-Herramienta (Gemini)

Denguard implementa un agente conversacional avanzado utilizando la infraestructura de **Gemini** mediante **Function Calling real**. El agente tiene prohibido generar respuestas libres o alucinar datos. Mapea la consulta en lenguaje natural del usuario y ejecuta forzosamente una de las 5 herramientas conectadas al core analítico:
1. `obtener_prediccion_municipio(municipio)`
2. `consultar_inventario_actual(municipio)`
3. `listar_municipios_criticos()`
4. `calcular_orden_despacho(municipio)`
5. `obtener_metricas_auditoria_hash(municipio)`

---

## 🔒 9. Integridad de Datos y Auditoría bajo Estándar ALCOA+

Para cumplir con el estándar de integridad internacional **ALCOA+**, el flujo de datos cuenta con:
* **Trazabilidad Criptográfica:** Cada payload crudo descargado desde `datos.gov.co` genera un **Hash MD5 único**.
* **Inmutabilidad:** El hash acompaña el ciclo de vida del dato (transformación -> inferencia -> orden logística -> respuesta del agente IA), garantizando que las órdenes de despacho no puedan ser alteradas externamente.

---

## 📈 10. Impacto Potencial

Utilizando la matriz de costos indexada por el **SISMED**, Denguard modela el ahorro al sustituir compras reactivas de emergencia por despachos preventivos planificados.

* **Eficiencia Financiera Media:** Optimización de **\$3,500,000 COP por municipio por semana** en periodos de brote.
* **Impacto Económico Regional:** Extrapolado a los 42 municipios del Valle del Cauca durante las 12 semanas críticas de la temporada alta de dengue, el sistema proyecta un ahorro neto estructural de:
$$\text{Ahorro Anual} = 42 \times 12 \times \$3,500,000 = \mathbf{\$1,764,000,000 \text{ COP}}$$

Este capital liberado representa eficiencia fiscal directa de alto impacto, permitiendo su reinversión por la autoridad sanitaria departamental en contención biológica del vector y equipamiento de la red hospitalaria de baja complejidad.

---

## 💻 11. Interfaces de Usuario de la Aplicación

El sistema segmenta su acceso a través de la aplicación en Streamlit:
* **Vista General (Ciudadana):** Mapas coropléticos interactivos de libre acceso.
* **Dashboard Técnico de Inventarios:** Tablas de control de ROP y sugeridos automáticos de despacho para el hub **SECCIONED**.
* **Panel de Auditoría ALCOA+:** Consola de verificación criptográfica de hashes MD5.
* **Agente conversacional IA:** Ventana flotante interactiva para consultas rápidas.

---

## 📂 12. Estructura del Repositorio

```text
denguard-platform/
│
├── .streamlit/
│   └── config.toml             # Configuración visual de la interfaz Streamlit
│
├── assets/
│   └── Denguardlogo.png        # Identidad visual de la plataforma (Logo en la raíz)
│
├── data/
│   ├── municipios_valle.json   # Base cartográfica y distancias IGAC
│   └── logistica_params.json   # Parámetros logísticos (Lead times, Z, Costos)
│
├── src/
│   ├── __init__.py
│   ├── model_inference.py      # Pipeline de ejecución de Random Forest
│   ├── logistic_engine.py      # Implementación matemática de Chopra & Meindl
│   ├── gemini_agent.py         # Configuración del LLM y binding de Function Calling
│   └── audit_logger.py         # Generador de hashes MD5 bajo estándar ALCOA+
│
├── app.py                      # Punto de entrada principal (Streamlit App)
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación técnica principal
