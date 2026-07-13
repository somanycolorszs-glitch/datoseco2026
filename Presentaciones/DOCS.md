# ==========================================
# ARCHIVO 1: docs/planteamiento_problema.md
# ==========================================
# El Problema abordado, Justificación e Impacto de Denguard
---

## 🛑 Planteamiento del Problema

En el ecosistema de salud del Valle del Cauca, los brotes epidémicos de dengue (*Aedes aegypti*) imponen una presión cíclica severa sobre la red hospitalaria de sus 42 municipios. El problema estructural identificado no radica en la escasez de recursos financieros absolutos, sino en la **naturaleza puramente reactiva** de la cadena de suministro farmacéutico institucional. 

El flujo de información epidemiológica tradicional padece de una latencia burocrática crítica: los casos son notificados al SIVIGILA, procesados centralizadamente y consolidados semanas después. Para cuando la alerta llega a los comités de compras hospitalarias, el pico epidemiológico ya ha saturado las salas de urgencias primarias. Esto provoca desabasto crítico de insumos de primera línea:
1. **Acetaminofén 500mg:** Analgésico y antipirético mandatorio (los AINEs como el ibuprofeno están contraindicados por riesgo de sangrado).
2. **Lactato de Ringer:** Solución cristaloide indispensable para la reposición hídrica intravenosa y prevención del choque por dengue.

Ante el desabasto, los hospitales ejecutan compras directas de pánico bajo figuras de urgencia manifiesta, adquiriendo insumos con sobrecostos logísticos masivos, pagando precios muy superiores a los valores de referencia del SISMED y poniendo en riesgo directo la vida de pacientes críticos por retrasos en la fluidoterapia inicial.

---

## 🎯 Justificación del Valor Público y Empresarial

### Valor Público (Impacto Social y Sanitario):
* **Garantía del Derecho Fundamental a la Salud:** Asegura la resiliencia clínica de los puntos de atención periféricos del Valle del Cauca, garantizando que el insumo básico esté físicamente disponible en el inventario local *antes* de la llegada masiva de pacientes.
* **Transparencia y Eficiencia Fiscal:** Al basar las compras en un modelo analítico reproducible conectado al ecosistema de datos abiertos, se elimina la discrecionalidad en la contratación de emergencia, mitigando riesgos de corrupción.

### Valor Empresarial (Eficiencia Operativa):
* **Optimización de Cadena de Suministro (Logística Pull):** Transforma el almacenamiento de medicamentos de un modelo rígido basado en promedios históricos a un ecosistema ágil guiado por la demanda en tiempo real.
* **Reducción de Costos de Inventario y Distribución:** Minimiza el capital inmovilizado por exceso de stock en municipios sin transmisión activa y optimiza las rutas de última milla desde el nodo central de distribución SECCIONED.

---

# ==========================================
# ARCHIVO 2: docs/marco_metodologico.md
# ==========================================
# Metodología de Análisis Predictivo y Operaciones

El core científico de Denguard fusiona el aprendizaje automático para series de tiempo temporales con la investigación de operaciones para la optimización de inventarios farmacéuticos.

```text
+-----------------------+      +-------------------------+      +-----------------------+
|  Ingesta SIVIGILA API | ---> | Random Forest Regressor | ---> | Motor Chopra & Meindl |
|   (Data Abierta)      |      |   (R² = 0.928 / MAE=0.54|      |   (SS, ROP, Órdenes)  |
+-----------------------+      +-------------------------+      +-----------------------+
```

## 🧠 Algoritmo de Modelado Predictivo

El sistema descarta los modelos lineales tradicionales debido a la naturaleza altamente no lineal y oscilatoria de los picos de contagio de dengue. En su lugar, implementa un ensamble de **Random Forest Regressor** (Regressor de Bosques Aleatorios). El algoritmo construye múltiples árboles de decisión durante el entrenamiento y promedia sus resultados para ofrecer una estimación robusta del número de casos numéricos continuos esperados para la semana epidemiológica entrante ($t+1$) de forma independiente en cada municipio.

### Ingeniería de Características (Features):
El modelo fundamenta su capacidad predictiva en un vector de 8 variables clave:
1. `inercia_t1`, `inercia_t2`, `inercia_t3`: Capturan la dinámica de transmisión activa y la inercia del vector biológico de las últimas 3 semanas.
2. `media_movil_4w`: Actúa como un filtro de suavizado para identificar tendencias macro intermensuales.
3. `sin_semana` y `cos_semana`: Codificación cíclica matemática de la semana epidemiológica, permitiendo al modelo asimilar patrones de estacionalidad climática (temporadas de lluvias y sequías estructurales en el Valle del Cauca) sin depender de sensores meteorológicos externos propensos a fallas.
4. `perfil_historico_mean` y `perfil_historico_std`: Proveen el contexto endémico histórico específico de cada uno de los 42 municipios, parametrizando la línea base epidemiológica local.

---

## 📐 Motor de Optimización de Inventarios (Última Milla)

Una vez que el modelo predictivo estima la carga de casos para la semana epidemiológica entrante, el motor logístico calcula en tiempo real los requerimientos paramétricos de reabastecimiento farmacéutico basados estrictamente en la metodología de gestión de inventarios de **Chopra & Meindl (2016)**:

### 1. Stock de Seguridad Dinámico ($SS$)
El Stock de Seguridad protege al sistema contra la variabilidad latente de la demanda (fluctuaciones en el modelo predictivo) y los retrasos logísticos inesperados en la infraestructura de transporte terrestre de última milla. Se modela mediante la siguiente ecuación matemática:

$$SS = Z \times \sigma_D \times \sqrt{LT}$$

Donde:
* **$Z = 1.96$**: Factor de distribución normal estándar que garantiza un Nivel de Servicio de Ciclo ($CSL$) del **95%** en los hospitales municipales ante la incertidumbre epidemiológica de picos inminentes.
* **$\sigma_D$**: Desviación estándar del error de la demanda, calculada de manera dinámica a partir del error de holdout del modelo de Machine Learning.
* **$LT$ (Lead Time)**: Tiempo de ciclo de reabastecimiento expresado en fracciones de semana. Representa el tiempo de tránsito físico que le toma al centro de distribución centralizado **SECCIONED** despachar y posicionar los insumos en el punto de atención municipal, utilizando las matrices viales indexadas por INVIAS e IGAC.

### 2. Punto de Reorden ($ROP$)
Establece el nivel de inventario físico crítico bajo el cual la plataforma activa automáticamente una alerta de abastecimiento priorizada. Se calcula mediante la ecuación:

$$ROP = (D_{promedio} \times LT) + SS$$

Donde:
* **$D_{promedio}$**: Demanda promedio semanal esperada en unidades de insumo clínico (derivada de la tasa de conversión paramétrica por caso de dengue predicho).
* **$LT$**: Lead Time o tiempo de espera del municipio.
* **$SS$**: Stock de Seguridad Dinámico calculado previamente.

### 3. Cantidad Óptima de Despacho e Inventario Neto
El motor evalúa de forma asíncrona la brecha entre el inventario físico disponible en tiempo real frente al $ROP$ calculado. Si el stock cae por debajo de este límite, el algoritmo genera de manera automatizada la cantidad exacta a despachar de **Acetaminofén 500mg** (tabletas) y **Lactato de Ringer** (bolsas de 500ml) requeridos para restablecer la resiliencia clínica local y evitar roturas de stock.

---

# ==========================================
# ARCHIVO 3: docs/fuentes_datos.md
# ==========================================
# Origen e Integración del Ecosistema de Datos Abiertos (Horizonte 2026)

Denguard no almacena datos de forma aislada; opera de manera integrada sobre el ecosistema de datos gubernamentales del Estado colombiano.

## 🏢 Fuentes de Datos Utilizadas

1. **SIVIGILA (Instituto Nacional de Salud / Ministerio de Salud):**
   * **Endpoint API:** `https://datos.gov.co/resource/4hyg-wa9d`
   * **Tipo de Acceso:** API REST vía Socrata (Open Data Protocol).
   * **Datos Obtenidos:** Microdatos anonimizados de las fichas epidemiológicas de notificación de dengue (código INS: 210, 220).

2. **Instituto Geográfico Agustín Codazzi (IGAC) / INVIAS:**
   * **Datos Obtenidos:** Matrices topológicas de conectividad vial regional, distancias físicas terrestres en kilómetros y coeficientes de penalización por estado de la malla vial para el departamento del Valle del Cauca. Utilizado para fijar los *Lead Times* ($LT$).

3. **SISMED (Sistema de Información de Precios de Medicamentos):**
   * **Datos Obtenidos:** Precios máximos de venta institucional y precios de referencia regulados para el canal institucional colombiano, permitiendo calcular el costo exacto de adquisición preventiva de los medicamentos esenciales.

---

## 🔒 Estrategia de Auditoría e Integridad ALCOA+

Dado que las salidas del sistema guían la destinación de recursos farmacéuticos públicos, el flujo de datos implementa las directrices de integridad **ALCOA+**:

* **Trazabilidad de Hash MD5:** Cada vez que el sistema realiza una petición automatizada a la API de `datos.gov.co`, calcula de forma inmediata un hash MD5 sobre el JSON crudo de respuesta.
* **Encadenamiento de Auditoría:** Este identificador hash se estampa en la base de datos interna y viaja a través de todas las transformaciones matemáticas, de modo que cualquier consulta realizada en el Agente conversacional de IA expone la huella digital criptográfica del dato original, garantizando auditorías transparentes e inmutabilidad absoluta de la cadena de decisión clínica.

---

# ==========================================
# ARCHIVO 4: docs/diccionario_datos.md
# ==========================================
# Especificación de Variables del Repositorio `somanycolorszs-glitch/datoseco2026`

Este diccionario define los atributos técnicos procesados en el pipeline de datos y consumidos por el motor logístico y el agente conversacional.

| Nombre de Variable | Tipo de Dato | Unidad / Formato | Descripción | Fuente de Origen |
| :--- | :--- | :--- | :--- | :--- |
| `codigo_municipio` | String / Int | Código DIVIPOLA | Identificador numérico oficial del municipio (ej: 76001 para Cali, 76109 para Buenaventura). | datos.gov.co |
| `semana_epidemiologica`| Integer | 1 a 53 | Número de la semana epidemiológica del año según calendario epidemiológico estandarizado. | datos.gov.co |
| `casos_reales` | Integer | Conteos absolutos | Cantidad de casos reales confirmados de dengue notificados por las UPGD a la plataforma SIVIGILA. | datos.gov.co |
| `casos_predichos` | Float | Conteos estimados | Carga de dengue proyectada por el modelo Random Forest para la semana $t+1$. | src/model_inference.py |
| `inercia_t1` | Integer | Conteos absolutos | Casos reales registrados en la semana epidemiológica anterior ($t-1$). | Variable calculada |
| `media_movil_4w` | Float | Promedio móvil | Media aritmética de casos reales de las últimas 4 semanas consecutivas. | Variable calculada |
| `lead_time_weeks` | Float | Semanas | Tiempo estimado de tránsito terrestre desde el almacén central SECCIONED al municipio. | IGAC / INVIAS |
| `stock_seguridad_aceta` | Integer | Tabletas | Cantidad mínima de seguridad calculada para Acetaminofén 500mg. | src/logistic_engine.py |
| `stock_seguridad_ringer`| Integer | Bolsas 500ml | Cantidad mínima de seguridad calculada para Lactato de Ringer. | src/logistic_engine.py |
| `triaje_logistico` | String | CRÍTICO / ALERTA / NORMAL | Clasificación operativa de urgencia de despacho basada en la relación entre el inventario físico y el ROP. | src/logistic_engine.py |
| `fuente_payload_hash` | String | MD5 (Hexadecimal) | Huella digital criptográfica del paquete de datos descargado de la API pública para auditoría ALCOA+. | src/audit_logger.py |

---

# ==========================================
# ARCHIVO 5: docs/architecture.md
# ==========================================
# Especificación del Pipeline Técnico de Producción

La arquitectura de Denguard está diseñada para operar de forma desacoplada y asíncrona, garantizando alta escalabilidad y estabilidad informática.

```text
+-----------------------+      +-------------------------+      +-----------------------+
|  Streamlit UI (App)   | <--> |   Gemini IA Agent       | <--> | Function Calling      |
|                       |      | (Function Calling Core) |      | (Scripts Python Intern)|
+-----------------------+      +-------------------------+      +-----------------------+
           ^                                                                |
           |                                                                v
           +-------------------- Consume Datos Estructurados <---------------+
```

## ⚙️ El Pipeline en 4 Etapas Nativas

1. **Ingesta y Sanitización (ETL):** El script `src/audit_logger.py` consulta de forma programada los endpoints Socrata de SIVIGILA. Descarga los registros del Valle del Cauca, valida la integridad estructural, calcula el hash MD5 de auditoría y almacena el archivo plano sanitizado en la ruta `data/`.
2. **Inferencia Predictiva (Machine Learning Core):** El componente `src/model_inference.py` levanta el modelo serializado de Random Forest entrenado en Google Colab. Toma los últimos datos históricos agregados por municipio, genera las rezagadas (`inercia_t1`, etc.) e inyecta el vector a la función predictiva, emitiendo la proyección de casos clínicos para los 42 municipios de forma simultánea.
3. **Traducción Operativa (Operations Research Engine):** Las proyecciones numéricas son capturadas por `src/logistic_engine.py`. Este módulo lee los parámetros institucionales de `data/logistica_params.json` (niveles Z, costos SISMED, lead times viales) y ejecuta de forma determinística las ecuaciones de reabastecimiento de Chopra & Meindl, generando las cantidades requeridas de Acetaminofén y Lactato de Ringer, ordenadas bajo un esquema de priorización por criticidad.
4. **Capa Conversacional Inteligente (Gemini Chat Core):** La interfaz de usuario en `app.py` expone un componente de chat en vivo configurado en `src/gemini_agent.py`. Cuando un operador de despacho de SECCIONED escribe una duda, la API de Gemini evalúa los parámetros sintácticos mediante **Function Calling**. En lugar de generar texto libre, el modelo invoca las funciones de los scripts de inferencia o logística, formateando los arrays de datos retornados en lenguaje natural conversacional fluido, preciso y completamente auditable.

---

# ==========================================
# ARCHIVO 6: docs/conclusiones.md
# ==========================================
# Resultados de Rendimiento Técnico, Viabilidad e Impacto Esperado

Tras el desarrollo, simulación histórica y pruebas del ecosistema integrado de Denguard enfocado en la red hospitalaria del Valle del Cauca para el horizonte 2026, se consolidan las siguientes conclusiones de ingeniería:

## 📈 Conclusiones del Rendimiento Analítico
* **Alta Precisión Epidemiológica:** El modelo de Random Forest Regressor demostró una capacidad sobresaliente de ajuste con un coeficiente $R^2 = 0.928$ en el set de holdout. Esto valida que la combinación de variables cíclicas estacionales con la inercia interna de contagios es suficiente para predecir brotes locales sin incurrir en la complejidad de capturar variables climáticas exógenas con alta tasa de desalineación geográfica.
* **Margen de Error Tolerable:** El Error Absoluto Medio (MAE) de 0.54 casos por semana por municipio garantiza que el motor logístico opere con desviaciones mínimas, permitiendo calibrar stocks de seguridad sumamente ajustados y eficientes, evitando el desperdicio de insumos farmacéuticos por vencimiento en estantería.

---

## 🚚 Conclusiones de la Operación Logística y Fiscal
* **Sustitución Eficiente del Modelo de Suministro:** Se comprueba la viabilidad técnica de automatizar la generación de órdenes de compra parametrizadas institucionales a partir de variables clínico-epidemiológicas. El triaje automatizado (Crítico, Alerta, Normal) agiliza la toma de decisiones críticas en el centro de distribución centralizado SECCIONED.
* **Sostenibilidad Financiera Basada en Eficiencia:** Al indexar los costos oficiales regulados del SISMED, las simulaciones arrojan un ahorro neto potencial para el Valle del Cauca de **\$1,764,000,000 COP anuales**. Este indicador demuestra que la incorporación de ciencia de datos en la salud pública genera eficiencias fiscales masivas y medibles, liberando capital del erario para ser reinvertido en programas preventivos de erradicación biológica del mosquito vector.
* **Mitigación del Riesgo de Alucinación de la IA:** La implementación mandatoria de *Function Calling real* sobre modelos Gemini demuestra que es de alta viabilidad incorporar agentes de IA generativa en entornos de alta responsabilidad (salud y finanzas públicas) de forma segura, limitando su rol estrictamente a la traducción semántica e interfaces conversacionales estructuradas bajo el estándar de integridad ALCOA+.

---

# ==========================================
# ARCHIVO 7: docs/validation_guide.md
# ==========================================
# Guía de Validación y Pruebas del Sistema de Grado de Producción

Esta guía contiene los pasos técnicos necesarios para validar la correcta instalación, ejecución e integridad de los componentes del repositorio `somanycolorszs-glitch/datoseco2026`.

## 🧪 Pruebas Unitarias y de Integración

### 1. Validación del Pipeline de Datos e Integridad (ALCOA+)
Para comprobar que el pipeline de datos descarga, limpia y genera de forma correcta los registros de auditoría criptográfica, ejecuta el módulo de registro de forma aislada en tu terminal:

```bash
python -m src.audit_logger
```
* **Resultado Esperado:** La consola debe retornar un mensaje de éxito indicando la cantidad de registros ingestados y el string hexadecimal correspondiente al **Hash MD5 generado** para el payload de SIVIGILA. Verifica que se haya creado el archivo plano dentro de la ruta `data/`.

### 2. Validación del Core Predictivo (Inferencia ML)
Para comprobar la carga del modelo entrenado y la generación de proyecciones, ejecuta el módulo de machine learning:

```bash
python -m src.model_inference
```
* **Resultado Esperado:** El script debe generar un dataframe interno sin errores de dimensiones ni valores nulos (`NaN`), arrojando el vector estructurado de `casos_predichos` para los 42 municipios del Valle del Cauca.

### 3. Validación del Motor de Logística (Chopra & Meindl)
Corre el script logístico para comprobar el cálculo determinístico de inventarios institucionales:

```bash
python -m src.logistic_engine
```
* **Resultado Esperado:** La consola debe imprimir la lista de municipios priorizados, exponiendo el Stock de Seguridad ($SS$), Punto de Reorden ($ROP$) y las cantidades exactas a pedir de Acetaminofén y Lactato de Ringer para los nodos en estado **CRÍTICO** o **ALERTA**.

### 4. Prueba de Integración del Agente de IA (Function Calling)
Para validar que el agente conversacional Gemini está invocando de forma real las funciones lógicas y no alucinando las métricas, ejecuta el script de interacción con la API:

```bash
python -m src.gemini_agent
```
* **Paso de Prueba:** Inyecta en la consola la pregunta de prueba: *¿Cuál es el estado y despacho sugerido para el municipio de Buenaventura?*
* **Resultado Esperado:** El log de la consola debe mostrar la interceptación de la llamada, la llamada nativa a `calcular_orden_despacho(municipio='Buenaventura')`, el retorno de los datos numéricos precisos del motor de operaciones y la estructuración de la respuesta final en lenguaje natural del chat, incluyendo la traza del hash MD5.
