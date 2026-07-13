# 🛡️ Denguard: Ecosistema Predictivo de Logística Farmacéutica de Última Milla
## Plataforma de Soporte a Decisiones Clínicas y Logísticas sobre el Ecosistema Colombiano de Datos Abiertos (Valle del Cauca • Colombia • 2026)

<p align="center">
  <img src="Denguardlogo.png" alt="Denguard Logo" width="550px">
</p>

<p align="center">
  <a href="https://observatory.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Desplegado en Streamlit"></a>
  <a href="https://colab.research.google.com/drive/1_ZHAxARnehdR7ifGCaTe-qMTHEg7ptrQ?usp=sharing"><img src="https://img.shields.io/badge/Google%20Colab-Training%20Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"></a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=for-the-badge" alt="Licencia: CC BY-NC-SA 4.0"></a>
  <img src="https://img.shields.io/badge/Audit-ALCOA%2B-blue?style=for-the-badge" alt="Estandár ALCOA+">
</p>

---

## 📌 1. Visión General y Arquitectura del Sistema

Denguard is una plataforma de software de grado de producción diseñada para resolver una de las fallas más críticas en la salud pública colombiana: el desabasto cíclico y reactivo de medicamentos e insumos esenciales durante los brotes de dengue. En lugar de operar bajo un esquema de reabastecimiento puramente empírico o reactivo (reaccionar cuando las urgencias hospitalarias ya están saturadas), Denguard fusiona la inteligencia epidemiológica cuantitativa con la ingeniería de la cadena de suministro de última milla, automatizando la toma de decisiones críticas para los 42 municipios del Valle del Cauca.

El sistema se fundamenta en un principio rector: el dato epidemiológico público debe transformarse inmediatamente en una orden de despacho farmacéutico parametrizada hacia cada punto de atención desde el centro de distribución centralizado SECCIONED.

---

## 🛑 2. Planteamiento del Problema Técnico y Logístico

En Colombia, las epidemias de dengue se abordan históricamente bajo esquemas de emergencia. Los hospitales y centros de atención primaria del Valle del Cauca reportan de forma obligatoria los casos al SIVIGILA (Sistema de Vigilancia en Salud Pública). Sin embargo, este flujo de datos padece de latencia burocrática y desarticulación operativa con los almacenes y operadores logísticos farmacéuticos.

El Patrón de Falla Estructural se compone de:
1. Pico Epidemiológico: Los casos aumentan de forma exponencial en municipios distantes o de alta endemicidad (ej. Buenaventura, Cartago, Tuluá).
2. Saturación Hospitalaria: El punto de atención agota su stock de seguridad de Acetaminofén 500mg (analgésico de primera línea libre de riesgo de sangrado) y Lactato de Ringer (solución cristaloide esencial para la reposición hídrica intravenosa y prevención del choque por dengue).
3. Compra de Pánico Reactiva: Al no prever la demanda de la semana epidemiológica entrante, las instituciones ejecutan compras directas bajo figuras de urgencia manifiesta, adquiriendo insumos a precios significativamente inflados respecto a los valores de referencia del SISMED, enfrentando sobrecostos logísticos de envío express y, en el peor de los escenarios, causando pérdidas humanas por retrasos en la fluidoterapia inicial.

---

## 🛠️ 3. Componentes Arquitectónicos de la Solución

### 3.1. Capa de Predicción Epidemiológica (Machine Learning)
El núcleo predictivo de Denguard está compuesto por un modelo matemático basado en el algoritmo Random Forest Regressor, entrenado y documentado exhaustivamente en el [Pipeline de Google Colab](https://colab.research.google.com/drive/1_ZHAxARnehdR7ifGCaTe-qMTHEg7ptrQ?usp=sharing). El entrenamiento se ejecutó sobre un dataset robusto de 11 años de registros históricos de SIVIGILA (2007–2018) provenientes del portal oficial de datos abiertos del Estado colombiano (datos.gov.co).

* Métricas de Rendimiento: El modelo alcanza un coeficiente de determinación R² = 0.928 en un esquema de holdout temporal estricto (datos de testeo correspondientes al año 2018 completo), con un Error Absoluto Medio (MAE) de 0.54 casos por semana por municipio.
* Ingeniería de Características (8 Features Fundamentales):
  1. `inercia_t1`: Carga epidemiológica real registrada en la semana t-1.
  2. `inercia_t2`: Carga epidemiológica real registrada en la semana t-2.
  3. `inercia_t3`: Carga epidemiológica real registrada en la semana t-3.
  4. `media_movil_4w`: Promedio móvil de las últimas 4 semanas para suavizar ruido estructural y capturar tendencias macro.
  5. `sin_semana`: Codificación cíclica estacional mediante la función seno de la semana epidemiológica: sin(2π · semana / 52).
  6. `cos_semana`: Codificación cíclica estacional mediante la función coseno de la semana epidemiológica: cos(2\pi · semana / 52).
  7. `perfil_historico_mean`: Promedio histórico absoluto de casos para el municipio específico objetivo.
  8. `perfil_historico_std`: Desviación estándar histórica de la carga epidemiológica del municipio, penalizando o ponderando la volatilidad endémica local.

Este enfoque de "Nowcasting" avanzado evita la dependencia de variables meteorológicas exógenas (que suelen sufrir de desalineación geográfica o latencia de sensores), apalancando la inercia del vector biológico y el histórico de contagios.

### 3.2. Motor Logístico de Última Milla (Operations Research)
Cada predicción semanal generada por el modelo predictivo se inyecta de forma síncrona en el motor analítico de inventarios, estructurado bajo el paradigma clásico de optimización de cadenas de suministro corporativas (Chopra & Meindl, 2016).

El sistema calcula dinámicamente las siguientes variables clave para cada uno de los 42 municipios:

1. Stock de Seguridad Dinámico (SS):
   SS = Z × σ_D × √LT
   Donde Z = 1.96 (Garantizando un Nivel de Servicio de Ciclo del 95% ante fluctuaciones), σ_D es la desviación estándar de la demanda (calculada a partir de las variaciones del modelo de machine learning) y LT (Lead Time) es el tiempo de tránsito específico en semanas desde el Hub Central SECCIONED hasta el centro hospitalario municipal, derivado de las matrices de conectividad e infraestructura vial de INVIAS e IGAC.

2. Punto de Reorden (ROP):
   ROP = (D_promedio × LT) + SS
   Establece el umbral de inventario físico por debajo del cual una orden se cataloga en estado crítico de desabastecimiento.

3. Cantidad Óptima de Pedido (Órdenes de Despacho):
   El motor contrasta el inventario disponible proyectado contra el ROP y calcula las unidades exactas de Acetaminofén 500mg (tabletas) y Lactato de Ringer (bolsas de 500ml) requeridas para restablecer la resiliencia clínica local.

4. Classification de Prioridad Logística:
   El motor automatiza el triaje logístico departamental en tres niveles de criticidad:
   * CRÍTICO: Inventario actual por debajo del stock de seguridad básico bajo el contexto de brote inminente. Requiere despacho prioritario escoltado o express en menos de 24 horas.
   * ALERTA: Inventario en zona de reorden. Despacho programado estándar dentro de la ventana de lead time (48-72 horas).
   * NORMAL: Niveles de existencias suficientes para cubrir la carga predictiva estacional.

### 3.3. Agente Conversacional Multi-Herramienta (Generative AI & Function Calling)
Para democratizar el acceso a la data técnica y agilizar la operación en centros de despacho hospitalarios, Denguard implementa un agente conversacional avanzado de Inteligencia Artificial utilizando Gemini (infraestructura API nativa).

Garantía de Cero Alucinación: El agente opera exclusivamente bajo un esquema estricto de Function Calling real. No tiene permitido inventar métricas, aproximar stock ni improvisar rutas. Cuando un operador realiza una consulta, el Modelo de Lenguaje (LLM) procesa la intención semántica, extrae las entidades requeridas y gatilla de forma mandatoria alguna de las siguientes 5 herramientas determinísticas escritas en código Python:
1. `obtener_prediccion_municipio(municipio)`: Retorna la carga exacta de dengue proyectada por el modelo de ML para la semana en curso.
2. `consultar_inventario_actual(municipio)`: Extrae en tiempo real las existencias físicas reportadas en el nodo local.
3. `listar_municipios_criticos()`: Devuelve el vector filtrado de localidades bajo triaje de urgencia máxima.
4. `calcular_orden_despacho(municipio)`: Ejecuta el motor matemático de Chopra & Meindl y retorna la cantidad de cajas de acetaminofén y bolsas de Lactato a despachar desde SECCIONED.
5. `obtener_metricas_auditoria_hash(municipio)`: Expone los metadatos de seguridad del dato para garantizar inmutabilidad.

---

## 📊 4. Consumo de Datos Abiertos e Integración de Fuentes

Denguard se acopla orgánicamente a la infraestructura del Ecosistema Colombiano de Datos Abiertos, realizando ingesta y mapeo bajo las siguientes directrices técnicas y regulatorias:

| Fuente Institucional | Recurso / Endpoint API | Propósito en el Ecosistema Denguard |
| :--- | :--- | :--- |
| SIVIGILA / MinSalud | resource/4hyg-wa9d (Socrata API) | Datos crudos históricos y microdatos anonimizados para entrenamiento offline y telemetría de nowcasting semanal. |
| IGAC / INVIAS | Infraestructura Geográfica Vial | Mapeo de distancias topológicas, estado de la malla vial y cálculo del parámetro dinámico de Lead Time (LT). |
| Ministerio de la Protección Social | Resolución 1403 de 2007 | Cumplimiento del marco regulatorio del Servicio Farmacéutico en Colombia, fijando las directrices técnicas de almacenamiento y distribución. |
| SISMED | Base de Precios de Referencia | Monitoreo de precios de adquisición regulados para el cálculo automatizado del costo de la orden de compra y cuantificación del ahorro preventivo. |

---

## 🔒 5. Integridad de Datos y Auditoría bajo Estándar ALCOA+

Dado que las decisiones automatizadas de Denguard impactan directamente la asignación de recursos públicos de salud y la vida de pacientes críticos, toda la tubería de datos (data pipeline) está blindada bajo el estándar de integridad internacional ALCOA+ (Attributable, Legible, Contemporaneous, Original, Accurate + Complete, Consistent, Enduring, Available):

* Trazabilidad Criptográfica de Fin a Fin: Cada vez que un dato ingresa desde la API pública de datos abiertos, el sistema genera de forma obligatoria un Hash MD5 único asociado al payload crudo de origen.
* Encadenamiento de Bloques de Auditoría: El hash acompaña el registro a través de la transformación de características, la ejecución de la inferencia en el modelo de Machine Learning, el cálculo de las ecuaciones logísticas en el motor de inventario, y queda estampado de forma permanente en el reporte final consultado por el Agente de IA. Esto garantiza que es matemáticamente imposible alterar una predicción o falsear una alerta logística sin romper el registro de auditoría del sistema.

---

## 📈 6. Modelo de Impacto Económico y Eficiencia Operativa

Utilizando las matrices de precios oficiales indexadas por el SISMED, Denguard sustituye la compra reactiva por la adquisición planificada.

El Algoritmo de Costeo Compara:
* Costo Reactivo (C_R): Precio de compra en contingencia + flete express prioritario + penalizaciones por rotura de stock.
* Costo Preventivo (C_P): Precio contratado bajo volumen institucional + logística integrada optimizada por SECCIONED.

Cuantificación Financiera del Proyecto:
El sistema genera una optimización financiera media de $3,500,000 COP por municipio por semana durante picos epidemiológicos. Extrapolado de manera conservadora a los 42 municipios del Valle del Cauca a lo largo de las 12 semanas de la temporada alta de transmisión de dengue identificada en la región, Denguard proyecta un ahorro neto de:
Ahorro Anual = 42 × 12 × $3,500,000 = $1,764,000,000 COP

Este capital de alta eficiencia fiscal puede ser reinyectado directamente por la Secretaría de Salud Departamental en estrategias de erradicación biológica del vector (Aedes aegypti), campañas de educación comunitaria o mejora de la infraestructura hospitalaria de atención primaria.

---

## 🚀 7. Escalabilidad Técnica y Pipeline de Re-Entrenamiento

La arquitectura de Denguard fue concebida bajo patrones de diseño desacoplados, lo que permite tres vectores de escalabilidad limpia:

1. Escalabilidad Geográfica: Para expandir el sistema fuera del Valle del Cauca (por ejemplo, a departamentos de alta complejidad como Antioquia, Santander o Meta), solo es requerido modificar el parámetro geográfico en las consultas API de SIVIGILA y actualizar la matriz vial local. El core del modelo de Machine Learning y el motor logístico asimilan los nuevos datos de forma nativa sin refactorización de código.
2. Pipeline de Re-Entrenamiento Continuo: Diseñado para la era post-pandemia. A medida que el ecosistema de datos abiertos consolide registros epidemiológicos con coberturas estandarizadas óptimas (>= 70%), el sistema activa un script automatizado que aplica una ventana deslizante (sliding window), incorporando los años recientes al dataset de entrenamiento y re-evaluando las métricas de holdout de manera autónoma para prevenir la degradación del modelo (data drift).
3. Flexibilidad de Catálogo Farmacéutico: El motor logístico es completamente paramétrico. Si la autoridad sanitaria decide incluir nuevos medicamentos al protocolo de manejo del dengue (ej. suero de rehidratación oral de densidades específicas o analgésicos alternativos regulados), basta con añadir un nuevo objeto estructurado dentro del archivo de configuración logistica_params.json especificando el lead time, stock base y costo unitario SISMED.

---

## 💻 8. Interfaces de Usuario de la Aplicación

El ecosistema Denguard segmenta su visualización y analítica según el perfil del usuario final, optimizando la experiencia de usuario (UX/UI):
* Vista General (Ciudadana): Panel interactivo simplificado con mapas de calor coropléticos del Valle del Cauca, permitiendo a la sociedad civil y periodistas monitorear la transparencia del abastecimiento.
* Dashboard Técnico de Inventarios: Diseñado para los directores de compras hospitalarias y gerentes del centro de distribución SECCIONED. Expone tablas dinámicas con filtros de criticidad, curvas de ROP y sugeridos de despacho automáticos descargables en formatos estructurados.
* Panel de Auditoría ALCOA+: Consola de visualización de hashes MD5 y trazas criptográficas orientada a organismos de control y revisores fiscales de salud pública.
* Agente conversacional IA: Interfaz de chat integrada en la esquina operativa para agilizar consultas críticas mediante procesamiento de lenguaje natural en tiempo real.

---

## 📂 9. Estructura del Repositorio

```text
denguard-platform/
│
├── .streamlit/
│   └── config.toml             # Configuración visual de la interfaz Streamlit
│
├── assets/
│   └── Denguardlogo.png        # Identidad visual de la plataforma
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
