import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de página
st.set_page_config(page_title="Data Sentinel - Última Milla", layout="wide")

st.title("🛡️ Data Sentinel: Logística Farmacéutica de Última Milla")
st.markdown("Ecosistema Predictivo (Spatial-Aware) para la prevención de desabastecimiento hospitalario focalizado.")

# 1. Carga del Modelo
@st.cache_resource
def load_model():
    return joblib.load('modelo_municipal_v2.pkl')

try:
    modelo = load_model()
    st.success("✅ Motor predictivo ALCOA+ en línea. Granularidad: Top 5 Municipios Críticos.")
except FileNotFoundError:
    st.error("⚠️ No se encontró 'modelo_municipal_v2.pkl'. Súbelo a tu repositorio.")
    st.stop()

# --- 2. PANEL LATERAL (ENTRADAS DEL USUARIO) ---
st.sidebar.header("📍 Parámetros Locales")
municipios_validos = ['BUGA', 'CALI', 'CARTAGO', 'PALMIRA', 'TULUA']
municipio_seleccionado = st.sidebar.selectbox("Seleccione Municipio Objetivo:", municipios_validos)

st.sidebar.subheader("Inercia Epidemiológica (SIVIGILA)")
casos_t1 = st.sidebar.number_input("Casos Semana Anterior (t-1)", min_value=0, value=25)
casos_t2 = st.sidebar.number_input("Casos Hace 2 Semanas (t-2)", min_value=0, value=18)

st.sidebar.subheader("Rezago Climático (Open-Meteo)")
lluvia_t3 = st.sidebar.slider("Lluvia hace 3 semanas (mm)", 0.0, 200.0, 50.0)
temp_t3 = st.sidebar.slider("Temperatura Promedio hace 3 semanas (°C)", 20.0, 35.0, 26.5)

# --- 3. PROCESAMIENTO ESPACIAL (ONE-HOT ENCODING) ---
# Inicializamos todas las variables en 0
datos_entrada = {
    'casos_t-1': [casos_t1],
    'casos_t-2': [casos_t2],
    'lluvia_t-3': [lluvia_t3],
    'temp_t-3': [temp_t3],
    'municipio_ocurrencia_BUGA': [0],
    'municipio_ocurrencia_CALI': [0],
    'municipio_ocurrencia_CARTAGO': [0],
    'municipio_ocurrencia_PALMIRA': [0],
    'municipio_ocurrencia_TULUA': [0]
}

# Activamos matemáticamente solo el municipio seleccionado
datos_entrada[f'municipio_ocurrencia_{municipio_seleccionado}'] = [1]
X_pred = pd.DataFrame(datos_entrada)

# Forzamos el orden estricto de las columnas para que el modelo no arroje error
columnas_esperadas = ['casos_t-1', 'casos_t-2', 'lluvia_t-3', 'temp_t-3',
                      'municipio_ocurrencia_BUGA', 'municipio_ocurrencia_CALI',
                      'municipio_ocurrencia_CARTAGO', 'municipio_ocurrencia_PALMIRA',
                      'municipio_ocurrencia_TULUA']
X_pred = X_pred[columnas_esperadas]

# Ejecución del pronóstico
prediccion_casos = int(np.round(modelo.predict(X_pred)[0]))
# Evitar predicciones negativas por fluctuaciones estadísticas extremas
prediccion_casos = max(0, prediccion_casos)

# --- 4. TRADUCCIÓN FARMACOECONÓMICA ---
# Precios estándar basados en consulta SISMED
COSTO_ACETA_NORMAL = 150
COSTO_ACETA_URGENCIA = 450
COSTO_RINGER_NORMAL = 3500
COSTO_RINGER_URGENCIA = 8000

# Cálculo de necesidades físicas
req_aceta = prediccion_casos * 20
casos_graves = int(prediccion_casos * 0.15)
req_ringer = casos_graves * 9

# Cálculo de impacto financiero
costo_preventivo = (req_aceta * COSTO_ACETA_NORMAL) + (req_ringer * COSTO_RINGER_NORMAL)
costo_reactivo = (req_aceta * COSTO_ACETA_URGENCIA) + (req_ringer * COSTO_RINGER_URGENCIA)
ahorro = costo_reactivo - costo_preventivo

# --- 5. DASHBOARD EJECUTIVO ---
st.markdown(f"### 📊 Reporte de Abastecimiento Preventivo: Hospitales de {municipio_seleccionado}")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🦠 Proyección de Pacientes")
    st.metric("Casos Estimados", f"{prediccion_casos} pacientes", f"Margen de error del modelo: +/- 3.6")

with col2:
    st.warning("💊 Insumos Críticos a Despachar")
    st.metric("Acetaminofén 500mg", f"{req_aceta:,} Tab.")
    st.metric("Lactato de Ringer", f"{req_ringer:,} Bolsas")

with col3:
    st.success("💰 Impacto Financiero en Compras")
    st.metric("Ahorro por Compra Temprana", f"${ahorro:,.0f} COP")
    st.caption("Diferencial entre abastecimiento planeado vs. compra reactiva por escasez.")

st.divider()

# Sección de justificación de calidad
st.subheader("Auditoría de Datos (Compliance)")
st.markdown(f"""
El algoritmo ha determinado la cuota de abastecimiento para **{municipio_seleccionado}** evaluando el comportamiento previo del virus (Inercia) y las condiciones termodinámicas del ciclo de incubación del vector ({lluvia_t3} mm de precipitación y {temp_t3}°C). El sistema aísla matemáticamente los datos de los demás municipios para evitar sesgos cruzados, garantizando la trazabilidad local.
""")
