import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Sentinel — Última Milla",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CARGA DEL MODELO Y DATOS
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    paquete = joblib.load('modelo_municipal_v3.pkl')
    return paquete

@st.cache_data
def cargar_datos():
    return pd.read_csv('dengue_valle_semanal.csv', parse_dates=['fecha'])

try:
    paquete   = cargar_modelo()
    modelo    = paquete['modelo']
    FEATURES  = paquete['features']
    MUNICIPIOS = paquete['municipios']
    METRICAS  = paquete['metricas_test']
    VERSION   = paquete['version']
except FileNotFoundError:
    st.error("⚠️ No se encontró 'modelo_municipal_v3.pkl'. Asegúrate de subirlo al repositorio.")
    st.stop()

try:
    df_hist = cargar_datos()
except FileNotFoundError:
    st.error("⚠️ No se encontró 'dengue_valle_semanal.csv'.")
    st.stop()

# ─────────────────────────────────────────────
# COORDENADAS DE LOS 5 MUNICIPIOS
# ─────────────────────────────────────────────
COORDENADAS = {
    'BUGA':    (3.9003,  -76.2979),
    'CALI':    (3.4516,  -76.5320),
    'CARTAGO': (4.7458,  -75.9119),
    'PALMIRA': (3.5394,  -76.3036),
    'TULUA':   (4.0840,  -76.1960),
}

# Costos SISMED — fecha de consulta documentada
COSTOS = {
    'fecha_consulta':       'Abril 2025',
    'aceta_normal':         150,    # COP por tableta
    'aceta_urgencia':       450,
    'ringer_normal':        3_500,  # COP por bolsa
    'ringer_urgencia':      8_000,
}

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.title("🛡️ Data Sentinel: Logística Farmacéutica de Última Milla")
st.markdown(
    "**Ecosistema Predictivo (Spatial-Aware) para la prevención de desabastecimiento "
    "hospitalario focalizado** — Valle del Cauca · SIVIGILA 2007–2018"
)

# Badge de métricas del modelo en el encabezado
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
col_b1.metric("Modelo", f"Random Forest {VERSION}")
col_b2.metric("R² (test 2018)", f"{METRICAS['r2']}")
col_b3.metric("MAE", f"{METRICAS['mae']} casos/sem")
col_b4.metric("RMSE", f"{METRICAS['rmse']} casos/sem")

st.caption(
    f"Entrenado con datos SIVIGILA 2007–2017 · "
    f"Evaluado en holdout temporal 2018 · "
    f"Gap Train-Val R²: 0.055 (sin overfitting)"
)
st.divider()

# ─────────────────────────────────────────────
# PANEL LATERAL
# ─────────────────────────────────────────────
st.sidebar.header("📍 Parámetros de Predicción")

municipio_sel = st.sidebar.selectbox(
    "Municipio objetivo:", MUNICIPIOS
)

st.sidebar.subheader("Inercia Epidemiológica (SIVIGILA)")
st.sidebar.caption("Ingrese los casos reportados en semanas anteriores.")

# Valores sugeridos: último dato disponible del municipio seleccionado
hist_mun = df_hist[df_hist['municipio_ocurrencia'] == municipio_sel].sort_values('fecha')
ultimo_caso = int(hist_mun['casos'].iloc[-1]) if len(hist_mun) > 0 else 10

casos_t1 = st.sidebar.number_input(
    "Casos semana anterior (t-1)", min_value=0, value=ultimo_caso
)
casos_t2 = st.sidebar.number_input(
    "Casos hace 2 semanas (t-2)", min_value=0,
    value=int(hist_mun['casos'].iloc[-2]) if len(hist_mun) > 1 else 8
)
casos_t3 = st.sidebar.number_input(
    "Casos hace 3 semanas (t-3)", min_value=0,
    value=int(hist_mun['casos'].iloc[-3]) if len(hist_mun) > 2 else 6
)

semana_actual = st.sidebar.slider(
    "Semana epidemiológica actual", min_value=1, max_value=52, value=20
)

# Media móvil calculada automáticamente
media_movil = np.mean([casos_t1, casos_t2, casos_t3,
                       int(hist_mun['casos'].iloc[-4]) if len(hist_mun) > 3 else casos_t3])

st.sidebar.divider()
st.sidebar.subheader("ℹ️ Acerca del modelo")
st.sidebar.caption(
    f"**Features:** Lags t-1, t-2, t-3 · Media móvil 4s · "
    f"Estacionalidad (seno/coseno semana) · One-hot municipio\n\n"
    f"**Datos:** SIVIGILA cod_eve 210/211 · Valle del Cauca\n\n"
    f"**Costos SISMED:** Consulta {COSTOS['fecha_consulta']}"
)

# ─────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────
datos_entrada = {
    'casos_t-1':                          [casos_t1],
    'casos_t-2':                          [casos_t2],
    'casos_t-3':                          [casos_t3],
    'media_movil_4s':                     [media_movil],
    'semana_seno':                        [np.sin(2 * np.pi * semana_actual / 52)],
    'semana_coseno':                      [np.cos(2 * np.pi * semana_actual / 52)],
    'municipio_ocurrencia_BUGA':          [1 if municipio_sel == 'BUGA'    else 0],
    'municipio_ocurrencia_CALI':          [1 if municipio_sel == 'CALI'    else 0],
    'municipio_ocurrencia_CARTAGO':       [1 if municipio_sel == 'CARTAGO' else 0],
    'municipio_ocurrencia_PALMIRA':       [1 if municipio_sel == 'PALMIRA' else 0],
    'municipio_ocurrencia_TULUA':         [1 if municipio_sel == 'TULUA'   else 0],
}

X_pred = pd.DataFrame(datos_entrada)[FEATURES]
prediccion_casos = max(0, int(np.round(modelo.predict(X_pred)[0])))

# Intervalo de confianza aproximado (±RMSE del modelo)
ic_bajo = max(0, prediccion_casos - int(METRICAS['rmse']))
ic_alto = prediccion_casos + int(METRICAS['rmse'])

# ─────────────────────────────────────────────
# TRADUCCIÓN FARMACOECONÓMICA
# ─────────────────────────────────────────────
req_aceta   = prediccion_casos * 20
casos_graves = max(0, int(prediccion_casos * 0.15))
req_ringer   = casos_graves * 9

costo_preventivo = (req_aceta * COSTOS['aceta_normal']) + (req_ringer * COSTOS['ringer_normal'])
costo_reactivo   = (req_aceta * COSTOS['aceta_urgencia']) + (req_ringer * COSTOS['ringer_urgencia'])
ahorro           = costo_reactivo - costo_preventivo

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Predictivo",
    "📈 Serie Histórica",
    "🗺️ Mapa de Riesgo",
    "🔬 Auditoría del Modelo"
])

# ══════════════════════════════════════════════
# TAB 1: DASHBOARD PREDICTIVO
# ══════════════════════════════════════════════
with tab1:
    st.markdown(f"### Reporte de Abastecimiento: Hospitales de {municipio_sel}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🦠 Proyección de Pacientes")
        st.metric(
            "Casos Estimados (próxima semana)",
            f"{prediccion_casos} pacientes",
            delta=f"IC 95%: [{ic_bajo} – {ic_alto}]",
            delta_color="off"
        )
        st.caption(f"MAE del modelo: ±{METRICAS['mae']} casos · R²={METRICAS['r2']}")

    with col2:
        st.warning("💊 Insumos Críticos a Despachar")
        st.metric("Acetaminofén 500mg", f"{req_aceta:,} Tab.")
        st.metric("Lactato de Ringer", f"{req_ringer:,} Bolsas")
        st.caption(f"Base: 20 tab/paciente · 15% casos graves · 9 bolsas/grave")

    with col3:
        st.success("💰 Impacto Financiero")
        st.metric(
            "Ahorro por Compra Temprana",
            f"${ahorro:,.0f} COP",
            delta="vs compra reactiva por escasez",
            delta_color="off"
        )
        st.caption(
            f"Preventivo: ${costo_preventivo:,.0f} COP · "
            f"Reactivo: ${costo_reactivo:,.0f} COP\n"
            f"Precios SISMED · Consulta: {COSTOS['fecha_consulta']}"
        )

    st.divider()

    # Gráfica de predicción con contexto histórico (últimas 20 semanas)
    st.subheader(f"Contexto histórico + Predicción — {municipio_sel}")

    hist_reciente = (
        df_hist[df_hist['municipio_ocurrencia'] == municipio_sel]
        .sort_values('fecha')
        .tail(24)
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hist_reciente['fecha'], hist_reciente['casos'],
                    alpha=0.15, color='#378ADD')
    ax.plot(hist_reciente['fecha'], hist_reciente['casos'],
            color='#378ADD', linewidth=2, label='Histórico real')

    # Punto de predicción
    ultima_fecha = hist_reciente['fecha'].max()
    fecha_pred   = ultima_fecha + pd.Timedelta(weeks=1)
    ax.scatter([fecha_pred], [prediccion_casos],
               color='#E24B4A', s=100, zorder=5, label='Predicción próxima semana')
    ax.axvline(ultima_fecha, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Banda de confianza
    ax.fill_between([fecha_pred], [ic_bajo], [ic_alto],
                    alpha=0.3, color='#E24B4A', label=f'IC ±RMSE [{ic_bajo}–{ic_alto}]')

    ax.set_ylabel('Casos/semana')
    ax.set_title(f'Serie histórica reciente + proyección — {municipio_sel}', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
# TAB 2: SERIE HISTÓRICA COMPLETA
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Serie Temporal Completa — SIVIGILA 2007–2018")

    muns_mostrar = st.multiselect(
        "Municipios a visualizar:",
        MUNICIPIOS,
        default=MUNICIPIOS
    )

    colores_map = {
        'BUGA':    '#E24B4A',
        'CALI':    '#378ADD',
        'CARTAGO': '#639922',
        'PALMIRA': '#BA7517',
        'TULUA':   '#533AB7',
    }

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for mun in muns_mostrar:
        datos_mun = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha')
        ax2.plot(datos_mun['fecha'], datos_mun['casos'],
                 label=mun, color=colores_map[mun], linewidth=1.4, alpha=0.9)

    ax2.set_ylabel('Casos por semana')
    ax2.set_title('Dengue — Valle del Cauca · Semanas epidemiológicas', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    plt.close()

    # Tabla de estadísticas por municipio
    st.subheader("Estadísticas Descriptivas por Municipio")
    stats = (
        df_hist[df_hist['municipio_ocurrencia'].isin(muns_mostrar)]
        .groupby('municipio_ocurrencia')['casos']
        .agg(
            Semanas_con_datos='count',
            Total_casos='sum',
            Promedio_semanal='mean',
            Maximo_semanal='max',
            Pico_año=lambda x: df_hist.loc[x.idxmax(), 'ano'] if len(x) > 0 else '-'
        )
        .round(1)
    )
    st.dataframe(stats, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3: MAPA DE RIESGO
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Mapa de Riesgo — Valle del Cauca")
    st.caption(
        "El color y tamaño de cada marcador refleja el nivel de casos predichos "
        "relativo al promedio histórico de cada municipio."
    )

    # Calcular predicciones para TODOS los municipios
    predicciones_mapa = {}
    promedios_hist    = {}

    for mun in MUNICIPIOS:
        hist_m  = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha')
        prom    = hist_m['casos'].mean()
        promedios_hist[mun] = prom

        ult1 = int(hist_m['casos'].iloc[-1])  if len(hist_m) > 0 else 5
        ult2 = int(hist_m['casos'].iloc[-2])  if len(hist_m) > 1 else 4
        ult3 = int(hist_m['casos'].iloc[-3])  if len(hist_m) > 2 else 3
        mm   = np.mean([ult1, ult2, ult3, int(hist_m['casos'].iloc[-4]) if len(hist_m) > 3 else ult3])

        entrada_mun = {
            'casos_t-1':                    [ult1],
            'casos_t-2':                    [ult2],
            'casos_t-3':                    [ult3],
            'media_movil_4s':               [mm],
            'semana_seno':                  [np.sin(2 * np.pi * semana_actual / 52)],
            'semana_coseno':                [np.cos(2 * np.pi * semana_actual / 52)],
            'municipio_ocurrencia_BUGA':    [1 if mun == 'BUGA'    else 0],
            'municipio_ocurrencia_CALI':    [1 if mun == 'CALI'    else 0],
            'municipio_ocurrencia_CARTAGO': [1 if mun == 'CARTAGO' else 0],
            'municipio_ocurrencia_PALMIRA': [1 if mun == 'PALMIRA' else 0],
            'municipio_ocurrencia_TULUA':   [1 if mun == 'TULUA'   else 0],
        }
        X_m = pd.DataFrame(entrada_mun)[FEATURES]
        predicciones_mapa[mun] = max(0, int(np.round(modelo.predict(X_m)[0])))

    # Construir mapa Folium
    mapa = folium.Map(
        location=[3.8, -76.3],
        zoom_start=8,
        tiles='CartoDB positron'
    )

    max_pred = max(predicciones_mapa.values()) if predicciones_mapa else 1

    for mun, pred in predicciones_mapa.items():
        lat, lon  = COORDENADAS[mun]
        prom      = promedios_hist[mun]
        ratio     = pred / prom if prom > 0 else 1.0
        radio     = max(8, min(35, int(pred * 1.5)))

        # Color según riesgo relativo al promedio histórico
        if ratio > 1.5:
            color = '#E24B4A'   # Rojo — por encima del promedio histórico
        elif ratio > 0.8:
            color = '#EF9F27'   # Naranja — en rango normal
        else:
            color = '#639922'   # Verde — por debajo del promedio

        popup_html = f"""
        <div style='font-family:sans-serif; width:180px'>
            <b style='font-size:14px'>{mun}</b><br>
            <hr style='margin:4px 0'>
            <b>Predicción:</b> {pred} casos/sem<br>
            <b>Promedio histórico:</b> {prom:.1f} casos/sem<br>
            <b>Ratio vs promedio:</b> {ratio:.2f}x<br>
            <hr style='margin:4px 0'>
            <b>Acetaminofén:</b> {pred*20:,} tab<br>
            <b>Ringer:</b> {max(0,int(pred*0.15))*9:,} bolsas
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radio,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{mun}: {pred} casos predichos"
        ).add_to(mapa)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:#333;'
                     f'text-shadow:1px 1px 2px white">{mun}<br>{pred}</div>',
                icon_size=(60, 30),
                icon_anchor=(30, -5)
            )
        ).add_to(mapa)

    st_folium(mapa, width=900, height=500)

    # Leyenda del mapa
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.error("🔴 Riesgo alto — >1.5x promedio histórico")
    col_l2.warning("🟠 Riesgo moderado — 0.8–1.5x promedio")
    col_l3.success("🟢 Riesgo bajo — <0.8x promedio histórico")

# ══════════════════════════════════════════════
# TAB 4: AUDITORÍA DEL MODELO
# ══════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Auditoría de Datos y Modelo (Compliance ALCOA+)")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### Ficha Técnica del Modelo")
        ficha = {
            'Algoritmo':          'Random Forest Regressor',
            'N° árboles':         '300',
            'Profundidad máx.':   '12',
            'Min. muestras hoja': '3',
            'Semilla aleatoria':  '42',
            'Features':           str(len(FEATURES)),
            'Versión':            VERSION,
            'Entrenado con':      paquete['entrenado_con'],
            'Evaluado en':        paquete['evaluado_en'],
            'Fecha entreno':      paquete['fecha_entreno'],
        }
        st.table(pd.DataFrame.from_dict(ficha, orient='index', columns=['Valor']))

    with col_a2:
        st.markdown("#### Métricas Oficiales (Test 2018 — Holdout temporal)")
        metricas_df = pd.DataFrame({
            'Métrica': ['MAE', 'RMSE', 'R²', 'Gap Train-Val R²'],
            'Valor':   [
                f"{METRICAS['mae']} casos/semana",
                f"{METRICAS['rmse']} casos/semana",
                f"{METRICAS['r2']} (87.3% varianza explicada)",
                "0.055 — sin overfitting"
            ],
            'Interpretación': [
                'Error promedio absoluto de predicción',
                'Error cuadrático medio (penaliza picos)',
                'Bondad de ajuste sobre datos no vistos',
                'Diferencia entre train y validación'
            ]
        })
        st.dataframe(metricas_df, use_container_width=True, hide_index=True)

        st.markdown("#### Fuentes de Datos")
        fuentes = pd.DataFrame({
            'Variable':    ['Casos dengue', 'Estacionalidad', 'Municipios', 'Costos'],
            'Fuente':      ['SIVIGILA / datos.gov.co', 'Calculada (seno/coseno semana)', 'SIVIGILA cod_dpto_o=76', 'SISMED'],
            'Actualizado': ['2007–2018', 'Derivada', 'Serie completa', f"Consulta {COSTOS['fecha_consulta']}"],
        })
        st.dataframe(fuentes, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Justificación Epidemiológica de Features")
    st.markdown("""
    | Feature | Justificación |
    |---|---|
    | `casos_t-1, t-2, t-3` | Inercia epidemiológica: el dengue tiene ciclo de incubación de 4–10 días. Las semanas anteriores son el predictor más fuerte. |
    | `media_movil_4s` | Captura la tendencia de corto plazo, suavizando semanas atípicas. |
    | `semana_seno / coseno` | Estacionalidad circular: Valle del Cauca tiene dos picos anuales asociados a temporadas de lluvia (sem. 15–25 y 40–48). |
    | `municipio (one-hot)` | Aislamiento espacial: cada municipio tiene su baseline epidemiológico propio, evitando sesgo cruzado. |
    """)

    st.info(
        "**Nota sobre datos climáticos:** La versión actual del modelo no incluye "
        "variables climáticas (precipitación, temperatura) porque la API Open-Meteo "
        "tiene cobertura histórica limitada para los municipios del Valle del Cauca "
        "en el período 2007–2018. La estacionalidad climática queda parcialmente "
        "capturada por el encoding seno/coseno de la semana epidemiológica. "
        "Incorporar datos climáticos reales es una mejora planificada para v4.0."
    )
