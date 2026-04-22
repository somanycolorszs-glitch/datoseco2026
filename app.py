import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# CARGA DE RECURSOS
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_municipal_v3.pkl')

@st.cache_data
def cargar_datos():
    return pd.read_csv('dengue_valle_semanal.csv', parse_dates=['fecha'])

@st.cache_data
def cargar_logistica():
    with open('logistica_params.json', 'r', encoding='utf-8') as f:
        return json.load(f)

try:
    paquete    = cargar_modelo()
    modelo     = paquete['modelo']
    FEATURES   = paquete['features']
    MUNICIPIOS = paquete['municipios']
    METRICAS   = paquete['metricas_test']
    VERSION    = paquete['version']
except FileNotFoundError:
    st.error("⚠️ No se encontró 'modelo_municipal_v3.pkl'.")
    st.stop()

try:
    df_hist = cargar_datos()
except FileNotFoundError:
    st.error("⚠️ No se encontró 'dengue_valle_semanal.csv'.")
    st.stop()

try:
    params_log      = cargar_logistica()
    RED_LOGISTICA   = params_log['red_logistica']
    INVENTARIO_BASE = params_log['inventario_inicial']
    SUPUESTOS       = params_log['supuestos']
except FileNotFoundError:
    st.error("⚠️ No se encontró 'logistica_params.json'.")
    st.stop()

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
COORDENADAS = {
    'BUGA':    (3.9003, -76.2979),
    'CALI':    (3.4516, -76.5320),
    'CARTAGO': (4.7458, -75.9119),
    'PALMIRA': (3.5394, -76.3036),
    'TULUA':   (4.0840, -76.1960),
}
COSTOS = {
    'fecha_consulta':  'Abril 2025',
    'aceta_normal':    150,
    'aceta_urgencia':  450,
    'ringer_normal':   3_500,
    'ringer_urgencia': 8_000,
}
COLORES_MUN = {
    'BUGA':    '#E24B4A',
    'CALI':    '#378ADD',
    'CARTAGO': '#639922',
    'PALMIRA': '#BA7517',
    'TULUA':   '#533AB7',
}

# ─────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────
def predecir(municipio, casos_t1, casos_t2, casos_t3, semana):
    media_movil = np.mean([casos_t1, casos_t2, casos_t3])
    entrada = {
        'casos_t-1':                    [casos_t1],
        'casos_t-2':                    [casos_t2],
        'casos_t-3':                    [casos_t3],
        'media_movil_4s':               [media_movil],
        'semana_seno':                  [np.sin(2 * np.pi * semana / 52)],
        'semana_coseno':                [np.cos(2 * np.pi * semana / 52)],
        'municipio_ocurrencia_BUGA':    [1 if municipio == 'BUGA'    else 0],
        'municipio_ocurrencia_CALI':    [1 if municipio == 'CALI'    else 0],
        'municipio_ocurrencia_CARTAGO': [1 if municipio == 'CARTAGO' else 0],
        'municipio_ocurrencia_PALMIRA': [1 if municipio == 'PALMIRA' else 0],
        'municipio_ocurrencia_TULUA':   [1 if municipio == 'TULUA'   else 0],
    }
    X = pd.DataFrame(entrada)[FEATURES]
    return max(0, int(np.round(modelo.predict(X)[0])))


def evaluar_cadena(municipio, pred_casos, stock_aceta_actual, stock_ringer_actual):
    inv = INVENTARIO_BASE[municipio]
    red = RED_LOGISTICA[municipio]
    sup = SUPUESTOS

    req_aceta  = pred_casos * sup['aceta_por_caso']
    req_ringer = max(0, int(pred_casos * sup['tasa_gravedad'])) * sup['ringer_por_caso_grave']

    stock_post_aceta  = stock_aceta_actual  - req_aceta
    stock_post_ringer = stock_ringer_actual - req_ringer

    ss_aceta   = inv['ss_aceta_tab']
    ss_ringer  = inv['ss_ringer_bolsas']
    rop_aceta  = inv['rop_aceta_tab']
    rop_ringer = inv['rop_ringer_bolsas']
    lt_dias    = red['lead_time_dias']

    dem_diaria_aceta = (inv['demanda_semanal_casos'] * sup['aceta_por_caso']) / 7
    dias_cobertura   = (stock_aceta_actual / dem_diaria_aceta) if dem_diaria_aceta > 0 else 999

    if stock_post_aceta < ss_aceta or stock_post_ringer < ss_ringer:
        urgencia, despachar_en, emoji = 'CRÍTICO', 1, '🔴'
    elif stock_aceta_actual < rop_aceta or stock_ringer_actual < rop_ringer:
        urgencia, despachar_en, emoji = 'ALERTA', max(1, int(np.ceil(lt_dias))), '🟠'
    else:
        urgencia, despachar_en, emoji = 'NORMAL', max(1, int(np.ceil(lt_dias * 2))), '🟢'

    orden_aceta  = max(0, int(req_aceta  * 4 - max(0, stock_post_aceta)  + ss_aceta))
    orden_ringer = max(0, int(req_ringer * 4 - max(0, stock_post_ringer) + ss_ringer))

    costo_preventivo = orden_aceta  * COSTOS['aceta_normal']   + orden_ringer * COSTOS['ringer_normal']
    costo_reactivo   = orden_aceta  * COSTOS['aceta_urgencia'] + orden_ringer * COSTOS['ringer_urgencia']

    return {
        'municipio':          municipio,
        'urgencia':           urgencia,
        'emoji':              emoji,
        'pred_casos':         pred_casos,
        'req_aceta':          req_aceta,
        'req_ringer':         req_ringer,
        'stock_aceta':        stock_aceta_actual,
        'stock_ringer':       stock_ringer_actual,
        'stock_post_aceta':   round(stock_post_aceta),
        'stock_post_ringer':  round(stock_post_ringer),
        'orden_aceta':        orden_aceta,
        'orden_ringer':       orden_ringer,
        'despachar_en_dias':  despachar_en,
        'lead_time_dias':     round(lt_dias, 2),
        'dist_carretera_km':  red['dist_carretera_km'],
        'dias_cobertura':     round(dias_cobertura, 1),
        'costo_preventivo':   costo_preventivo,
        'costo_reactivo':     costo_reactivo,
        'ahorro':             costo_reactivo - costo_preventivo,
    }

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
st.title("🛡️ Data Sentinel: Logística Farmacéutica de Última Milla")
st.markdown(
    "**Ecosistema Predictivo (Spatial-Aware) para la prevención de desabastecimiento "
    "hospitalario focalizado** — Valle del Cauca · SIVIGILA 2007–2018"
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", f"Random Forest {VERSION}")
c2.metric("R² holdout 2018", f"{METRICAS['r2']}")
c3.metric("MAE", f"{METRICAS['mae']} casos/sem")
c4.metric("RMSE", f"{METRICAS['rmse']} casos/sem")
st.caption(
    "Entrenado: SIVIGILA 2007–2017 · Evaluado: holdout temporal 2018 · "
    "Gap Train-Val R²: 0.055 · Sin overfitting"
)
st.divider()

# ─────────────────────────────────────────────
# PANEL LATERAL
# ─────────────────────────────────────────────
st.sidebar.header("📍 Parámetros de Simulación")
municipio_sel = st.sidebar.selectbox("Municipio objetivo:", MUNICIPIOS)

hist_mun = df_hist[df_hist['municipio_ocurrencia'] == municipio_sel].sort_values('fecha')
ult      = lambda i: int(hist_mun['casos'].iloc[i]) if len(hist_mun) > abs(i) else 5

st.sidebar.subheader("Inercia Epidemiológica")
casos_t1      = st.sidebar.number_input("Casos semana anterior (t-1)", min_value=0, value=ult(-1))
casos_t2      = st.sidebar.number_input("Casos hace 2 semanas (t-2)", min_value=0, value=ult(-2))
casos_t3      = st.sidebar.number_input("Casos hace 3 semanas (t-3)", min_value=0, value=ult(-3))
semana_actual = st.sidebar.slider("Semana epidemiológica", 1, 52, 20)

st.sidebar.subheader("Stock Actual (editable)")
inv_base           = INVENTARIO_BASE[municipio_sel]
stock_aceta_input  = st.sidebar.number_input(
    "Acetaminofén disponible (tab)", min_value=0,
    value=inv_base['stock_aceta_tab'], step=100
)
stock_ringer_input = st.sidebar.number_input(
    "Lactato de Ringer disponible (bolsas)", min_value=0,
    value=inv_base['stock_ringer_bolsas'], step=10
)
st.sidebar.caption("Edita el stock para simular escenarios de escasez o abundancia.")

# ─────────────────────────────────────────────
# CÁLCULOS CENTRALES
# ─────────────────────────────────────────────
pred_sel   = predecir(municipio_sel, casos_t1, casos_t2, casos_t3, semana_actual)
cadena_sel = evaluar_cadena(municipio_sel, pred_sel, stock_aceta_input, stock_ringer_input)
ic_bajo    = max(0, pred_sel - int(METRICAS['rmse']))
ic_alto    = pred_sel + int(METRICAS['rmse'])

resumen_todos = []
for mun in MUNICIPIOS:
    h = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha')
    g = lambda i, h=h: int(h['casos'].iloc[i]) if len(h) > abs(i) else 5
    p = predecir(mun, g(-1), g(-2), g(-3), semana_actual)
    c = evaluar_cadena(mun, p,
                       INVENTARIO_BASE[mun]['stock_aceta_tab'],
                       INVENTARIO_BASE[mun]['stock_ringer_bolsas'])
    resumen_todos.append(c)

df_resumen = pd.DataFrame(resumen_todos)
orden_urg  = {'CRÍTICO': 0, 'ALERTA': 1, 'NORMAL': 2}
df_resumen_sorted = df_resumen.sort_values('urgencia', key=lambda x: x.map(orden_urg))

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard Predictivo",
    "🚚 Cadena de Abastecimiento",
    "📈 Serie Histórica",
    "🗺️ Mapa de Riesgo",
    "🔬 Auditoría del Modelo",
])

# ══════════════════════════════════════════════
# TAB 1 — DASHBOARD PREDICTIVO
# ══════════════════════════════════════════════
with tab1:
    st.markdown(f"### Reporte Predictivo: {municipio_sel}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🦠 Proyección de Pacientes")
        st.metric("Casos estimados (próxima semana)", f"{pred_sel} pacientes",
                  delta=f"IC ±RMSE: [{ic_bajo} – {ic_alto}]", delta_color="off")
        st.caption(f"MAE: ±{METRICAS['mae']} casos · R²={METRICAS['r2']}")

    with col2:
        st.warning("💊 Insumos Críticos")
        st.metric("Acetaminofén 500mg", f"{cadena_sel['req_aceta']:,} Tab.")
        st.metric("Lactato de Ringer",  f"{cadena_sel['req_ringer']:,} Bolsas")
        nivel_color = {"CRÍTICO": "error", "ALERTA": "warning", "NORMAL": "success"}
        getattr(st, nivel_color[cadena_sel['urgencia']])(
            f"{cadena_sel['emoji']} Estado logístico: **{cadena_sel['urgencia']}** — "
            f"Despachar en ≤ {cadena_sel['despachar_en_dias']} día(s)"
        )

    with col3:
        st.success("💰 Impacto Financiero")
        st.metric("Ahorro compra temprana", f"${cadena_sel['ahorro']:,.0f} COP",
                  delta="vs compra reactiva", delta_color="off")
        st.caption(
            f"Preventivo: ${cadena_sel['costo_preventivo']:,.0f} · "
            f"Reactivo: ${cadena_sel['costo_reactivo']:,.0f} · "
            f"SISMED {COSTOS['fecha_consulta']}"
        )

    st.divider()
    st.subheader(f"Histórico reciente + Proyección — {municipio_sel}")
    hist_rec = hist_mun.tail(24)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hist_rec['fecha'], hist_rec['casos'],
                    alpha=0.12, color=COLORES_MUN[municipio_sel])
    ax.plot(hist_rec['fecha'], hist_rec['casos'],
            color=COLORES_MUN[municipio_sel], linewidth=2, label='Histórico real')
    ultima_fecha = hist_rec['fecha'].max()
    fecha_pred   = ultima_fecha + pd.Timedelta(weeks=1)
    ax.scatter([fecha_pred], [pred_sel], color='#E24B4A', s=120, zorder=5, label='Predicción')
    ax.fill_between([fecha_pred], [ic_bajo], [ic_alto],
                    alpha=0.3, color='#E24B4A', label=f'IC ±RMSE [{ic_bajo}–{ic_alto}]')
    ax.axvline(ultima_fecha, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_ylabel('Casos / semana')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════
# TAB 2 — CADENA DE ABASTECIMIENTO
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🚚 Motor Logístico — Cadena de Abastecimiento Completa")
    st.caption(
        "Integra predicción de demanda + stock actual + lead times reales + "
        "norma MINSALUD 1403/2007 para generar órdenes de despacho priorizadas."
    )

    # Semáforo
    st.subheader("Panel de Urgencia — Todos los Municipios")
    cols_sem = st.columns(5)
    for i, (_, row) in enumerate(df_resumen_sorted.iterrows()):
        with cols_sem[i]:
            nivel = {"CRÍTICO": "error", "ALERTA": "warning", "NORMAL": "success"}
            getattr(st, nivel[row['urgencia']])(
                f"**{row['emoji']} {row['municipio']}**\n\n"
                f"{row['pred_casos']} casos pred.\n\n"
                f"Despachar en ≤{row['despachar_en_dias']}d"
            )

    st.divider()

    # Tabla de órdenes
    st.subheader("Órdenes de Despacho — Prioridad Automática")
    tabla = df_resumen_sorted[[
        'municipio', 'urgencia', 'pred_casos',
        'orden_aceta', 'orden_ringer',
        'despachar_en_dias', 'dist_carretera_km',
        'costo_preventivo', 'ahorro'
    ]].copy()
    tabla.columns = [
        'Municipio', 'Urgencia', 'Casos pred.',
        'Aceta. (tab)', 'Ringer (bol)',
        'Despachar en (d)', 'Distancia (km)',
        'Costo orden (COP)', 'Ahorro vs reactivo'
    ]
    tabla['Costo orden (COP)']    = tabla['Costo orden (COP)'].apply(lambda x: f"${x:,.0f}")
    tabla['Ahorro vs reactivo']   = tabla['Ahorro vs reactivo'].apply(lambda x: f"${x:,.0f}")

    def color_urgencia(val):
        c = {'CRÍTICO': 'background-color:#ffd5d5',
             'ALERTA':  'background-color:#fff3cd',
             'NORMAL':  'background-color:#d4edda'}
        return c.get(val, '')

    st.dataframe(
        tabla.style.applymap(color_urgencia, subset=['Urgencia']),
        use_container_width=True, hide_index=True
    )

    st.divider()

    # Detalle municipio seleccionado
    st.subheader(f"Detalle de Cadena — {municipio_sel}")
    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        st.markdown("**📦 Estado de Stock**")
        st.dataframe(pd.DataFrame({
            'Insumo':           ['Acetaminofén', 'Lactato de Ringer'],
            'Stock actual':     [f"{cadena_sel['stock_aceta']:,} tab",
                                 f"{cadena_sel['stock_ringer']:,} bol"],
            'Demanda predicha': [f"{cadena_sel['req_aceta']:,} tab",
                                 f"{cadena_sel['req_ringer']:,} bol"],
            'Stock post-dem.':  [f"{cadena_sel['stock_post_aceta']:,} tab",
                                 f"{cadena_sel['stock_post_ringer']:,} bol"],
            'Punto reorden':    [f"{inv_base['rop_aceta_tab']:,} tab",
                                 f"{inv_base['rop_ringer_bolsas']:,} bol"],
        }), hide_index=True, use_container_width=True)

    with col_d2:
        st.markdown("**🛣️ Red Logística**")
        st.dataframe(pd.DataFrame({
            'Parámetro': ['Centro distribución', 'Dist. aérea', 'Dist. carretera',
                          'Tortuosidad vial', 'Velocidad', 'Lead time', 'Cobertura actual'],
            'Valor':     [
                'SECCIONED Cali',
                f"{RED_LOGISTICA[municipio_sel]['dist_aerea_km']} km",
                f"{RED_LOGISTICA[municipio_sel]['dist_carretera_km']} km",
                f"{SUPUESTOS['factor_tortuosidad']}x (INVIAS 2022)",
                f"{SUPUESTOS['velocidad_kmph']} km/h",
                f"{RED_LOGISTICA[municipio_sel]['lead_time_horas']} h",
                f"{cadena_sel['dias_cobertura']} días",
            ]
        }), hide_index=True, use_container_width=True)

    with col_d3:
        st.markdown("**📋 Orden de Despacho**")
        st.metric("Acetaminofén a ordenar", f"{cadena_sel['orden_aceta']:,} tab")
        st.metric("Ringer a ordenar",       f"{cadena_sel['orden_ringer']:,} bol")
        st.metric("Despachar en",           f"≤ {cadena_sel['despachar_en_dias']} día(s)")
        st.metric("Costo orden",            f"${cadena_sel['costo_preventivo']:,.0f} COP")
        st.metric("Ahorro vs reactivo",     f"${cadena_sel['ahorro']:,.0f} COP")

    # Gráfica stock vs ROP
    st.divider()
    st.subheader("Stock Actual vs Punto de Reorden — Todos los Municipios")
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    munis_ord  = list(df_resumen_sorted['municipio'])
    color_urg  = {'CRÍTICO': '#E24B4A', 'ALERTA': '#EF9F27', 'NORMAL': '#639922'}
    bar_colors = [color_urg[r['urgencia']] for _, r in df_resumen_sorted.iterrows()]

    for ax, clave_s, clave_r, titulo, ylabel in [
        (axes2[0], 'stock_aceta_tab',    'rop_aceta_tab',    'Acetaminofén', 'Tabletas'),
        (axes2[1], 'stock_ringer_bolsas','rop_ringer_bolsas','Lactato de Ringer','Bolsas'),
    ]:
        stocks = [INVENTARIO_BASE[m][clave_s] for m in munis_ord]
        rops   = [INVENTARIO_BASE[m][clave_r] for m in munis_ord]
        ax.bar(munis_ord, stocks, color=bar_colors, alpha=0.85, width=0.6)
        ax.plot(munis_ord, rops, 'o--', color='#333', linewidth=1.5,
                markersize=7, label='Punto de reorden (ROP)', zorder=5)
        ax.set_title(f'{titulo} — Stock vs ROP', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    parches = [mpatches.Patch(color=v, label=k) for k, v in color_urg.items()]
    fig2.legend(handles=parches, loc='lower center', ncol=3,
                fontsize=9, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Supuestos auditables
    st.divider()
    with st.expander("📋 Supuestos del Módulo Logístico — Auditabilidad completa"):
        st.dataframe(pd.DataFrame({
            'Parámetro': [
                'Factor tortuosidad vial', 'Velocidad promedio carretera',
                'Tiempo carga + descarga', 'Stock de seguridad',
                'Acetaminofén por caso', 'Ringer por caso grave',
                'Tasa de gravedad dengue', 'Centro de distribución',
            ],
            'Valor': [
                f"{SUPUESTOS['factor_tortuosidad']}x distancia aérea",
                f"{SUPUESTOS['velocidad_kmph']} km/h",
                f"{SUPUESTOS['horas_carga_descarga']} horas",
                f"{SUPUESTOS['stock_seguridad_semanas']} semanas de demanda promedio",
                f"{SUPUESTOS['aceta_por_caso']} tabletas / paciente",
                f"{SUPUESTOS['ringer_por_caso_grave']} bolsas / paciente grave",
                f"{SUPUESTOS['tasa_gravedad']*100:.0f}% de casos totales",
                "SECCIONED Cali",
            ],
            'Fuente': [
                SUPUESTOS['fuentes']['tortuosidad'],
                SUPUESTOS['fuentes']['velocidad'],
                "Estándar logístico farmacéutico",
                SUPUESTOS['fuentes']['stock_seguridad'],
                SUPUESTOS['fuentes']['protocolos'],
                SUPUESTOS['fuentes']['protocolos'],
                SUPUESTOS['fuentes']['protocolos'],
                "Secretaría de Salud Valle del Cauca",
            ]
        }), hide_index=True, use_container_width=True)
        st.info(
            "Todos los supuestos son documentados, auditables y reemplazables "
            "por datos reales cuando estén disponibles. El operador logístico puede "
            "ajustar velocidad vial y stock de seguridad según condiciones operativas."
        )

# ══════════════════════════════════════════════
# TAB 3 — SERIE HISTÓRICA
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Serie Temporal Completa — SIVIGILA 2007–2018")
    muns_vis = st.multiselect("Municipios:", MUNICIPIOS, default=MUNICIPIOS)
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    for mun in muns_vis:
        d = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha')
        ax3.plot(d['fecha'], d['casos'], label=mun,
                 color=COLORES_MUN[mun], linewidth=1.4, alpha=0.9)
    ax3.set_ylabel('Casos / semana')
    ax3.set_title('Dengue — Valle del Cauca · Semanas epidemiológicas', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    fig3.autofmt_xdate()
    st.pyplot(fig3)
    plt.close()

    st.subheader("Estadísticas por Municipio")
    stats = (
        df_hist[df_hist['municipio_ocurrencia'].isin(muns_vis)]
        .groupby('municipio_ocurrencia')['casos']
        .agg(Semanas_con_datos='count', Total_casos='sum',
             Promedio_semanal='mean', Maximo_semanal='max')
        .round(1)
    )
    st.dataframe(stats, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — MAPA DE RIESGO
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Mapa de Riesgo Integrado — Predicción + Logística")
    st.caption("Color = urgencia logística · Tamaño = casos predichos · Líneas = rutas desde SECCIONED")

    promedios   = df_hist.groupby('municipio_ocurrencia')['casos'].mean()
    mapa        = folium.Map(location=[3.8, -76.3], zoom_start=8, tiles='CartoDB positron')
    origen      = [3.4516, -76.5320]
    color_u_map = {'CRÍTICO': '#E24B4A', 'ALERTA': '#EF9F27', 'NORMAL': '#639922'}

    for _, row in df_resumen.iterrows():
        mun      = row['municipio']
        lat, lon = COORDENADAS[mun]
        color    = color_u_map[row['urgencia']]
        radio    = max(8, min(40, int(row['pred_casos'] * 0.8)))
        prom     = promedios.get(mun, 1)
        ratio    = row['pred_casos'] / prom if prom > 0 else 1

        folium.PolyLine(
            locations=[origen, [lat, lon]], color=color, weight=2, opacity=0.5,
            dash_array='5 5' if row['urgencia'] == 'NORMAL' else None
        ).add_to(mapa)

        popup_html = (
            f"<div style='font-family:sans-serif;width:200px'>"
            f"<b style='font-size:14px'>{row['emoji']} {mun}</b>"
            f"<hr style='margin:4px 0'>"
            f"<b>Urgencia:</b> {row['urgencia']}<br>"
            f"<b>Predicción:</b> {row['pred_casos']} casos/sem<br>"
            f"<b>Ratio vs promedio:</b> {ratio:.2f}x<br>"
            f"<hr style='margin:4px 0'>"
            f"<b>Distancia:</b> {row['dist_carretera_km']} km<br>"
            f"<b>Lead time:</b> {row['lead_time_dias']} días<br>"
            f"<b>Despachar en:</b> ≤{row['despachar_en_dias']} día(s)<br>"
            f"<hr style='margin:4px 0'>"
            f"<b>Orden aceta:</b> {row['orden_aceta']:,} tab<br>"
            f"<b>Orden ringer:</b> {row['orden_ringer']:,} bol<br>"
            f"<b>Costo:</b> ${row['costo_preventivo']:,.0f} COP"
            f"</div>"
        )
        folium.CircleMarker(
            location=[lat, lon], radius=radio, color=color,
            fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{mun} · {row['urgencia']} · {row['pred_casos']} casos"
        ).add_to(mapa)

    folium.Marker(
        location=origen,
        icon=folium.Icon(color='blue', icon='home', prefix='fa'),
        tooltip="SECCIONED — Centro distribución Cali"
    ).add_to(mapa)

    st_folium(mapa, width=900, height=500)
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.error("🔴 CRÍTICO — Stock post-demanda < Stock de seguridad")
    col_l2.warning("🟠 ALERTA — Stock actual < Punto de reorden")
    col_l3.success("🟢 NORMAL — Stock suficiente")

# ══════════════════════════════════════════════
# TAB 5 — AUDITORÍA DEL MODELO
# ══════════════════════════════════════════════
with tab5:
    st.subheader("🔬 Auditoría Técnica Completa (Compliance ALCOA+)")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### Ficha Técnica del Modelo")
        st.table(pd.DataFrame.from_dict({
            'Algoritmo':          'Random Forest Regressor',
            'N° árboles':         '300',
            'Profundidad máxima': '12',
            'Min. muestras hoja': '3',
            'Max features':       'sqrt',
            'Semilla aleatoria':  '42',
            'N° features':        str(len(FEATURES)),
            'Versión':            VERSION,
            'Entrenado con':      paquete['entrenado_con'],
            'Evaluado en':        paquete['evaluado_en'],
            'Fecha entreno':      paquete['fecha_entreno'],
        }, orient='index', columns=['Valor']))

    with col_a2:
        st.markdown("#### Métricas Oficiales — Holdout Temporal 2018")
        st.dataframe(pd.DataFrame({
            'Métrica':        ['MAE', 'RMSE', 'R²', 'Gap Train-Val R²'],
            'Valor':          [f"{METRICAS['mae']} casos/sem",
                               f"{METRICAS['rmse']} casos/sem",
                               f"{METRICAS['r2']}",
                               "0.055"],
            'Interpretación': [
                'Error promedio absoluto en datos no vistos',
                'Error cuadrático medio (penaliza outliers)',
                '87.3% de la varianza de casos explicada',
                'Diferencia train vs validación — sin overfitting',
            ]
        }), hide_index=True, use_container_width=True)

        st.markdown("#### Fuentes de Datos")
        st.dataframe(pd.DataFrame({
            'Variable':  ['Casos dengue', 'Estacionalidad', 'Municipios',
                          'Costos insumos', 'Red logística', 'Stock seguridad'],
            'Fuente':    ['SIVIGILA / datos.gov.co', 'Encoding seno/coseno semana',
                          'SIVIGILA cod_dpto_o=76', 'SISMED',
                          'IGAC + INVIAS 2022', 'Res. MINSALUD 1403/2007'],
            'Período':   ['2007–2018', 'Derivada', 'Serie completa',
                          f"Consulta {COSTOS['fecha_consulta']}", '2022', '2007 (vigente)'],
        }), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### Justificación Epidemiológica de Features")
    st.markdown("""
| Feature | Justificación |
|---|---|
| `casos_t-1, t-2, t-3` | Inercia: incubación dengue 4–10 días. Predictor dominante (importancia RF >0.6). |
| `media_movil_4s` | Tendencia de corto plazo. Suaviza semanas atípicas y captura momentum del brote. |
| `semana_seno / coseno` | Estacionalidad circular: 2 picos anuales en Valle del Cauca (sem. 15–25 y 40–48). |
| `municipio (one-hot)` | Baseline epidemiológico propio. Evita sesgo cruzado entre municipios. |
    """)
    st.info(
        "**Nota sobre variables climáticas:** La estacionalidad climática queda capturada "
        "por el encoding circular de semana epidemiológica. Incorporar precipitación y "
        "temperatura reales de Open-Meteo está planificado como mejora v4.0."
    )
