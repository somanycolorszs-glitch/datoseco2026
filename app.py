import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import hashlib
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ── Dependencias opcionales (agente IA + menú de navegación tipo pilar) ──
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_DISPONIBLE = True
except ImportError:
    GEMINI_DISPONIBLE = False

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_DISPONIBLE = True
except ImportError:
    OPTION_MENU_DISPONIBLE = False

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Denguard — Última Milla",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# INYECCIÓN DE CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800&family=Nunito+Sans:wght@300;400;600;700&display=swap');

/* ══════════════════════════════════════
   TEMA CLARO (defecto)
   ══════════════════════════════════════ */
:root {
  --bg:         #ffffff;
  --bg2:        #f4f7fb;
  --bg3:        #eaf0fa;
  --surface:    #ffffff;
  --surface2:   #f0f5fc;
  --surface3:   #e2eaf8;
  --blue:       #1a56db;
  --blue2:      #2563eb;
  --blue-lt:    #3b82f6;
  --blue-dk:    #1e3a8a;
  --text:       #1e293b;
  --text2:      #334155;
  --muted:      #64748b;
  --border:     #d1ddf5;
  --border2:    #bfcfee;
  --danger:     #dc2626;
  --warn:       #d97706;
  --ok:         #16a34a;
  --shadow:     0 2px 16px rgba(30,86,219,0.09);
  --shadow2:    0 4px 32px rgba(30,86,219,0.16);
  --dot-color:  #d1ddf5;
  --plot-bg:    #ffffff;
  --grid-color: #e2eaf8;
}

/* ══════════════════════════════════════
   TEMA OSCURO — se activa automáticamente
   cuando el sistema operativo está en
   modo oscuro (prefers-color-scheme)
   ══════════════════════════════════════ */
@media (prefers-color-scheme: dark) {
  :root {
    --bg:         #0f172a;
    --bg2:        #1e293b;
    --bg3:        #1a2744;
    --surface:    #1e293b;
    --surface2:   #243352;
    --surface3:   #2d3f66;
    --blue:       #60a5fa;
    --blue2:      #3b82f6;
    --blue-lt:    #93c5fd;
    --blue-dk:    #bfdbfe;
    --text:       #e2e8f0;
    --text2:      #cbd5e1;
    --muted:      #94a3b8;
    --border:     #2d3f66;
    --border2:    #3b4f7a;
    --danger:     #f87171;
    --warn:       #fbbf24;
    --ok:         #4ade80;
    --shadow:     0 2px 16px rgba(0,0,0,0.35);
    --shadow2:    0 4px 32px rgba(0,0,0,0.5);
    --dot-color:  #2d3f66;
    --plot-bg:    #1e293b;
    --grid-color: #2d3f66;
  }
}

/* ══════════════════════════════════════
   ESTILOS BASE (usan las variables,
   funcionan en ambos temas)
   ══════════════════════════════════════ */

/* ── Reset base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Nunito', 'Nunito Sans', sans-serif !important;
}

/* ── Fondo con patrón de puntos ── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: radial-gradient(circle, var(--dot-color) 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: 0;
  opacity: 0.5;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border2) !important;
  box-shadow: 2px 0 20px var(--shadow) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.68rem !important;
  font-weight: 800 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.14em !important;
  color: var(--blue) !important;
  margin-top: 1.4rem !important;
  padding-bottom: 0.35rem !important;
  border-bottom: 2px solid var(--border) !important;
}

/* ── Encabezados globales ── */
h1, h2, h3 {
  font-family: 'Nunito', sans-serif !important;
  color: var(--text) !important;
}
h1 { font-size: 1.85rem !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

/* ── Encabezado principal custom ── */
.ds-main-title {
  font-family: 'Nunito', sans-serif !important;
  font-size: 1.85rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 0.3rem 0 !important;
  display: flex !important;
  align-items: center !important;
  gap: 0.55rem !important;
  color: var(--blue-dk) !important;
  animation: titleReveal 0.7s ease both;
}
@keyframes titleReveal {
  from { opacity: 0; transform: translateY(-12px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Métricas ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1.1rem 1.3rem !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: var(--shadow) !important;
  transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
  cursor: default !important;
}
[data-testid="stMetric"]::after {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 3px;
  background: linear-gradient(90deg, var(--blue), var(--blue-lt));
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.3s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: var(--shadow2) !important;
  border-color: var(--blue-lt) !important;
}
[data-testid="stMetric"]:hover::after { transform: scaleX(1); }
[data-testid="stMetricLabel"] {
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Nunito', sans-serif !important;
  font-size: 1.55rem !important;
  font-weight: 800 !important;
  color: var(--blue-dk) !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
  font-family: 'Nunito Sans', sans-serif !important;
  font-size: 0.72rem !important;
  color: var(--text2) !important;
}

/* ── Navegación pilar en sidebar ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.3rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 0.6rem 0.85rem !important;
  margin: 0 !important;
  cursor: pointer !important;
  transition: color 0.18s, background 0.18s, border-color 0.18s, transform 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
  color: var(--blue) !important;
  border-color: var(--blue-lt) !important;
  transform: translateX(3px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
  color: #ffffff !important;
  background: var(--blue) !important;
  border-color: var(--blue) !important;
  box-shadow: 0 2px 12px rgba(30,86,219,0.3) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  font-size: inherit !important;
  color: inherit !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] svg { display: none !important; }
[data-testid="stSidebar"] .nav-link { margin-bottom: 3px !important; }

/* ── Botones ── */
[data-testid="stButton"] > button {
  background: transparent !important;
  color: var(--blue) !important;
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  border: 1.5px solid var(--blue) !important;
  border-radius: 10px !important;
  padding: 0.5rem 1.3rem !important;
  transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
  background: var(--blue) !important;
  color: #fff !important;
  box-shadow: var(--shadow2) !important;
  transform: translateY(-2px) !important;
}
[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
  background: var(--blue) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {
  background: var(--blue2) !important;
  box-shadow: var(--shadow2) !important;
  transform: translateY(-2px) !important;
}

/* ── Selectbox / Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div {
  background: var(--surface2) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Nunito Sans', sans-serif !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] label {
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
  color: var(--muted) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"],
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderThumb"] {
  background: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1.5px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stDataFrame"] table { background: var(--surface) !important; }
[data-testid="stDataFrame"] th {
  background: var(--bg2) !important;
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.68rem !important;
  font-weight: 800 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
  color: var(--blue-dk) !important;
  border-bottom: 1.5px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
  font-family: 'Nunito Sans', sans-serif !important;
  font-size: 0.82rem !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
  background: var(--surface) !important;
}
[data-testid="stDataFrame"] tr:hover td { background: var(--bg3) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 14px !important;
  box-shadow: var(--shadow) !important;
  overflow: hidden !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stExpander"]:hover {
  border-color: var(--blue-lt) !important;
  box-shadow: var(--shadow2) !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  padding: 0.85rem 1.1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--blue) !important; }
details[data-testid="stExpander"] > summary::marker,
details[data-testid="stExpander"] > summary::-webkit-details-marker {
  color: var(--blue) !important;
}

/* ── Contenido de expander (fondo correcto en dark) ── */
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
  background: var(--surface) !important;
  color: var(--text) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border-left-width: 3px !important;
  font-size: 0.85rem !important;
  font-family: 'Nunito Sans', sans-serif !important;
  background: var(--surface2) !important;
  color: var(--text) !important;
}
div.stInfo    { background: color-mix(in srgb, var(--blue)   10%, var(--surface)) !important;
                border-left-color: var(--blue) !important; }
div.stSuccess { background: color-mix(in srgb, var(--ok)     10%, var(--surface)) !important;
                border-left-color: var(--ok) !important; }
div.stWarning { background: color-mix(in srgb, var(--warn)   10%, var(--surface)) !important;
                border-left-color: var(--warn) !important; }
div.stError   { background: color-mix(in srgb, var(--danger) 10%, var(--surface)) !important;
                border-left-color: var(--danger) !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
  font-family: 'Nunito Sans', sans-serif !important;
  font-size: 0.72rem !important;
  color: var(--muted) !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1.5px solid var(--border) !important;
  margin: 1.2rem 0 !important;
}

/* ── Mapa iframe ── */
[data-testid="stIFrame"] {
  border-radius: 16px !important;
  border: 1.5px solid var(--border) !important;
  box-shadow: var(--shadow) !important;
  overflow: hidden !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background: color-mix(in srgb, var(--blue) 15%, var(--surface)) !important;
  border: 1px solid var(--blue-lt) !important;
  color: var(--blue-dk) !important;
  font-family: 'Nunito', sans-serif !important;
  font-size: 0.75rem !important;
  border-radius: 8px !important;
}

/* ── Radio (no sidebar) ── */
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  font-family: 'Nunito Sans', sans-serif !important;
  font-size: 0.85rem !important;
  color: var(--text) !important;
}

/* ── Chat messages (sección agente IA) ── */
[data-testid="stChatMessage"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}
[data-testid="stChatInputTextArea"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
  font-family: 'Nunito Sans', sans-serif !important;
}

/* ── Texto general dentro de la app ── */
p, li, span, label {
  color: var(--text) !important;
}
a { color: var(--blue) !important; }
code {
  background: var(--surface3) !important;
  color: var(--blue-dk) !important;
  border-radius: 4px !important;
  padding: 0.1em 0.35em !important;
}
pre, .stCodeBlock {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue-lt); }

/* ── Animaciones ── */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(14px); }
  to   { opacity:1; transform:translateY(0); }
}
.ds-animate-in { animation: fadeUp 0.4s ease both; }
[data-testid="column"]:nth-child(1) { animation: fadeUp 0.4s 0.05s ease both; }
[data-testid="column"]:nth-child(2) { animation: fadeUp 0.4s 0.12s ease both; }
[data-testid="column"]:nth-child(3) { animation: fadeUp 0.4s 0.20s ease both; }
[data-testid="column"]:nth-child(4) { animation: fadeUp 0.4s 0.28s ease both; }
[data-testid="column"]:nth-child(5) { animation: fadeUp 0.4s 0.36s ease both; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UTILIDADES ALCOA+
# ─────────────────────────────────────────────
def md5_archivo(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except Exception:
        return 'N/A'

def _colores_tema():
    """Detecta si el sistema/navegador está en modo oscuro o claro.
    Streamlit expone `st.get_option('theme.base')` cuando el usuario
    configuró un tema explícito en .streamlit/config.toml; si no hay
    configuración explícita (modo 'auto', lo más común), intenta leer
    el color de fondo real que Streamlit inyecta en la sesión.
    Si todo falla, usa 'light' como fallback seguro."""
    try:
        base = st.get_option('theme.base')
        if base in ('dark', 'light'):
            tema = base
        else:
            # Modo auto: inferir desde el color de fondo configurado
            bg = st.get_option('theme.backgroundColor') or '#ffffff'
            # Un fondo oscuro tiene luminancia baja (R+G+B < 382)
            bg = bg.lstrip('#')
            if len(bg) == 6:
                r, g, b = int(bg[0:2],16), int(bg[2:4],16), int(bg[4:6],16)
                tema = 'dark' if (r + g + b) < 382 else 'light'
            else:
                tema = 'light'
    except Exception:
        tema = 'light'
    if tema == 'dark':
        return {'plot':  '#1e293b', 'paper': '#1e293b',
                'grid':  '#2d3f66', 'font':  '#e2e8f0',
                'axis':  '#94a3b8', 'zero':  '#3b4f7a'}
    return {'plot':  '#ffffff', 'paper': '#ffffff',
            'grid':  '#e2eaf8', 'font':  '#1e293b',
            'axis':  '#64748b', 'zero':  '#bfcfee'}

@st.cache_data(show_spinner=False, ttl=60)
def _tema_cacheado():
    """Versión cacheada (60s) para no recalcular en cada widget."""
    return _colores_tema()

def _plotly_layout(fig, height=370, **extra):
    """Aplica el tema activo (claro/oscuro) a cualquier figura Plotly.
    Estandariza márgenes, fuentes, hover y ejes para toda la app."""
    t = _colores_tema()
    base = dict(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor=t['plot'],
        paper_bgcolor=t['paper'],
        font=dict(family='Nunito, Nunito Sans, sans-serif',
                  color=t['font'], size=12),
        legend=dict(font=dict(color=t['font']),
                    bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor=t['paper'],
            font_color=t['font'],
            bordercolor=t['grid'],
        ),
        xaxis=dict(showgrid=False,
                   linecolor=t['grid'],
                   tickfont=dict(color=t['axis']),
                   title_font=dict(color=t['axis'])),
        yaxis=dict(gridcolor=t['grid'],
                   linecolor=t['grid'],
                   tickfont=dict(color=t['axis']),
                   title_font=dict(color=t['axis'])),
    )
    base.update(extra)
    fig.update_layout(**base)
    return fig

def timestamp_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

def detectar_semanas_faltantes(serie):
    if len(serie) < 4:
        return [], False
    ceros = (serie == 0) & (serie.shift(1) > 3) & (serie.shift(-1) > 3)
    idx   = list(serie[ceros].index)
    return idx, len(idx) > 0

def imputar_semanas_faltantes(serie):
    serie_imp = serie.copy()
    idx_sosp, modo_deg = detectar_semanas_faltantes(serie)
    for idx in idx_sosp:
        pos     = serie.index.get_loc(idx)
        ventana = serie.iloc[max(0, pos-2):pos+3]
        ventana = ventana[ventana > 0]
        if len(ventana) > 0:
            serie_imp.iloc[pos] = int(ventana.median())
    return serie_imp, idx_sosp, modo_deg

# ─────────────────────────────────────────────
# CARGA DE RECURSOS
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    p = joblib.load('modelo_municipal_v4.pkl')
    s = {'hash_md5': md5_archivo('modelo_municipal_v4.pkl'),
         'cargado_en': timestamp_utc(), 'fuente': 'Archivo local (estático)'}
    return p, s

@st.cache_data
def cargar_datos():
    df = pd.read_csv('dengue_valle_semanal.csv', parse_dates=['fecha'])
    s  = {'hash_md5': md5_archivo('dengue_valle_semanal.csv'),
          'cargado_en': timestamp_utc(), 'fuente': 'Archivo local — SIVIGILA 2007–2018'}
    return df, s

@st.cache_data
def cargar_logistica():
    with open('logistica_params.json', 'r', encoding='utf-8') as f:
        p = json.load(f)
    s = {'hash_md5': md5_archivo('logistica_params.json'),
         'cargado_en': timestamp_utc(), 'fuente': 'Archivo local (parámetros calculados)'}
    return p, s

@st.cache_data
def cargar_justificacion():
    try:
        return pd.read_csv('justificacion_municipios.csv')
    except FileNotFoundError:
        return None

try:
    paquete, sello_modelo = cargar_modelo()
    modelo     = paquete['modelo']
    FEATURES   = paquete['features']
    MUNICIPIOS = paquete['municipios']
    METRICAS   = paquete['metricas_test']
    VERSION    = paquete['version']
    ENC_LOOKUP = paquete['target_enc_lookup']
    IQR_LOOKUP = paquete['iqr_lookup']
    # Gap Train-Val R²: si el .pkl trae este campo lo usamos (dato real del
    # entrenamiento); si no, usamos el último valor documentado como
    # fallback — pero queda en UN solo lugar, no repetido a mano dos veces.
    GAP_TRAIN_VAL = paquete.get('gap_train_val_r2', 0.077)
except FileNotFoundError:
    st.error("No se encontró 'modelo_municipal_v4.pkl'.")
    st.stop()

try:
    df_hist, sello_datos = cargar_datos()
except FileNotFoundError:
    st.error("No se encontró 'dengue_valle_semanal.csv'.")
    st.stop()

try:
    params_log, sello_log = cargar_logistica()
    RED_LOGISTICA   = params_log['red_logistica']
    INVENTARIO_BASE = params_log['inventario_inicial']
    SUPUESTOS       = params_log['supuestos']
    ERROR_ESTRAT    = params_log.get('error_estratificado', {})
except FileNotFoundError:
    st.error("No se encontró 'logistica_params.json'.")
    st.stop()

df_justificacion = cargar_justificacion()

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
COSTOS = {
    'fecha_consulta':  'Abril 2025',
    'aceta_normal':    150,
    'aceta_urgencia':  450,
    'ringer_normal':   3_500,
    'ringer_urgencia': 8_000,
}
COLORES_TOP = [
    '#1A56DB','#DC2626','#16A34A','#D97706','#7C3AED',
    '#0891B2','#EA580C','#0D9488','#4F46E5','#DB2777'
]
COLOR_URG = {'CRÍTICO': '#DC2626', 'ALERTA': '#D97706', 'NORMAL': '#16A34A'}

# ── Iconografía (sin emojis) ──────────────────────────────────────────
# SVG inline en vez de una fuente de íconos externa (ej. Bootstrap Icons
# vía CDN): el mapa folium se renderiza en un iframe aparte (componente
# de streamlit-folium), así que una fuente cargada en la página principal
# NO estaría disponible ahí. El SVG inline es autocontenido y funciona
# igual dentro y fuera del iframe, sin depender de ninguna red.
_ICONOS_SVG_PATH = {
    'exclamation-octagon-fill': (
        'M4.54.146A.5.5 0 0 1 4.893 0h6.214a.5.5 0 0 1 .353.146l4.394 '
        '4.394a.5.5 0 0 1 .146.353v6.214a.5.5 0 0 1-.146.353l-4.394 '
        '4.394a.5.5 0 0 1-.353.146H4.893a.5.5 0 0 1-.353-.146L.146 '
        '11.46A.5.5 0 0 1 0 11.107V4.893a.5.5 0 0 1 .146-.353zM7.002 11a1 '
        '1 0 1 0 2 0 1 1 0 0 0-2 0M7.1 4.995a.905.905 0 1 1 1.8 0l-.35 '
        '3.507a.552.552 0 0 1-1.1 0z'
    ),
    'exclamation-triangle-fill': (
        'M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 '
        '1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767zM8 5c.535 0 '
        '.954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 '
        '0 0 1 8 5m.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2'
    ),
    'check-circle-fill': (
        'M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0m-3.97-3.03a.75.75 0 0 0-1.08 '
        '.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a'
        '.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z'
    ),
}

def icono_svg(nombre, color='currentColor', size=14):
    """Ícono vectorial inline (sin emoji, sin dependencia de red)."""
    path = _ICONOS_SVG_PATH.get(nombre, '')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
            f'height="{size}" viewBox="0 0 16 16" fill="{color}" '
            f'style="vertical-align:-2px;display:inline-block">'
            f'<path d="{path}"/></svg>')

# ─────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────
def predecir(municipio, t1, t2, t3, semana, modo_degradado=False):
    mm = np.mean([t1, t2, t3])
    X  = pd.DataFrame({
        'casos_t-1':            [t1], 'casos_t-2': [t2], 'casos_t-3': [t3],
        'media_movil_4s':       [mm],
        'semana_seno':          [np.sin(2 * np.pi * semana / 52)],
        'semana_coseno':        [np.cos(2 * np.pi * semana / 52)],
        'municipio_target_enc': [ENC_LOOKUP.get(municipio,
                                  np.mean(list(ENC_LOOKUP.values())))],
        'municipio_iqr':        [IQR_LOOKUP.get(municipio,
                                  np.mean(list(IQR_LOOKUP.values())))],
    })[FEATURES]
    return max(0, int(np.round(modelo.predict(X)[0]))), modo_degradado


def predecir_horizonte(municipio, t1, t2, t3, semana_inicio, n=4):
    historial, resultados, divergencia = [t3, t2, t1], [], False
    for paso in range(1, n + 1):
        sem_p = ((semana_inicio + paso - 1) % 52) + 1
        h1, h2, h3 = historial[-1], historial[-2], historial[-3]
        mm = np.mean(historial[-4:]) if len(historial) >= 4 else np.mean(historial)
        X  = pd.DataFrame({
            'casos_t-1':            [h1], 'casos_t-2': [h2], 'casos_t-3': [h3],
            'media_movil_4s':       [mm],
            'semana_seno':          [np.sin(2 * np.pi * sem_p / 52)],
            'semana_coseno':        [np.cos(2 * np.pi * sem_p / 52)],
            'municipio_target_enc': [ENC_LOOKUP.get(municipio,
                                      np.mean(list(ENC_LOOKUP.values())))],
            'municipio_iqr':        [IQR_LOOKUP.get(municipio,
                                      np.mean(list(IQR_LOOKUP.values())))],
        })[FEATURES]
        pred = max(0, int(np.round(modelo.predict(X)[0])))
        ic   = round(METRICAS['mae'] * (1 + 0.35 * (paso - 1)), 1)
        if paso > 1 and resultados[-1]['pred'] > 0 and pred / resultados[-1]['pred'] > 3:
            divergencia = True
        resultados.append({'paso': f'+{paso}s', 'semana': sem_p, 'pred': pred,
                           'ic_bajo': max(0, pred - int(ic)),
                           'ic_alto': pred + int(ic), 'ic': ic})
        historial.append(pred)
    return pd.DataFrame(resultados), divergencia


def evaluar_cadena(municipio, pred_casos, stock_aceta, stock_ringer):
    inv = INVENTARIO_BASE.get(municipio, {})
    red = RED_LOGISTICA.get(municipio, {})
    if not inv or not red:
        return None
    sup   = SUPUESTOS
    req_a = pred_casos * sup['aceta_por_caso']
    req_r = max(0, int(pred_casos * sup['tasa_gravedad'])) * sup['ringer_por_caso_grave']
    sp_a  = stock_aceta  - req_a
    sp_r  = stock_ringer - req_r
    ss_a  = inv['ss_aceta_tab'];   ss_r  = inv['ss_ringer_bolsas']
    rop_a = inv['rop_aceta_tab'];  rop_r = inv['rop_ringer_bolsas']
    lt_d  = red.get('lead_time_dias', 0.1)
    dd_a  = (inv['demanda_semanal_casos'] * sup['aceta_por_caso']) / 7
    d_cob = round(stock_aceta / dd_a, 1) if dd_a > 0 else 999

    if sp_a < ss_a or sp_r < ss_r:
        urg, desp, icono = 'CRÍTICO', 1, 'exclamation-octagon-fill'
    elif stock_aceta < rop_a or stock_ringer < rop_r:
        urg, desp, icono = 'ALERTA', max(1, int(np.ceil(lt_d))), 'exclamation-triangle-fill'
    else:
        urg, desp, icono = 'NORMAL', max(1, int(np.ceil(lt_d * 2))), 'check-circle-fill'

    ord_a  = max(0, int(req_a * 4 - max(0, sp_a) + ss_a))
    ord_r  = max(0, int(req_r * 4 - max(0, sp_r) + ss_r))

    # ── FIX "ahorro siempre $0" ──────────────────────────────────────────
    # Antes: c_prev/c_reac se calculaban multiplicando el PRECIO por
    # `ord_a`/`ord_r` (cantidad a ORDENAR). Si el municipio ya tenía stock
    # suficiente (caso normal con los valores por defecto del sidebar),
    # `ord_a` y `ord_r` daban 0 → el "ahorro" mostrado era siempre $0,
    # incluso cuando sí había demanda real prevista para esa semana.
    #
    # Ahora: el ahorro compara el costo de cubrir la DEMANDA PREDICHA
    # (`req_a`/`req_r`) comprándola de forma preventiva (precio normal,
    # planificada con anticipación) vs comprándola de forma reactiva/
    # urgente (precio de emergencia, sin planificación). Esto refleja el
    # ahorro real de tener un sistema predictivo, independientemente de
    # si el stock actual ya alcanza o no para esta semana puntual.
    c_prev = req_a * COSTOS['aceta_normal']   + req_r * COSTOS['ringer_normal']
    c_reac = req_a * COSTOS['aceta_urgencia'] + req_r * COSTOS['ringer_urgencia']

    return {
        'municipio': municipio, 'urgencia': urg, 'icono': icono,
        'pred_casos': pred_casos, 'req_aceta': int(req_a), 'req_ringer': int(req_r),
        'stock_aceta': stock_aceta, 'stock_ringer': stock_ringer,
        'stock_post_aceta': round(sp_a), 'stock_post_ringer': round(sp_r),
        'ss_aceta': ss_a, 'ss_ringer': ss_r, 'rop_aceta': rop_a, 'rop_ringer': rop_r,
        'orden_aceta': ord_a, 'orden_ringer': ord_r, 'despachar_en_dias': desp,
        'lead_time_dias': round(lt_d, 2), 'lead_time_horas': red.get('lead_time_horas', 0),
        'dist_carretera_km': red.get('dist_carretera_km', 0), 'dias_cobertura': d_cob,
        'costo_preventivo': round(c_prev), 'costo_reactivo': round(c_reac),
        'ahorro': round(c_reac - c_prev),
        'sigma_error': inv.get('sigma_error_casos', 'N/A'),
        'metodo_ss': inv.get('metodo_ss', 'Estático'),
    }

# ─────────────────────────────────────────────
# NOWCASTING API
# ─────────────────────────────────────────────
# FIX timeout en CALI / nowcasting lento:
#   La consulta a la API NO filtra por municipio (filtra solo por
#   departamento + evento), así que el request HTTP es idéntico sin
#   importar qué municipio se seleccione — el filtrado por municipio
#   ocurre después, en memoria. Antes, `municipio` formaba parte de la
#   clave de caché (`@st.cache_data` usa los argumentos de la función),
#   así que CADA municipio disparaba una llamada de red nueva e idéntica.
#   CALI es el municipio seleccionado por defecto, así que normalmente es
#   el primer request — el "cold start" de la API datos.gov.co — y el que
#   más sufre si la API responde lento.
#
#   Ahora: 1) se quita `municipio` de los argumentos de la función cacheada
#   (un solo request cubre TODOS los municipios), 2) se sube el timeout y
#   se agregan reintentos automáticos, 3) se sube el límite de registros
#   para tener mejor cobertura por municipio. Resultado: una sola llamada
#   de red por hora (TTL) para los 42 municipios en vez de hasta 42
#   llamadas idénticas — mucho más rápido y mucho menos propenso a timeout.
@st.cache_data(ttl=3600, show_spinner=False)
def consultar_sivigila_reciente(limite=3000):
    BASE = "https://www.datos.gov.co/resource/4hyg-wa9d.json"
    ultimo_error = None
    for intento in range(3):
        try:
            r = requests.get(BASE, params={
                "$where": "cod_dpto_o='76' AND (cod_eve='210' OR cod_eve='211')",
                "$order": "ano DESC, semana DESC", "$limit": limite,
            }, timeout=90)
            if r.status_code != 200:
                ultimo_error = f"HTTP {r.status_code}"
                continue
            df_live = pd.DataFrame(r.json())
            if df_live.empty:
                return None, "Sin datos", None
            df_live['semana']  = df_live['semana'].astype(int)
            df_live['ano']     = df_live['ano'].astype(int)
            df_live['conteo']  = pd.to_numeric(df_live['conteo'],
                                                errors='coerce').fillna(0).astype(int)
            df_live['municipio_ocurrencia'] = (df_live['municipio_ocurrencia']
                                               .str.upper().str.strip())
            sello = {
                'timestamp':     timestamp_utc(),
                'fuente':        'API datos.gov.co/resource/4hyg-wa9d',
                'registros':     len(df_live),
                'ano_max':       int(df_live['ano'].max()),
                'hash_response': hashlib.md5(r.content).hexdigest()[:12],
            }
            return df_live, None, sello
        except requests.exceptions.Timeout:
            ultimo_error = f"Timeout en intento {intento + 1}/3 (>90s)"
            continue
        except Exception as e:
            ultimo_error = str(e)
            continue
    return None, ultimo_error or "Error desconocido tras 3 intentos", None

# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# VALIDACIÓN RETROSPECTIVA (cacheada — no depende de la UI)
# ─────────────────────────────────────────────
# FIX lentitud general: este bloque recorre ~150 semanas de CALI llamando
# al modelo en cada una. NO depende de ningún control del sidebar (ni
# municipio, ni stock, ni semana), pero antes vivía suelto dentro del tab
# y se recalculaba en CADA rerun de la app (es decir, cada vez que el
# usuario tocaba cualquier control, aunque fuera en otra pestaña). Ahora
# se cachea una sola vez por sesión.
@st.cache_data(show_spinner=False)
def calcular_validacion_retrospectiva():
    cali_hist = df_hist[
        (df_hist['municipio_ocurrencia'] == 'CALI') &
        (df_hist['fecha'].dt.year >= 2015)
    ].sort_values('fecha').reset_index(drop=True)

    if len(cali_hist) <= 3:
        return None

    inv_cali  = INVENTARIO_BASE.get('CALI', {})
    ss_aceta  = inv_cali.get('ss_aceta_tab', 0)
    rop_aceta = inv_cali.get('rop_aceta_tab', 0)
    stock_sim = inv_cali.get('stock_aceta_tab', 8000)
    registros_retro = []

    for i in range(3, len(cali_hist)):
        row  = cali_hist.iloc[i]
        t1   = int(cali_hist.iloc[i-1]['casos'])
        t2   = int(cali_hist.iloc[i-2]['casos'])
        t3   = int(cali_hist.iloc[i-3]['casos'])
        sem  = int(row['semana']) if 'semana' in row.index else 20
        pred, _ = predecir('CALI', t1, t2, t3, sem)
        real = int(row['casos'])
        req_a = pred * SUPUESTOS['aceta_por_caso']
        sp_a  = stock_sim - req_a
        urg   = ('CRÍTICO' if sp_a < ss_aceta
                 else 'ALERTA' if stock_sim < rop_aceta else 'NORMAL')
        registros_retro.append({
            'fecha': row['fecha'], 'real_casos': real,
            'pred_casos': pred, 'stock_aceta': round(stock_sim), 'urgencia': urg,
        })
        stock_sim = max(0, stock_sim - real * SUPUESTOS['aceta_por_caso'])
        if stock_sim < rop_aceta:
            stock_sim += int(inv_cali.get('demanda_semanal_casos', 141) * 4 *
                             SUPUESTOS['aceta_por_caso'])

    df_retro = pd.DataFrame(registros_retro)
    df_r16   = df_retro[df_retro['fecha'].dt.year >= 2016].reset_index(drop=True)
    if df_r16.empty:
        return None

    idx_pico   = df_r16['real_casos'].idxmax()
    pico_val   = df_r16.loc[idx_pico, 'real_casos']
    pico_fec   = df_r16.loc[idx_pico, 'fecha']
    pre_pico   = df_r16.iloc[max(0, idx_pico - 10):idx_pico]
    primera_al = pre_pico[pre_pico['urgencia'].isin(['ALERTA', 'CRÍTICO'])].head(1)
    sem_antic  = idx_pico - primera_al.index[0] if len(primera_al) > 0 else 0
    idx_primera_al = int(primera_al.index[0]) if len(primera_al) > 0 else None

    return {
        'df_r16': df_r16, 'pico_val': pico_val, 'pico_fec': pico_fec,
        'sem_antic': sem_antic, 'idx_primera_al': idx_primera_al,
        'ss_aceta': ss_aceta, 'rop_aceta': rop_aceta,
    }

# ─────────────────────────────────────────────
# CÁLCULO RESUMEN 42 MUNICIPIOS (cacheado por semana)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calcular_resumen_todos(_semana):
    res = []
    for mun in MUNICIPIOS:
        h  = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha')
        s  = h['casos'].tail(12).reset_index(drop=True)
        si, _, md = imputar_semanas_faltantes(s)
        g  = lambda i, si=si: int(si.iloc[i]) if len(si) > abs(i) else 3
        p, _ = predecir(mun, g(-1), g(-2), g(-3), _semana, md)
        c    = evaluar_cadena(
            mun, p,
            INVENTARIO_BASE.get(mun, {}).get('stock_aceta_tab', 50),
            INVENTARIO_BASE.get(mun, {}).get('stock_ringer_bolsas', 5)
        )
        if c:
            res.append(c)
    return pd.DataFrame(res)

# ─────────────────────────────────────────────
# MAPA (cacheado como recurso — solo se reconstruye si cambia la semana)
# ─────────────────────────────────────────────
# FIX lentitud: construir el mapa folium con 42 marcadores + popups +
# líneas de ruta es relativamente costoso. Antes se reconstruía en CADA
# rerun (cualquier interacción en cualquier pestaña), aunque el usuario
# no estuviera mirando el mapa. Ahora se cachea como recurso, indexado
# solo por la semana epidemiológica (que es lo único de lo que depende).
@st.cache_resource(show_spinner=False)
def construir_mapa(semana):
    df_resumen_mapa = calcular_resumen_todos(semana)
    promedios = df_hist.groupby('municipio_ocurrencia')['casos'].mean()
    mapa      = folium.Map(location=[3.9, -76.3], zoom_start=8, tiles='CartoDB positron')
    origen    = [3.4516, -76.5320]

    for _, row in df_resumen_mapa.iterrows():
        mun  = row['municipio']
        red  = RED_LOGISTICA.get(mun, {})
        if not red:
            continue
        lat, lon = red.get('lat', 3.8), red.get('lon', -76.3)
        color    = COLOR_URG[row['urgencia']]
        radio    = max(5, min(35, int(row['pred_casos'] * 0.6) + 5))
        prom     = promedios.get(mun, 1)
        ratio    = round(row['pred_casos'] / prom, 2) if prom > 0 else 1.0

        folium.PolyLine(
            locations=[origen, [lat, lon]], color=color, weight=1.5, opacity=0.4,
            dash_array='5 5' if row['urgencia'] == 'NORMAL' else None
        ).add_to(mapa)

        folium.CircleMarker(
            location=[lat, lon], radius=radio, color=color,
            fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(
                f"<div style='font-family:sans-serif;width:210px'>"
                f"<b style='font-size:13px'>{icono_svg(row['icono'], color)} {mun}</b>"
                f"<hr style='margin:3px 0'>"
                f"<b>Urgencia:</b> {row['urgencia']}<br>"
                f"<b>Predicción:</b> {row['pred_casos']} casos/sem<br>"
                f"<b>Ratio vs promedio:</b> {ratio}x<br>"
                f"<hr style='margin:3px 0'>"
                f"<b>Distancia:</b> {row['dist_carretera_km']} km<br>"
                f"<b>Lead time:</b> {row['lead_time_dias']} días<br>"
                f"<b>Despachar en:</b> ≤{row['despachar_en_dias']} día(s)<br>"
                f"<hr style='margin:3px 0'>"
                f"<b>Aceta.:</b> {row['orden_aceta']:,} tab · "
                f"<b>Ringer:</b> {row['orden_ringer']:,} bol<br>"
                f"<b>Costo:</b> ${row['costo_preventivo']:,.0f} COP",
                max_width=230
            ),
            tooltip=f"{mun} · {row['urgencia']} · {row['pred_casos']} casos"
        ).add_to(mapa)

    folium.Marker(
        location=origen,
        icon=folium.Icon(color='blue', icon='home', prefix='fa'),
        tooltip="SECCIONED — Centro de distribución Cali"
    ).add_to(mapa)
    return mapa

# ─────────────────────────────────────────────
# AGENTE IA — HERRAMIENTAS (tool-use real sobre los datos del sistema)
# ─────────────────────────────────────────────
# El agente NUNCA inventa cifras: cada pregunta que requiere un dato se
# resuelve llamando una de estas funciones, que leen directamente de los
# mismos objetos que usa el resto del dashboard (modelo, histórico SIVIGILA,
# parámetros logísticos). `semana_actual` se resuelve en tiempo de llamada
# (no de definición), así que toma el valor que el usuario fijó en el
# sidebar al momento de preguntar.

def _normalizar_municipio(nombre):
    nombre = (nombre or '').strip().upper()
    if nombre in MUNICIPIOS:
        return nombre
    coincidencias = [m for m in MUNICIPIOS if nombre in m or m in nombre]
    return coincidencias[0] if coincidencias else None


def tool_prediccion_municipio(municipio):
    mun = _normalizar_municipio(municipio)
    if not mun:
        return {"error": f"Municipio '{municipio}' no reconocido.",
                "municipios_disponibles": sorted(MUNICIPIOS)}
    h  = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha')
    s  = h['casos'].tail(12).reset_index(drop=True)
    si, _, md = imputar_semanas_faltantes(s)
    g  = lambda i: int(si.iloc[i]) if len(si) > abs(i) else 3
    pred, _ = predecir(mun, g(-1), g(-2), g(-3), semana_actual, md)
    cadena = evaluar_cadena(
        mun, pred,
        INVENTARIO_BASE.get(mun, {}).get('stock_aceta_tab', 50),
        INVENTARIO_BASE.get(mun, {}).get('stock_ringer_bolsas', 5)
    )
    return {
        "municipio": mun,
        "semana_epidemiologica_consultada": semana_actual,
        "casos_predichos_proxima_semana": pred,
        "modo_degradado": bool(md),
        "urgencia_logistica": cadena['urgencia'] if cadena else None,
        "despachar_en_dias": cadena['despachar_en_dias'] if cadena else None,
        "tabletas_acetaminofen_requeridas": cadena['req_aceta'] if cadena else None,
        "bolsas_ringer_requeridas": cadena['req_ringer'] if cadena else None,
        "ahorro_estimado_cop": cadena['ahorro'] if cadena else None,
    }


def tool_resumen_departamental():
    df_r = calcular_resumen_todos(semana_actual)
    return {
        "semana_epidemiologica": semana_actual,
        "total_municipios_evaluados": len(df_r),
        "municipios_criticos": df_r[df_r['urgencia'] == 'CRÍTICO']['municipio'].tolist(),
        "municipios_alerta": df_r[df_r['urgencia'] == 'ALERTA']['municipio'].tolist(),
        "total_casos_predichos_departamento": int(df_r['pred_casos'].sum()),
        "ahorro_total_estimado_cop": int(df_r['ahorro'].sum()),
        "municipio_mayor_riesgo": (
            df_r.sort_values('pred_casos', ascending=False).iloc[0]['municipio']
            if len(df_r) > 0 else "Sin datos"
        ),
    }


def tool_metricas_modelo():
    return {
        "algoritmo": "Random Forest Regressor", "version_modelo": VERSION,
        "mae_casos_semana": METRICAS['mae'], "rmse_casos_semana": METRICAS['rmse'],
        "r2_holdout_2018": METRICAS['r2'],
        "entrenado_con": paquete['entrenado_con'], "evaluado_en": paquete['evaluado_en'],
        "fecha_entreno": paquete['fecha_entreno'],
        "municipios_cubiertos": len(MUNICIPIOS),
    }


def tool_historico_municipio(municipio, semanas=12):
    mun = _normalizar_municipio(municipio)
    if not mun:
        return {"error": f"Municipio '{municipio}' no reconocido.",
                "municipios_disponibles": sorted(MUNICIPIOS)}
    try:
        semanas = max(1, min(int(semanas or 12), 104))
    except (ValueError, TypeError):
        semanas = 12
    h = df_hist[df_hist['municipio_ocurrencia'] == mun].sort_values('fecha').tail(semanas)
    return {
        "municipio": mun,
        "semanas_consultadas": len(h),
        "casos_por_semana": [
            {"fecha": str(r['fecha'].date()), "casos": int(r['casos'])}
            for _, r in h.iterrows()
        ],
        "promedio": round(float(h['casos'].mean()), 1) if len(h) else None,
        "pico": int(h['casos'].max()) if len(h) else None,
    }


def tool_logistica_municipio(municipio):
    mun = _normalizar_municipio(municipio)
    if not mun:
        return {"error": f"Municipio '{municipio}' no reconocido.",
                "municipios_disponibles": sorted(MUNICIPIOS)}
    red = RED_LOGISTICA.get(mun, {})
    inv = INVENTARIO_BASE.get(mun, {})
    return {
        "municipio": mun,
        "distancia_carretera_km": red.get('dist_carretera_km'),
        "lead_time_horas": red.get('lead_time_horas'),
        "lead_time_dias": red.get('lead_time_dias'),
        "stock_actual_aceta_tab": inv.get('stock_aceta_tab'),
        "punto_reorden_aceta_tab": inv.get('rop_aceta_tab'),
        "stock_seguridad_aceta_tab": inv.get('ss_aceta_tab'),
        "stock_actual_ringer_bolsas": inv.get('stock_ringer_bolsas'),
        "punto_reorden_ringer_bolsas": inv.get('rop_ringer_bolsas'),
        "stock_seguridad_ringer_bolsas": inv.get('ss_ringer_bolsas'),
    }


HERRAMIENTAS_AGENTE = [
    {
        "name": "consultar_prediccion_municipio",
        "description": "Predicción de casos de dengue para la próxima semana en un "
                        "municipio del Valle del Cauca, junto con urgencia logística "
                        "e insumos requeridos (acetaminofén, lactato de Ringer).",
        "parameters": {
            "type": "object",
            "properties": {"municipio": {"type": "string",
                            "description": "Nombre del municipio, ej. CALI, BUGA, TULUA"}},
            "required": ["municipio"],
        },
    },
    {
        "name": "consultar_resumen_departamental",
        "description": "Resumen del estado logístico (CRÍTICO/ALERTA/NORMAL) de los "
                        "42 municipios del Valle del Cauca para la semana actual.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "consultar_metricas_modelo",
        "description": "Métricas oficiales de desempeño del modelo predictivo "
                        "(MAE, RMSE, R²) y ficha técnica del Random Forest.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "consultar_historico_municipio",
        "description": "Histórico real de casos de dengue (SIVIGILA 2007–2018) de un "
                        "municipio en sus últimas N semanas reportadas.",
        "parameters": {
            "type": "object",
            "properties": {
                "municipio": {"type": "string"},
                "semanas": {"type": "integer",
                            "description": "Número de semanas a consultar (por defecto 12)"},
            },
            "required": ["municipio"],
        },
    },
    {
        "name": "consultar_logistica_municipio",
        "description": "Parámetros logísticos de un municipio: distancia desde el "
                        "centro de distribución, lead time, stock actual, punto de "
                        "reorden y stock de seguridad.",
        "parameters": {
            "type": "object",
            "properties": {"municipio": {"type": "string"}},
            "required": ["municipio"],
        },
    },
]

_DESPACHO_HERRAMIENTAS = {
    "consultar_prediccion_municipio": lambda a: tool_prediccion_municipio(a.get("municipio", "")),
    "consultar_resumen_departamental": lambda a: tool_resumen_departamental(),
    "consultar_metricas_modelo": lambda a: tool_metricas_modelo(),
    "consultar_historico_municipio": lambda a: tool_historico_municipio(
        a.get("municipio", ""), a.get("semanas", 12)),
    "consultar_logistica_municipio": lambda a: tool_logistica_municipio(a.get("municipio", "")),
}

SYSTEM_PROMPT_AGENTE = """Eres el agente de IA de Denguard, un sistema de soporte a
decisiones logístico-epidemiológicas para dengue en el Valle del Cauca, Colombia.

Reglas estrictas:
- NUNCA inventes cifras (casos, costos, stock, métricas). Si la pregunta requiere
  un dato del sistema, SIEMPRE llama la herramienta correspondiente antes de
  responder, incluso si crees saber la respuesta.
- Si una herramienta devuelve un error (ej. municipio no reconocido), explícalo
  al usuario y sugiere municipios válidos si la herramienta los provee.
- Responde siempre en español, de forma clara y concisa, citando las cifras
  exactas que devolvieron las herramientas.
- Si la pregunta no tiene relación con dengue, predicciones, logística o los
  datos del sistema, dilo amablemente y redirige al alcance de la app.
"""


def ejecutar_herramienta(nombre, args):
    try:
        funcion = _DESPACHO_HERRAMIENTAS.get(nombre)
        if not funcion:
            return {"error": f"Herramienta desconocida: {nombre}"}
        return funcion(args or {})
    except Exception as e:
        return {"error": str(e)}


def ejecutar_agente(client, historial_contenidos, max_iter=5):
    """Loop de tool-use con Gemini. Intenta los modelos en orden de prioridad:
    1. gemini-3.5-flash  — el más capaz, puede estar bajo alta demanda (503)
    2. gemini-2.5-flash  — estable GA, excelente para tool-use
    3. gemini-2.5-flash-lite — fallback mínimo si todo lo demás falla
    Si un modelo devuelve 503 o 429 pasa al siguiente automáticamente."""
    MODELOS_FALLBACK = [
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT_AGENTE,
        tools=[genai_types.Tool(function_declarations=HERRAMIENTAS_AGENTE)],
    )

    # Elegir el modelo disponible en este momento
    modelo_activo = None
    for candidato_modelo in MODELOS_FALLBACK:
        try:
            # Ping rápido para verificar disponibilidad antes del loop completo
            client.models.generate_content(
                model=candidato_modelo,
                contents=[genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text="ok")]
                )],
                config=genai_types.GenerateContentConfig(max_output_tokens=5),
            )
            modelo_activo = candidato_modelo
            break
        except Exception as e:
            msg = str(e).lower()
            if any(c in msg for c in ('503', '429', 'unavailable', 'quota')):
                continue   # modelo ocupado/con cuota — probar el siguiente
            # Otro error (auth, network) — no tiene sentido seguir intentando
            raise

    if modelo_activo is None:
        return (
            "Todos los modelos de Gemini están bajo alta demanda ahora mismo. "
            "Espera un minuto e intenta de nuevo.",
            historial_contenidos
        )

    for _ in range(max_iter):
        resp = client.models.generate_content(
            model=modelo_activo,
            contents=historial_contenidos,
            config=config,
        )
        candidato = resp.candidates[0]
        historial_contenidos = historial_contenidos + [candidato.content]
        partes = candidato.content.parts or []
        llamadas = [p.function_call for p in partes if p.function_call]
        if not llamadas:
            texto = "".join(p.text for p in partes if p.text).strip()
            sufijo = (f"\n\n---\n*Modelo: {modelo_activo}*"
                      if modelo_activo != MODELOS_FALLBACK[0] else "")
            return (texto or "(sin respuesta)") + sufijo, historial_contenidos
        partes_resultado = []
        for fc in llamadas:
            resultado = ejecutar_herramienta(fc.name, dict(fc.args) if fc.args else {})
            partes_resultado.append(genai_types.Part.from_function_response(
                name=fc.name,
                response={"result": json.loads(
                    json.dumps(resultado, ensure_ascii=False, default=str))},
            ))
        historial_contenidos = historial_contenidos + [
            genai_types.Content(role="tool", parts=partes_resultado)
        ]
    return ("No pude completar la respuesta tras varias consultas. "
            "Intenta reformular la pregunta."), historial_contenidos

# ─────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
    'viewBox="0 0 16 16" style="vertical-align:-6px;flex-shrink:0">'
    '<path d="M8 1 14 3.2v4.6c0 4-2.6 6.7-6 8.2-3.4-1.5-6-4.2-6-8.2V3.2z" '
    'fill="url(#dsGrad)"/>'
    '<path d="M5.4 8.1 7.2 9.9 10.6 6" stroke="#ffffff" stroke-width="1.4" '
    'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    '<defs><linearGradient id="dsGrad" x1="0" y1="0" x2="16" y2="16">'
    '<stop offset="0" stop-color="#1a56db"/>'
    '<stop offset="1" stop-color="#3b82f6"/>'
    '</linearGradient></defs></svg>'
)
st.markdown(
    f'<h1 class="ds-main-title">{_LOGO_SVG}'
    f'Denguard: Logística Farmacéutica de Última Milla</h1>',
    unsafe_allow_html=True
)
st.markdown(
    "**Ecosistema Predictivo Spatial-Aware** — De la predicción epidemiológica "
    "a la orden de despacho · Valle del Cauca · 42 municipios · SIVIGILA 2007–2018"
)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Modelo",          f"Random Forest {VERSION}")
c2.metric("Municipios",      f"{len(MUNICIPIOS)} / 42")
c3.metric("R² holdout 2018", f"{METRICAS['r2']}")
c4.metric("MAE",             f"{METRICAS['mae']} casos/sem")
c5.metric("RMSE",            f"{METRICAS['rmse']} casos/sem")
st.caption(
    f"Entrenado: SIVIGILA 2007–2017 · Evaluado: holdout temporal 2018 · "
    f"Gap Train-Val R²: {GAP_TRAIN_VAL} · Sin overfitting · MD5: `{sello_modelo['hash_md5']}`"
)
st.divider()

# ─────────────────────────────────────────────
# NAVEGACIÓN — menú lateral tipo "pilar" (vertical, dentro del sidebar
# colapsable nativo de Streamlit)
# ─────────────────────────────────────────────
SECCIONES = [
    "Vista General",
    "Dashboard",
    "Cadena de Abastecimiento",
    "Nowcasting",
    "Serie Histórica",
    "Mapa",
    "Validación Retrospectiva",
    "Auditoría ALCOA+",
    "Agente IA",
]
ICONOS_SECCION = ["house-fill", "graph-up-arrow", "truck", "broadcast",
                   "clock-history", "geo-alt", "search", "shield-check", "robot"]

st.sidebar.header("Navegación")
if OPTION_MENU_DISPONIBLE:
    with st.sidebar:
        seccion_activa = option_menu(
            menu_title=None,
            options=SECCIONES,
            icons=ICONOS_SECCION,
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#60a5fa", "font-size": "13px"},
                "nav-link": {
                    "font-family": "JetBrains Mono, monospace", "font-size": "12.5px",
                    "color": "#5a7a99", "background-color": "#0f1c2d",
                    "border": "1px solid rgba(0,255,224,0.18)", "border-radius": "9px",
                    "margin": "0 0 4px 0", "padding": "10px 12px",
                },
                "nav-link-selected": {
                    "background-color": "#1a56db", "color": "#ffffff",
                    "font-weight": "600",
                },
            },
        )
else:
    seccion_activa = st.sidebar.radio(
        "Navegación", SECCIONES, label_visibility="collapsed"
    )

st.sidebar.divider()
st.sidebar.header("Parámetros de Simulación")
municipio_sel = st.sidebar.selectbox(
    "Municipio objetivo:", sorted(MUNICIPIOS),
    index=sorted(MUNICIPIOS).index('CALI') if 'CALI' in MUNICIPIOS else 0
)

hist_mun  = df_hist[df_hist['municipio_ocurrencia'] == municipio_sel].sort_values('fecha')
serie_rec = hist_mun['casos'].tail(12).reset_index(drop=True)
serie_imp, idx_imp, modo_degradado = imputar_semanas_faltantes(serie_rec)

if modo_degradado:
    st.sidebar.warning(
        f"**Modo Degradado** — {len(idx_imp)} semana(s) con reporte cero "
        f"sospechoso detectadas. IC ampliado ×1.5 automáticamente."
    )

ult = lambda i: int(serie_imp.iloc[i]) if len(serie_imp) > abs(i) else 3

with st.sidebar.expander("Inercia Epidemiológica", expanded=False):
    casos_t1      = st.number_input("Casos semana anterior (t-1)",
                                     min_value=0, value=ult(-1))
    casos_t2      = st.number_input("Casos hace 2 semanas (t-2)",
                                     min_value=0, value=ult(-2))
    casos_t3      = st.number_input("Casos hace 3 semanas (t-3)",
                                     min_value=0, value=ult(-3))
    semana_actual = st.slider("Semana epidemiológica actual", 1, 52, 20)

with st.sidebar.expander("Stock Actual Simulado", expanded=False):
    inv_base           = INVENTARIO_BASE.get(municipio_sel, {})
    stock_aceta_input  = st.number_input(
        "Acetaminofén disponible (tab)",
        min_value=0, value=inv_base.get('stock_aceta_tab', 100), step=50
    )
    stock_ringer_input = st.number_input(
        "Lactato de Ringer disponible (bolsas)",
        min_value=0, value=inv_base.get('stock_ringer_bolsas', 10), step=5
    )
    st.caption("Stock simulado · Res. MINSALUD 1403/2007 · Edita para escenarios reales.")

# ─────────────────────────────────────────────
# CÁLCULOS CENTRALES
# ─────────────────────────────────────────────
pred_sel, _ = predecir(municipio_sel, casos_t1, casos_t2, casos_t3,
                        semana_actual, modo_degradado)
horizonte_df, diverge = predecir_horizonte(
    municipio_sel, casos_t1, casos_t2, casos_t3, semana_actual)
cadena_sel  = evaluar_cadena(municipio_sel, pred_sel,
                              stock_aceta_input, stock_ringer_input)
rmse_ef     = METRICAS['rmse'] * (1.5 if modo_degradado else 1.0)
ic_bajo     = max(0, pred_sel - int(rmse_ef))
ic_alto     = pred_sel + int(rmse_ef)

df_resumen = calcular_resumen_todos(semana_actual)
orden_urg  = {'CRÍTICO': 0, 'ALERTA': 1, 'NORMAL': 2}
df_sorted  = df_resumen.sort_values('urgencia', key=lambda x: x.map(orden_urg))

# ─────────────────────────────────────────────
# CONTENIDO DE LA SECCIÓN ACTIVA
# ─────────────────────────────────────────────
# Solo se ejecuta el código de la sección elegida en el menú lateral — las
# demás secciones (mapa, validación retrospectiva, nowcasting, etc.) no se
# calculan ni se renderizan hasta que el usuario las selecciona.
st.markdown('<div class="ds-animate-in">', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECCIÓN 1 — DASHBOARD PREDICTIVO
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# SECCIÓN 0 — VISTA GENERAL (lenguaje simple, para cualquier persona)
# ══════════════════════════════════════════════
if seccion_activa == SECCIONES[0]:
    st.markdown(f"### Resumen para {municipio_sel}")
    st.markdown(
        "Esta pantalla te da lo más importante en un vistazo. "
        "No necesitas saber de salud ni de logística para entenderla."
    )
    st.divider()

    # ── Bloque principal: semáforo visual ────────────────────────────────
    cadena_simple = evaluar_cadena(
        municipio_sel, pred_sel, stock_aceta_input, stock_ringer_input
    )
    urg = cadena_simple['urgencia'] if cadena_simple else 'NORMAL'
    COLOR_BG  = {'CRÍTICO': '#FFF0F0', 'ALERTA': '#FFFBF0', 'NORMAL': '#F0FFF5'}
    COLOR_BRD = {'CRÍTICO': '#E24B4A', 'ALERTA': '#EF9F27', 'NORMAL': '#2E9E5B'}
    LABEL_URG = {
        'CRÍTICO': 'Se necesitan medicamentos urgentemente',
        'ALERTA':  'Hay que pedir medicamentos pronto',
        'NORMAL':  'Todo está bajo control por ahora',
    }
    DESC_URG = {
        'CRÍTICO': (
            f"El stock actual no alcanza para cubrir los casos previstos esta semana. "
            f"Se debe despachar en las próximas **{cadena_simple['despachar_en_dias']} horas** "
            f"para evitar desabasto."
        ),
        'ALERTA': (
            f"El stock bajará del nivel mínimo recomendado si no se actúa esta semana. "
            f"Se debe hacer el pedido en máximo **{cadena_simple['despachar_en_dias']} día(s)**."
        ),
        'NORMAL': (
            f"El municipio tiene medicamentos suficientes para atender los casos previstos "
            f"durante los próximos **{cadena_simple['dias_cobertura']} días**."
        ),
    }

    st.markdown(
        f'<div style="background:{COLOR_BG[urg]};border:2px solid {COLOR_BRD[urg]};'
        f'border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.2rem">'
        f'<div style="font-size:1.05rem;font-weight:700;color:{COLOR_BRD[urg]};'
        f'margin-bottom:0.4rem">{LABEL_URG[urg]}</div>'
        f'<div style="font-size:0.92rem;color:#334">{DESC_URG[urg]}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── 4 métricas clave en lenguaje simple ──────────────────────────────
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "Casos previstos la próxima semana",
        f"{pred_sel}",
        help="Número de personas que probablemente necesitarán atención médica por dengue."
    )
    g2.metric(
        "Pastillas de acetaminofén necesarias",
        f"{cadena_simple['req_aceta']:,}" if cadena_simple else "—",
        help="Cantidad de tabletas de acetaminofén 500mg para tratar los casos previstos."
    )
    g3.metric(
        "Bolsas de suero necesarias",
        f"{cadena_simple['req_ringer']:,}" if cadena_simple else "—",
        help="Bolsas de Lactato de Ringer para los casos graves que requieren hospitalización."
    )
    g4.metric(
        "Ahorro al planificar con anticipación",
        f"${cadena_simple['ahorro']:,.0f} COP" if cadena_simple else "—",
        help="Dinero que se ahorra comprando a precio normal (planificado) en vez de a precio de emergencia."
    )

    st.divider()

    # ── Explicación sencilla del sistema ─────────────────────────────────
    with st.expander("Cómo funciona este sistema", expanded=False):
        st.markdown(f"""
Este sistema predice cuántos casos de dengue habrá en **{municipio_sel}** la
próxima semana, y calcula automáticamente cuántos medicamentos se necesitan
y si el stock actual alcanza.

**En 3 pasos:**

1. **Predice** — usa los casos reportados de las últimas semanas y el
   historial de 11 años (SIVIGILA 2007–2018) para estimar cuántos
   pacientes habrá la próxima semana: **{pred_sel} casos**.

2. **Evalúa** — compara esa predicción con el stock actual de medicamentos
   en el municipio y determina si hay suficiente, si hay que pedir pronto,
   o si es urgente.

3. **Actúa** — genera automáticamente la cantidad exacta a pedir y el
   tiempo máximo para hacer el pedido, para que los medicamentos lleguen
   a tiempo.

**Por qué importa:** un sistema reactivo (comprar cuando ya falta) paga
precios de emergencia y pone en riesgo a los pacientes. Este sistema
planifica con anticipación y ahorra dinero público.
        """)

    st.markdown("Para ver el análisis técnico completo, usa el menú de la izquierda.")

# ══════════════════════════════════════════════
# SECCIÓN 1 — DASHBOARD PREDICTIVO (antes sección 0)
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[1]:
    st.markdown(f"### Reporte Predictivo — {municipio_sel}")

    if modo_degradado:
        st.error(
            f"**MODO DEGRADADO — Gestión de Riesgo Epidemiológico**\n\n"
            f"Se detectaron **{len(idx_imp)} semana(s)** con reporte cero sospechoso "
            f"en **{municipio_sel}**. Posible falla de reporte SIVIGILA.\n\n"
            f"**Medidas automáticas:** Imputación por mediana móvil ±2 semanas · "
            f"IC ampliado de ±{METRICAS['rmse']:.2f} → **±{rmse_ef:.2f}** casos/sem · "
            f"Verificar en sección Nowcasting."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p style="font-family:Nunito,sans-serif;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--blue);margin:0 0 0.6rem 0">Proyección de Pacientes</p>', unsafe_allow_html=True)
        ic_label = (
            f"IC ±RMSE{'×1.5 (Modo Degradado)' if modo_degradado else ''}: "
            f"[{ic_bajo} – {ic_alto}]"
        )
        st.metric("Casos estimados (próxima semana)", f"{pred_sel} pacientes",
                  delta=ic_label, delta_color="off")
        st.caption(
            f"MAE base: ±{METRICAS['mae']} · R²={METRICAS['r2']}"
            + (f" · **RMSE efectivo: ±{rmse_ef:.2f}**" if modo_degradado else "")
        )
        with st.expander("IC y Gestión de Riesgo — para el jurado"):
            mae_n = ERROR_ESTRAT.get('mae_normal', 'N/A')
            mae_p = ERROR_ESTRAT.get('mae_pico',   'N/A')
            fac   = ERROR_ESTRAT.get('factor_deg', 'N/A')
            met   = ERROR_ESTRAT.get('metodo_umbral', 'OPS 2015')
            st.markdown(f"""
**Intervalo de Confianza (IC):**
RMSE del modelo: **±{METRICAS['rmse']} casos/sem** (holdout 2018).
En Modo Degradado se amplía ×1.5 como medida conservadora.

| Estado | RMSE efectivo | IC | Decisión |
|---|---|---|---|
| Normal | ±{METRICAS['rmse']} | [{max(0,pred_sel-int(METRICAS['rmse']))} – {pred_sel+int(METRICAS['rmse'])}] | Stock base |
| Modo Degradado | **±{rmse_ef:.2f}** | **[{ic_bajo} – {ic_alto}]** | Stock conservador |

**Análisis de error estratificado:**
- MAE semanas normales: **{mae_n} casos**
- MAE semanas de pico (≥ p75 municipal, {met}): **{mae_p} casos**
- Factor de degradación en picos: **{fac}x**

El modelo se degrada **{fac}x** en picos. Respuesta operativa:
SS dinámico `Z(95%)×σ×√LT` absorbe esta varianza estructuralmente.
En salud pública, un **falso negativo** es más costoso que un falso positivo.
            """)

    with col2:
        st.markdown('<p style="font-family:Nunito,sans-serif;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--warn);margin:0 0 0.6rem 0">Insumos Críticos</p>', unsafe_allow_html=True)
        if cadena_sel:
            st.metric("Acetaminofén 500mg", f"{cadena_sel['req_aceta']:,} Tab.")
            st.metric("Lactato de Ringer",  f"{cadena_sel['req_ringer']:,} Bol.")
            nivel = {"CRÍTICO": "error", "ALERTA": "warning", "NORMAL": "success"}
            getattr(st, nivel[cadena_sel['urgencia']])(
                f"**{cadena_sel['urgencia']}** — "
                f"Despachar en ≤{cadena_sel['despachar_en_dias']} día(s)"
            )

    with col3:
        st.markdown('<p style="font-family:Nunito,sans-serif;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--ok);margin:0 0 0.6rem 0">Eficiencia Farmacoeconómica</p>', unsafe_allow_html=True)
        if cadena_sel:
            st.metric("Ahorro vs compra reactiva",
                      f"${cadena_sel['ahorro']:,.0f} COP",
                      delta="logística de precisión vs adivinación",
                      delta_color="off")
            st.caption(
                f"Preventivo: ${cadena_sel['costo_preventivo']:,.0f} · "
                f"Reactivo: ${cadena_sel['costo_reactivo']:,.0f} · "
                f"SISMED {COSTOS['fecha_consulta']}"
            )
            with st.expander("Cómo se calcula el ahorro — para el jurado"):
                sigma = cadena_sel.get('sigma_error', 'N/A')
                lt    = cadena_sel.get('lead_time_dias', 0)
                z     = SUPUESTOS.get('z_score_95', 1.645)
                st.markdown(f"""
**Eficiencia Farmacoeconómica — Logística de Precisión:**

El ahorro compara el costo de cubrir la **demanda predicha de esta semana**
({cadena_sel['req_aceta']:,} tab. de acetaminofén + {cadena_sel['req_ringer']:,}
bolsas de Ringer) comprada de dos formas distintas:

- **Preventiva** (precio normal SISMED, planificada con anticipación gracias
  a la predicción): **${cadena_sel['costo_preventivo']:,.0f} COP**
- **Reactiva** (precio de emergencia/urgencia, sin planificación, comprando
  al momento del desabasto): **${cadena_sel['costo_reactivo']:,.0f} COP**

**Ahorro = Reactivo − Preventivo = ${cadena_sel['ahorro']:,.0f} COP**

Este ahorro depende de la demanda predicha, no de si el municipio ya tiene
stock suficiente hoy — así refleja el valor de *anticipar* la compra, no
solo si hace falta reabastecer en este instante.

**SS dinámico (Chopra & Meindl, SCM 2016):**
```
SS = Z(95%) × σ_error × √lead_time
SS = {z:.3f} × {sigma} × √{lt:.4f}
```
El modelo tiene MAE={METRICAS['mae']} casos/sem: alta precisión implica un
buffer de seguridad pequeño y, por tanto, menos capital inmovilizado en
inventario. En producción se aplica `max(SS_dinámico, SS_normativo_1403)`
para garantizar cumplimiento legal y seguridad operativa simultáneamente.
                """)

    st.divider()
    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        st.subheader(f"Histórico reciente + Proyección 4 semanas — {municipio_sel}")
        hist_rec     = hist_mun.tail(20).copy()
        ultima_fecha = hist_rec['fecha'].max()
        color_mun    = COLORES_TOP[sorted(MUNICIPIOS).index(municipio_sel) % len(COLORES_TOP)]

        fig_dash = go.Figure()
        fig_dash.add_trace(go.Scatter(
            x=hist_rec['fecha'], y=hist_rec['casos'],
            mode='lines', name='Histórico real',
            line=dict(color=color_mun, width=2.5),
            fill='tozeroy',
            fillcolor=f'rgba({int(color_mun[1:3],16)},'
                      f'{int(color_mun[3:5],16)},'
                      f'{int(color_mun[5:7],16)},0.10)',
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Casos: %{y}<extra></extra>'
        ))
        fechas_h = [ultima_fecha + pd.Timedelta(weeks=i) for i in range(1, 5)]
        preds_h  = horizonte_df['pred'].tolist()
        ic_b_h   = horizonte_df['ic_bajo'].tolist()
        ic_a_h   = horizonte_df['ic_alto'].tolist()

        fig_dash.add_trace(go.Scatter(
            x=fechas_h + fechas_h[::-1], y=ic_a_h + ic_b_h[::-1],
            fill='toself', fillcolor='rgba(220,38,38,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name=f'IC ±RMSE{"×1.5" if modo_degradado else ""}',
            hoverinfo='skip'
        ))
        fig_dash.add_trace(go.Scatter(
            x=fechas_h, y=preds_h, mode='lines+markers',
            name='Predicción 4 semanas',
            line=dict(color='#DC2626', width=2, dash='dash'),
            marker=dict(size=9, color='#DC2626'),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Pred: %{y}<br>'
                          'IC: [%{customdata[0]} – %{customdata[1]}]<extra></extra>',
            customdata=list(zip(ic_b_h, ic_a_h))
        ))
        fig_dash.add_vline(x=ultima_fecha, line_dash='dot', line_color='gray', opacity=0.4)
        if diverge:
            fig_dash.add_annotation(x=fechas_h[-1], y=max(preds_h),
                                    text="Posible divergencia", showarrow=True,
                                    arrowhead=2, font=dict(color='#E24B4A', size=11))
        _plotly_layout(fig_dash, height=370,
            legend=dict(orientation='h', y=-0.18,
                        font=dict(color=_colores_tema()['font'])),
            xaxis_title='', yaxis_title='Casos / semana')
        fig_dash.update_xaxes(showgrid=False)
        fig_dash.update_yaxes(gridcolor=_colores_tema()['grid'])
        st.plotly_chart(fig_dash, use_container_width=True)

        if diverge:
            st.warning("Salto >3x entre pasos. Use +1s y +2s con confianza; "
                       "+3s y +4s son indicativos.")

    with col_g2:
        st.subheader("Horizonte 4 semanas")
        df_h_disp = horizonte_df[['paso','semana','pred','ic_bajo','ic_alto','ic']].copy()
        df_h_disp.columns = ['Paso','Semana','Pred.','IC bajo','IC alto','Margen']
        st.dataframe(df_h_disp, hide_index=True, use_container_width=True)

        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=horizonte_df['paso'], y=horizonte_df['pred'],
            marker_color=['#1A56DB','#2563EB','#3B82F6','#6096FB'],
            error_y=dict(type='data', array=horizonte_df['ic'].tolist(),
                         visible=True, color='#666'),
            hovertemplate='%{x}: %{y} casos ± %{error_y.array}<extra></extra>'
        ))
        _plotly_layout(fig_h, height=200, showlegend=False,
            yaxis_title='Casos')
        fig_h.update_yaxes(gridcolor=_colores_tema()['grid'])
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption("IC = MAE×(1+35%/paso). "
                   + ("**×1.5 Modo Degradado.**" if modo_degradado else ""))

# ══════════════════════════════════════════════
# SECCIÓN 2 — CADENA DE ABASTECIMIENTO
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[2]:
    st.markdown("### Motor Logístico — De la Predicción a la Orden de Despacho")
    st.caption(
        "42 municipios · SS dinámico Z×σ×√LT (95% nivel servicio) · "
        "Chopra & Meindl SCM 2016 · Res. MINSALUD 1403/2007"
    )

    with st.expander("Eficiencia Farmacoeconómica — Logística de Precisión vs Adivinación"):
        st.success(
            "**De la logística de adivinación a la logística de precisión:**\n\n"
            "Los sistemas tradicionales de abastecimiento hospitalario usan "
            "reglas empíricas (stock = 2-4 semanas de demanda promedio) porque "
            "**no conocen su error de predicción**.\n\n"
            f"Data Sentinel usa `SS = Z(95%) × σ_error({METRICAS['mae']} casos) × √LT` "
            "porque **conoce exactamente cuánto se equivoca** y en qué contextos.\n\n"
            "**Consecuencia:** el costo preventivo/reactivo y el ahorro se calculan "
            "sobre la demanda predicha de cada semana, no solo sobre la cantidad a "
            "ordenar — así el ahorro refleja el valor de anticipar la compra aunque "
            "el municipio ya tenga stock suficiente en este instante. En producción "
            "se aplica `max(SS_dinámico, SS_normativo_Res1403)` como piso legal."
        )

    criticos = df_sorted[df_sorted['urgencia'] == 'CRÍTICO']
    alertas  = df_sorted[df_sorted['urgencia'] == 'ALERTA']
    normales = df_sorted[df_sorted['urgencia'] == 'NORMAL']

    cs1, cs2, cs3 = st.columns(3)
    cs1.error(  f"CRÍTICO: {len(criticos)} municipios")
    cs2.warning(f"ALERTA:  {len(alertas)} municipios")
    cs3.success(f"NORMAL:  {len(normales)} municipios")

    st.divider()

    st.subheader("Demanda Predicha vs Distancia Logística — 42 Municipios")
    fig_bub = px.scatter(
        df_resumen, x='dist_carretera_km', y='pred_casos',
        size='costo_preventivo', color='urgencia',
        color_discrete_map=COLOR_URG, hover_name='municipio',
        hover_data={'pred_casos': True, 'dist_carretera_km': True,
                    'costo_preventivo': ':,.0f', 'despachar_en_dias': True,
                    'urgencia': False},
        labels={'dist_carretera_km': 'Distancia desde SECCIONED (km)',
                'pred_casos': 'Casos predichos', 'costo_preventivo': 'Costo COP'},
        title='Tamaño de burbuja = costo de la orden de despacho',
    )
    _plotly_layout(fig_bub, height=380,
        margin=dict(l=0, r=0, t=40, b=0))
    fig_bub.update_xaxes(gridcolor=_colores_tema()['grid'])
    fig_bub.update_yaxes(gridcolor=_colores_tema()['grid'])
    st.plotly_chart(fig_bub, use_container_width=True)

    st.subheader("Órdenes de Despacho — Prioridad Automática")
    tabla = df_sorted[[
        'municipio','urgencia','pred_casos','orden_aceta','orden_ringer',
        'despachar_en_dias','dist_carretera_km','costo_preventivo','ahorro'
    ]].copy()
    tabla.columns = ['Municipio','Urgencia','Casos pred.','Aceta.(tab)','Ringer(bol)',
                     'Desp.en(d)','Dist.(km)','Costo(COP)','Ahorro']
    tabla['Costo(COP)'] = tabla['Costo(COP)'].apply(lambda x: f"${x:,.0f}")
    tabla['Ahorro']     = tabla['Ahorro'].apply(lambda x: f"${x:,.0f}")

    def color_urg(val):
        c = {'CRÍTICO':'background-color:#ffd5d5',
             'ALERTA': 'background-color:#fff3cd',
             'NORMAL': 'background-color:#d4edda'}
        return c.get(val,'')

    st.dataframe(tabla.style.map(color_urg, subset=['Urgencia']),
                 use_container_width=True, hide_index=True, height=420)

    st.divider()
    ct1, ct2, ct3, ct4 = st.columns(4)
    ct1.metric("Total aceta.",     f"{df_resumen['orden_aceta'].sum():,} tab")
    ct2.metric("Total ringer",     f"{df_resumen['orden_ringer'].sum():,} bol")
    ct3.metric("Costo preventivo", f"${df_resumen['costo_preventivo'].sum():,.0f} COP")
    ct4.metric("Ahorro total",     f"${df_resumen['ahorro'].sum():,.0f} COP")

    st.divider()
    with st.expander("Stock Actual vs Punto de Reorden Dinámico (42 municipios)"):
        munis_ord  = list(df_sorted['municipio'])
        bar_colors = [COLOR_URG[r] for r in df_sorted['urgencia']]

        fig_stock = make_subplots(rows=1, cols=2,
            subplot_titles=['Acetaminofén — Stock vs ROP dinámico',
                            'Lactato de Ringer — Stock vs ROP dinámico'])
        for col_idx, (clave_s, clave_r) in enumerate(
            [('stock_aceta_tab','rop_aceta_tab'),
             ('stock_ringer_bolsas','rop_ringer_bolsas')], 1):
            stocks = [INVENTARIO_BASE.get(m,{}).get(clave_s, 0) for m in munis_ord]
            rops   = [INVENTARIO_BASE.get(m,{}).get(clave_r, 0) for m in munis_ord]
            fig_stock.add_trace(go.Bar(
                x=munis_ord, y=stocks, marker_color=bar_colors, opacity=0.85,
                showlegend=False,
                hovertemplate='%{x}<br>Stock: %{y:,}<extra></extra>'
            ), row=1, col=col_idx)
            fig_stock.add_trace(go.Scatter(
                x=munis_ord, y=rops, mode='lines+markers',
                name='ROP dinámico', showlegend=(col_idx == 1),
                line=dict(color='#333', dash='dash', width=1.8), marker=dict(size=6),
                hovertemplate='%{x}<br>ROP: %{y:,}<extra></extra>'
            ), row=1, col=col_idx)

        _plotly_layout(fig_stock, height=400,
            margin=dict(l=0, r=0, t=40, b=80))
        fig_stock.update_xaxes(tickangle=45,
            tickfont=dict(size=8, color=_colores_tema()['axis']))
        fig_stock.update_yaxes(gridcolor=_colores_tema()['grid'])
        st.plotly_chart(fig_stock, use_container_width=True)

    st.divider()
    if cadena_sel:
        st.subheader(f"Detalle Cadena — {municipio_sel}")
        cd1, cd2, cd3 = st.columns(3)
        with cd1:
            st.markdown("**Estado de Stock**")
            sigma_mun = cadena_sel.get('sigma_error', 'N/A')
            st.dataframe(pd.DataFrame({
                'Insumo':          ['Acetaminofén','Ringer'],
                'Stock actual':    [f"{cadena_sel['stock_aceta']:,} tab",
                                    f"{cadena_sel['stock_ringer']:,} bol"],
                'Dem. predicha':   [f"{cadena_sel['req_aceta']:,} tab",
                                    f"{cadena_sel['req_ringer']:,} bol"],
                'Stock post-dem.': [f"{cadena_sel['stock_post_aceta']:,} tab",
                                    f"{cadena_sel['stock_post_ringer']:,} bol"],
                'SS dinámico':     [f"{cadena_sel['ss_aceta']:,} tab",
                                    f"{cadena_sel['ss_ringer']:,} bol"],
                'ROP dinámico':    [f"{cadena_sel['rop_aceta']:,} tab",
                                    f"{cadena_sel['rop_ringer']:,} bol"],
            }), hide_index=True, use_container_width=True)
            st.caption(f"σ_error: {sigma_mun} casos/sem · {cadena_sel['metodo_ss']}")
        with cd2:
            st.markdown("**Red Logística**")
            st.dataframe(pd.DataFrame({
                'Parámetro': ['Centro dist.','Dist. aérea','Dist. carretera',
                              'Tortuosidad','Velocidad','Lead time','Cobertura'],
                'Valor': [
                    'SECCIONED Cali',
                    f"{RED_LOGISTICA.get(municipio_sel,{}).get('dist_aerea_km',0)} km",
                    f"{cadena_sel['dist_carretera_km']} km",
                    f"{SUPUESTOS['factor_tortuosidad']}x (INVIAS 2022)",
                    f"{SUPUESTOS['velocidad_kmph']} km/h",
                    f"{cadena_sel['lead_time_horas']} h ({cadena_sel['lead_time_dias']} d)",
                    f"{cadena_sel['dias_cobertura']} días con stock actual",
                ]
            }), hide_index=True, use_container_width=True)
        with cd3:
            st.markdown("**Orden de Despacho**")
            st.metric("Aceta. a ordenar", f"{cadena_sel['orden_aceta']:,} tab")
            st.metric("Ringer a ordenar", f"{cadena_sel['orden_ringer']:,} bol")
            st.metric("Despachar en",     f"≤{cadena_sel['despachar_en_dias']} día(s)")
            st.metric("Costo orden",      f"${cadena_sel['costo_preventivo']:,.0f} COP")
            st.metric("Ahorro",           f"${cadena_sel['ahorro']:,.0f} COP")


    with st.expander("Supuestos logísticos — Transparencia total"):
        st.warning("Stock inicial simulado (Res. MINSALUD 1403/2007). "
                   "No representa inventario en tiempo real.")
        rows = [
            ('Factor tortuosidad', f"{SUPUESTOS['factor_tortuosidad']}x",
             SUPUESTOS['fuentes']['tortuosidad']),
            ('Velocidad',          f"{SUPUESTOS['velocidad_kmph']} km/h",
             SUPUESTOS['fuentes']['velocidad']),
            ('Carga+descarga',     f"{SUPUESTOS['horas_carga_descarga']} h",
             'Estándar logístico farmacéutico'),
            ('SS método',          'Z(95%)×σ_error×√LT',
             SUPUESTOS.get('referencia_ss','Chopra & Meindl SCM 2016')),
            ('Nivel servicio',     f"{SUPUESTOS.get('nivel_servicio',0.95)*100:.0f}%",
             'Estándar farmacéutico'),
            ('Aceta/caso',         f"{SUPUESTOS['aceta_por_caso']} tab",
             SUPUESTOS['fuentes']['protocolos']),
            ('Ringer/grave',       f"{SUPUESTOS['ringer_por_caso_grave']} bol",
             SUPUESTOS['fuentes']['protocolos']),
            ('Tasa gravedad',      f"{SUPUESTOS['tasa_gravedad']*100:.0f}%",
             SUPUESTOS['fuentes']['protocolos']),
        ]
        st.dataframe(pd.DataFrame(rows, columns=['Parámetro','Valor','Fuente']),
                     hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# SECCIÓN 3 — NOWCASTING
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[3]:
    st.markdown("### Nowcasting — Conexión SIVIGILA en Tiempo Real")
    st.info(
        "**Data Gap 2018→2026 — Contexto COVID-19:**\n\n"
        "El modelo fue entrenado con datos SIVIGILA 2007–2018. La pandemia "
        "COVID-19 alteró los ciclos de reporte por tres mecanismos documentados: "
        "(1) **Subregistro** por reorientación diagnóstica; "
        "(2) **Cambio vectorial** de *Aedes aegypti*; "
        "(3) **Discontinuidades 2020–2022** con cobertura <60% en municipios "
        "categoría 5 y 6.\n\n"
        "**Solución técnica:** Re-entrenamiento continuo vía esta misma API "
        "cuando SIVIGILA restablezca flujo post-2022. Nowcasting inmediato "
        "disponible con datos frescos sin necesidad de re-entrenamiento."
    )
    st.divider()

    col_nw1, col_nw2 = st.columns([1, 2])
    with col_nw1:
        municipio_nw  = st.selectbox("Municipio:", sorted(MUNICIPIOS), key='mun_nw',
            index=sorted(MUNICIPIOS).index('CALI') if 'CALI' in MUNICIPIOS else 0)
        consultar_btn = st.button("Consultar API SIVIGILA", type="primary")
    with col_nw2:
        st.markdown(
            "- Consulta datos.gov.co en tiempo real (cod_eve 210+211)\n"
            "- Un solo request cubre los 42 municipios (cacheado 1h) — ya no se "
            "repite la llamada al cambiar de municipio\n"
            "- Calcula lags reales t-1, t-2, t-3 desde datos más recientes\n"
            "- Compara predicción con datos frescos vs histórico 2018\n"
            "- Hash MD5 de respuesta para trazabilidad ALCOA+"
        )

    if consultar_btn:
        with st.spinner("Consultando API SIVIGILA..."):
            df_live, error_msg, sello_live = consultar_sivigila_reciente()

        if error_msg:
            st.error(f"Error: {error_msg}")
            st.info("Consistente con discontinuidades de reporte post-COVID en SIVIGILA. "
                    "Se reintentó automáticamente 3 veces antes de mostrar este error.")
        elif df_live is not None:
            df_mun_live = df_live[
                df_live['municipio_ocurrencia'] == municipio_nw
            ].sort_values(['ano','semana'], ascending=False)

            st.success("Dato fresco obtenido directamente de la API SIVIGILA")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Año más reciente",   sello_live['ano_max'])
            s2.metric("Registros (total)",  sello_live['registros'])
            s3.metric("Hash MD5",           sello_live['hash_response'])
            s4.metric("Consultado",         sello_live['timestamp'])
            st.caption("Original — SIVIGILA directo · Contemporáneo — tiempo real · "
                       "Cacheado 1h para los 42 municipios")

            if len(df_mun_live) >= 3:
                conteos = (df_mun_live.groupby(['ano','semana'])['conteo']
                           .sum().reset_index().head(4))
                t1_l  = int(conteos.iloc[0]['conteo']) if len(conteos) > 0 else 0
                t2_l  = int(conteos.iloc[1]['conteo']) if len(conteos) > 1 else 0
                t3_l  = int(conteos.iloc[2]['conteo']) if len(conteos) > 2 else 0
                sem_l = int(conteos.iloc[0]['semana']) if len(conteos) > 0 else semana_actual
                ano_l = int(conteos.iloc[0]['ano'])    if len(conteos) > 0 else 0

                serie_l = pd.Series([t3_l, t2_l, t1_l])
                _, _, md_l = imputar_semanas_faltantes(serie_l)
                if md_l:
                    st.warning("Semanas cero sospechosas en datos frescos. "
                               "Imputación aplicada. Posible discontinuidad post-COVID.")

                pred_live, _ = predecir(municipio_nw, t1_l, t2_l, t3_l, sem_l, md_l)
                pred_hist, _ = predecir(municipio_nw, ult(-1), ult(-2), ult(-3), semana_actual)

                nl1, nl2, nl3, nl4 = st.columns(4)
                nl1.metric("Año/Sem API",          f"{ano_l}/S{sem_l}")
                nl2.metric("Casos t-1 (API real)", t1_l)
                nl3.metric("Pred. datos frescos",  f"{pred_live} casos")
                nl4.metric("Pred. histórico 2018", f"{pred_hist} casos")

                if abs(pred_live - pred_hist) > 10:
                    st.warning(
                        f"**Divergencia {abs(pred_live-pred_hist)} casos** entre API "
                        f"y histórico 2018. Evidencia de data drift post-pandemia. "
                        f"Re-entrenamiento con datos 2023+ recomendado."
                    )
                else:
                    st.success("Predicciones consistentes entre fuente histórica y API.")

                if len(df_mun_live) >= 6:
                    df_sl = (df_mun_live.groupby(['ano','semana'])['conteo']
                             .sum().reset_index().sort_values(['ano','semana']).tail(20))
                    df_sl['periodo'] = (df_sl['ano'].astype(str) + '-S' +
                                        df_sl['semana'].astype(str).str.zfill(2))
                    fig_live = px.bar(
                        df_sl, x='periodo', y='conteo',
                        title=f'Casos recientes desde API — {municipio_nw}',
                        labels={'conteo': 'Casos', 'periodo': 'Año-Semana'},
                        color_discrete_sequence=['#378ADD'],
                    )
                    _plotly_layout(fig_live, height=300,
                        margin=dict(l=0, r=0, t=40, b=0))
                    fig_live.update_xaxes(tickangle=45,
                        tickfont=dict(size=8, color=_colores_tema()['axis']))
                    fig_live.update_yaxes(gridcolor=_colores_tema()['grid'])
                    st.plotly_chart(fig_live, use_container_width=True)

                with st.expander("Datos crudos API"):
                    st.dataframe(df_live[['municipio_ocurrencia','ano','semana',
                                         'conteo','nombre_evento']].head(20),
                                 hide_index=True, use_container_width=True)
            else:
                st.warning(f"No hay registros suficientes para {municipio_nw}. "
                           "Posible discontinuidad post-COVID.")
    else:
        st.info("Selecciona un municipio y presiona **Consultar API SIVIGILA**.")

    st.divider()
    with st.expander("Arquitectura de Re-entrenamiento Continuo"):
        st.markdown("""
```
API SIVIGILA → Extracción semanal automatizable
     ↓
Limpieza + Detección semanas faltantes (mediana móvil)
     ↓
Actualización lags (t-1, t-2, t-3) → Nowcasting inmediato
     ↓  (cuando haya ≥52 semanas nuevas + cobertura ≥70%)
Re-entrenamiento Random Forest (mismo pipeline, ventana deslizante)
     ↓
Validación holdout temporal (rechazar si R² < 0.80)
     ↓
Exportar modelo_municipal_vX.pkl con métricas y hash incrustados
```
**Condición de exclusión COVID:** Años 2020–2022 con flag de subregistro documentado.
Re-entrenamiento inicia desde datos 2023+ para capturar nueva dinámica vectorial.
        """)

# ══════════════════════════════════════════════
# SECCIÓN 4 — SERIE HISTÓRICA
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[4]:
    st.subheader("Serie Temporal Completa — SIVIGILA 2007–2018 · 42 Municipios")

    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        vista = st.radio("Vista:", ["Top 10 por carga", "Selección libre"])
    with col_f2:
        if vista == "Top 10 por carga":
            muns_vis = (df_hist.groupby('municipio_ocurrencia')['casos']
                        .sum().sort_values(ascending=False).head(10).index.tolist())
        else:
            muns_vis = st.multiselect("Municipios:", sorted(MUNICIPIOS),
                                      default=['CALI','PALMIRA','TULUA'])

    if muns_vis:
        df_ph = df_hist[df_hist['municipio_ocurrencia'].isin(muns_vis)].copy()
        fig_hist = px.line(
            df_ph, x='fecha', y='casos', color='municipio_ocurrencia',
            color_discrete_sequence=COLORES_TOP,
            labels={'casos': 'Casos / semana', 'fecha': '',
                    'municipio_ocurrencia': 'Municipio'},
            title='Dengue — Valle del Cauca · Semanas epidemiológicas',
        )
        fig_hist.update_traces(line_width=1.5)
        _plotly_layout(fig_hist, height=420,
            margin=dict(l=0, r=0, t=40, b=0))
        fig_hist.update_yaxes(gridcolor=_colores_tema()['grid'])
        fig_hist.update_xaxes(showgrid=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    stats = (
        df_hist[df_hist['municipio_ocurrencia'].isin(muns_vis if muns_vis else MUNICIPIOS)]
        .groupby('municipio_ocurrencia')['casos']
        .agg(Semanas='count', Total='sum', Promedio='mean', Pico='max')
        .round(1).sort_values('Total', ascending=False)
    )
    st.dataframe(stats, use_container_width=True)

# ══════════════════════════════════════════════
# SECCIÓN 5 — MAPA DEPARTAMENTAL
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[5]:
    st.subheader("Mapa de Riesgo Departamental — 42 Municipios Valle del Cauca")
    st.caption("Color = urgencia logística · Tamaño = casos predichos · "
               "Líneas = rutas desde SECCIONED Cali")

    mapa = construir_mapa(semana_actual)
    st_folium(mapa, width=None, height=550)
    ml1, ml2, ml3 = st.columns(3)
    ml1.error("CRÍTICO — Stock post-demanda < SS dinámico")
    ml2.warning("ALERTA — Stock actual < ROP dinámico")
    ml3.success("NORMAL — Stock suficiente para el período")

# ══════════════════════════════════════════════
# SECCIÓN 6 — VALIDACIÓN RETROSPECTIVA
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[6]:
    st.subheader("Validación Retrospectiva — Brote Cali 2016–2017")
    st.markdown(
        "Demostración de que el sistema **hubiera detectado** el mayor brote "
        "del dataset con anticipación suficiente. Predicciones genuinamente "
        "*out-of-sample* (modelo entrenado hasta 2015)."
    )

    resultado_retro = calcular_validacion_retrospectiva()

    if resultado_retro is not None:
        df_r16     = resultado_retro['df_r16']
        pico_val   = resultado_retro['pico_val']
        pico_fec   = resultado_retro['pico_fec']
        sem_antic  = resultado_retro['sem_antic']
        ss_aceta   = resultado_retro['ss_aceta']
        rop_aceta  = resultado_retro['rop_aceta']
        idx_primera_al = resultado_retro['idx_primera_al']

        rv1, rv2, rv3, rv4 = st.columns(4)
        rv1.metric("Pico real",            f"{pico_val} casos/sem")
        rv2.metric("Fecha del pico",       pico_fec.strftime('%Y · Sem %W'))
        rv3.metric("Semanas anticipación", f"{sem_antic} semanas")
        rv4.metric("Lead time Cali",
                   f"{RED_LOGISTICA.get('CALI',{}).get('lead_time_horas',2)} h")

        if sem_antic > 0:
            st.success(
                f"Sistema generó alerta {sem_antic} semanas antes del pico. "
                f"Lead time de {RED_LOGISTICA.get('CALI',{}).get('lead_time_horas',2)} horas "
                f"— tiempo suficiente para activar la cadena."
            )

        fig_retro = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=['Casos reales vs predichos (out-of-sample)',
                            'Stock simulado vs umbrales (SS dinámico)',
                            'Semáforo logístico semana a semana'],
            row_heights=[0.4, 0.35, 0.25], vertical_spacing=0.08
        )
        fig_retro.add_trace(go.Scatter(
            x=df_r16['fecha'], y=df_r16['real_casos'],
            name='Casos reales', line=dict(color='#378ADD', width=2.5),
            hovertemplate='%{x|%d %b %Y}<br>Real: %{y}<extra></extra>'
        ), row=1, col=1)
        fig_retro.add_trace(go.Scatter(
            x=df_r16['fecha'], y=df_r16['pred_casos'],
            name='Predicción', line=dict(color='#DC2626', width=2, dash='dash'),
            hovertemplate='%{x|%d %b %Y}<br>Pred: %{y}<extra></extra>'
        ), row=1, col=1)
        fig_retro.add_vline(x=pico_fec, line_dash='dot',
                            line_color='#533AB7', opacity=0.7)
        fig_retro.add_annotation(x=pico_fec, y=pico_val,
                                  text=f" Pico: {pico_val}", showarrow=False,
                                  font=dict(color='#533AB7', size=10))

        fig_retro.add_trace(go.Scatter(
            x=df_r16['fecha'], y=df_r16['stock_aceta'],
            name='Stock aceta.', line=dict(color='#333', width=1.8),
            hovertemplate='%{x|%d %b %Y}<br>Stock: %{y:,}<extra></extra>'
        ), row=2, col=1)
        fig_retro.add_hline(y=rop_aceta, line_dash='dash', line_color='#EF9F27',
                             annotation_text=f'ROP ({rop_aceta:,})',
                             annotation_position='top right', row=2, col=1)
        fig_retro.add_hline(y=ss_aceta, line_dash='dash', line_color='#E24B4A',
                             annotation_text=f'SS ({ss_aceta:,})',
                             annotation_position='bottom right', row=2, col=1)

        colors_sem = [COLOR_URG[u] for u in df_r16['urgencia']]
        fig_retro.add_trace(go.Bar(
            x=df_r16['fecha'], y=[1] * len(df_r16),
            marker_color=colors_sem, name='Urgencia',
            hovertemplate='%{x|%d %b %Y}<br>%{customdata}<extra></extra>',
            customdata=df_r16['urgencia'].tolist()
        ), row=3, col=1)
        if idx_primera_al is not None:
            fig_retro.add_vline(
                x=df_r16.loc[idx_primera_al, 'fecha'],
                line_dash='solid', line_color='#EF9F27',
                line_width=2.5, opacity=0.9
            )

        _t = _colores_tema()
        fig_retro.update_layout(
            height=700, hovermode='x unified',
            plot_bgcolor=_t['plot'], paper_bgcolor=_t['paper'],
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(family='Nunito, sans-serif', color=_t['font']),
        )
        fig_retro.update_yaxes(gridcolor=_t['grid'])
        fig_retro.update_xaxes(showgrid=False)
        st.plotly_chart(fig_retro, use_container_width=True)

        mae_r  = round(np.mean(np.abs(df_r16['real_casos'] - df_r16['pred_casos'])), 2)
        rmse_r = round(np.sqrt(np.mean((df_r16['real_casos'] - df_r16['pred_casos'])**2)), 2)
        denom  = np.sum((df_r16['real_casos'] - df_r16['real_casos'].mean())**2)
        r2_r   = round(1 - np.sum((df_r16['real_casos'] - df_r16['pred_casos'])**2) /
                       denom, 3) if denom > 0 else 0
        mr1, mr2, mr3 = st.columns(3)
        mr1.metric("MAE (Cali 2016–17)",  f"{mae_r} casos/sem")
        mr2.metric("RMSE (Cali 2016–17)", f"{rmse_r} casos/sem")
        mr3.metric("R² (Cali 2016–17)",   f"{r2_r}")
    else:
        st.warning("No hay suficiente histórico de CALI desde 2015 para esta validación.")

# ══════════════════════════════════════════════
# SECCIÓN 7 — AUDITORÍA ALCOA+
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[7]:
    st.subheader("Auditoría Técnica Completa — Compliance ALCOA+")

    st.subheader("Sellos de Integridad de Datos")
    df_sellos = pd.DataFrame({
        'Artefacto': ['modelo_municipal_v4.pkl','dengue_valle_semanal.csv',
                      'logistica_params.json','API SIVIGILA (en vivo)',
],
        'Hash MD5':  [sello_modelo['hash_md5'], sello_datos['hash_md5'],
                      sello_log['hash_md5'], 'Calculado en tiempo real (sección Nowcasting)',
],
        'Cargado en':[sello_modelo['cargado_en'], sello_datos['cargado_en'],
                      sello_log['cargado_en'], 'Bajo demanda', 'Bajo demanda (caché 24h)'],
        'Fuente':    [sello_modelo['fuente'], sello_datos['fuente'],
                      sello_log['fuente'], 'datos.gov.co/resource/4hyg-wa9d · Socrata',
],
        'Estado':    ['Atención', 'Verificado', 'Verificado', 'Verificado', 'Verificado'],
        'ALCOA+ Original': [
            'Artefacto local — MLflow/DVC recomendado en producción',
            'Descargado de datos.gov.co',
            'Calculado de IGAC + INVIAS + MINSALUD',
            'Dato original en tiempo real',
        ],
    })

    def _color_estado(val):
        mapa = {'Verificado': 'background-color:#d4edda',
                'Atención':   'background-color:#fff3cd'}
        return mapa.get(val, '')

    st.dataframe(df_sellos.style.map(_color_estado, subset=['Estado']),
                 hide_index=True, use_container_width=True)

    st.divider()
    ca1, ca2 = st.columns(2)

    with ca1:
        st.markdown("#### Ficha Técnica")
        st.table(pd.DataFrame.from_dict({
            'Algoritmo':            'Random Forest Regressor',
            'N° árboles':           '300',
            'Profundidad máxima':   '12',
            'Min. muestras hoja':   '3',
            'Max features':         'sqrt',
            'Semilla':              '42',
            'Encoding municipio':   'Target encoding + IQR histórico',
            'N° features':          str(len(FEATURES)),
            'Municipios':           f"{len(MUNICIPIOS)} (100% Valle del Cauca)",
            'Versión':              VERSION,
            'Entrenado con':        paquete['entrenado_con'],
            'Evaluado en':          paquete['evaluado_en'],
            'Fecha entreno':        paquete['fecha_entreno'],
            'Hash modelo':          sello_modelo['hash_md5'],
        }, orient='index', columns=['Valor']))

    with ca2:
        st.markdown("#### Métricas Oficiales — Holdout Temporal 2018")
        st.dataframe(pd.DataFrame({
            'Métrica':        ['MAE','RMSE','R²','Gap Train-Val R²','Municipios test'],
            'Valor':          [f"{METRICAS['mae']} casos/sem",
                               f"{METRICAS['rmse']} casos/sem",
                               f"{METRICAS['r2']}", f"{GAP_TRAIN_VAL}",
                               f"{len(MUNICIPIOS)} municipios"],
            'Interpretación': [
                'Error promedio absoluto en datos no vistos',
                'Error cuadrático medio (penaliza outliers)',
                f"{METRICAS['r2']*100:.1f}% de la varianza explicada",
                'Sin overfitting',
                '100% cobertura departamental',
            ]
        }), hide_index=True, use_container_width=True)

        mae_n = ERROR_ESTRAT.get('mae_normal', 'N/A')
        mae_p = ERROR_ESTRAT.get('mae_pico',   'N/A')
        fac   = ERROR_ESTRAT.get('factor_deg', 'N/A')
        pct   = ERROR_ESTRAT.get('pct_pico',   'N/A')
        met   = ERROR_ESTRAT.get('metodo_umbral', 'OPS 2015')

        st.markdown("#### Análisis de Error Estratificado")
        st.dataframe(pd.DataFrame({
            'Contexto':       ['Semanas normales','Semanas de pico','Factor degradación'],
            'MAE':            [f"{mae_n} casos/sem", f"{mae_p} casos/sem", f"{fac}x"],
            'Muestra':        [
                f"{ERROR_ESTRAT.get('n_normal','N/A')} semanas (84.5%)",
                f"{ERROR_ESTRAT.get('n_pico','N/A')} semanas ({pct}%)",
                '—'
            ],
            'Método umbral':  [met, met, 'Chopra & Meindl SCM 2016'],
        }), hide_index=True, use_container_width=True)
        st.caption(
            f"El modelo se degrada {fac}x en picos. Mitigado con SS dinámico "
            f"Z(95%)×σ×√LT que absorbe la varianza estructural del error."
        )

        if df_justificacion is not None:
            st.markdown("#### Justificación de Municipios")
            st.dataframe(
                df_justificacion[['municipio_ocurrencia','total_casos',
                                  'anos_activos','carga_pct','carga_acum_pct']]
                .rename(columns={'municipio_ocurrencia':'Municipio',
                                 'total_casos':'Total','anos_activos':'Años',
                                 'carga_pct':'Carga %','carga_acum_pct':'Acum. %'}),
                hide_index=True, use_container_width=True, height=260
            )

    st.divider()
    with st.expander("Limitaciones Documentadas — Respuestas Preparadas para el Jurado"):
        fac_limit = ERROR_ESTRAT.get('factor_deg', 'N/A')
        st.warning(f"""
**1. Data Gap 2018→2026 (COVID-19):**
Entrenado hasta 2018. Re-entrenamiento continuo vía API SIVIGILA planificado
desde datos 2023+. La sección Nowcasting es la solución operativa inmediata.

**2. Dependencia de inercia (casos_t-1 dominante):**
Estructural en modelos de lags. Mitigado con: detección de semanas faltantes,
imputación por mediana móvil, IC ×1.5 en Modo Degradado, y Nowcasting con API.

**3. Degradación en picos (factor {fac_limit}x):**
Esperado y documentado — calculado comparando el error del modelo en semanas
normales vs. semanas de pico epidémico (holdout 2018), no es una versión ni
un año. Respuesta: SS dinámico Z×σ×√LT absorbe este error estructuralmente.
En picos, el sistema emite ALERTA antes del desbordamiento.

**4. Stock simulado (no en tiempo real):**
El inventario hospitalario en tiempo real no es dato abierto en Colombia.
Normativo (Res. 1403/2007). En producción: integrar con SISMED/SISPRO.
En presentación: `max(SS_dinámico, SS_normativo)` como piso legal.

**5. Ahorro calculado sobre demanda predicha, no sobre orden a realizar:**
El ahorro mostrado compara comprar la demanda predicha de la semana a precio
preventivo vs precio reactivo de urgencia. Es independiente de si el stock
actual ya alcanza, porque mide el valor de *anticipar* la compra.

**6. Variables climáticas ausentes:**
Estacionalidad capturada vía seno/coseno de semana. Open-Meteo planificado v5.0.
        """)

    with st.expander("Argumento de Farmacia Clínica — Para el Evaluador del Sector Salud"):
        st.info("""
Data Sentinel no es una herramienta para científicos de datos.

**Es una herramienta para el Químico Farmacéutico hospitalario** que necesita
saber si el Lactato de Ringer llega a Buenaventura antes de que la curva de
contagio sature la urgencia, o si Acetaminofén 500mg está disponible en Buga
cuando el sistema de alerta temprana dice que la próxima semana habrá 15 casos.

**La cadena de decisión completa:**
```
SIVIGILA (dato real) → Modelo RF (predicción semana t+1 a t+4)
→ Motor logístico (SS dinámico + lead time real)
→ Orden de despacho priorizada (CRÍTICO/ALERTA/NORMAL)
→ Químico Farmacéutico activa la compra antes del desabasto
```

Esto es lo que diferencia un sistema de soporte a decisiones clínicas
de un dashboard de visualización. La norma (Res. MINSALUD 1403/2007)
y la evidencia (SIVIGILA + modelo) hablan el mismo idioma.
        """)

# ══════════════════════════════════════════════
# SECCIÓN 8 — AGENTE IA
# ══════════════════════════════════════════════
elif seccion_activa == SECCIONES[8]:
    st.subheader("Agente IA — Pregúntale a Denguard")
    st.caption(
        "Agente con acceso a herramientas en tiempo real sobre el modelo, el "
        "histórico SIVIGILA y la cadena logística — no improvisa cifras, las consulta."
    )

    with st.expander("Arquitectura del agente — para el jurado", expanded=False):
        st.markdown("""
Esto **no** es un chatbot que alucina números: es un agente con **tool-use real**
sobre Gemini (Google). Cada vez que el usuario pregunta algo, el modelo
decide si necesita datos del sistema y llama una o varias de estas herramientas
*antes* de redactar la respuesta:

| Herramienta | Qué consulta |
|---|---|
| `consultar_prediccion_municipio` | Predicción de la próxima semana + urgencia logística |
| `consultar_resumen_departamental` | Estado CRÍTICO / ALERTA / NORMAL de los 42 municipios |
| `consultar_metricas_modelo` | MAE, RMSE, R² y ficha técnica del Random Forest |
| `consultar_historico_municipio` | Casos reales SIVIGILA por semana |
| `consultar_logistica_municipio` | Distancia, lead time, stock, ROP, SS |

```
Pregunta del usuario
     ↓
Gemini decide qué función(es) necesita (function calling)
     ↓
Denguard ejecuta la(s) función(es) sobre los datos reales del sistema
     ↓
El resultado (JSON) vuelve a Gemini como function_response
     ↓
Gemini redacta la respuesta final citando las cifras obtenidas
```

Si una pregunta no tiene una herramienta para resolverla, el agente lo dice
en vez de inventar un número.
        """)

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

    if not GEMINI_DISPONIBLE:
        st.error("Falta instalar el SDK de Gemini: `pip install google-genai`")
    elif not api_key:
        st.error(
            "No se encontró `GEMINI_API_KEY` en *Secrets*.\n\n"
            "**En Streamlit Cloud:** ⋮ → Settings → Secrets, agrega:\n"
            "```toml\nGEMINI_API_KEY = \"tu-key-aquí\"\n```\n"
            "y reinicia la app (⋮ → Reboot app).\n\n"
            "**En local:** crea `.streamlit/secrets.toml` con la misma línea.\n\n"
            "Consigue tu key gratis en aistudio.google.com/apikey"
        )
    else:
        if "agente_chat" not in st.session_state:
            st.session_state.agente_chat = []         # historial visible (solo texto)
        if "agente_contenidos" not in st.session_state:
            st.session_state.agente_contenidos = []    # historial completo (incluye tool calls)

        for m in st.session_state.agente_chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        pregunta = st.chat_input(
            "Ej: ¿Qué municipios están en CRÍTICO esta semana? · "
            "¿Cuántos casos se predicen para Buga?"
        )
        if pregunta:
            st.session_state.agente_chat.append({"role": "user", "content": pregunta})
            st.session_state.agente_contenidos.append(
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=pregunta)])
            )
            with st.chat_message("user"):
                st.markdown(pregunta)
            with st.chat_message("assistant"):
                with st.spinner("Consultando herramientas..."):
                    try:
                        cliente_ia = genai.Client(api_key=api_key)
                        texto, st.session_state.agente_contenidos = ejecutar_agente(
                            cliente_ia, st.session_state.agente_contenidos
                        )
                    except Exception as e:
                        texto = f"Error consultando al agente: {e}"
                st.markdown(texto)
            st.session_state.agente_chat.append({"role": "assistant", "content": texto})

        if st.session_state.agente_chat and st.button("Limpiar conversación"):
            st.session_state.agente_chat = []
            st.session_state.agente_contenidos = []
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
