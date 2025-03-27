import streamlit as st
import joblib
import numpy as np

# Cargar modelos y scaler
nb_model = joblib.load('nb_model.joblib')
knn_model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

def predecir_calidad(modelo, datos):
    datos_escalados = scaler.transform([datos])
    return modelo.predict(datos_escalados)[0]

# Aplicar estilos CSS personalizados para el fondo en tonos suaves
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #A8E6CE, #DCEDC1);
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput, .stNumberInput, .stButton {
        border-radius: 10px;
    }
    .stMarkdown h2 {
        color: #333;
    }
    .stButton > button {
        background-color: #76B39D;
        color: white;
        border-radius: 8px;
        padding: 8px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #5A9E87;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicializar session_state si no existe
if "modelo_elegido" not in st.session_state:
    st.session_state["modelo_elegido"] = None

if "pm10" not in st.session_state:
    st.session_state["pm10"] = 0.0
    st.session_state["pm25"] = 0.0
    st.session_state["no2"] = 0.0
    st.session_state["o3"] = 0.0
    
# -------------------------------------------------------------------------
# Barra superior
# -------------------------------------------------------------------------
st.markdown(
    """
    <div style="
         background-color: rgba(255, 255, 255, 0.6); 
         padding: 10px;
         margin-bottom: 10px;
         border-radius: 5px;
         text-align: right;
         font-size: 18px;">
        <strong>Made by: Cesar Solano, UNAB 2025</strong>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# Título principal
# -------------------------------------------------------------------------
st.title("🌍 Calidad de Aire en Bucaramanga")

# -------------------------------------------------------------------------
# Entradas de usuario
# -------------------------------------------------------------------------
st.subheader("📊 Ingrese los valores de calidad del aire:")

pm10 = st.number_input("PM10", value=st.session_state["pm10"], key="pm10")
pm25 = st.number_input("PM2.5", value=st.session_state["pm25"], key="pm25")
no2 = st.number_input("NO2", value=st.session_state["no2"], key="no2")
o3 = st.number_input("O3", value=st.session_state["o3"], key="o3")


datos = [pm10, pm25, no2, o3]

# -------------------------------------------------------------------------
# Botones de selección de modelo
# -------------------------------------------------------------------------
st.subheader("🤖 Seleccione el modelo de predicción:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Naive Bayes"):
        st.session_state["modelo_elegido"] = nb_model

with col2:
    if st.button("KNN"):
        st.session_state["modelo_elegido"] = knn_model

with col3:
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# -------------------------------------------------------------------------
# Predicción
# -------------------------------------------------------------------------
if st.session_state["modelo_elegido"]:
    resultado = predecir_calidad(st.session_state["modelo_elegido"], datos)

    emoji = "😊" if resultado == "Buena" else "👍" if resultado == "Moderada" else "⚠️"
    
    st.subheader("📢 Predicción de Calidad de Aire:")
    st.markdown(f"<h2 style='color: #2E7D32;'>{resultado} {emoji}</h2>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Mapa de calidad del aire
# -------------------------------------------------------------------------
st.subheader("🗺️ Mapa de Calidad del Aire en Tiempo Real")

iframe_code = """
<iframe 
    src="https://embed.windy.com/embed2.html?lat=7.1254&lon=-73.1198&zoom=8&level=surface&overlay=airquality&product=ecmwf&menu=&message=true&marker=&calendar=now" 
    frameborder="0" 
    width="650" 
    height="450">
</iframe>
"""
st.markdown(iframe_code, unsafe_allow_html=True)
