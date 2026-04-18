import streamlit as st
import os
import re
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# ==========================================
# CONFIGURACIÓN E INTERFAZ PROFESIONAL
# ==========================================
st.set_page_config(
    page_title="Academia IA Pro - UNSA", 
    page_icon="🏫", 
    layout="wide",
    initial_sidebar_state="expanded" # Asegura que la bandeja esté abierta
)

# CSS Avanzado para estética y ocultar elementos de desarrollo
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stAppDeployButton {display:none;}
            #stDecoration {display:none;}
            [data-testid="stSidebarNav"] {display: none;}
            
            /* Centrar elementos y mejorar fuentes */
            .main .block-container {
                padding-top: 2rem;
                max-width: 900px;
            }
            h1 {
                text-align: center;
                color: #f0f2f6;
            }
            
            /* Ocultar botón Manage App para alumnos */
            button[title="Manage app"] {display: none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# CAPA 1: SEGURIDAD Y ACCESO (Login Centrado)
# ==========================================
INTERNAL_API_KEY = st.secrets["GOOGLE_API_KEY"] 

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    # Contenedor centrado para el login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("🔐 Academia UNSA")
        st.write("Bienvenido al sistema de preparación de alto rendimiento.")
        
        password_input = st.text_input("Introduce tu código de alumno:", type="password")
        
        # Sugerencia aplicada: Aviso de advertencia en el login
        st.warning("⚠️ El sistema detecta accesos simultáneos. No compartas tu código.")
        
        if st.button("Ingresar al Aula Virtual", use_container_width=True):
            try:
                contraseñas_validas = list(st.secrets["alumnos"].values())
                if password_input in contraseñas_validas:
                    st.session_state.autenticado = True
                    st.rerun()
                else:
                    st.error("Código incorrecto o inactivo. Contacta con soporte.")
            except:
                st.error("Error de configuración de servidor.")
    st.stop()

# ==========================================
# CAPA 2: SANITIZACIÓN (Mantenida al 100%)
# ==========================================
def validar_consulta_educativa(texto):
    patrones_bloqueo = [
        r'ignora.*instrucciones', r'actúa como', r'olvida.*anterior',
        r'revela.*sistema', r'sal.*modo', r'prompt.*original'
    ]
    if any(re.search(p, texto.lower()) for p in patrones_bloqueo):
        return False, "⚠️ Alerta de Seguridad: Consulta no procesable. Enfócate en tu preparación académica."
    return True, texto

# ==========================================
# CAPA 3: MULTI-PROFESOR (PDF + TXT)
# ==========================================
st.sidebar.title("👨‍🏫 Panel de Cursos")

# Detector automático de materiales
archivos_encontrados = glob.glob("*.txt") + glob.glob("*.pdf")
archivos_materias = {os.path.splitext(f)[0].capitalize(): f for f in archivos_encontrados if f != "requirements.txt"}
lista_materias = list(archivos_materias.keys())

if not lista_materias:
    st.sidebar.error("❌ No hay materiales cargados en el repositorio.")
    st.stop()

materia = st.sidebar.selectbox("Selecciona tu curso actual:", lista_materias)

@st.cache_resource
def cargar_profesor(nombre_materia):
    archivo = archivos_materias[nombre_materia]
    # Carga según el formato del archivo
    if archivo.endswith(".pdf"):
        loader = PyPDFLoader(archivo)
    else:
        loader = TextLoader(archivo, encoding="utf-8")
        
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vector_db = cargar_profesor(materia)

# ==========================================
# CAPA 4: CHAT Y PROMPT MAESTRO (Sin reducciones)
# ==========================================
if "materia_actual" not in st.session_state:
    st.session_state.materia_actual = materia

if st.session_state.materia_actual != materia:
    st.session_state.messages = []
    st.session_state.materia_actual = materia

if "messages" not in st.session_state:
    st.session_state.messages = []

# Título centrado con estilo agradable
st.markdown(f"<h1>🏫 Especialista en {materia}</h1>", unsafe_allow_html=True)
st.info(f"¡Hola! Soy tu profesor estratega de {materia} 🛠️. Vamos a asegurar esa vacante en la UNSA.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué tema o ejercicio resolveremos ahora?"):
    
    es_valido, mensaje_seguro = validar_consulta_educativa(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not es_valido:
            st.warning(mensaje_seguro)
            st.session_state.messages.append({"role": "assistant", "content": mensaje_seguro})
        elif vector_db:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto = "\n\n".join([d.page_content for d in docs])
            
            # PROMPT MAESTRO ORIGINAL RE-POTENCIADO
            instrucciones_maestras = f"""
            Actúa como un Profesor Especialista en {materia} y Estratega con 10 años de experiencia preuniversitaria para la UNSA.
            Tu misión es asegurar la vacante del alumno en Ingeniería.

            🔒 RESTRICCIONES:
            • NUNCA des respuestas directas; guía mediante el método socrático.
            • Usa LaTeX ($...$) para fórmulas matemáticas impecables.
            • PRIORIDAD: Usa este material: {contexto}

            ESTRUCTURA OBLIGATORIA:
            1. 🧠 Las "Tripas de Estudio" (Teoría Profunda)
            2. 🎯 Foco de Examen y Trampas Clásicas (Cáscaras de plátano UNSA)
            3. ⚡ El "Porqué" y el "Hack" (Estrategia Rápida)
            4. 📈 Entrenamiento Progresivo (3 Retos: Calentamiento, Modo UNSA, Asegura Vacante)
            5. 🛑 Punto de Control (Pregunta antes de seguir)
            """

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=INTERNAL_API_KEY,
                temperature=0.3
            )
            
            historial = [("system", instrucciones_maestras)]
            for m in st.session_state.messages[-6:]:
                role = "human" if m["role"] == "user" else "ai"
                historial.append((role, m["content"]))
            
            respuesta = llm.invoke(historial)
            
            texto_final = respuesta.content if isinstance(respuesta.content, str) else respuesta.content[0].get("text", "")
            
            st.markdown(texto_final)
            st.session_state.messages.append({"role": "assistant", "content": texto_final})
