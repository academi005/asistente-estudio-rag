import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Academia IA - Profesor de Álgebra", page_icon="🎓")
st.title("🎓 Profesor Virtual de Álgebra")
st.markdown("Consulta tus dudas sobre el material de la academia.")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración")
    # Es mejor usar st.secrets en producción, pero aquí lo dejamos para que pegues tu llave
    api_key = st.text_input("Ingresa tu Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# --- FUNCIÓN PARA CARGAR EL CEREBRO (CON CACHÉ) ---
@st.cache_resource
def inicializar_sistema():
    if not os.path.exists("algebra.txt"):
        return None
    
    # Carga y fragmentación
    loader = TextLoader("algebra.txt", encoding="utf-8")
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    
    # Embeddings locales (HuggingFace) para evitar errores de cuota
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- LÓGICA DE LA APP ---
if api_key:
    vector_db = inicializar_sistema()
    
    if vector_db:
        pregunta = st.text_input("¿Qué quieres aprender hoy?", placeholder="Ej: ¿Qué es un polinomio?")
        
        if pregunta:
            with st.spinner("El profesor está revisando el libro..."):
                # Búsqueda en el material
                docs = vector_db.similarity_search(pregunta, k=3)
                contexto = "\n\n".join([d.page_content for d in docs])
                
                # Generación con Gemini 3 Flash
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
                prompt = f"Eres un profesor de Álgebra. Responde de forma didáctica usando solo este material:\n\n{contexto}\n\nPregunta: {pregunta}"
                
                respuesta = llm.invoke(prompt)
                
                st.subheader("🎓 Respuesta del Profesor:")
                st.write(respuesta.content)
    else:
        st.error("No se encontró el archivo 'algebra.txt'. Asegúrate de que esté en la misma carpeta que este código.")
else:
    st.warning("Por favor, ingresa tu API Key en la barra lateral para comenzar.")
