import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd

st.set_page_config(
    page_title="🚗 Detector de Objetos em Veículos",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    """
    Carrega o modelo YOLOv8n. A biblioteca ultralytics fará o download 
    automaticamente no ambiente do Streamlit Cloud na primeira execução.
    """
    try:
        model = YOLO('yolov8n.pt') 
        return model
    except Exception as e:
        st.error(f"Erro ao carregar ou baixar o modelo YOLOv8n: {e}")
        return None

def process_image(image, model):
    """Processa uma imagem, realiza a detecção de objetos e retorna os resultados."""
    results = model(image)
    
    detections = []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            detection = {
                'class': model.names[int(boxes.cls[i])],
                'confidence': float(boxes.conf[i]),
            }
            detections.append(detection)
    
    # A função plot() retorna uma imagem em formato numpy array (BGR)
    annotated_img_bgr = results[0].plot()
    # Convertemos de BGR para RGB para exibição correta no Streamlit
    annotated_img_rgb = annotated_img_bgr[..., ::-1]
    
    return detections, annotated_img_rgb

st.title("🚗 Sistema de Detecção de Objetos em Veículos")
st.markdown("**Protótipo para Carglass, implantado via Streamlit Community Cloud e GitHub.**")
st.info("ℹ️ **Nota:** Este protótipo usa um modelo genérico (YOLOv8n) que detecta objetos comuns (carros, pessoas, etc.), não danos específicos. Ele serve para demonstrar a funcionalidade da aplicação online.")

# Carrega o modelo e exibe o status
model = load_model()

if model is None:
    st.error("❌ O modelo de detecção não pôde ser carregado. A aplicação não pode continuar.")
else:
    st.success("✅ Modelo de Inteligência Artificial (YOLOv8n) carregado com sucesso!")

    # Opção de Upload
    st.header("1. Envie uma Imagem para Análise")
    uploaded_file = st.file_uploader(
        "Selecione uma imagem de um veículo:",
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        st.divider()
        st.header("2. Resultados da Análise")

        # Análise da imagem
        with st.spinner("🔍 Analisando a imagem com a IA... Por favor, aguarde."):
            detections, annotated_img = process_image(image, model)

        # Exibição dos resultados
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Objetos Detectados")
            st.image(annotated_img, use_column_width=True)

        # Tabela de detecções
        if detections:
            st.header("3. Detalhes da Detecção")
            df = pd.DataFrame(detections)
            df_display = df.rename(columns={'class': 'Objeto Detectado', 'confidence': 'Confiança'})
            df_display['Confiança'] = df_display['Confiança'].map('{:.1%}'.format)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum objeto do modelo padrão foi detectado na imagem.")
    else:
        st.info("👆 Aguardando o envio de uma imagem para iniciar a análise.")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido como prova de conceito para a **Carglass**.")
