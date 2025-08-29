import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import plotly.express as px
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

st.set_page_config(
    page_title="Detector de Danos Veiculares",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        import urllib.request
        model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
        if not os.path.exists('yolo11m.pt'):
            urllib.request.urlretrieve(model_url, 'yolo11m.pt')
        model = YOLO('yolo11m.pt')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

def process_image(image, model):
    img_array = np.array(image)
    results = model(img_array)
    
    detections = []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            detection = {
                'class': results[0].names[int(boxes.cls[i])],
                'confidence': float(boxes.conf[i]),
                'bbox': boxes.xyxy[i].cpu().numpy()
            }
            detections.append(detection)
    
    try:
        annotated_img = results[0].plot()
        if cv2 is not None:
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    except:
        annotated_img = img_array
    
    return detections, annotated_img

def create_detection_summary(detections):
    if not detections:
        return "Nenhum dano detectado na imagem."
    
    damage_counts = {}
    for detection in detections:
        damage_type = detection['class']
        if damage_type not in damage_counts:
            damage_counts[damage_type] = []
        damage_counts[damage_type].append(detection['confidence'])
    
    summary = []
    total_damages = len(detections)
    
    summary.append(f"**Total de danos detectados: {total_damages}**\n")
    
    for damage_type, confidences in damage_counts.items():
        count = len(confidences)
        avg_confidence = np.mean(confidences)
        summary.append(f"• **{damage_type.replace('_', ' ').title()}**: {count} ocorrência(s) - Confiança média: {avg_confidence:.1%}")
    
    return "\n".join(summary)

def create_confidence_chart(detections):
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    df['class_clean'] = df['class'].str.replace('_', ' ').str.title()
    
    fig = px.bar(
        df, 
        x='class_clean', 
        y='confidence',
        title='Confiança das Detecções por Tipo de Dano',
        labels={'confidence': 'Confiança (%)', 'class_clean': 'Tipo de Dano'},
        color='confidence',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    fig.update_yaxis(tickformat='.0%')
    
    return fig

def main():
    st.title("Detector de Danos Veiculares")
    st.markdown("**Powered by YOLO11m** - Detecta automaticamente danos em veículos usando Inteligência Artificial")
    
    with st.sidebar:
        st.header("Sobre o Sistema")
        st.markdown("""
        Este sistema utiliza um modelo YOLO11m treinado especificamente para detectar:
        
        - Amassados (Dents)
        - Riscos (Scratches) 
        - Rachaduras (Cracks)
        - Vidros Quebrados
        - Lâmpadas Quebradas
        - Pneus Vazios
        """)
        
        st.header("Performance do Modelo")
        st.markdown("""
        **mAP50 por Classe:**
        - Vidros quebrados: 99.4%
        - Pneus vazios: 95.9%
        - Lâmpadas quebradas: 89.5%
        - Riscos: 90.5%
        - Amassados: 69.2%
        - Rachaduras: 62.0%
        """)
    
    model = load_model()
    if model is None:
        st.error("Não foi possível carregar o modelo. Verifique se o arquivo 'trained.pt' está disponível.")
        return
    
    st.success("Modelo carregado com sucesso!")
    
    st.header("Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem do veículo:",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos aceitos: PNG, JPG, JPEG"
    )
    
    st.header("Ou teste com exemplos:")
    
    example_images = {
        "examples/1.png": "Dent - Amassado",
        "examples/2.png": "Múltiplos Danos", 
        "examples/3.png": "Vidro Estilhaçado",
        "examples/4.png": "Lâmpada Quebrada",
        "examples/5.png": "Dent - Lateral",
        "examples/6.png": "Riscos",
        "examples/7.png": "Múltiplos Riscos"
    }
    
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    for i, (img_path, label) in enumerate(example_images.items()):
        with cols[i % 3]:
            if st.button(label, key=f"example_{i}"):
                try:
                    if os.path.exists(img_path):
                        example_image = Image.open(img_path)
                        st.session_state['uploaded_example'] = example_image
                        st.session_state['example_name'] = label
                        st.rerun()
                except:
                    st.error(f"Exemplo não encontrado: {img_path}")
    
    image_source = None
    image_name = None
    
    if 'uploaded_example' in st.session_state:
        image_source = st.session_state['uploaded_example']
        image_name = st.session_state['example_name']
        st.info(f"Usando exemplo: {image_name}")
    elif uploaded_file is not None:
        image_source = Image.open(uploaded_file)
        image_name = "Imagem enviada"
    
    if image_source is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagem Original")
            st.image(image_source, caption=image_name, use_column_width=True)
        
        with st.spinner("Analisando imagem..."):
            detections, annotated_img = process_image(image_source, model)
        
        with col2:
            st.subheader("Detecções Encontradas")
            st.image(annotated_img, caption="Danos detectados", use_column_width=True)
        
        st.header("Resumo da Análise")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Detalhes dos Danos")
            summary = create_detection_summary(detections)
            st.markdown(summary)
        
        with col2:
            st.markdown("### Gráfico de Confiança")
            if detections:
                chart = create_confidence_chart(detections)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Nenhum dano detectado para exibir no gráfico.")
        
        if detections:
            st.header("Detalhes Técnicos")
            df_detections = pd.DataFrame(detections)
            df_detections['confidence'] = df_detections['confidence'].apply(lambda x: f"{x:.1%}")
            df_detections['class'] = df_detections['class'].str.replace('_', ' ').str.title()
            df_detections = df_detections[['class', 'confidence']].rename(columns={
                'class': 'Tipo de Dano',
                'confidence': 'Confiança'
            })
            st.dataframe(df_detections, use_container_width=True)
        
        if detections:
            st.header("Recomendações")
            if any(d['class'] == 'shattered_glass' for d in detections):
                st.warning("**Vidro quebrado detectado** - Reparo urgente necessário por questões de segurança.")
            if any(d['class'] == 'flat_tire' for d in detections):
                st.warning("**Pneu vazio detectado** - Verifique o pneu antes de dirigir.")
            if any(d['class'] == 'broken_lamp' for d in detections):
                st.info("**Lâmpada quebrada** - Substitua para manter a segurança no trânsito.")
            
            high_conf_damages = [d for d in detections if d['confidence'] > 0.8]
            if high_conf_damages:
                st.success(f"{len(high_conf_damages)} dano(s) detectado(s) com alta confiança.")
        
        if st.button("Testar Nova Imagem"):
            if 'uploaded_example' in st.session_state:
                del st.session_state['uploaded_example']
            if 'example_name' in st.session_state:
                del st.session_state['example_name']
            st.rerun()
    
    else:
        st.info("Faça upload de uma imagem ou selecione um exemplo para começar a análise.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Sistema de Detecção de Danos Veiculares</strong></p>
        <p>Desenvolvido com YOLO11m + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
