import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="🚗 Detector de Danos Veiculares",
    page_icon="🚗",
    layout="wide"
)

# Cache do modelo para evitar recarregamento
@st.cache_resource
def load_model():
    """Carrega o modelo YOLO treinado"""
    try:
        model = YOLO('trained.pt')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

def process_image(image, model):
    """Processa a imagem e retorna os resultados da detecção"""
    # Converte PIL para array numpy
    img_array = np.array(image)
    
    # Executa a inferência
    results = model(img_array)
    
    # Extrai informações das detecções
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
    
    # Obtém imagem com annotations
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return detections, annotated_img

def create_detection_summary(detections):
    """Cria resumo das detecções encontradas"""
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
    """Cria gráfico de barras com as confidências das detecções"""
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

# Interface principal
def main():
    st.title("🚗 Detector de Danos Veiculares")
    st.markdown("**Powered by YOLO11m** - Detecta automaticamente danos em veículos usando Inteligência Artificial")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Sobre o Sistema")
        st.markdown("""
        Este sistema utiliza um modelo YOLO11m treinado especificamente para detectar:
        
        - 🔹 **Amassados (Dents)**
        - 🔹 **Riscos (Scratches)** 
        - 🔹 **Rachaduras (Cracks)**
        - 🔹 **Vidros Quebrados**
        - 🔹 **Lâmpadas Quebradas**
        - 🔹 **Pneus Vazios**
        """)
        
        st.header("🎯 Performance do Modelo")
        st.markdown("""
        **mAP50 por Classe:**
        - Vidros quebrados: 99.4%
        - Pneus vazios: 95.9%
        - Lâmpadas quebradas: 89.5%
        - Riscos: 90.5%
        - Amassados: 69.2%
        - Rachaduras: 62.0%
        """)
    
    # Carrega o modelo
    model = load_model()
    if model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique se o arquivo 'trained.pt' está disponível.")
        return
    
    st.success("✅ Modelo carregado com sucesso!")
    
    # Upload da imagem
    st.header("📸 Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem do veículo:",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos aceitos: PNG, JPG, JPEG"
    )
    
    # Exemplos de imagens (opcional)
    st.header("🖼️ Ou teste com exemplos:")
    col1, col2, col3 = st.columns(3)
    
    example_images = {
        "Carro com amassados": "https://via.placeholder.com/300x200?text=Exemplo+1",
        "Vidro quebrado": "https://via.placeholder.com/300x200?text=Exemplo+2", 
        "Riscos na lataria": "https://via.placeholder.com/300x200?text=Exemplo+3"
    }
    
    if uploaded_file is not None:
        # Processa a imagem
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagem Original")
            st.image(image, caption="Imagem enviada", use_column_width=True)
        
        # Executa a detecção
        with st.spinner("🔍 Analisando imagem..."):
            detections, annotated_img = process_image(image, model)
        
        with col2:
            st.subheader("🎯 Detecções Encontradas")
            st.image(annotated_img, caption="Danos detectados", use_column_width=True)
        
        # Resumo das detecções
        st.header("📊 Resumo da Análise")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📋 Detalhes dos Danos")
            summary = create_detection_summary(detections)
            st.markdown(summary)
        
        with col2:
            st.markdown("### 📈 Gráfico de Confiança")
            if detections:
                chart = create_confidence_chart(detections)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Nenhum dano detectado para exibir no gráfico.")
        
        # Tabela detalhada
        if detections:
            st.header("📑 Detalhes Técnicos")
            df_detections = pd.DataFrame(detections)
            df_detections['confidence'] = df_detections['confidence'].apply(lambda x: f"{x:.1%}")
            df_detections['class'] = df_detections['class'].str.replace('_', ' ').str.title()
            df_detections = df_detections[['class', 'confidence']].rename(columns={
                'class': 'Tipo de Dano',
                'confidence': 'Confiança'
            })
            st.dataframe(df_detections, use_container_width=True)
        
        # Recomendações
        if detections:
            st.header("💡 Recomendações")
            if any(d['class'] == 'shattered_glass' for d in detections):
                st.warning("⚠️ **Vidro quebrado detectado** - Reparo urgente necessário por questões de segurança.")
            if any(d['class'] == 'flat_tire' for d in detections):
                st.warning("⚠️ **Pneu vazio detectado** - Verifique o pneu antes de dirigir.")
            if any(d['class'] == 'broken_lamp' for d in detections):
                st.info("🔧 **Lâmpada quebrada** - Substitua para manter a segurança no trânsito.")
            
            high_conf_damages = [d for d in detections if d['confidence'] > 0.8]
            if high_conf_damages:
                st.success(f"✅ {len(high_conf_damages)} dano(s) detectado(s) com alta confiança.")
    
    else:
        st.info("👆 Faça upload de uma imagem para começar a análise.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>🤖 Sistema de Detecção de Danos Veiculares</strong></p>
        <p>Desenvolvido com YOLO11m + Streamlit | <a href='https://github.com/seu-usuario/vehicle-damage-detector'>📁 Código no GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
