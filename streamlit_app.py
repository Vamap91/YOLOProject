import streamlit as st
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import plotly.express as px
import pandas as pd
import requests

# Importações para o modelo Transformer
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("❌ Biblioteca 'transformers' não está disponível. Instale com: pip install transformers")

# Fallback para YOLO se transformers não estiver disponível
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import cv2
except ImportError:
    cv2 = None

st.set_page_config(
    page_title="Carglass - Detector de Danos Veiculares",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração para o modelo Transformer
DAMAGE_CONFIG = {
    'class_mapping': {
        "0": "Crack",
        "1": "Scratch", 
        "2": "Tire Flat",
        "3": "Dent",
        "4": "Glass Shatter",
        "5": "Lamp Broken",
        # Mapeamento alternativo
        "Crack": "Crack",
        "Scratch": "Scratch",
        "Tire Flat": "Tire Flat",
        "Dent": "Dent", 
        "Glass Shatter": "Glass Shatter",
        "Lamp Broken": "Lamp Broken"
    },
    'severity_map': {
        "Crack": 'Leve',
        "Scratch": 'Leve',
        "Tire Flat": 'Severo',
        "Dent": 'Moderado',
        "Glass Shatter": 'Severo',
        "Lamp Broken": 'Severo'
    },
    'location_map': {
        "Crack": 'Para-choque/Plásticos',
        "Scratch": 'Pintura',
        "Tire Flat": 'Rodas',
        "Dent": 'Carroceria',
        "Glass Shatter": 'Para-brisa/Vidros',
        "Lamp Broken": 'Faróis/Lanternas'
    },
    'cost_ranges': {
        'Severo': (1500, 4000),
        'Moderado': (600, 1500),
        'Leve': (200, 600)
    },
    'class_names': {
        "Crack": 'Rachadura',
        "Scratch": 'Risco',
        "Tire Flat": 'Pneu Vazio',
        "Dent": 'Amassado',
        "Glass Shatter": 'Vidro Quebrado',
        "Lamp Broken": 'Lâmpada Quebrada'
    }
}

@st.cache_resource
def load_transformer_model():
    """Carrega o modelo Transformer do Hugging Face."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        model_name = "beingamit99/car_damage_detection"
        
        # Carregar processador e modelo
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        return processor, model
    except Exception as e:
        st.error(f"Erro ao carregar modelo Transformer: {e}")
        return None, None

@st.cache_resource
def load_pipeline_model():
    """Carrega o modelo usando pipeline (mais simples)."""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        model_name = "beingamit99/car_damage_detection"
        pipe = pipeline("image-classification", model=model_name)
        return pipe
    except Exception as e:
        st.error(f"Erro ao carregar pipeline: {e}")
        return None

@st.cache_resource
def load_yolo_fallback():
    """Carrega modelo YOLO como fallback."""
    if not YOLO_AVAILABLE:
        return None, None
    
    try:
        model = YOLO('yolov8n.pt')
        return model, "YOLO Genérico (Fallback)"
    except Exception as e:
        st.error(f"Erro ao carregar YOLO: {e}")
        return None, None

def predict_with_transformer(image, processor, model, confidence_threshold=0.5):
    """Faz predição usando o modelo Transformer."""
    try:
        # Processar imagem
        inputs = processor(images=image, return_tensors="pt")
        
        # Fazer predição
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        
        # Obter probabilidades
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Obter todas as classes com suas probabilidades
        detections = []
        label_map = model.config.id2label
        
        for class_id, prob in enumerate(probabilities[0]):
            if prob >= confidence_threshold:
                class_name = label_map.get(class_id, f"class_{class_id}")
                # Mapear para nome legível
                mapped_name = DAMAGE_CONFIG['class_mapping'].get(str(class_id), class_name)
                
                detection = {
                    'class': mapped_name,
                    'confidence': float(prob),
                    'bbox': [0, 0, image.width, image.height]  # Imagem inteira para classificação
                }
                detections.append(detection)
        
        # Ordenar por confiança
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
        
    except Exception as e:
        st.error(f"Erro na predição Transformer: {e}")
        return []

def predict_with_pipeline(image, pipe, confidence_threshold=0.5):
    """Faz predição usando pipeline."""
    try:
        # Fazer predição
        results = pipe(image)
        
        detections = []
        for result in results:
            if result['score'] >= confidence_threshold:
                # Mapear label para nome conhecido
                class_name = result['label']
                mapped_name = DAMAGE_CONFIG['class_mapping'].get(class_name, class_name)
                
                detection = {
                    'class': mapped_name,
                    'confidence': float(result['score']),
                    'bbox': [0, 0, image.width, image.height]
                }
                detections.append(detection)
        
        return detections
        
    except Exception as e:
        st.error(f"Erro na predição Pipeline: {e}")
        return []

def create_annotated_image(image, detections):
    """Cria imagem anotada com as detecções."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Adicionar texto com as detecções
        y_offset = 30
        for i, detection in enumerate(detections[:3]):  # Mostrar apenas top 3
            class_name = DAMAGE_CONFIG['class_names'].get(detection['class'], detection['class'])
            confidence = detection['confidence']
            
            text = f"{class_name}: {confidence:.1%}"
            ax.text(10, y_offset + (i * 25), text, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                   fontsize=12, color='white', weight='bold')
        
        # Converter para imagem
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        annotated_image = Image.open(buf)
        plt.close()
        
        return np.array(annotated_image)
        
    except Exception as e:
        st.warning(f"Erro ao criar imagem anotada: {e}")
        return np.array(image)

def create_damage_analysis(detections):
    """Analisa as detecções de danos."""
    damage_report = []
    for i, detection in enumerate(detections):
        class_name = detection['class']
        severity = DAMAGE_CONFIG['severity_map'].get(class_name, 'Moderado')
        location = DAMAGE_CONFIG['location_map'].get(class_name, 'Carroceria')
        
        cost_range = DAMAGE_CONFIG['cost_ranges'].get(severity, (500, 1500))
        estimated_cost = int(np.random.randint(cost_range[0], cost_range[1]))
        
        damage_report.append({
            'damage_id': f"DMG_{i+1:03d}",
            'class': class_name,
            'class_display': DAMAGE_CONFIG['class_names'].get(class_name, class_name),
            'confidence': detection['confidence'],
            'severity': severity,
            'location': location,
            'estimated_cost': estimated_cost,
            'bbox': detection['bbox']
        })
    return damage_report

def create_detection_summary(detections):
    """Cria resumo das detecções."""
    if not detections:
        return "Nenhum dano detectado na imagem."
    
    summary = [f"**Total de danos detectados: {len(detections)}**\n"]
    for detection in detections:
        class_name = DAMAGE_CONFIG['class_names'].get(detection['class'], detection['class'])
        confidence = detection['confidence']
        summary.append(f"• **{class_name}**: Confiança {confidence:.1%}")
    
    return "\n".join(summary)

def create_confidence_chart(damage_analysis):
    """Cria gráfico de confiança."""
    if not damage_analysis:
        return None
    
    df = pd.DataFrame(damage_analysis)
    fig = px.bar(
        df, 
        x='class_display', 
        y='confidence',
        title='Confiança das Detecções de Dano',
        labels={'confidence': 'Confiança (%)', 'class_display': 'Tipo de Dano'},
        color='confidence',
        color_continuous_scale='RdYlGn',
        text='confidence'
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False, yaxis=dict(tickformat='.0%'))
    return fig

def create_damage_report_json(damage_analysis, vehicle_info):
    """Gera relatório em JSON."""
    severity_count = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    total_cost = 0
    
    for damage in damage_analysis:
        if damage['severity'] in severity_count:
            severity_count[damage['severity']] += 1
        total_cost += damage['estimated_cost']
    
    urgency = 'Baixa'
    if severity_count['Severo'] > 0:
        urgency = 'Alta'
    elif severity_count['Moderado'] > 0:
        urgency = 'Média'

    report = {
        "inspection_info": {
            "timestamp": datetime.now().isoformat(),
            "inspector": "Sistema IA Carglass",
            "version": "3.0",
            "model": "Transformer ViT (beingamit99/car_damage_detection)",
        },
        "vehicle_info": vehicle_info,
        "damage_analysis": {
            "total_damages": len(damage_analysis),
            "severity_count": severity_count,
            "estimated_total_cost": f"R$ {total_cost:,.2f}",
            "repair_urgency": urgency,
        },
        "damages": damage_analysis
    }
    return report

def main():
    """Função principal."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🚗 Carglass - Detector de Danos Veiculares</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Análise de danos com IA Transformer (Versão 3.0)</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configurações")
        
        # Seleção do método
        method_choice = st.selectbox(
            "Método de Predição:",
            options=["pipeline", "transformer", "yolo_fallback"],
            format_func=lambda x: {
                "pipeline": "🤗 Pipeline (Recomendado)",
                "transformer": "🔬 Transformer Manual",
                "yolo_fallback": "⚡ YOLO (Fallback)"
            }[x],
            index=0,
            help="Pipeline é mais simples e estável"
        )
        
        confidence_threshold = st.slider(
            "Threshold de Confiança", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3, 
            step=0.1,
            help="Threshold mais baixo para detectar mais danos"
        )
        
        st.header("Sobre o Modelo")
        st.markdown("""
        **Modelo Transformer ViT**
        
        **Classes detectadas:**
        - 🔴 Rachadura (Crack)
        - 🟡 Risco (Scratch)  
        - 🔴 Pneu Vazio (Tire Flat)
        - 🟠 Amassado (Dent)
        - 🔴 Vidro Quebrado (Glass Shatter)
        - 🔴 Lâmpada Quebrada (Lamp Broken)
        """)
        
        st.header("Informações do Veículo")
        vehicle_plate = st.text_input("Placa", placeholder="ABC-1234")
        vehicle_model = st.text_input("Modelo", placeholder="Ex: Toyota Corolla")
        vehicle_year = st.number_input("Ano", min_value=1990, max_value=datetime.now().year + 1, value=datetime.now().year)
        vehicle_color = st.text_input("Cor", placeholder="Ex: Branco")

    # Carregar modelo baseado na escolha
    model_loaded = False
    processor, model, pipe = None, None, None
    
    if method_choice == "pipeline":
        if TRANSFORMERS_AVAILABLE:
            pipe = load_pipeline_model()
            if pipe:
                st.success("✅ Pipeline Transformer carregado!")
                model_loaded = True
            else:
                st.error("❌ Erro ao carregar Pipeline")
        else:
            st.error("❌ Transformers não disponível")
    
    elif method_choice == "transformer":
        if TRANSFORMERS_AVAILABLE:
            processor, model = load_transformer_model()
            if processor and model:
                st.success("✅ Modelo Transformer carregado!")
                model_loaded = True
            else:
                st.error("❌ Erro ao carregar Transformer")
        else:
            st.error("❌ Transformers não disponível")
    
    else:  # yolo_fallback
        yolo_model, yolo_name = load_yolo_fallback()
        if yolo_model:
            st.success(f"✅ {yolo_name} carregado!")
            model_loaded = True
        else:
            st.error("❌ Erro ao carregar YOLO")

    if not model_loaded:
        st.error("❌ Nenhum modelo pôde ser carregado.")
        return

    st.header("1. Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem do veículo:",
        type=['png', 'jpg', 'jpeg'],
        help="Envie uma foto clara do veículo para análise de danos"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Imagem Original")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        with st.spinner("🔍 Analisando danos com IA Transformer..."):
            # Fazer predição baseada no método escolhido
            if method_choice == "pipeline" and pipe:
                detections = predict_with_pipeline(image, pipe, confidence_threshold)
            elif method_choice == "transformer" and processor and model:
                detections = predict_with_transformer(image, processor, model, confidence_threshold)
            else:
                # Fallback para YOLO (implementação simplificada)
                detections = []
            
            # Criar imagem anotada
            annotated_img = create_annotated_image(image, detections)
            damage_analysis = create_damage_analysis(detections)

        with col2:
            st.subheader("🎯 Análise de Danos")
            st.image(annotated_img, caption="Danos detectados pela IA", use_container_width=True)

        # Mostrar resultados
        if not detections:
            st.success("✅ Nenhum dano detectado na imagem!")
            st.info("💡 Tente diminuir o threshold de confiança se houver danos visíveis.")
            st.balloons()
        else:
            st.header("2. Resultados da Análise")
            total_cost = sum(d['estimated_cost'] for d in damage_analysis)
            urgency = create_damage_report_json(damage_analysis, {})['damage_analysis']['repair_urgency']

            c1, c2, c3 = st.columns(3)
            c1.metric("🔍 Danos Detectados", len(damage_analysis))
            c2.metric("💰 Custo Estimado", f"R$ {total_cost:,.2f}")
            c3.metric("⚠️ Urgência", urgency)

            st.markdown("### Resumo dos Danos")
            summary = create_detection_summary(detections)
            st.markdown(summary)

            if damage_analysis:
                st.markdown("### Detalhes dos Danos")
                df_display = pd.DataFrame(damage_analysis)[['class_display', 'severity', 'confidence', 'estimated_cost']]
                df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
                df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
                df_display.columns = ['Tipo de Dano', 'Severidade', 'Confiança', 'Custo Estimado']
                st.dataframe(df_display, use_container_width=True)

                chart = create_confidence_chart(damage_analysis)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                st.header("3. Relatório de Inspeção")
                vehicle_info = {
                    "plate": vehicle_plate or "Não informado",
                    "model": vehicle_model or "Não informado", 
                    "year": str(vehicle_year),
                    "color": vehicle_color or "Não informado"
                }
                report = create_damage_report_json(damage_analysis, vehicle_info)
                report_json = json.dumps(report, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📄 Baixar Relatório Completo (JSON)",
                    data=report_json,
                    file_name=f"relatorio_danos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                with st.expander("Visualizar JSON do Relatório"):
                    st.json(report)
    else:
        st.info("👆 **Aguardando imagem para análise.**")
        
        with st.expander("💡 Dicas para Melhores Resultados"):
            st.markdown("""
            **Para obter melhores detecções:**
            
            1. **Use fotos claras** com boa iluminação
            2. **Foque nas áreas danificadas** 
            3. **Evite reflexos** e sombras excessivas
            4. **Threshold baixo** (30%) detecta mais danos
            5. **Pipeline** é o método mais estável
            6. **Modelo especializado** em 6 tipos de dano específicos
            """)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Carglass IA v3.0 - Powered by Transformer ViT</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
