import streamlit as st
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import plotly.express as px
import pandas as pd
import requests
import base64
from io import BytesIO

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

# Configuração para o modelo via API
DAMAGE_CONFIG = {
    'class_mapping': {
        "LABEL_0": "Crack",
        "LABEL_1": "Scratch", 
        "LABEL_2": "Tire Flat",
        "LABEL_3": "Dent",
        "LABEL_4": "Glass Shatter",
        "LABEL_5": "Lamp Broken",
        # Variações possíveis
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

def download_custom_model():
    """Baixa o modelo customizado como fallback."""
    model_path = "car_damage_best.pt"
    
    if not os.path.exists(model_path):
        st.info("🔄 Baixando modelo customizado...")
        
        model_url = "https://github.com/Vamap91/YOLOProject/releases/download/v2.0.0/car_damage_best.pt"
        
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Baixando: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB")
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Modelo customizado baixado!")
            
        except Exception as e:
            st.error(f"❌ Erro ao baixar modelo customizado: {e}")
            return None
    
    return model_path

@st.cache_resource
def load_yolo_model():
    """Carrega modelo YOLO como fallback."""
    if not YOLO_AVAILABLE:
        return None, None
    
    try:
        # Tentar carregar modelo customizado primeiro
        model_path = download_custom_model()
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            return model, "Customizado (car_damage_best.pt)"
        else:
            # Fallback para modelo genérico
            model = YOLO('yolov8n.pt')
            return model, "Genérico (YOLOv8n)"
    except Exception as e:
        st.error(f"Erro ao carregar YOLO: {e}")
        return None, None

def image_to_base64(image):
    """Converte imagem PIL para base64."""
    try:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        st.error(f"Erro ao converter imagem: {e}")
        return None

def predict_with_huggingface_api(image, confidence_threshold=0.3):
    """Faz predição usando a API do Hugging Face."""
    try:
        # URL da API do Hugging Face
        API_URL = "https://api-inference.huggingface.co/models/beingamit99/car_damage_detection"
        
        # Converter imagem para bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        
        # Fazer requisição para a API
        response = requests.post(API_URL, data=img_bytes)
        
        if response.status_code == 200:
            results = response.json()
            
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
            
            return detections, "API Hugging Face"
            
        elif response.status_code == 503:
            st.warning("⏳ Modelo está carregando na API do Hugging Face. Tente novamente em alguns segundos.")
            return [], "API Hugging Face (Carregando)"
        else:
            st.error(f"Erro na API: {response.status_code} - {response.text}")
            return [], "API Hugging Face (Erro)"
            
    except Exception as e:
        st.error(f"Erro na API do Hugging Face: {e}")
        return [], "API Hugging Face (Erro)"

def predict_with_yolo(image, model, confidence_threshold=0.5):
    """Faz predição usando YOLO como fallback."""
    try:
        img_array = np.array(image)
        results = model(img_array, conf=confidence_threshold)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if hasattr(boxes, '__len__') and len(boxes) > 0:
                    if hasattr(boxes, 'xyxy'):
                        for i in range(len(boxes.xyxy)):
                            try:
                                class_id = int(boxes.cls[i])
                                confidence = float(boxes.conf[i])
                                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                                class_name = model.names.get(class_id, f"class_{class_id}")
                                
                                detection = {
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': bbox
                                }
                                detections.append(detection)
                            except Exception:
                                continue
        
        return detections, "YOLO"
        
    except Exception as e:
        st.error(f"Erro na predição YOLO: {e}")
        return [], "YOLO (Erro)"

def create_annotated_image(image, detections, model_type):
    """Cria imagem anotada com as detecções."""
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Adicionar título
        ax.set_title(f"Análise: {model_type}", fontsize=16, weight='bold', pad=20)
        
        # Adicionar texto com as detecções
        y_offset = 30
        for i, detection in enumerate(detections[:5]):  # Mostrar apenas top 5
            class_name = DAMAGE_CONFIG['class_names'].get(detection['class'], detection['class'])
            confidence = detection['confidence']
            severity = DAMAGE_CONFIG['severity_map'].get(detection['class'], 'Moderado')
            
            # Cor baseada na severidade
            color = {'Leve': 'yellow', 'Moderado': 'orange', 'Severo': 'red'}.get(severity, 'blue')
            
            text = f"{class_name}: {confidence:.1%} ({severity})"
            ax.text(10, y_offset + (i * 30), text, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   fontsize=11, color='black', weight='bold')
        
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

def filter_valid_detections(detections):
    """Filtra detecções válidas."""
    valid_detections = []
    
    for detection in detections:
        class_name = detection['class']
        # Aceitar apenas classes conhecidas de dano
        if class_name in DAMAGE_CONFIG['class_names']:
            valid_detections.append(detection)
    
    return valid_detections

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
        severity = DAMAGE_CONFIG['severity_map'].get(detection['class'], 'Moderado')
        summary.append(f"• **{class_name}**: {confidence:.1%} ({severity})")
    
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
        color='severity',
        color_discrete_map={'Leve': 'yellow', 'Moderado': 'orange', 'Severo': 'red'},
        text='confidence'
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=400, yaxis=dict(tickformat='.0%'))
    return fig

def create_damage_report_json(damage_analysis, vehicle_info, model_type):
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
            "version": "3.1",
            "model": f"{model_type} (beingamit99/car_damage_detection)",
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
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Análise de danos com IA via API (Versão 3.1)</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configurações")
        
        # Seleção do método
        method_choice = st.selectbox(
            "Método de Análise:",
            options=["api_hf", "yolo_fallback"],
            format_func=lambda x: {
                "api_hf": "🤗 API Hugging Face (Recomendado)",
                "yolo_fallback": "⚡ YOLO (Fallback)"
            }[x],
            index=0,
            help="API do Hugging Face usa modelo especializado em danos"
        )
        
        confidence_threshold = st.slider(
            "Threshold de Confiança", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3, 
            step=0.1,
            help="Threshold mais baixo detecta mais danos"
        )
        
        st.header("Sobre o Sistema")
        st.markdown("""
        **Versão 3.1 - API Hugging Face**
        
        **Modelo Especializado:**
        - 🔴 Rachadura (Crack)
        - 🟡 Risco (Scratch)  
        - 🔴 Pneu Vazio (Tire Flat)
        - 🟠 Amassado (Dent)
        - 🔴 Vidro Quebrado (Glass Shatter)
        - 🔴 Lâmpada Quebrada (Lamp Broken)
        
        **Sem instalações extras!**
        """)
        
        st.header("Informações do Veículo")
        vehicle_plate = st.text_input("Placa", placeholder="ABC-1234")
        vehicle_model = st.text_input("Modelo", placeholder="Ex: Toyota Corolla")
        vehicle_year = st.number_input("Ano", min_value=1990, max_value=datetime.now().year + 1, value=datetime.now().year)
        vehicle_color = st.text_input("Cor", placeholder="Ex: Branco")

    # Carregar modelo YOLO se necessário
    yolo_model, yolo_name = None, None
    if method_choice == "yolo_fallback":
        yolo_model, yolo_name = load_yolo_model()
        if yolo_model:
            st.success(f"✅ {yolo_name} carregado!")
        else:
            st.error("❌ Erro ao carregar YOLO")
            return
    else:
        st.success("✅ API Hugging Face pronta!")

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
        
        with st.spinner("🔍 Analisando danos com IA especializada..."):
            # Fazer predição baseada no método escolhido
            if method_choice == "api_hf":
                detections, model_type = predict_with_huggingface_api(image, confidence_threshold)
            else:
                detections, model_type = predict_with_yolo(image, yolo_model, confidence_threshold)
            
            # Filtrar detecções válidas
            valid_detections = filter_valid_detections(detections)
            
            # Criar imagem anotada
            annotated_img = create_annotated_image(image, valid_detections, model_type)
            damage_analysis = create_damage_analysis(valid_detections)

        with col2:
            st.subheader("🎯 Análise de Danos")
            st.image(annotated_img, caption=f"Análise: {model_type}", use_container_width=True)

        # Mostrar resultados
        if not valid_detections:
            st.success("✅ Nenhum dano detectado na imagem!")
            if len(detections) > 0:
                st.info(f"💡 {len(detections)} detecções foram filtradas. Tente diminuir o threshold.")
            else:
                st.info("💡 Tente diminuir o threshold de confiança se houver danos visíveis.")
            st.balloons()
        else:
            st.header("2. Resultados da Análise")
            total_cost = sum(d['estimated_cost'] for d in damage_analysis)
            urgency = create_damage_report_json(damage_analysis, {}, model_type)['damage_analysis']['repair_urgency']

            c1, c2, c3 = st.columns(3)
            c1.metric("🔍 Danos Detectados", len(damage_analysis))
            c2.metric("💰 Custo Estimado", f"R$ {total_cost:,.2f}")
            c3.metric("⚠️ Urgência", urgency)

            st.markdown("### Resumo dos Danos")
            summary = create_detection_summary(valid_detections)
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
                report = create_damage_report_json(damage_analysis, vehicle_info, model_type)
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
            
            1. **API Hugging Face** é o método recomendado
            2. **Use fotos claras** com boa iluminação
            3. **Foque nas áreas danificadas** 
            4. **Threshold 30%** é ideal para detectar mais danos
            5. **Modelo especializado** em 6 tipos específicos de dano
            6. **Sem instalações** - funciona direto no Streamlit Cloud
            
            **Se a API estiver carregando, aguarde alguns segundos e tente novamente.**
            """)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Carglass IA v3.1 - Powered by Hugging Face API</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
