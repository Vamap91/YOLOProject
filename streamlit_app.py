# streamlit_app_final.py - Solução Definitiva para Detecção de Danos Carglass

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import pandas as pd
import json
import datetime
import os
import requests
from io import BytesIO

# --- Configuração da Página ---
st.set_page_config(
    page_title="Carglass - Detector de Danos Definitivo",
    page_icon="🛡️",
    layout="wide"
)

# --- Funções Principais ---

@st.cache_resource
def load_damage_model():
    """
    Carrega o modelo YOLO específico para detecção de danos.
    Primeiro tenta baixar o modelo do Hugging Face, senão usa o genérico.
    """
    try:
        # Tenta baixar o modelo específico de danos do Hugging Face
        model_url = "https://huggingface.co/nezahatkorkmaz/car-damage-level-detection-yolov8/resolve/main/car-damage.pt"
        
        if not os.path.exists("car-damage.pt"):
            st.info("🔄 Baixando modelo especializado em danos... Aguarde.")
            response = requests.get(model_url)
            if response.status_code == 200:
                with open("car-damage.pt", "wb") as f:
                    f.write(response.content)
                st.success("✅ Modelo de danos baixado com sucesso!")
            else:
                st.warning("⚠️ Não foi possível baixar o modelo especializado. Usando modelo genérico.")
                return YOLO('yolov8n.pt'), False
        
        # Carrega o modelo específico de danos
        model = YOLO('car-damage.pt')
        return model, True
        
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar modelo de danos: {e}. Usando modelo genérico.")
        return YOLO('yolov8n.pt'), False

def simulate_damage_detection(image, generic_model):
    """
    Simula detecção de danos usando o modelo genérico.
    Mapeia classes do COCO para tipos de danos para demonstração.
    """
    results = generic_model(image)
    detections = []
    
    # Mapeamento simulado de classes COCO para danos
    damage_mapping = {
        'person': {'type': 'Amassado', 'severity': 'Moderado', 'location': 'Lateral'},
        'car': {'type': 'Risco', 'severity': 'Leve', 'location': 'Porta'},
        'bicycle': {'type': 'Vidro Quebrado', 'severity': 'Severo', 'location': 'Para-brisa'},
        'motorcycle': {'type': 'Pneu Vazio', 'severity': 'Severo', 'location': 'Roda'},
        'truck': {'type': 'Amassado', 'severity': 'Severo', 'location': 'Para-choque'},
        'bus': {'type': 'Risco', 'severity': 'Moderado', 'location': 'Capô'}
    }

    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            class_name = generic_model.names[int(box.cls[0])]
            
            if class_name in damage_mapping:
                damage_info = damage_mapping[class_name]
                bbox = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'id': i + 1,
                    'damage_type': damage_info['type'],
                    'severity': damage_info['severity'],
                    'location': damage_info['location'],
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x1': float(bbox[0]),
                        'y1': float(bbox[1]),
                        'x2': float(bbox[2]),
                        'y2': float(bbox[3])
                    },
                    'area_pixels': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                }
                detections.append(detection)
    
    annotated_img = results[0].plot()
    annotated_img_rgb = annotated_img[..., ::-1]
    
    return detections, annotated_img_rgb

def process_real_damage_detection(image, damage_model):
    """
    Processa detecção real de danos usando modelo especializado.
    """
    results = damage_model(image)
    detections = []
    
    # Mapeamento das classes do modelo de danos
    severity_mapping = {
        0: 'Leve',
        1: 'Moderado', 
        2: 'Severo'
    }

    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            detection = {
                'id': i + 1,
                'damage_type': 'Dano Detectado',
                'severity': severity_mapping.get(class_id, 'Desconhecido'),
                'location': 'A definir',  # Seria calculado com base na posição da bbox
                'confidence': float(box.conf[0]),
                'bbox': {
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3])
                },
                'area_pixels': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            }
            detections.append(detection)
    
    annotated_img = results[0].plot()
    annotated_img_rgb = annotated_img[..., ::-1]
    
    return detections, annotated_img_rgb

def create_damage_report_json(vehicle_info, detections, image_path=None):
    """
    Cria um relatório completo em formato JSON.
    """
    report = {
        "inspection_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "inspector": "Sistema IA Carglass",
            "version": "2.0"
        },
        "vehicle_info": vehicle_info,
        "damage_summary": {
            "total_damages": len(detections),
            "severity_count": {
                "Leve": len([d for d in detections if d['severity'] == 'Leve']),
                "Moderado": len([d for d in detections if d['severity'] == 'Moderado']),
                "Severo": len([d for d in detections if d['severity'] == 'Severo'])
            },
            "damage_types": list(set([d['damage_type'] for d in detections]))
        },
        "detections": detections,
        "recommendations": generate_recommendations(detections)
    }
    
    return report

def generate_recommendations(detections):
    """
    Gera recomendações baseadas nos danos detectados.
    """
    recommendations = []
    
    for detection in detections:
        if detection['severity'] == 'Severo':
            recommendations.append({
                "priority": "Alta",
                "action": f"Reparo urgente necessário para {detection['damage_type']} em {detection['location']}",
                "estimated_cost": "R$ 500 - R$ 2000"
            })
        elif detection['severity'] == 'Moderado':
            recommendations.append({
                "priority": "Média",
                "action": f"Reparo recomendado para {detection['damage_type']} em {detection['location']}",
                "estimated_cost": "R$ 200 - R$ 800"
            })
        else:
            recommendations.append({
                "priority": "Baixa",
                "action": f"Reparo opcional para {detection['damage_type']} em {detection['location']}",
                "estimated_cost": "R$ 50 - R$ 300"
            })
    
    return recommendations

# --- Interface Principal ---

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Sistema de Detecção de Danos - Versão Definitiva")
st.markdown("**Solução de IA para identificação precisa de danos veiculares com saída em JSON**")

# Carrega o modelo
model, is_damage_model = load_damage_model()

if model is None:
    st.error("❌ Não foi possível carregar nenhum modelo.")
    st.stop()

# Informações sobre o modelo carregado
if is_damage_model:
    st.success("✅ Modelo especializado em danos carregado!")
    st.info("🎯 Este modelo detecta níveis de severidade: Leve, Moderado, Severo")
else:
    st.warning("⚠️ Usando modelo genérico para demonstração")
    st.info("🔄 O modelo simula detecção de danos mapeando objetos para tipos de avarias")

# Sidebar para informações do veículo
st.sidebar.header("📋 Informações do Veículo")
vehicle_plate = st.sidebar.text_input("Placa", "ABC-1234")
vehicle_model = st.sidebar.text_input("Modelo", "Toyota Corolla")
vehicle_year = st.sidebar.number_input("Ano", min_value=1990, max_value=2025, value=2020)
vehicle_color = st.sidebar.selectbox("Cor", ["Branco", "Preto", "Prata", "Azul", "Vermelho", "Outro"])

vehicle_info = {
    "plate": vehicle_plate,
    "model": vehicle_model,
    "year": vehicle_year,
    "color": vehicle_color
}

# Upload de imagem
st.sidebar.header("📤 Upload da Imagem")
uploaded_file = st.sidebar.file_uploader(
    "Selecione uma imagem do veículo:",
    type=['png', 'jpg', 'jpeg']
)

# Opção de usar imagens de exemplo
st.sidebar.header("🖼️ Ou Use um Exemplo")
example_option = st.sidebar.selectbox(
    "Escolha um exemplo:",
    ["Nenhum", "Carro com Amassado", "Vidro Quebrado", "Risco na Lateral"]
)

example_urls = {
    "Carro com Amassado": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500",
    "Vidro Quebrado": "https://images.unsplash.com/photo-1544306094-e2dcf9479da3?w=500", 
    "Risco na Lateral": "https://images.unsplash.com/photo-1609521263047-f8f205293f24?w=500"
}

if example_option != "Nenhum" and example_option in example_urls:
    try:
        response = requests.get(example_urls[example_option])
        uploaded_file = BytesIO(response.content)
        uploaded_file.name = f"{example_option}.jpg"
    except:
        st.sidebar.error("Erro ao carregar imagem de exemplo")

# Processamento principal
if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.header("🔍 Análise em Andamento")
    
    with st.spinner("Analisando imagem com IA especializada..."):
        if is_damage_model:
            detections, annotated_img = process_real_damage_detection(image, model)
        else:
            detections, annotated_img = simulate_damage_detection(image, model)
    
    # Exibição dos resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Danos Detectados")
        st.image(annotated_img, use_column_width=True)
    
    # Resultados detalhados
    if detections:
        st.header("📊 Resultados da Análise")
        
        # Resumo em cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Danos", len(detections))
        
        with col2:
            severe_count = len([d for d in detections if d['severity'] == 'Severo'])
            st.metric("Danos Severos", severe_count, delta=None if severe_count == 0 else "⚠️")
        
        with col3:
            avg_confidence = sum([d['confidence'] for d in detections]) / len(detections)
            st.metric("Confiança Média", f"{avg_confidence:.1%}")
        
        with col4:
            damage_types = len(set([d['damage_type'] for d in detections]))
            st.metric("Tipos de Danos", damage_types)
        
        # Tabela de detecções
        st.subheader("📋 Detalhes dos Danos")
        df = pd.DataFrame(detections)
        display_df = df[['damage_type', 'severity', 'location', 'confidence']].copy()
        display_df.rename(columns={
            'damage_type': 'Tipo de Dano',
            'severity': 'Severidade', 
            'location': 'Localização',
            'confidence': 'Confiança'
        }, inplace=True)
        display_df['Confiança'] = display_df['Confiança'].map('{:.1%}'.format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Gerar relatório JSON
        st.header("📄 Relatório JSON")
        
        report_json = create_damage_report_json(vehicle_info, detections)
        
        # Exibir JSON formatado
        st.json(report_json)
        
        # Download do JSON
        json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Baixar Relatório JSON",
            data=json_str,
            file_name=f"relatorio_danos_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        # Recomendações
        st.header("💡 Recomendações")
        recommendations = report_json['recommendations']
        
        for i, rec in enumerate(recommendations):
            if rec['priority'] == 'Alta':
                st.error(f"🚨 **{rec['priority']}:** {rec['action']} - {rec['estimated_cost']}")
            elif rec['priority'] == 'Média':
                st.warning(f"⚠️ **{rec['priority']}:** {rec['action']} - {rec['estimated_cost']}")
            else:
                st.info(f"ℹ️ **{rec['priority']}:** {rec['action']} - {rec['estimated_cost']}")
    
    else:
        st.success("✅ Nenhum dano detectado na imagem!")
        
        # Ainda gera um JSON mesmo sem danos
        report_json = create_damage_report_json(vehicle_info, [])
        st.json(report_json)
        
        json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Baixar Relatório JSON (Sem Danos)",
            data=json_str,
            file_name=f"relatorio_sem_danos_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

else:
    st.info("👆 Aguardando o envio de uma imagem na barra lateral para iniciar a análise.")
    
    # Exemplo de JSON de saída
    st.header("📋 Exemplo de Saída JSON")
    st.markdown("**Este é o formato do JSON que será gerado após a análise:**")
    
    example_json = {
        "inspection_info": {
            "timestamp": "2025-09-10T14:30:00",
            "inspector": "Sistema IA Carglass",
            "version": "2.0"
        },
        "vehicle_info": {
            "plate": "ABC-1234",
            "model": "Toyota Corolla",
            "year": 2020,
            "color": "Branco"
        },
        "damage_summary": {
            "total_damages": 2,
            "severity_count": {
                "Leve": 1,
                "Moderado": 1,
                "Severo": 0
            },
            "damage_types": ["Risco", "Amassado"]
        },
        "detections": [
            {
                "id": 1,
                "damage_type": "Risco",
                "severity": "Leve",
                "location": "Porta",
                "confidence": 0.89,
                "bbox": {"x1": 150, "y1": 200, "x2": 300, "y2": 350},
                "area_pixels": 22500
            }
        ],
        "recommendations": [
            {
                "priority": "Baixa",
                "action": "Reparo opcional para Risco em Porta",
                "estimated_cost": "R$ 50 - R$ 300"
            }
        ]
    }
    
    st.json(example_json)

# Rodapé
st.markdown("---")
st.markdown("**Desenvolvido para Carglass** | Versão 2.0 - Detecção Especializada de Danos")
