import streamlit as st
from PIL import Image
import pandas as pd
import json
import datetime
import requests
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="Carglass - Detector de Danos Real",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_damage_model():
    model_url = "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector/raw/main/trained.pt"
    model_path = "trained.pt"
    
    if not os.path.exists(model_path):
        st.info("🔄 Baixando modelo YOLO11m especializado em danos...")
        try:
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Modelo baixado com sucesso!")
            else:
                st.error("❌ Erro ao baixar modelo. Usando modelo genérico.")
                return YOLO('yolov8n.pt'), False
        except Exception as e:
            st.error(f"❌ Erro ao baixar modelo: {e}")
            return YOLO('yolov8n.pt'), False
    
    try:
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return YOLO('yolov8n.pt'), False

def process_damage_detection(image, model, is_damage_model):
    results = model(image)
    detections = []
    
    if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                try:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    if confidence > 0.3:
                        if is_damage_model:
                            class_name = model.names[class_id]
                            damage_type = class_name.replace('_', ' ').title()
                            
                            severity_map = {
                                'shattered_glass': 'Severo',
                                'flat_tire': 'Severo', 
                                'broken_lamp': 'Moderado',
                                'dent': 'Moderado',
                                'scratch': 'Leve',
                                'crack': 'Leve'
                            }
                            
                            location_map = {
                                'shattered_glass': 'Para-brisa/Vidros',
                                'flat_tire': 'Rodas',
                                'broken_lamp': 'Faróis/Lanternas', 
                                'dent': 'Carroceria',
                                'scratch': 'Pintura',
                                'crack': 'Para-choque/Plásticos'
                            }
                            
                            severity = severity_map.get(class_name, 'Moderado')
                            location = location_map.get(class_name, 'Carroceria')
                            
                        else:
                            damage_type = "Dano Detectado"
                            severity = "Moderado"
                            location = "A definir"
                        
                        detection = {
                            'id': i + 1,
                            'damage_type': damage_type,
                            'severity': severity,
                            'location': location,
                            'confidence': confidence,
                            'bbox': {
                                'x1': float(bbox[0]),
                                'y1': float(bbox[1]),
                                'x2': float(bbox[2]),
                                'y2': float(bbox[3])
                            },
                            'area_pixels': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                        }
                        detections.append(detection)
                        
                except Exception as e:
                    continue
    
    try:
        annotated_img = results[0].plot()
        annotated_img_rgb = annotated_img[..., ::-1]
    except:
        import numpy as np
        annotated_img_rgb = np.array(image)
    
    return detections, annotated_img_rgb

def create_damage_report_json(vehicle_info, detections):
    cost_map = {
        'Severo': 2000,
        'Moderado': 800,
        'Leve': 300
    }
    
    total_cost = sum([cost_map.get(d['severity'], 500) for d in detections])
    
    report = {
        "inspection_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "inspector": "Sistema IA Carglass",
            "version": "5.0",
            "model": "YOLO11m Car Damage Detector"
        },
        "vehicle_info": vehicle_info,
        "damage_summary": {
            "total_damages": len(detections),
            "severity_count": {
                "Leve": len([d for d in detections if d['severity'] == 'Leve']),
                "Moderado": len([d for d in detections if d['severity'] == 'Moderado']),
                "Severo": len([d for d in detections if d['severity'] == 'Severo'])
            },
            "damage_types": list(set([d['damage_type'] for d in detections])),
            "estimated_total_cost": f"R$ {total_cost:,.2f}",
            "repair_urgency": "Alta" if any(d['severity'] == 'Severo' for d in detections) else "Média"
        },
        "detections": detections,
        "recommendations": generate_recommendations(detections)
    }
    return report

def generate_recommendations(detections):
    recommendations = []
    
    for detection in detections:
        severity = detection['severity']
        damage_type = detection['damage_type']
        location = detection['location']
        
        if severity == 'Severo':
            recommendations.append({
                "priority": "Urgente",
                "action": f"Reparo imediato necessário: {damage_type}",
                "location": location,
                "estimated_cost": "R$ 1.500 - R$ 3.500",
                "timeframe": "1-2 dias úteis"
            })
        elif severity == 'Moderado':
            recommendations.append({
                "priority": "Recomendado", 
                "action": f"Reparo recomendado: {damage_type}",
                "location": location,
                "estimated_cost": "R$ 500 - R$ 1.500",
                "timeframe": "3-7 dias úteis"
            })
        else:
            recommendations.append({
                "priority": "Opcional",
                "action": f"Reparo estético: {damage_type}",
                "location": location,
                "estimated_cost": "R$ 200 - R$ 600",
                "timeframe": "1-3 dias úteis"
            })
    
    return recommendations

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Sistema de Detecção de Danos - YOLO11m")
st.markdown("**Detecção real de danos com modelo especializado YOLO11m**")

model, is_damage_model = load_damage_model()

if model is None:
    st.error("❌ Não foi possível carregar o modelo.")
    st.stop()

if is_damage_model:
    st.success("✅ Modelo YOLO11m especializado carregado!")
    st.info("🎯 **Classes detectadas:** Amassados, Riscos, Rachaduras, Vidros Quebrados, Lâmpadas Quebradas, Pneus Vazios")
else:
    st.warning("⚠️ Usando modelo genérico para demonstração")

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

st.sidebar.header("📤 Upload da Imagem")
uploaded_file = st.sidebar.file_uploader("Selecione uma imagem do veículo:", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.header("🔍 Análise com YOLO11m")
    
    with st.spinner("Analisando danos com modelo especializado..."):
        detections, annotated_img = process_damage_detection(image, model, is_damage_model)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Danos Detectados")
        st.image(annotated_img, use_column_width=True)
    
    if detections:
        st.header("📊 Resultados da Análise")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Danos", len(detections))
        
        with col2:
            severe_count = len([d for d in detections if d['severity'] == 'Severo'])
            st.metric("Danos Severos", severe_count)
        
        with col3:
            avg_confidence = sum([d['confidence'] for d in detections]) / len(detections)
            st.metric("Confiança Média", f"{avg_confidence:.1%}")
        
        with col4:
            damage_types = len(set([d['damage_type'] for d in detections]))
            st.metric("Tipos de Danos", damage_types)
        
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
        
        st.header("📄 Relatório JSON")
        report_json = create_damage_report_json(vehicle_info, detections)
        st.json(report_json)
        
        json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Baixar Relatório JSON",
            data=json_str,
            file_name=f"relatorio_danos_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        st.header("💡 Recomendações")
        recommendations = report_json['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"Recomendação {i}: {rec['priority']} - {rec['location']}"):
                st.write(f"**Ação:** {rec['action']}")
                st.write(f"**Custo Estimado:** {rec['estimated_cost']}")
                st.write(f"**Prazo:** {rec['timeframe']}")
    
    else:
        st.success("✅ Nenhum dano detectado na imagem!")
        report_json = create_damage_report_json(vehicle_info, [])
        st.json(report_json)

else:
    st.info("👆 Aguardando o envio de uma imagem na barra lateral.")

st.markdown("---")
st.markdown("**Carglass - YOLO11m Car Damage Detector** | Modelo Especializado")
