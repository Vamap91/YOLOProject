import streamlit as st
from PIL import Image
import pandas as pd
import json
import datetime
import requests
import os
from ultralytics import YOLO
import numpy as np

st.set_page_config(
    page_title="Carglass - Sistema Híbrido Inteligente",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_damage_model():
    model_url = "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector/raw/main/trained.pt"
    model_path = "trained.pt"
    
    if not os.path.exists(model_path):
        st.info("🔄 Baixando modelo YOLO11m especializado...")
        try:
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Modelo YOLO11m carregado!")
            else:
                st.warning("⚠️ Usando modelo genérico")
                return YOLO('yolov8n.pt'), False
        except:
            st.warning("⚠️ Usando modelo genérico")
            return YOLO('yolov8n.pt'), False
    
    try:
        model = YOLO(model_path)
        return model, True
    except:
        return YOLO('yolov8n.pt'), False

def intelligent_damage_analysis(detections, image_size):
    """Análise inteligente dos danos detectados"""
    
    width, height = image_size
    enhanced_detections = []
    
    damage_intelligence = {
        'dent': {
            'portuguese_name': 'Amassado',
            'severity_logic': lambda area: 'Severo' if area > 5000 else 'Moderado' if area > 2000 else 'Leve',
            'repair_cost': lambda severity: {'Severo': 1500, 'Moderado': 800, 'Leve': 400}[severity],
            'urgency': lambda severity: {'Severo': 'Alta', 'Moderado': 'Média', 'Leve': 'Baixa'}[severity],
            'description': lambda severity, location: f"Amassado {severity.lower()} detectado na região {location}. {'Reparo urgente necessário.' if severity == 'Severo' else 'Reparo recomendado para restaurar aparência.'}"
        },
        'scratch': {
            'portuguese_name': 'Risco',
            'severity_logic': lambda area: 'Moderado' if area > 3000 else 'Leve',
            'repair_cost': lambda severity: {'Moderado': 600, 'Leve': 250}[severity],
            'urgency': lambda severity: {'Moderado': 'Média', 'Leve': 'Baixa'}[severity],
            'description': lambda severity, location: f"Risco {severity.lower()} na pintura da região {location}. {'Retoque recomendado para evitar oxidação.' if severity == 'Moderado' else 'Retoque estético opcional.'}"
        },
        'crack': {
            'portuguese_name': 'Rachadura',
            'severity_logic': lambda area: 'Severo' if area > 2000 else 'Moderado',
            'repair_cost': lambda severity: {'Severo': 1200, 'Moderado': 700}[severity],
            'urgency': lambda severity: {'Severo': 'Alta', 'Moderado': 'Média'}[severity],
            'description': lambda severity, location: f"Rachadura {severity.lower()} em {location}. {'Substituição necessária por questões de segurança.' if severity == 'Severo' else 'Reparo recomendado.'}"
        },
        'shattered_glass': {
            'portuguese_name': 'Vidro Quebrado',
            'severity_logic': lambda area: 'Severo',
            'repair_cost': lambda severity: 2500,
            'urgency': lambda severity: 'Crítica',
            'description': lambda severity, location: f"Vidro quebrado em {location}. Substituição imediata obrigatória por segurança."
        },
        'broken_lamp': {
            'portuguese_name': 'Lâmpada Quebrada',
            'severity_logic': lambda area: 'Moderado',
            'repair_cost': lambda severity: 400,
            'urgency': lambda severity: 'Média',
            'description': lambda severity, location: f"Lâmpada danificada em {location}. Substituição necessária para conformidade legal."
        },
        'flat_tire': {
            'portuguese_name': 'Pneu Vazio',
            'severity_logic': lambda area: 'Severo',
            'repair_cost': lambda severity: 300,
            'urgency': lambda severity: 'Alta',
            'description': lambda severity, location: f"Pneu vazio detectado. Verificação e possível substituição necessária."
        }
    }
    
    def determine_location(bbox, width, height):
        x_center = (bbox['x1'] + bbox['x2']) / 2
        y_center = (bbox['y1'] + bbox['y2']) / 2
        
        x_ratio = x_center / width
        y_ratio = y_center / height
        
        if y_ratio > 0.7:
            return "Para-choque frontal"
        elif y_ratio < 0.3:
            return "Teto/Para-brisa"
        elif x_ratio < 0.3:
            return "Lateral esquerda"
        elif x_ratio > 0.7:
            return "Lateral direita"
        elif y_ratio > 0.5:
            return "Frente"
        else:
            return "Centro do veículo"
    
    for detection in detections:
        damage_type = detection['damage_type'].lower().replace(' ', '_')
        
        if damage_type in damage_intelligence:
            intel = damage_intelligence[damage_type]
            area = detection['area_pixels']
            location = determine_location(detection['bbox'], width, height)
            
            severity = intel['severity_logic'](area)
            cost = intel['repair_cost'](severity)
            urgency = intel['urgency'](severity)
            description = intel['description'](severity, location)
            
            enhanced_detection = {
                'id': detection['id'],
                'damage_type': intel['portuguese_name'],
                'severity': severity,
                'location': location,
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'area_pixels': area,
                'repair_cost': cost,
                'urgency': urgency,
                'description': description,
                'technical_details': {
                    'area_cm2': round(area * 0.01, 2),
                    'position_ratio': {
                        'x': round((detection['bbox']['x1'] + detection['bbox']['x2']) / (2 * width), 3),
                        'y': round((detection['bbox']['y1'] + detection['bbox']['y2']) / (2 * height), 3)
                    }
                }
            }
            enhanced_detections.append(enhanced_detection)
    
    return enhanced_detections

def process_intelligent_detection(image, model, is_damage_model):
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
                        class_name = model.names[class_id]
                        
                        detection = {
                            'id': i + 1,
                            'damage_type': class_name,
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
        annotated_img_rgb = np.array(image)
    
    enhanced_detections = intelligent_damage_analysis(detections, image.size)
    
    return enhanced_detections, annotated_img_rgb

def create_comprehensive_report(vehicle_info, detections):
    total_cost = sum([d['repair_cost'] for d in detections])
    
    urgency_priority = {'Crítica': 4, 'Alta': 3, 'Média': 2, 'Baixa': 1}
    max_urgency = max([urgency_priority.get(d['urgency'], 1) for d in detections]) if detections else 1
    overall_urgency = {4: 'Crítica', 3: 'Alta', 2: 'Média', 1: 'Baixa'}[max_urgency]
    
    report = {
        "inspection_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "inspector": "Sistema Híbrido Carglass",
            "version": "7.0",
            "model": "YOLO11m + Análise Inteligente",
            "analysis_method": "Detecção automática com interpretação contextual"
        },
        "vehicle_info": vehicle_info,
        "damage_summary": {
            "total_damages": len(detections),
            "total_repair_cost": f"R$ {total_cost:,.2f}",
            "overall_urgency": overall_urgency,
            "severity_distribution": {
                "Crítico": len([d for d in detections if d['urgency'] == 'Crítica']),
                "Alto": len([d for d in detections if d['urgency'] == 'Alta']),
                "Médio": len([d for d in detections if d['urgency'] == 'Média']),
                "Baixo": len([d for d in detections if d['urgency'] == 'Baixa'])
            },
            "damage_types": list(set([d['damage_type'] for d in detections])),
            "affected_areas": list(set([d['location'] for d in detections]))
        },
        "detections": detections,
        "recommendations": generate_smart_recommendations(detections),
        "repair_timeline": generate_repair_timeline(detections)
    }
    return report

def generate_smart_recommendations(detections):
    recommendations = []
    
    critical_damages = [d for d in detections if d['urgency'] == 'Crítica']
    high_damages = [d for d in detections if d['urgency'] == 'Alta']
    
    if critical_damages:
        recommendations.append({
            "priority": "URGENTE - 24h",
            "action": "Reparo imediato obrigatório",
            "damages": [d['damage_type'] for d in critical_damages],
            "reason": "Questões de segurança e conformidade legal",
            "estimated_cost": f"R$ {sum([d['repair_cost'] for d in critical_damages]):,.2f}"
        })
    
    if high_damages:
        recommendations.append({
            "priority": "Alta - 1 semana",
            "action": "Reparo recomendado",
            "damages": [d['damage_type'] for d in high_damages],
            "reason": "Prevenção de agravamento e manutenção do valor",
            "estimated_cost": f"R$ {sum([d['repair_cost'] for d in high_damages]):,.2f}"
        })
    
    return recommendations

def generate_repair_timeline(detections):
    timeline = []
    
    for detection in sorted(detections, key=lambda x: {'Crítica': 4, 'Alta': 3, 'Média': 2, 'Baixa': 1}[x['urgency']], reverse=True):
        timeframes = {
            'Crítica': '24-48 horas',
            'Alta': '1-2 semanas',
            'Média': '2-4 semanas',
            'Baixa': '1-3 meses'
        }
        
        timeline.append({
            "damage": detection['damage_type'],
            "location": detection['location'],
            "timeframe": timeframes[detection['urgency']],
            "cost": f"R$ {detection['repair_cost']:,.2f}"
        })
    
    return timeline

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Sistema Híbrido Inteligente Carglass")
st.markdown("**YOLO11m + Análise Contextual = Precisão Máxima**")

model, is_damage_model = load_damage_model()

if model is None:
    st.error("❌ Erro ao carregar modelo")
    st.stop()

if is_damage_model:
    st.success("🎯 **Modelo YOLO11m Especializado Ativo** + Análise Inteligente")
    st.info("🧠 **Capacidades:** Detecção precisa + Interpretação contextual + Estimativas realistas")
else:
    st.warning("⚠️ **Modo Demonstração** - Usando modelo genérico com análise inteligente")

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
    
    st.header("🔍 Análise Híbrida Inteligente")
    
    with st.spinner("🧠 Processando com sistema híbrido..."):
        detections, annotated_img = process_intelligent_detection(image, model, is_damage_model)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Danos Detectados")
        st.image(annotated_img, use_column_width=True)
    
    if detections:
        st.header("📊 Análise Inteligente")
        
        total_cost = sum([d['repair_cost'] for d in detections])
        avg_confidence = sum([d['confidence'] for d in detections]) / len(detections)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Danos", len(detections))
        
        with col2:
            critical_count = len([d for d in detections if d['urgency'] in ['Crítica', 'Alta']])
            st.metric("Danos Críticos/Altos", critical_count, delta="⚠️" if critical_count > 0 else "✅")
        
        with col3:
            st.metric("Confiança Média", f"{avg_confidence:.1%}")
        
        with col4:
            st.metric("Custo Total", f"R$ {total_cost:,.2f}")
        
        st.subheader("📋 Análise Detalhada dos Danos")
        
        for detection in detections:
            with st.expander(f"{detection['damage_type']} - {detection['location']} (Urgência: {detection['urgency']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Tipo:** {detection['damage_type']}")
                    st.write(f"**Severidade:** {detection['severity']}")
                    st.write(f"**Localização:** {detection['location']}")
                    st.write(f"**Confiança:** {detection['confidence']:.1%}")
                
                with col2:
                    st.write(f"**Custo Estimado:** R$ {detection['repair_cost']:,.2f}")
                    st.write(f"**Urgência:** {detection['urgency']}")
                    st.write(f"**Área:** {detection['technical_details']['area_cm2']} cm²")
                
                st.write(f"**Análise:** {detection['description']}")
        
        st.header("📄 Relatório Completo JSON")
        report_json = create_comprehensive_report(vehicle_info, detections)
        st.json(report_json)
        
        json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Baixar Relatório Completo",
            data=json_str,
            file_name=f"relatorio_hibrido_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        st.header("💡 Recomendações Inteligentes")
        recommendations = report_json['recommendations']
        
        for rec in recommendations:
            if 'URGENTE' in rec['priority']:
                st.error(f"🚨 **{rec['priority']}:** {rec['action']}")
                st.write(f"**Danos:** {', '.join(rec['damages'])}")
                st.write(f"**Motivo:** {rec['reason']}")
                st.write(f"**Custo:** {rec['estimated_cost']}")
            else:
                st.warning(f"⚠️ **{rec['priority']}:** {rec['action']}")
                st.write(f"**Danos:** {', '.join(rec['damages'])}")
                st.write(f"**Motivo:** {rec['reason']}")
                st.write(f"**Custo:** {rec['estimated_cost']}")
    
    else:
        st.success("✅ Nenhum dano detectado!")
        report_json = create_comprehensive_report(vehicle_info, [])
        st.json(report_json)

else:
    st.info("👆 Aguardando o envio de uma imagem na barra lateral.")

st.markdown("---")
st.markdown("**Carglass - Sistema Híbrido Inteligente** | YOLO11m + IA Contextual")
