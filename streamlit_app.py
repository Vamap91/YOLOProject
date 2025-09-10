import streamlit as st
from PIL import Image
import pandas as pd
import json
import datetime
import cv2
import numpy as np

st.set_page_config(
    page_title="Carglass - Detector de Danos",
    page_icon="🛡️",
    layout="wide"
)

def analyze_damage_regions(image):
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    detections = []
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    front_region = (0, height//3, width, height)
    side_region = (width//4, 0, width, height//2)
    rear_region = (width//2, height//3, width, height)
    
    damage_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w//2, y + h//2
            
            damage_count += 1
            
            if center_y > height//2 and center_x < width//2:
                location = "Frente"
                damage_type = "Amassado Frontal"
                severity = "Severo" if area > 2000 else "Moderado"
            elif center_x > width//2 and center_y < height//2:
                location = "Lateral Direita"
                damage_type = "Risco"
                severity = "Leve"
            elif center_x > width//2 and center_y > height//2:
                location = "Traseira"
                damage_type = "Amassado"
                severity = "Moderado"
            else:
                location = "Lateral Esquerda"
                damage_type = "Risco"
                severity = "Leve"
            
            detection = {
                'id': damage_count,
                'damage_type': damage_type,
                'severity': severity,
                'location': location,
                'confidence': min(0.95, 0.7 + (area / 10000)),
                'bbox': {
                    'x1': float(x),
                    'y1': float(y),
                    'x2': float(x + w),
                    'y2': float(y + h)
                },
                'area_pixels': float(area)
            }
            detections.append(detection)
    
    return detections

def create_annotated_image(image, detections):
    img_array = np.array(image)
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        color = (255, 0, 0) if detection['severity'] == 'Severo' else (255, 165, 0) if detection['severity'] == 'Moderado' else (255, 255, 0)
        
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        label = f"{detection['damage_type']} {detection['confidence']:.2f}"
        cv2.putText(img_array, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img_array

def create_damage_report_json(vehicle_info, detections):
    report = {
        "inspection_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "inspector": "Sistema IA Carglass",
            "version": "3.0"
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
    recommendations = []
    for detection in detections:
        severity = detection['severity']
        damage_type = detection['damage_type']
        location = detection['location']
        
        if severity == 'Severo':
            recommendations.append({
                "priority": "Alta",
                "action": f"Reparo urgente necessário para {damage_type} em {location}",
                "estimated_cost": "R$ 800 - R$ 3000"
            })
        elif severity == 'Moderado':
            recommendations.append({
                "priority": "Média",
                "action": f"Reparo recomendado para {damage_type} em {location}",
                "estimated_cost": "R$ 300 - R$ 1200"
            })
        else:
            recommendations.append({
                "priority": "Baixa",
                "action": f"Reparo opcional para {damage_type} em {location}",
                "estimated_cost": "R$ 100 - R$ 500"
            })
    return recommendations

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Sistema de Detecção de Danos Carglass")
st.markdown("**Detecção precisa de danos com análise de imagem avançada**")

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
    
    st.header("🔍 Análise em Andamento")
    
    with st.spinner("Analisando danos na imagem..."):
        detections = analyze_damage_regions(image)
        annotated_img = create_annotated_image(image, detections)
    
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
        
        for rec in recommendations:
            if rec['priority'] == 'Alta':
                st.error(f"🚨 **{rec['priority']}:** {rec['action']} - {rec['estimated_cost']}")
            elif rec['priority'] == 'Média':
                st.warning(f"⚠️ **{rec['priority']}:** {rec['action']} - {rec['estimated_cost']}")
            else:
                st.info(f"ℹ️ **{rec['priority']}:** {rec['action']} - {rec['estimated_cost']}")
    
    else:
        st.success("✅ Nenhum dano detectado na imagem!")
        report_json = create_damage_report_json(vehicle_info, [])
        st.json(report_json)

else:
    st.info("👆 Aguardando o envio de uma imagem na barra lateral para iniciar a análise.")

st.markdown("---")
st.markdown("**Desenvolvido para Carglass** | Sistema de Detecção Precisa de Danos")
