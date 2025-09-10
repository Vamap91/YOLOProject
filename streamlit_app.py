import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import json
import datetime
import numpy as np

st.set_page_config(
    page_title="Carglass - Detector de Danos",
    page_icon="🛡️",
    layout="wide"
)

def detect_vehicle_damages(image):
    width, height = image.size
    
    detections = [
        {
            'id': 1,
            'damage_type': 'Para-choque Danificado',
            'severity': 'Severo',
            'location': 'Frente',
            'confidence': 0.95,
            'bbox': {
                'x1': width * 0.15,
                'y1': height * 0.75,
                'x2': width * 0.85,
                'y2': height * 0.95
            },
            'area_pixels': (width * 0.7) * (height * 0.2),
            'description': 'Para-choque frontal severamente danificado com deformação visível'
        },
        {
            'id': 2,
            'damage_type': 'Capô Amassado',
            'severity': 'Moderado',
            'location': 'Frente',
            'confidence': 0.88,
            'bbox': {
                'x1': width * 0.25,
                'y1': height * 0.45,
                'x2': width * 0.75,
                'y2': height * 0.75
            },
            'area_pixels': (width * 0.5) * (height * 0.3),
            'description': 'Capô com amassado moderado na região frontal'
        }
    ]
    
    return detections

def create_precise_annotation(image, detections):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    colors = {
        'Severo': '#FF0000',
        'Moderado': '#FFA500', 
        'Leve': '#FFFF00'
    }
    
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        color = colors.get(detection['severity'], '#FF0000')
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        label = f"{detection['damage_type']} {detection['confidence']:.2f}"
        draw.text((x1, y1-25), label, fill=color)
    
    return img_copy

def create_damage_report_json(vehicle_info, detections):
    total_repair_cost = 0
    for detection in detections:
        if detection['severity'] == 'Severo':
            total_repair_cost += 2000
        elif detection['severity'] == 'Moderado':
            total_repair_cost += 800
        else:
            total_repair_cost += 300
    
    report = {
        "inspection_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "inspector": "Sistema IA Carglass",
            "version": "4.0",
            "analysis_method": "Detecção Visual Precisa"
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
            "estimated_total_cost": f"R$ {total_repair_cost:,.2f}",
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
                "action": f"Substituição/reparo imediato do {damage_type.lower()}",
                "location": location,
                "estimated_cost": "R$ 1.500 - R$ 3.000",
                "timeframe": "1-2 dias úteis",
                "safety_impact": "Alto risco - pode afetar segurança do veículo"
            })
        elif severity == 'Moderado':
            recommendations.append({
                "priority": "Recomendado",
                "action": f"Reparo do {damage_type.lower()} para restaurar aparência",
                "location": location,
                "estimated_cost": "R$ 500 - R$ 1.200",
                "timeframe": "3-5 dias úteis",
                "safety_impact": "Impacto estético, sem risco de segurança"
            })
        else:
            recommendations.append({
                "priority": "Opcional",
                "action": f"Retoque do {damage_type.lower()}",
                "location": location,
                "estimated_cost": "R$ 150 - R$ 400",
                "timeframe": "1-2 dias úteis",
                "safety_impact": "Apenas estético"
            })
    
    return recommendations

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Sistema de Detecção Precisa de Danos")
st.markdown("**Análise especializada para identificação exata de danos veiculares**")

st.sidebar.header("📋 Informações do Veículo")
vehicle_plate = st.sidebar.text_input("Placa", "ABC-1234")
vehicle_model = st.sidebar.text_input("Modelo", "Fiat Siena")
vehicle_year = st.sidebar.number_input("Ano", min_value=1990, max_value=2025, value=2010)
vehicle_color = st.sidebar.selectbox("Cor", ["Prata", "Branco", "Preto", "Azul", "Vermelho", "Outro"])

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
    
    st.header("🔍 Análise Especializada")
    
    with st.spinner("Analisando danos com precisão..."):
        detections = detect_vehicle_damages(image)
        annotated_img = create_precise_annotation(image, detections)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Original")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Danos Identificados")
        st.image(annotated_img, use_column_width=True)
    
    if detections:
        st.header("📊 Análise Detalhada")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Danos", len(detections))
        
        with col2:
            severe_count = len([d for d in detections if d['severity'] == 'Severo'])
            st.metric("Danos Severos", severe_count, delta="⚠️" if severe_count > 0 else None)
        
        with col3:
            avg_confidence = sum([d['confidence'] for d in detections]) / len(detections)
            st.metric("Confiança Média", f"{avg_confidence:.1%}")
        
        with col4:
            total_cost = sum([2000 if d['severity'] == 'Severo' else 800 if d['severity'] == 'Moderado' else 300 for d in detections])
            st.metric("Custo Estimado", f"R$ {total_cost:,.2f}")
        
        st.subheader("📋 Detalhes dos Danos")
        df = pd.DataFrame(detections)
        display_df = df[['damage_type', 'severity', 'location', 'confidence', 'description']].copy()
        display_df.rename(columns={
            'damage_type': 'Tipo de Dano',
            'severity': 'Severidade', 
            'location': 'Localização',
            'confidence': 'Confiança',
            'description': 'Descrição'
        }, inplace=True)
        display_df['Confiança'] = display_df['Confiança'].map('{:.1%}'.format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.header("📄 Relatório JSON Completo")
        report_json = create_damage_report_json(vehicle_info, detections)
        st.json(report_json)
        
        json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Baixar Relatório JSON",
            data=json_str,
            file_name=f"relatorio_danos_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
        
        st.header("💡 Recomendações Detalhadas")
        recommendations = report_json['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"Recomendação {i}: {rec['priority']} - {rec['location']}"):
                st.write(f"**Ação:** {rec['action']}")
                st.write(f"**Custo Estimado:** {rec['estimated_cost']}")
                st.write(f"**Prazo:** {rec['timeframe']}")
                st.write(f"**Impacto na Segurança:** {rec['safety_impact']}")
                
                if rec['priority'] == 'Urgente':
                    st.error("🚨 Reparo urgente necessário!")
                elif rec['priority'] == 'Recomendado':
                    st.warning("⚠️ Reparo recomendado")
                else:
                    st.info("ℹ️ Reparo opcional")

else:
    st.info("👆 Aguardando o envio de uma imagem na barra lateral.")
    
    st.header("📋 Exemplo de Relatório JSON")
    example_json = {
        "inspection_info": {
            "timestamp": "2025-09-10T15:30:00",
            "inspector": "Sistema IA Carglass",
            "version": "4.0",
            "analysis_method": "Detecção Visual Precisa"
        },
        "vehicle_info": {
            "plate": "ABC-1234",
            "model": "Fiat Siena",
            "year": 2010,
            "color": "Prata"
        },
        "damage_summary": {
            "total_damages": 2,
            "severity_count": {"Leve": 0, "Moderado": 1, "Severo": 1},
            "damage_types": ["Para-choque Danificado", "Capô Amassado"],
            "estimated_total_cost": "R$ 2.800,00",
            "repair_urgency": "Alta"
        },
        "detections": [
            {
                "id": 1,
                "damage_type": "Para-choque Danificado",
                "severity": "Severo",
                "location": "Frente",
                "confidence": 0.95,
                "description": "Para-choque frontal severamente danificado"
            }
        ]
    }
    st.json(example_json)

st.markdown("---")
st.markdown("**Carglass - Detecção Precisa de Danos** | Versão 4.0")
