import streamlit as st
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import plotly.express as px
import pandas as pd

from ultralytics import YOLO

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

DAMAGE_CONFIG = {
    'severity_map': {
        'shattered_glass': 'Severo',
        'broken_lamp': 'Severo', 
        'flat_tire': 'Severo',
        'dent': 'Moderado',
        'scratch': 'Leve',
        'crack': 'Leve'
    },
    'location_map': {
        'shattered_glass': 'Para-brisa/Vidros',
        'flat_tire': 'Rodas',
        'broken_lamp': 'Faróis/Lanternas',
        'dent': 'Carroceria',
        'scratch': 'Pintura',
        'crack': 'Para-choque/Plásticos'
    },
    'cost_ranges': {
        'Severo': (1500, 3500),
        'Moderado': (500, 1500),
        'Leve': (200, 600)
    },
    'class_names': {
        'shattered_glass': 'Vidro Quebrado',
        'broken_lamp': 'Lâmpada Quebrada',
        'flat_tire': 'Pneu Vazio',
        'dent': 'Amassado',
        'scratch': 'Risco',
        'crack': 'Rachadura'
    }
}

@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8m.pt')
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
        return "Nenhum objeto detectado na imagem."
    
    object_counts = {}
    for detection in detections:
        obj_type = detection['class']
        if obj_type not in object_counts:
            object_counts[obj_type] = []
        object_counts[obj_type].append(detection['confidence'])
    
    summary = []
    total_objects = len(detections)
    
    summary.append(f"**Total de objetos detectados: {total_objects}**\n")
    
    for obj_type, confidences in object_counts.items():
        count = len(confidences)
        avg_confidence = np.mean(confidences)
        summary.append(f"• **{obj_type.replace('_', ' ').title()}**: {count} detectado(s) - Confiança média: {avg_confidence:.1%}")
    
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
        title='Confiança das Detecções por Tipo de Objeto',
        labels={'confidence': 'Confiança (%)', 'class_clean': 'Tipo de Objeto'},
        color='confidence',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    fig.update_layout(yaxis=dict(tickformat='.1%'))
    
    return fig

def simulate_damage_analysis(detections):
    simulated_damages = []
    
    car_detected = any(d['class'].lower() in ['car', 'truck', 'bus'] for d in detections)
    
    if car_detected:
        max_confidence = max([d['confidence'] for d in detections if d['class'].lower() in ['car', 'truck', 'bus']])
        
        if max_confidence > 0.5:
            simulated_damages.append({
                'damage_id': "SIM_001",
                'class': 'dent',
                'class_display': 'Amassado (Análise Simulada)',
                'confidence': float(max_confidence * 0.8),
                'severity': 'Moderado',
                'location': 'Carroceria',
                'estimated_cost': int(np.random.randint(500, 1500)),
                'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
            })
        
        if max_confidence > 0.7:
            simulated_damages.append({
                'damage_id': "SIM_002",
                'class': 'scratch',
                'class_display': 'Possível Risco (Análise Simulada)',
                'confidence': float(max_confidence * 0.6),
                'severity': 'Leve',
                'location': 'Pintura',
                'estimated_cost': int(np.random.randint(200, 600)),
                'bbox': {'x1': 150, 'y1': 150, 'x2': 250, 'y2': 180}
            })
    
    return simulated_damages

def create_damage_report_json(detections, simulated_damages, vehicle_info=None):
    if vehicle_info is None:
        vehicle_info = {
            "plate": "Não informado",
            "model": "Não informado", 
            "year": "Não informado",
            "color": "Não informado"
        }
    
    severity_count = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    damage_types = []
    total_cost = 0
    
    for damage in simulated_damages:
        severity_count[damage['severity']] += 1
        if damage['class_display'] not in damage_types:
            damage_types.append(damage['class_display'])
        total_cost += damage['estimated_cost']
    
    urgency = 'Baixa'
    if severity_count['Severo'] > 0:
        urgency = 'Alta'
    elif severity_count['Moderado'] > 1:
        urgency = 'Média'
    
    report = {
        "inspection_info": {
            "timestamp": datetime.now().isoformat(),
            "inspector": "Sistema IA Carglass",
            "version": "5.0",
            "model": "YOLOv8m + Análise Simulada",
            "confidence_threshold": 0.25
        },
        "vehicle_info": vehicle_info,
        "objects_detected": detections,
        "damage_analysis": {
            "total_damages": len(simulated_damages),
            "severity_count": severity_count,
            "damage_types": damage_types,
            "estimated_total_cost": f"R$ {total_cost:,.2f}",
            "repair_urgency": urgency,
            "note": "Análise simulada baseada na detecção de veículos"
        },
        "simulated_damages": simulated_damages
    }
    
    return report

def main():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🚗 Carglass - Detector de Danos Veiculares</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Sistema IA para Detecção de Objetos + Análise Simulada de Danos</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Sobre o Sistema")
        st.markdown("""
        **Versão Atual**: Detecção de objetos com YOLOv8m
        
        **Como funciona:**
        1. Detecta objetos na imagem (carros, pessoas, etc.)
        2. Se detectar veículos, simula análise de danos
        3. Gera relatório baseado na análise
        
        **Nota**: Esta é uma versão de demonstração que simula a detecção de danos até o modelo especializado estar pronto.
        """)
        
        st.header("Informações do Veículo")
        vehicle_plate = st.text_input("Placa", placeholder="ABC-1234")
        vehicle_model = st.text_input("Modelo", placeholder="Ex: Toyota Corolla")
        vehicle_year = st.number_input("Ano", min_value=1990, max_value=2025, value=2020)
        vehicle_color = st.text_input("Cor", placeholder="Ex: Branco")
    
    model = load_model()
    if model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique a instalação do Ultralytics.")
        return
    
    st.success("✅ Modelo YOLOv8m carregado com sucesso!")
    
    st.header("Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem do veículo:",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos aceitos: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Imagem Original")
            st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        with st.spinner("🔍 Analisando imagem..."):
            detections, annotated_img = process_image(image, model)
        
        with col2:
            st.subheader("🎯 Objetos Detectados")
            st.image(annotated_img, caption="Detecções encontradas", use_column_width=True)
        
        st.header("📊 Resumo da Detecção")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🔍 Objetos Encontrados")
            summary = create_detection_summary(detections)
            st.markdown(summary)
        
        with col2:
            st.markdown("### 📈 Gráfico de Confiança")
            if detections:
                chart = create_confidence_chart(detections)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Nenhum objeto detectado para exibir no gráfico.")
        
        simulated_damages = simulate_damage_analysis(detections)
        
        if simulated_damages:
            st.header("🔧 Análise Simulada de Danos")
            st.info("⚠️ **Nota**: Esta é uma análise simulada baseada na presença de veículos detectados. Para análise real de danos, aguarde o modelo especializado.")
            
            total_cost = sum([d['estimated_cost'] for d in simulated_damages])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔍 Danos Simulados", len(simulated_damages))
            with col2:
                st.metric("💰 Custo Estimado", f"R$ {total_cost:,.2f}")
            with col3:
                if simulated_damages:
                    avg_conf = np.mean([d['confidence'] for d in simulated_damages])
                    st.metric("📈 Confiança Média", f"{avg_conf:.1%}")
                else:
                    st.metric("📈 Confiança Média", "N/A")
            
            df_damages = pd.DataFrame(simulated_damages)
            df_display = df_damages[['damage_id', 'class_display', 'severity', 'estimated_cost']].copy()
            df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['confidence'] = df_damages['confidence'].apply(lambda x: f"{x:.1%}")
            df_display.columns = ['ID', 'Tipo de Dano', 'Severidade', 'Custo Est.', 'Confiança']
            
            st.dataframe(df_display, use_container_width=True)
        
        if detections:
            st.header("📄 Relatório Completo")
            
            vehicle_info = {
                "plate": vehicle_plate or "Não informado",
                "model": vehicle_model or "Não informado",
                "year": str(vehicle_year),
                "color": vehicle_color or "Não informado"
            }
            
            report = create_damage_report_json(detections, simulated_damages, vehicle_info)
            
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 Baixar Relatório JSON",
                data=report_json,
                file_name=f"relatorio_carglass_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            with st.expander("📋 Resumo do Relatório"):
                st.write(f"**Objetos detectados**: {len(detections)}")
                st.write(f"**Danos simulados**: {len(simulated_damages)}")
                if simulated_damages:
                    st.write(f"**Custo total estimado**: R$ {sum([d['estimated_cost'] for d in simulated_damages]):,.2f}")
                st.write(f"**Timestamp**: {report['inspection_info']['timestamp']}")
        
        else:
            st.warning("⚠️ Nenhum objeto foi detectado na imagem. Tente com uma imagem diferente.")
    
    else:
        st.info("👆 **Faça upload de uma imagem** para começar a análise")
        
        st.markdown("### 💡 Como funciona:")
        st.markdown("""
        1. **Upload**: Envie uma foto do veículo
        2. **Detecção**: Sistema identifica objetos (carros, pessoas, etc.)  
        3. **Análise**: Se detectar veículos, simula análise de danos
        4. **Relatório**: Gera relatório com estimativas de custo
        
        **Aguardando**: Modelo especializado em danos veiculares em treinamento
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Carglass - Sistema de Detecção de Objetos + Análise Simulada</strong></p>
        <p>Powered by YOLOv8m + Streamlit | Versão 5.0 Demo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
