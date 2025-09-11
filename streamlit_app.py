import streamlit as st
import numpy as np
from PIL import Image
import os
import json
import requests
import torch
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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

original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

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
def load_damage_model():
    try:
        if os.path.exists('trained.pt'):
            model = YOLO('trained.pt')
            st.success("✅ Modelo personalizado carregado com sucesso!")
            return model, "Personalizado"
        else:
            url = "https://drive.google.com/uc?export=download&id=1ey-QZYRu-SgbT_PF1nXb0ag9V8tOAVU5"
            response = requests.get(url, allow_redirects=True, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000000:
                with open('trained.pt', 'wb') as f:
                    f.write(response.content)
                model = YOLO('trained.pt')
                st.success("✅ Modelo personalizado baixado e carregado!")
                return model, "Personalizado"
            else:
                raise Exception("Falha no download")
                
    except Exception as e:
        st.warning(f"⚠️ Usando modelo base YOLOv8m: {str(e)}")
        try:
            model = YOLO('yolov8m.pt')
            return model, "Base YOLOv8m"
        except Exception as e2:
            st.error(f"❌ Erro ao carregar qualquer modelo: {str(e2)}")
            return None, None

def process_damage_detection(image, model, confidence_threshold=0.25):
    img_array = np.array(image)
    results = model(img_array, conf=confidence_threshold)
    
    detections = []
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            class_name = results[0].names[int(boxes.cls[i])]
            confidence = float(boxes.conf[i])
            bbox = boxes.xyxy[i].cpu().numpy()
            
            severity = DAMAGE_CONFIG['severity_map'].get(class_name, 'Leve')
            location = DAMAGE_CONFIG['location_map'].get(class_name, 'Desconhecida')
            cost_range = DAMAGE_CONFIG['cost_ranges'][severity]
            estimated_cost = np.random.randint(cost_range[0], cost_range[1])
            
            detection = {
                'damage_id': f"DMG_{i+1:03d}",
                'class': class_name,
                'class_display': DAMAGE_CONFIG['class_names'].get(class_name, class_name),
                'confidence': confidence,
                'severity': severity,
                'location': location,
                'estimated_cost': estimated_cost,
                'bbox': {
                    'x1': int(bbox[0]), 'y1': int(bbox[1]),
                    'x2': int(bbox[2]), 'y2': int(bbox[3])
                }
            }
            detections.append(detection)
    
    try:
        annotated_img = results[0].plot()
        if cv2 is not None:
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    except:
        annotated_img = img_array
    
    return detections, annotated_img

def create_damage_report_json(detections, vehicle_info=None):
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
    
    for detection in detections:
        severity_count[detection['severity']] += 1
        if detection['class_display'] not in damage_types:
            damage_types.append(detection['class_display'])
        total_cost += detection['estimated_cost']
    
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
            "model": "YOLOv8m Personalizado",
            "confidence_threshold": 0.25
        },
        "vehicle_info": vehicle_info,
        "damage_summary": {
            "total_damages": len(detections),
            "severity_count": severity_count,
            "damage_types": damage_types,
            "estimated_total_cost": f"R$ {total_cost:,.2f}",
            "repair_urgency": urgency
        },
        "detections": detections,
        "recommendations": generate_recommendations(detections)
    }
    
    return report

def generate_recommendations(detections):
    recommendations = []
    
    severe_damages = [d for d in detections if d['severity'] == 'Severo']
    moderate_damages = [d for d in detections if d['severity'] == 'Moderado']
    
    if severe_damages:
        recommendations.append({
            "priority": "URGENTE",
            "message": f"Foram detectados {len(severe_damages)} dano(s) severo(s). Recomenda-se reparo imediato.",
            "damages": [d['class_display'] for d in severe_damages]
        })
    
    if moderate_damages:
        recommendations.append({
            "priority": "IMPORTANTE", 
            "message": f"Foram detectados {len(moderate_damages)} dano(s) moderado(s). Agende reparo em breve.",
            "damages": [d['class_display'] for d in moderate_damages]
        })
    
    glass_damages = [d for d in detections if 'glass' in d['class']]
    if glass_damages:
        recommendations.append({
            "priority": "SEGURANÇA",
            "message": "Vidros danificados comprometem a segurança. Procure assistência Carglass imediatamente.",
            "damages": [d['class_display'] for d in glass_damages]
        })
    
    if len(detections) == 0:
        recommendations.append({
            "priority": "OK",
            "message": "Nenhum dano significativo detectado. Veículo em boas condições visuais.",
            "damages": []
        })
    
    return recommendations

def create_severity_chart(detections):
    if not detections:
        return None
    
    severity_counts = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    for detection in detections:
        severity_counts[detection['severity']] += 1
    
    df = pd.DataFrame(list(severity_counts.items()), columns=['Severidade', 'Quantidade'])
    df = df[df['Quantidade'] > 0]
    
    colors = {'Leve': '#28a745', 'Moderado': '#ffc107', 'Severo': '#dc3545'}
    
    fig = px.pie(
        df, 
        values='Quantidade', 
        names='Severidade',
        title='Distribuição por Severidade',
        color='Severidade',
        color_discrete_map=colors
    )
    
    fig.update_layout(height=300)
    return fig

def create_confidence_chart(detections):
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    df['confidence_pct'] = df['confidence'] * 100
    
    fig = px.bar(
        df, 
        x='class_display', 
        y='confidence_pct',
        color='severity',
        title='Confiança das Detecções',
        labels={'confidence_pct': 'Confiança (%)', 'class_display': 'Tipo de Dano'},
        color_discrete_map={'Leve': '#28a745', 'Moderado': '#ffc107', 'Severo': '#dc3545'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        xaxis_title="Tipo de Dano",
        yaxis_title="Confiança (%)"
    )
    
    return fig

def main():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🚗 Carglass - Detector de Danos Veiculares</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Sistema IA para Detecção Automática de Danos</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🔧 Configurações")
        
        confidence_threshold = st.slider(
            "Limite de Confiança", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.25, 
            step=0.05,
            help="Ajuste a sensibilidade da detecção"
        )
        
        st.markdown("### 📊 Tipos de Danos Detectados")
        st.markdown("""
        - **Severos**: Vidros quebrados, Lâmpadas quebradas, Pneus vazios
        - **Moderados**: Amassados
        - **Leves**: Riscos, Rachaduras
        """)
        
        st.markdown("### 💰 Estimativas de Custo")
        st.markdown("""
        - **Severo**: R$ 1.500 - R$ 3.500
        - **Moderado**: R$ 500 - R$ 1.500  
        - **Leve**: R$ 200 - R$ 600
        """)
        
        st.markdown("### ℹ️ Informações do Veículo")
        vehicle_plate = st.text_input("Placa", placeholder="ABC-1234")
        vehicle_model = st.text_input("Modelo", placeholder="Ex: Toyota Corolla")
        vehicle_year = st.number_input("Ano", min_value=1990, max_value=2025, value=2020)
        vehicle_color = st.text_input("Cor", placeholder="Ex: Branco")
    
    model, model_type = load_damage_model()
    if model is None:
        st.error("❌ Não foi possível carregar nenhum modelo. Verifique a conexão.")
        return
    
    st.info(f"🤖 Modelo carregado: {model_type}")
    
    tab1, tab2 = st.tabs(["📷 Análise de Imagem", "📄 Relatório Completo"])
    
    with tab1:
        st.markdown("### Upload da Imagem")
        uploaded_file = st.file_uploader(
            "Escolha uma imagem do veículo:",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos aceitos: PNG, JPG, JPEG"
        )
        
        col1, col2, col3 = st.columns(3)
        example_selected = None
        
        with col1:
            if st.button("🚗 Exemplo: Amassado", use_container_width=True):
                example_selected = "examples/1.png"
        with col2:
            if st.button("🔍 Exemplo: Múltiplos Danos", use_container_width=True):
                example_selected = "examples/2.png"
        with col3:
            if st.button("💥 Exemplo: Vidro Quebrado", use_container_width=True):
                example_selected = "examples/3.png"
        
        image_source = None
        image_name = "Imagem"
        
        if example_selected and os.path.exists(example_selected):
            image_source = Image.open(example_selected)
            image_name = f"Exemplo: {example_selected.split('/')[-1]}"
        elif uploaded_file is not None:
            image_source = Image.open(uploaded_file)
            image_name = uploaded_file.name
        
        if image_source is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📸 Imagem Original")
                st.image(image_source, caption=image_name, use_column_width=True)
            
            with st.spinner("🔍 Analisando imagem..."):
                detections, annotated_img = process_damage_detection(
                    image_source, model, confidence_threshold
                )
            
            with col2:
                st.markdown("#### 🎯 Detecções Encontradas")
                st.image(annotated_img, caption="Danos detectados", use_column_width=True)
            
            if detections:
                st.markdown("### 📊 Análise dos Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    severity_chart = create_severity_chart(detections)
                    if severity_chart:
                        st.plotly_chart(severity_chart, use_container_width=True)
                
                with col2:
                    confidence_chart = create_confidence_chart(detections)
                    if confidence_chart:
                        st.plotly_chart(confidence_chart, use_container_width=True)
                
                st.markdown("### 📋 Resumo Executivo")
                
                total_cost = sum([d['estimated_cost'] for d in detections])
                severity_counts = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
                for d in detections:
                    severity_counts[d['severity']] += 1
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Danos", len(detections))
                with col2:
                    st.metric("Custo Estimado", f"R$ {total_cost:,.2f}")
                with col3:
                    st.metric("Danos Severos", severity_counts['Severo'])
                with col4:
                    confidence_avg = np.mean([d['confidence'] for d in detections])
                    st.metric("Confiança Média", f"{confidence_avg:.1%}")
                
                st.markdown("### 🚨 Recomendações")
                vehicle_info = {
                    "plate": vehicle_plate or "Não informado",
                    "model": vehicle_model or "Não informado",
                    "year": str(vehicle_year),
                    "color": vehicle_color or "Não informado"
                }
                
                report = create_damage_report_json(detections, vehicle_info)
                
                for rec in report['recommendations']:
                    if rec['priority'] == 'URGENTE':
                        st.error(f"🚨 **{rec['priority']}**: {rec['message']}")
                    elif rec['priority'] == 'IMPORTANTE':
                        st.warning(f"⚠️ **{rec['priority']}**: {rec['message']}")
                    elif rec['priority'] == 'SEGURANÇA':
                        st.error(f"🛡️ **{rec['priority']}**: {rec['message']}")
                    else:
                        st.success(f"✅ **{rec['priority']}**: {rec['message']}")
                
                st.session_state['report'] = report
                
            else:
                st.success("✅ Nenhum dano detectado na imagem!")
                st.balloons()
    
    with tab2:
        if 'report' in st.session_state:
            report = st.session_state['report']
            
            st.markdown("### 📄 Relatório Completo de Inspeção")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🚗 Informações do Veículo")
                st.write(f"**Placa:** {report['vehicle_info']['plate']}")
                st.write(f"**Modelo:** {report['vehicle_info']['model']}")
                st.write(f"**Ano:** {report['vehicle_info']['year']}")
                st.write(f"**Cor:** {report['vehicle_info']['color']}")
            
            with col2:
                st.markdown("#### 🔍 Informações da Inspeção")
                st.write(f"**Data:** {datetime.fromisoformat(report['inspection_info']['timestamp']).strftime('%d/%m/%Y %H:%M')}")
                st.write(f"**Versão:** {report['inspection_info']['version']}")
                st.write(f"**Modelo IA:** {report['inspection_info']['model']}")
            
            if report['detections']:
                st.markdown("#### 📊 Detalhes dos Danos")
                
                df_detections = pd.DataFrame(report['detections'])
                df_display = df_detections[['damage_id', 'class_display', 'severity', 'location', 'estimated_cost']].copy()
                df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
                df_display.columns = ['ID', 'Tipo', 'Severidade', 'Localização', 'Custo Est.']
                
                st.dataframe(df_display, use_container_width=True)
            
            st.markdown("#### 📥 Download do Relatório")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_json = json.dumps(report, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Baixar Relatório JSON",
                    data=report_json,
                    file_name=f"relatorio_danos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                report_text = f"""
RELATÓRIO DE INSPEÇÃO CARGLASS
{'='*40}

VEÍCULO:
Placa: {report['vehicle_info']['plate']}
Modelo: {report['vehicle_info']['model']} 
Ano: {report['vehicle_info']['year']}
Cor: {report['vehicle_info']['color']}

RESUMO:
Total de Danos: {report['damage_summary']['total_damages']}
Custo Total Estimado: {report['damage_summary']['estimated_total_cost']}
Urgência: {report['damage_summary']['repair_urgency']}

DANOS DETECTADOS:
"""
                for detection in report['detections']:
                    report_text += f"- {detection['class_display']} ({detection['severity']}) - R$ {detection['estimated_cost']:,.2f}\n"
                
                st.download_button(
                    label="📝 Baixar Relatório TXT",
                    data=report_text,
                    file_name=f"relatorio_resumo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("📷 Faça a análise de uma imagem na aba anterior para gerar o relatório completo.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Carglass - Sistema de Detecção Automática de Danos</strong></p>
        <p>Powered by YOLOv8 + Streamlit | Versão 5.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
