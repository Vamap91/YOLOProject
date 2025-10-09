import streamlit as st
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
import requests
from pathlib import Path

try:
    import cv2
except ImportError:
    st.warning("OpenCV não está instalado, a anotação de imagens pode não funcionar como esperado.")
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
    'class_names': {
        'shattered_glass': 'Vidro Quebrado',
        'broken_lamp': 'Lâmpada Quebrada',
        'flat_tire': 'Pneu Vazio',
        'dent': 'Amassado',
        'scratch': 'Risco',
        'crack': 'Rachadura'
    },
    'cost_estimate': {
        'shattered_glass': (800, 2500),
        'broken_lamp': (300, 1500),
        'flat_tire': (200, 800),
        'dent': (500, 2000),
        'scratch': (150, 800),
        'crack': (200, 1000)
    }
}

def download_model_from_release():
    MODEL_VERSION = "v2.0.0"
    model_path = f"car_damage_best_{MODEL_VERSION}.pt"
    
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
            st.info("🔄 Removendo modelo antigo do cache...")
        except:
            pass
    
    if not os.path.exists(model_path):
        st.info("📄 Baixando modelo... (primeira execução, pode levar alguns minutos)")
        
        model_url = f"https://github.com/Vamap91/YOLOProject/releases/download/{MODEL_VERSION}/car_damage_best.pt"
        
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
            st.success("✅ Modelo baixado com sucesso!")
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Erro ao baixar o modelo: {e}")
            st.error("Verifique se a URL do modelo está correta no código.")
            return None
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")
            return None
    
    return model_path

@st.cache_resource
def load_model():
    model_path = download_model_from_release()
    
    if model_path is None:
        return None
        
    if not os.path.exists(model_path):
        st.error(f"Modelo '{model_path}' não encontrado após o download.")
        return None
        
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

def process_image(image, model):
    img_array = np.array(image)
    results = model(img_array)
    
    detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            detection = {
                'class': class_name,
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].cpu().numpy().tolist()
            }
            detections.append(detection)
    
    annotated_frame = results[0].plot()
    if cv2 is not None:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        for detection in detections:
            class_display = DAMAGE_CONFIG['class_names'].get(detection['class'], detection['class'])
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            label = f"{class_display} {detection['confidence']:.0%}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return detections, annotated_frame

def create_damage_analysis(detections):
    damage_report = []
    for i, detection in enumerate(detections):
        class_name = detection['class']
        severity = DAMAGE_CONFIG['severity_map'].get(class_name, 'Indefinido')
        location = DAMAGE_CONFIG['location_map'].get(class_name, 'N/A')
        cost_range = DAMAGE_CONFIG['cost_estimate'].get(class_name, (0, 0))
        estimated_cost = np.mean(cost_range)
        
        damage_report.append({
            'damage_id': f"DMG_{i+1:03d}",
            'class': class_name,
            'class_display': DAMAGE_CONFIG['class_names'].get(class_name, class_name.replace('_', ' ').title()),
            'confidence': detection['confidence'],
            'severity': severity,
            'location': location,
            'bbox': detection['bbox'],
            'estimated_cost': estimated_cost
        })
    return damage_report

def create_detection_summary(detections):
    if not detections:
        return "Nenhum dano detectado na imagem."
    
    damage_counts = {}
    for detection in detections:
        class_name = DAMAGE_CONFIG['class_names'].get(detection['class'], detection['class'])
        if class_name not in damage_counts:
            damage_counts[class_name] = []
        damage_counts[class_name].append(detection['confidence'])
    
    summary = [f"**Total de danos detectados: {len(detections)}**\n"]
    for damage_type, confidences in damage_counts.items():
        count = len(confidences)
        avg_confidence = np.mean(confidences)
        summary.append(f"• **{damage_type}**: {count} detectado(s) - Confiança média: {avg_confidence:.1%}")
    
    return "\n".join(summary)

def create_confidence_chart(damage_analysis):
    if not damage_analysis:
        return None
    
    df = pd.DataFrame(damage_analysis)
    fig = px.bar(
        df, 
        x='class_display', 
        y='confidence',
        title='Confiança das Detecções por Tipo de Dano',
        labels={'confidence': 'Confiança (%)', 'class_display': 'Tipo de Dano'},
        color='confidence',
        color_continuous_scale='RdYlGn',
        text='confidence'
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False, yaxis=dict(tickformat='.0%'))
    return fig

def create_damage_report_json(damage_analysis, vehicle_info):
    severity_count = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    damage_types = []
    total_cost = 0
    
    for damage in damage_analysis:
        severity = damage.get('severity', 'Indefinido')
        if severity in severity_count:
            severity_count[severity] += 1
        if damage['class_display'] not in damage_types:
            damage_types.append(damage['class_display'])
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
            "version": "2.0",
            "model": "YOLOv8 (car_damage_best.pt)",
        },
        "vehicle_info": vehicle_info,
        "damage_analysis": {
            "total_damages": len(damage_analysis),
            "severity_count": severity_count,
            "damage_types": sorted(list(set(d['class_display'] for d in damage_analysis))),
            "estimated_total_cost": f"R$ {total_cost:,.2f}",
            "repair_urgency": urgency,
        },
        "damages": damage_analysis
    }
    return report

def main():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🚗 Carglass - Detector de Danos Veiculares</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Análise de danos em tempo real com Inteligência Artificial</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Sobre o Sistema")
        st.markdown("""
        **Versão 2.0**
        
        Este sistema utiliza um modelo de IA (YOLOv8) treinado especificamente para **identificar e classificar danos em veículos** a partir de imagens.
        
        **Funcionalidades:**
        1. Detecção de 6 tipos de danos.
        2. Classificação de severidade.
        3. Estimativa de custo de reparo.
        4. Geração de relatório detalhado.
        """)
        
        st.header("⚙️ Configuração do Modelo")
        st.success("✅ URL do modelo configurada: Vamap91/YOLOProject v2.0.0")
        
        st.header("Informações do Veículo (Opcional)")
        vehicle_plate = st.text_input("Placa", placeholder="ABC-1234")
        vehicle_model = st.text_input("Modelo", placeholder="Ex: Toyota Corolla")
        vehicle_year = st.number_input("Ano", min_value=1990, max_value=datetime.now().year + 1, value=datetime.now().year)
        vehicle_color = st.text_input("Cor", placeholder="Ex: Branco")

    model = load_model()
    if model is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique a configuração.")
        return

    st.success("✅ Modelo carregado com sucesso!")

    st.header("1. Upload da Imagem do Veículo")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem para análise:",
        type=['png', 'jpg', 'jpeg'],
        help="Envie uma imagem de um veículo para que a IA possa detectar os danos."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📸 Imagem Original")
            st.image(image, caption=f"Imagem original: {uploaded_file.name}", use_column_width=True)
        
        with st.spinner("🔍 Analisando imagem em busca de danos..."):
            detections, annotated_img = process_image(image, model)
            damage_analysis = create_damage_analysis(detections)

        with col2:
            st.subheader("🎯 Danos Detectados")
            st.image(annotated_img, caption="Imagem com os danos destacados pela IA.", use_column_width=True)

        if not detections:
            st.success("✅ Nenhuma avaria detectada na imagem!")
            st.balloons()
        else:
            st.header("2. Resultados da Análise")
            total_cost = sum(d['estimated_cost'] for d in damage_analysis)
            urgency = create_damage_report_json(damage_analysis, {})['damage_analysis']['repair_urgency']

            c1, c2, c3 = st.columns(3)
            c1.metric("🔍 Total de Danos", len(damage_analysis))
            c2.metric("💰 Custo Total Estimado", f"R$ {total_cost:,.2f}")
            c3.metric("⚠️ Urgência de Reparo", urgency)

            st.markdown("### Resumo dos Danos")
            summary = create_detection_summary(detections)
            st.markdown(summary)

            st.markdown("### Detalhes dos Danos")
            df_display = pd.DataFrame(damage_analysis)[['class_display', 'severity', 'confidence', 'estimated_cost']]
            df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
            df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
            df_display.columns = ['Tipo de Dano', 'Severidade', 'Confiança', 'Custo Estimado']
            st.dataframe(df_display, use_container_width=True)

            st.markdown("### Gráfico de Confiança")
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

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Desenvolvido com ❤️ pela equipe de IA da Carglass</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
