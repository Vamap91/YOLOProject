# streamlit_app.py - Versão Corrigida e Robusta

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
    page_title="Carglass - Detector de Danos",
    page_icon="🛡️",
    layout="wide"
)

# --- Funções Principais ---

@st.cache_resource
def load_damage_model():
    """
    Carrega o modelo YOLO. Primeiro tenta o modelo genérico que sempre funciona.
    """
    try:
        model = YOLO('yolov8n.pt')
        return model, False  # False = modelo genérico
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, False

def safe_process_image(image, model):
    """
    Processa a imagem de forma segura, tratando todos os casos possíveis.
    """
    try:
        results = model(image)
        detections = []
        
        # Verifica se há resultados e se há boxes
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Verifica se há detecções
            if len(boxes) > 0:
                # Mapeamento simulado para demonstração
                damage_mapping = {
                    'person': {'type': 'Amassado', 'severity': 'Moderado', 'location': 'Lateral'},
                    'car': {'type': 'Risco', 'severity': 'Leve', 'location': 'Porta'},
                    'bicycle': {'type': 'Vidro Quebrado', 'severity': 'Severo', 'location': 'Para-brisa'},
                    'motorcycle': {'type': 'Pneu Vazio', 'severity': 'Severo', 'location': 'Roda'},
                    'truck': {'type': 'Amassado', 'severity': 'Severo', 'location': 'Para-choque'},
                    'bus': {'type': 'Risco', 'severity': 'Moderado', 'location': 'Capô'}
                }

                for i, box in enumerate(boxes):
                    try:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Só processa se a confiança for alta o suficiente
                        if confidence > 0.3:
                            if class_name in damage_mapping:
                                damage_info = damage_mapping[class_name]
                                
                                detection = {
                                    'id': i + 1,
                                    'damage_type': damage_info['type'],
                                    'severity': damage_info['severity'],
                                    'location': damage_info['location'],
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
                    except Exception as box_error:
                        st.warning(f"Erro ao processar detecção {i}: {box_error}")
                        continue
        
        # Gera imagem anotada de forma segura
        try:
            annotated_img = results[0].plot()
            annotated_img_rgb = annotated_img[..., ::-1]
        except:
            # Se falhar, retorna a imagem original
            import numpy as np
            annotated_img_rgb = np.array(image)
        
        return detections, annotated_img_rgb
        
    except Exception as e:
        st.error(f"Erro no processamento da imagem: {e}")
        import numpy as np
        return [], np.array(image)

def create_damage_report_json(vehicle_info, detections):
    """
    Cria um relatório completo em formato JSON.
    """
    try:
        report = {
            "inspection_info": {
                "timestamp": datetime.datetime.now().isoformat(),
                "inspector": "Sistema IA Carglass",
                "version": "2.1"
            },
            "vehicle_info": vehicle_info,
            "damage_summary": {
                "total_damages": len(detections),
                "severity_count": {
                    "Leve": len([d for d in detections if d.get('severity') == 'Leve']),
                    "Moderado": len([d for d in detections if d.get('severity') == 'Moderado']),
                    "Severo": len([d for d in detections if d.get('severity') == 'Severo'])
                },
                "damage_types": list(set([d.get('damage_type', 'Desconhecido') for d in detections]))
            },
            "detections": detections,
            "recommendations": generate_recommendations(detections)
        }
        return report
    except Exception as e:
        st.error(f"Erro ao criar relatório JSON: {e}")
        return {"error": str(e)}

def generate_recommendations(detections):
    """
    Gera recomendações baseadas nos danos detectados.
    """
    recommendations = []
    
    try:
        for detection in detections:
            severity = detection.get('severity', 'Desconhecido')
            damage_type = detection.get('damage_type', 'Dano')
            location = detection.get('location', 'Local não especificado')
            
            if severity == 'Severo':
                recommendations.append({
                    "priority": "Alta",
                    "action": f"Reparo urgente necessário para {damage_type} em {location}",
                    "estimated_cost": "R$ 500 - R$ 2000"
                })
            elif severity == 'Moderado':
                recommendations.append({
                    "priority": "Média",
                    "action": f"Reparo recomendado para {damage_type} em {location}",
                    "estimated_cost": "R$ 200 - R$ 800"
                })
            else:
                recommendations.append({
                    "priority": "Baixa",
                    "action": f"Reparo opcional para {damage_type} em {location}",
                    "estimated_cost": "R$ 50 - R$ 300"
                })
    except Exception as e:
        recommendations.append({
            "priority": "Erro",
            "action": f"Erro ao gerar recomendação: {e}",
            "estimated_cost": "A definir"
        })
    
    return recommendations

# --- Interface Principal ---

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Sistema de Detecção de Danos Carglass")
st.markdown("**Solução de IA para identificação de danos veiculares com saída em JSON**")

# Carrega o modelo
model, is_damage_model = load_damage_model()

if model is None:
    st.error("❌ Não foi possível carregar o modelo.")
    st.stop()

st.warning("⚠️ **Modo de Demonstração:** Este protótipo simula a detecção de danos mapeando objetos detectados para tipos de avarias. Para uso em produção, um modelo específico de danos deve ser treinado.")

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

# Processamento principal
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        
        st.header("🔍 Análise em Andamento")
        
        with st.spinner("Analisando imagem com IA..."):
            detections, annotated_img = safe_process_image(image, model)
        
        # Exibição dos resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagem Original")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Análise de Danos")
            st.image(annotated_img, use_column_width=True)
        
        # Resultados detalhados
        if detections:
            st.header("📊 Resultados da Análise")
            
            # Resumo em cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Danos", len(detections))
            
            with col2:
                severe_count = len([d for d in detections if d.get('severity') == 'Severo'])
                st.metric("Danos Severos", severe_count)
            
            with col3:
                avg_confidence = sum([d.get('confidence', 0) for d in detections]) / len(detections)
                st.metric("Confiança Média", f"{avg_confidence:.1%}")
            
            with col4:
                damage_types = len(set([d.get('damage_type', 'Desconhecido') for d in detections]))
                st.metric("Tipos de Danos", damage_types)
            
            # Tabela de detecções
            st.subheader("📋 Detalhes dos Danos")
            try:
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
            except Exception as e:
                st.error(f"Erro ao exibir tabela: {e}")
            
            # Gerar relatório JSON
            st.header("📄 Relatório JSON")
            
            report_json = create_damage_report_json(vehicle_info, detections)
            
            # Exibir JSON formatado
            st.json(report_json)
            
            # Download do JSON
            try:
                json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Baixar Relatório JSON",
                    data=json_str,
                    file_name=f"relatorio_danos_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Erro ao gerar download: {e}")
            
            # Recomendações
            st.header("💡 Recomendações")
            recommendations = report_json.get('recommendations', [])
            
            for rec in recommendations:
                priority = rec.get('priority', 'Desconhecida')
                if priority == 'Alta':
                    st.error(f"🚨 **{priority}:** {rec.get('action', 'Ação não especificada')} - {rec.get('estimated_cost', 'Custo a definir')}")
                elif priority == 'Média':
                    st.warning(f"⚠️ **{priority}:** {rec.get('action', 'Ação não especificada')} - {rec.get('estimated_cost', 'Custo a definir')}")
                else:
                    st.info(f"ℹ️ **{priority}:** {rec.get('action', 'Ação não especificada')} - {rec.get('estimated_cost', 'Custo a definir')}")
        
        else:
            st.success("✅ Nenhum dano detectado na imagem!")
            
            # Ainda gera um JSON mesmo sem danos
            report_json = create_damage_report_json(vehicle_info, [])
            st.json(report_json)
            
            try:
                json_str = json.dumps(report_json, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Baixar Relatório JSON (Sem Danos)",
                    data=json_str,
                    file_name=f"relatorio_sem_danos_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Erro ao gerar download: {e}")
    
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")

else:
    st.info("👆 Aguardando o envio de uma imagem na barra lateral para iniciar a análise.")
    
    # Exemplo de JSON de saída
    st.header("📋 Exemplo de Saída JSON")
    st.markdown("**Este é o formato do JSON que será gerado após a análise:**")
    
    example_json = {
        "inspection_info": {
            "timestamp": "2025-09-10T14:30:00",
            "inspector": "Sistema IA Carglass",
            "version": "2.1"
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
st.markdown("**Desenvolvido para Carglass** | Sistema de Detecção de Danos com IA")
