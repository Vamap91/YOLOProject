import streamlit as st
import numpy as np
from PIL import Image
import os
import json
import requests
from datetime import datetime
import plotly.express as px
import pandas as pd

# IMPORTAÇÃO LIMPA - SEM PATCH PROBLEMÁTICO
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
    layout="wide"
)

# Configuração do GitHub Release
GITHUB_CONFIG = {
    'user': 'Vamap91',
    'repo': 'yoloproject',
    'release_tag': 'v1.0.0',
    'model_filename': 'yolov8m.pt'
}

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

def download_from_github():
    """Download direto do GitHub sem complicações"""
    url = f"https://github.com/{GITHUB_CONFIG['user']}/{GITHUB_CONFIG['repo']}/releases/download/{GITHUB_CONFIG['release_tag']}/{GITHUB_CONFIG['model_filename']}"
    
    try:
        st.info(f"📥 Baixando de: {url}")
        
        response = requests.get(url, stream=True)
        
        if response.status_code == 404:
            st.error("❌ Arquivo não encontrado no GitHub!")
            return False
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(GITHUB_CONFIG['model_filename'], 'wb') as f:
            downloaded = 0
            progress_bar = st.progress(0)
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(downloaded / total_size)
            
            progress_bar.empty()
        
        if os.path.exists(GITHUB_CONFIG['model_filename']):
            size_mb = os.path.getsize(GITHUB_CONFIG['model_filename']) / (1024 * 1024)
            st.success(f"✅ Download concluído! ({size_mb:.1f} MB)")
            return True
        
        return False
        
    except Exception as e:
        st.error(f"❌ Erro no download: {str(e)}")
        return False

def load_model_simple():
    """Carregamento super simples sem cache problemático"""
    
    if not YOLO_AVAILABLE:
        st.error("❌ Ultralytics não instalado!")
        st.code("pip install ultralytics")
        return None, "Erro", False
    
    model_path = GITHUB_CONFIG['model_filename']
    
    # Verifica se existe local
    if os.path.exists(model_path):
        st.info(f"📂 Arquivo {model_path} encontrado localmente")
    else:
        st.info(f"📂 Arquivo {model_path} não encontrado, baixando...")
        if not download_from_github():
            st.error("❌ Falha no download, usando modelo web")
            model_path = 'yolov8n'  # Modelo mais leve
    
    # Carregamento direto
    try:
        st.info(f"🔄 Carregando modelo: {model_path}")
        
        # SEM CACHE, SEM PATCH, DIRETO
        model = YOLO(model_path)
        
        # Verifica classes
        if hasattr(model, 'names') and model.names:
            class_names = list(model.names.values())
            damage_classes = ['dent', 'scratch', 'crack', 'shattered_glass', 'broken_lamp', 'flat_tire']
            
            has_damage_classes = any(dc in str(class_names).lower() for dc in damage_classes)
            
            if has_damage_classes:
                st.success(f"✅ Modelo de danos carregado! Classes: {len(class_names)}")
                return model, "Modelo Personalizado", True
            else:
                st.warning(f"⚠️ Modelo genérico carregado. Classes: {len(class_names)}")
                return model, "Modelo Genérico", False
        else:
            st.warning("⚠️ Modelo carregado mas sem informações de classes")
            return model, "Modelo Desconhecido", False
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar: {str(e)}")
        
        # Último recurso - modelo nano
        try:
            st.info("🔄 Tentando YOLOv8n como último recurso...")
            model = YOLO('yolov8n')
            st.warning("⚠️ YOLOv8n carregado como fallback")
            return model, "YOLOv8n Fallback", False
        except Exception as e2:
            st.error(f"❌ Falha total: {str(e2)}")
            return None, "Erro Total", False

def process_image_simple(image, model, confidence=0.25, is_damage_model=False):
    """Processamento simples de imagem"""
    
    if model is None:
        # Modo demo
        return [{
            'damage_id': 'DEMO_001',
            'class': 'dent',
            'class_display': 'Amassado (Demo)',
            'confidence': 0.85,
            'severity': 'Moderado',
            'location': 'Carroceria',
            'estimated_cost': 750,
            'bbox': {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
        }], np.array(image)
    
    try:
        img_array = np.array(image)
        results = model(img_array, conf=confidence, verbose=False)
        
        detections = []
        
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                try:
                    class_id = int(boxes.cls[i])
                    class_name = results[0].names.get(class_id, f"class_{class_id}")
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].cpu().numpy()
                    
                    if not is_damage_model:
                        # Modelo genérico - simula se detectar veículo
                        if class_name.lower() in ['car', 'truck', 'bus', 'vehicle']:
                            detections.append({
                                'damage_id': f'SIM_{i+1:03d}',
                                'class': 'dent',
                                'class_display': 'Amassado (Simulado)',
                                'confidence': conf * 0.8,
                                'severity': 'Moderado',
                                'location': 'Carroceria',
                                'estimated_cost': np.random.randint(500, 1200),
                                'bbox': {'x1': int(bbox[0]), 'y1': int(bbox[1]), 'x2': int(bbox[2]), 'y2': int(bbox[3])}
                            })
                    else:
                        # Modelo real de danos
                        severity = DAMAGE_CONFIG['severity_map'].get(class_name, 'Leve')
                        location = DAMAGE_CONFIG['location_map'].get(class_name, 'Desconhecida')
                        cost_range = DAMAGE_CONFIG['cost_ranges'][severity]
                        
                        detections.append({
                            'damage_id': f'DMG_{i+1:03d}',
                            'class': class_name,
                            'class_display': DAMAGE_CONFIG['class_names'].get(class_name, class_name),
                            'confidence': conf,
                            'severity': severity,
                            'location': location,
                            'estimated_cost': np.random.randint(cost_range[0], cost_range[1]),
                            'bbox': {'x1': int(bbox[0]), 'y1': int(bbox[1]), 'x2': int(bbox[2]), 'y2': int(bbox[3])}
                        })
                        
                except Exception as e:
                    st.warning(f"Erro ao processar detecção {i}: {str(e)[:50]}")
                    continue
        
        # Imagem anotada
        try:
            if results and len(results) > 0:
                annotated = results[0].plot()
                if cv2 is not None:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                else:
                    annotated = img_array
            else:
                annotated = img_array
        except:
            annotated = img_array
        
        return detections, annotated
        
    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        return [], np.array(image)

def create_report(detections, vehicle_info=None):
    """Cria relatório simples"""
    if not vehicle_info:
        vehicle_info = {"plate": "N/A", "model": "N/A", "year": "N/A", "color": "N/A"}
    
    severity_count = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    total_cost = 0
    damage_types = []
    
    for det in detections:
        severity_count[det['severity']] += 1
        total_cost += det['estimated_cost']
        if det['class_display'] not in damage_types:
            damage_types.append(det['class_display'])
    
    urgency = 'Alta' if severity_count['Severo'] > 0 else 'Média' if severity_count['Moderado'] > 1 else 'Baixa'
    
    return {
        "inspection_info": {
            "timestamp": datetime.now().isoformat(),
            "inspector": "Sistema Carglass",
            "version": "6.0 Simplificado"
        },
        "vehicle_info": vehicle_info,
        "damage_summary": {
            "total_damages": len(detections),
            "severity_count": severity_count,
            "damage_types": damage_types,
            "estimated_total_cost": f"R$ {total_cost:,.2f}",
            "repair_urgency": urgency
        },
        "detections": detections
    }

def create_charts(detections):
    """Gráficos simples"""
    if not detections:
        return None, None
    
    # Gráfico de severidade
    severity_counts = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    for det in detections:
        severity_counts[det['severity']] += 1
    
    df_sev = pd.DataFrame(list(severity_counts.items()), columns=['Severidade', 'Quantidade'])
    df_sev = df_sev[df_sev['Quantidade'] > 0]
    
    colors = {'Leve': '#28a745', 'Moderado': '#ffc107', 'Severo': '#dc3545'}
    
    fig_sev = px.pie(df_sev, values='Quantidade', names='Severidade', 
                     title='Distribuição por Severidade', color='Severidade',
                     color_discrete_map=colors, height=300)
    
    # Gráfico de confiança
    df_conf = pd.DataFrame(detections)
    df_conf['confidence_pct'] = df_conf['confidence'] * 100
    
    fig_conf = px.bar(df_conf, x='class_display', y='confidence_pct', color='severity',
                      title='Confiança das Detecções', color_discrete_map=colors,
                      labels={'confidence_pct': 'Confiança (%)', 'class_display': 'Tipo'},
                      height=400)
    fig_conf.update_layout(xaxis_tickangle=-45)
    
    return fig_sev, fig_conf

def main():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🚗 Carglass - Detector SIMPLIFICADO</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Versão Limpa - Sem Problemas de Recursão</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configurações Simples")
        
        # Informações do GitHub
        st.success("✅ GitHub configurado")
        st.info(f"👤 {GITHUB_CONFIG['user']}/{GITHUB_CONFIG['repo']}")
        st.info(f"🏷️ {GITHUB_CONFIG['release_tag']}")
        
        # Upload manual
        uploaded = st.file_uploader("📁 Upload Manual", type=['pt'])
        if uploaded:
            with open(GITHUB_CONFIG['model_filename'], 'wb') as f:
                f.write(uploaded.getbuffer())
            st.success("✅ Upload concluído!")
            st.rerun()
        
        # Configurações
        confidence = st.slider("Confiança", 0.1, 0.9, 0.25, 0.05)
        
        # Dados do veículo
        st.markdown("### 🚗 Dados do Veículo")
        plate = st.text_input("Placa", "ABC-1234")
        model_name = st.text_input("Modelo", "Toyota Corolla")
        year = st.number_input("Ano", 1990, 2025, 2020)
        color = st.text_input("Cor", "Branco")
    
    # Carregamento do modelo
    model, model_type, is_damage_model = load_model_simple()
    
    if model is None:
        st.error("❌ Nenhum modelo carregado - Usando modo demonstração")
        model_status = "🎭 Modo Demo"
    elif is_damage_model:
        model_status = f"🎯 {model_type} - Detecta danos reais"
    else:
        model_status = f"⚠️ {model_type} - Resultados simulados"
    
    st.info(model_status)
    
    # Interface principal
    st.markdown("### 📤 Upload da Imagem")
    uploaded_file = st.file_uploader("Imagem do veículo:", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Redimensiona se necessário
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
            st.info("🔄 Imagem redimensionada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Original")
            st.image(image, use_column_width=True)
        
        # Processamento
        with st.spinner("🔍 Analisando..."):
            detections, annotated = process_image_simple(image, model, confidence, is_damage_model)
        
        with col2:
            st.markdown("#### 🎯 Detecções")
            st.image(annotated, use_column_width=True)
        
        if detections:
            st.markdown("### 📊 Resultados")
            
            # Métricas
            total_cost = sum(d['estimated_cost'] for d in detections)
            avg_conf = np.mean([d['confidence'] for d in detections])
            severe_count = sum(1 for d in detections if d['severity'] == 'Severo')
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔍 Danos", len(detections))
            col2.metric("💰 Custo", f"R$ {total_cost:,.2f}")
            col3.metric("🚨 Severos", severe_count)
            col4.metric("📈 Confiança", f"{avg_conf:.1%}")
            
            # Gráficos
            fig_sev, fig_conf = create_charts(detections)
            
            col1, col2 = st.columns(2)
            if fig_sev:
                col1.plotly_chart(fig_sev, use_container_width=True)
            if fig_conf:
                col2.plotly_chart(fig_conf, use_container_width=True)
            
            # Tabela
            df = pd.DataFrame(detections)
            df_display = df[['damage_id', 'class_display', 'severity', 'estimated_cost']].copy()
            df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
            df_display['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
            df_display.columns = ['ID', 'Tipo', 'Severidade', 'Custo', 'Confiança']
            
            st.dataframe(df_display, use_container_width=True)
            
            # Relatório
            vehicle_info = {"plate": plate, "model": model_name, "year": str(year), "color": color}
            report = create_report(detections, vehicle_info)
            
            # Download
            report_json = json.dumps(report, indent=2, ensure_ascii=False)
            st.download_button(
                "📄 Baixar Relatório JSON",
                report_json,
                f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
            
        else:
            st.success("✅ Nenhum dano detectado!")
            st.balloons()
    
    else:
        st.info("👆 Faça upload de uma imagem para começar")
        
        with st.expander("💡 Dicas"):
            st.markdown("""
            - **Boa iluminação**: Fotos claras funcionam melhor
            - **Foco no dano**: Aproxime da área danificada  
            - **Múltiplos ângulos**: Teste diferentes perspectivas
            - **Formato**: PNG, JPG ou JPEG até 200MB
            """)
    
    st.markdown("---")
    st.markdown("**🚗 Carglass Detector v6.0 Simplificado** | Sem recursão, sem complicação!")

if __name__ == "__main__":
    main()
