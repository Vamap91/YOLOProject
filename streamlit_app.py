import streamlit as st
import numpy as np
from PIL import Image
import os
import json
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
    """Carrega modelo YOLO com estratégia simples e robusta"""
    
    try:
        if os.path.exists('trained.pt'):
            st.info("🔄 Carregando modelo personalizado...")
            model = YOLO('trained.pt')
            st.success("✅ Modelo personalizado carregado!")
            return model, "Modelo Personalizado"
    except Exception as e:
        st.warning(f"⚠️ Erro no modelo personalizado: {str(e)}")
    
    try:
        if os.path.exists('yolov8m.pt'):
            st.info("🔄 Carregando YOLOv8m local...")
            model = YOLO('yolov8m.pt')
            st.success("✅ YOLOv8m local carregado!")
            return model, "YOLOv8m Local"
    except Exception as e:
        st.warning(f"⚠️ Erro no YOLOv8m local: {str(e)}")
    
    try:
        st.info("🔄 Baixando YOLOv8m da web...")
        model = YOLO('yolov8m')
        st.success("✅ YOLOv8m web carregado!")
        return model, "YOLOv8m Web"
    except Exception as e:
        st.warning(f"⚠️ Erro no YOLOv8m web: {str(e)}")
    
    try:
        st.info("🔄 Tentando YOLOv8s como fallback...")
        model = YOLO('yolov8s')
        st.success("✅ YOLOv8s carregado como fallback!")
        return model, "YOLOv8s Fallback"
    except Exception as e:
        st.error(f"❌ Erro crítico: {str(e)}")
        return None, None

def process_damage_detection(image, model, confidence_threshold=0.25):
    """Processa detecção de danos na imagem"""
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
    """Cria relatório JSON estruturado"""
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
            "model": "Sistema YOLO",
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
    """Gera recomendações baseadas nos danos detectados"""
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
    
    glass_damages = [d for d in detections if 'glass' in d['class'] or 'vidro' in d['class_display'].lower()]
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

def create_charts(detections):
    """Cria gráficos de análise"""
    if not detections:
        return None, None
    
    # Gráfico de severidade
    severity_counts = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
    for detection in detections:
        severity_counts[detection['severity']] += 1
    
    df_severity = pd.DataFrame(list(severity_counts.items()), columns=['Severidade', 'Quantidade'])
    df_severity = df_severity[df_severity['Quantidade'] > 0]
    
    colors = {'Leve': '#28a745', 'Moderado': '#ffc107', 'Severo': '#dc3545'}
    
    fig_severity = px.pie(
        df_severity, 
        values='Quantidade', 
        names='Severidade',
        title='Distribuição por Severidade',
        color='Severidade',
        color_discrete_map=colors,
        height=300
    )
    
    # Gráfico de confiança
    df_confidence = pd.DataFrame(detections)
    df_confidence['confidence_pct'] = df_confidence['confidence'] * 100
    
    fig_confidence = px.bar(
        df_confidence, 
        x='class_display', 
        y='confidence_pct',
        color='severity',
        title='Confiança das Detecções',
        labels={'confidence_pct': 'Confiança (%)', 'class_display': 'Tipo de Dano'},
        color_discrete_map=colors,
        height=400
    )
    
    fig_confidence.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Tipo de Dano",
        yaxis_title="Confiança (%)"
    )
    
    return fig_severity, fig_confidence

def main():
    """Função principal da aplicação"""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🚗 Carglass - Detector de Danos Veiculares</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0;'>Sistema IA para Detecção Automática de Danos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar com configurações
    with st.sidebar:
        st.markdown("### 🔧 Configurações")
        
        # Upload manual do modelo
        st.markdown("#### 📁 Upload do Modelo (Opcional)")
        uploaded_model = st.file_uploader(
            "Faça upload do modelo .pt", 
            type=['pt'],
            help="Carregue trained.pt ou yolov8m.pt se disponível"
        )
        
        if uploaded_model is not None:
            model_name = uploaded_model.name
            with open(model_name, 'wb') as f:
                f.write(uploaded_model.getbuffer())
            st.success(f"✅ {model_name} carregado via upload!")
            st.rerun()
        
        # Configuração de confiança
        confidence_threshold = st.slider(
            "Limite de Confiança", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.25, 
            step=0.05,
            help="Ajuste a sensibilidade da detecção"
        )
        
        st.markdown("### 📊 Tipos de Danos")
        st.markdown("""
        **Severos:**
        - Vidros quebrados
        - Lâmpadas quebradas  
        - Pneus vazios
        
        **Moderados:**
        - Amassados
        
        **Leves:**
        - Riscos
        - Rachaduras
        """)
        
        st.markdown("### 💰 Estimativas")
        st.markdown("""
        - **Severo**: R$ 1.500 - R$ 3.500
        - **Moderado**: R$ 500 - R$ 1.500  
        - **Leve**: R$ 200 - R$ 600
        """)
        
        # Informações do veículo
        st.markdown("### ℹ️ Dados do Veículo")
        vehicle_plate = st.text_input("Placa", placeholder="ABC-1234")
        vehicle_model = st.text_input("Modelo", placeholder="Toyota Corolla")
        vehicle_year = st.number_input("Ano", min_value=1990, max_value=2025, value=2020)
        vehicle_color = st.text_input("Cor", placeholder="Branco")
    
    # Carregamento do modelo
    model, model_type = load_damage_model()
    if model is None:
        st.error("❌ Não foi possível carregar nenhum modelo YOLO!")
        st.info("💡 **Soluções:**")
        st.info("1. Faça upload manual do modelo na barra lateral")
        st.info("2. Verifique sua conexão com a internet")
        st.info("3. Reinstale: `pip install ultralytics==8.0.196`")
        return
    
    st.info(f"🤖 **Modelo ativo:** {model_type}")
    
    # Interface principal com tabs
    tab1, tab2 = st.tabs(["📷 Análise de Imagem", "📄 Relatório Completo"])
    
    with tab1:
        st.markdown("### 📤 Upload da Imagem")
        uploaded_file = st.file_uploader(
            "Escolha uma imagem do veículo:",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos aceitos: PNG, JPG, JPEG (máx. 200MB)"
        )
        
        # Exemplos de teste
        st.markdown("### 🎯 Ou teste com exemplos:")
        col1, col2, col3 = st.columns(3)
        
        example_selected = None
        with col1:
            if st.button("🚗 Exemplo: Amassado", use_container_width=True):
                example_selected = "examples/1.png"
        with col2:
            if st.button("🔍 Exemplo: Múltiplos", use_container_width=True):
                example_selected = "examples/2.png"
        with col3:
            if st.button("💥 Exemplo: Vidro", use_container_width=True):
                example_selected = "examples/3.png"
        
        # Processamento da imagem
        image_source = None
        image_name = "Imagem"
        
        if example_selected and os.path.exists(example_selected):
            image_source = Image.open(example_selected)
            image_name = f"Exemplo: {example_selected.split('/')[-1]}"
            st.info(f"🎯 Usando: {image_name}")
        elif uploaded_file is not None:
            image_source = Image.open(uploaded_file)
            image_name = uploaded_file.name
        
        if image_source is not None:
            # Redimensiona imagem se muito grande
            max_size = (1024, 1024)
            if image_source.size[0] > max_size[0] or image_source.size[1] > max_size[1]:
                image_source.thumbnail(max_size, Image.Resampling.LANCZOS)
                st.info("🔄 Imagem redimensionada para otimizar processamento")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📸 Imagem Original")
                st.image(image_source, caption=image_name, use_column_width=True)
            
            # Processamento
            with st.spinner("🔍 Analisando imagem... Aguarde..."):
                detections, annotated_img = process_damage_detection(
                    image_source, model, confidence_threshold, is_damage_model
                )
                
                # Aviso se usando simulação
                if not is_damage_model and detections:
                    st.warning("⚠️ **Atenção:** Resultados simulados - usando modelo genérico")
                    st.info("📋 Para detecção real, carregue o modelo personalizado `trained.pt`")
            
            with col2:
                st.markdown("#### 🎯 Detecções Encontradas")
                st.image(annotated_img, caption="Danos detectados", use_column_width=True)
            
            # Resultados da análise
            if detections:
                st.markdown("### 📊 Resultados da Análise")
                
                # Métricas principais
                total_cost = sum([d['estimated_cost'] for d in detections])
                severity_counts = {'Leve': 0, 'Moderado': 0, 'Severo': 0}
                for d in detections:
                    severity_counts[d['severity']] += 1
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔍 Total Danos", len(detections))
                with col2:
                    st.metric("💰 Custo Estimado", f"R$ {total_cost:,.2f}")
                with col3:
                    st.metric("🚨 Danos Severos", severity_counts['Severo'])
                with col4:
                    confidence_avg = np.mean([d['confidence'] for d in detections])
                    st.metric("📈 Confiança Média", f"{confidence_avg:.1%}")
                
                # Gráficos
                st.markdown("### 📈 Análise Visual")
                fig_severity, fig_confidence = create_charts(detections)
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig_severity:
                        st.plotly_chart(fig_severity, use_container_width=True)
                
                with col2:
                    if fig_confidence:
                        st.plotly_chart(fig_confidence, use_container_width=True)
                
                # Tabela detalhada
                st.markdown("### 📋 Detalhes dos Danos")
                df_detections = pd.DataFrame(detections)
                df_display = df_detections[['damage_id', 'class_display', 'severity', 'location', 'estimated_cost']].copy()
                df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
                df_display['confidence'] = df_detections['confidence'].apply(lambda x: f"{x:.1%}")
                df_display.columns = ['ID', 'Tipo', 'Severidade', 'Localização', 'Custo', 'Confiança']
                
                st.dataframe(df_display, use_container_width=True)
                
                # Recomendações
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
                
                # Salva relatório na sessão
                st.session_state['report'] = report
                
            else:
                st.success("### ✅ Nenhum dano detectado!")
                st.markdown("**Parabéns!** Seu veículo parece estar em excelentes condições visuais.")
                st.balloons()
                
                # Relatório vazio para veículos sem danos
                vehicle_info = {
                    "plate": vehicle_plate or "Não informado",
                    "model": vehicle_model or "Não informado",
                    "year": str(vehicle_year),
                    "color": vehicle_color or "Não informado"
                }
                st.session_state['report'] = create_damage_report_json([], vehicle_info)
        
        else:
            st.info("👆 **Faça upload de uma imagem** ou **selecione um exemplo** para começar a análise")
            
            # Instruções de uso
            with st.expander("📖 **Como usar este sistema**"):
                st.markdown("""
                1. **Upload**: Envie uma foto clara do seu veículo
                2. **Aguarde**: O sistema processará automaticamente 
                3. **Analise**: Veja os danos detectados com bounding boxes
                4. **Relatório**: Acesse informações detalhadas na aba "Relatório"
                5. **Download**: Baixe o relatório completo em JSON ou TXT
                
                **💡 Dicas para melhores resultados:**
                - Use fotos com boa iluminação
                - Foque nas áreas danificadas
                - Evite reflexos e sombras excessivas
                - Mantenha o veículo limpo para melhor detecção
                """)
    
    with tab2:
        if 'report' in st.session_state:
            report = st.session_state['report']
            
            st.markdown("### 📄 Relatório Completo de Inspeção")
            
            # Cabeçalho do relatório
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🚗 Informações do Veículo")
                st.write(f"**Placa:** {report['vehicle_info']['plate']}")
                st.write(f"**Modelo:** {report['vehicle_info']['model']}")
                st.write(f"**Ano:** {report['vehicle_info']['year']}")
                st.write(f"**Cor:** {report['vehicle_info']['color']}")
            
            with col2:
                st.markdown("#### 🔍 Dados da Inspeção")
                inspection_date = datetime.fromisoformat(report['inspection_info']['timestamp'])
                st.write(f"**Data:** {inspection_date.strftime('%d/%m/%Y')}")
                st.write(f"**Hora:** {inspection_date.strftime('%H:%M:%S')}")
                st.write(f"**Sistema:** {report['inspection_info']['inspector']}")
                st.write(f"**Versão:** {report['inspection_info']['version']}")
            
            # Resumo executivo
            st.markdown("#### 📊 Resumo Executivo")
            summary = report['damage_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Danos", summary['total_damages'])
            with col2:
                st.metric("Custo Total", summary['estimated_total_cost'])
            with col3:
                st.metric("Urgência", summary['repair_urgency'])
            with col4:
                severity_critical = summary['severity_count']['Severo']
                st.metric("Críticos", severity_critical)
            
            # Detalhes dos danos (se houver)
            if report['detections']:
                st.markdown("#### 📋 Lista Detalhada de Danos")
                
                df_detections = pd.DataFrame(report['detections'])
                df_display = df_detections[['damage_id', 'class_display', 'severity', 'location', 'estimated_cost']].copy()
                df_display['estimated_cost'] = df_display['estimated_cost'].apply(lambda x: f"R$ {x:,.2f}")
                df_display['confidence'] = df_detections['confidence'].apply(lambda x: f"{x:.1%}")
                df_display.columns = ['ID do Dano', 'Tipo', 'Severidade', 'Localização', 'Custo Estimado', 'Confiança']
                
                st.dataframe(df_display, use_container_width=True)
            
            # Recomendações detalhadas
            st.markdown("#### 🎯 Recomendações Detalhadas")
            for i, rec in enumerate(report['recommendations'], 1):
                with st.expander(f"{i}. {rec['priority']} - {rec['message'][:50]}..."):
                    st.write(f"**Prioridade:** {rec['priority']}")
                    st.write(f"**Recomendação:** {rec['message']}")
                    if rec['damages']:
                        st.write(f"**Danos Relacionados:** {', '.join(rec['damages'])}")
            
            # Download do relatório
            st.markdown("#### 📥 Download do Relatório")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_json = json.dumps(report, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Baixar Relatório JSON",
                    data=report_json,
                    file_name=f"relatorio_carglass_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Relatório em texto
                report_text = f"""
RELATÓRIO DE INSPEÇÃO CARGLASS
{'='*50}

INFORMAÇÕES DO VEÍCULO:
- Placa: {report['vehicle_info']['plate']}
- Modelo: {report['vehicle_info']['model']} 
- Ano: {report['vehicle_info']['year']}
- Cor: {report['vehicle_info']['color']}

RESUMO DA INSPEÇÃO:
- Data: {inspection_date.strftime('%d/%m/%Y %H:%M:%S')}
- Total de Danos: {report['damage_summary']['total_damages']}
- Custo Total Estimado: {report['damage_summary']['estimated_total_cost']}
- Nível de Urgência: {report['damage_summary']['repair_urgency']}

DANOS DETECTADOS:
"""
                if report['detections']:
                    for detection in report['detections']:
                        report_text += f"- {detection['damage_id']}: {detection['class_display']} ({detection['severity']}) - R$ {detection['estimated_cost']:,.2f} - Confiança: {detection['confidence']:.1%}\n"
                else:
                    report_text += "- Nenhum dano detectado\n"
                
                report_text += f"""
RECOMENDAÇÕES:
"""
                for i, rec in enumerate(report['recommendations'], 1):
                    report_text += f"{i}. [{rec['priority']}] {rec['message']}\n"
                
                report_text += f"""
---
Relatório gerado pelo Sistema Carglass v{report['inspection_info']['version']}
Tecnologia: {report['inspection_info']['model']}
"""
                
                st.download_button(
                    label="📝 Baixar Relatório TXT",
                    data=report_text,
                    file_name=f"relatorio_carglass_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        else:
            st.info("📷 **Realize a análise de uma imagem** na aba anterior para gerar o relatório completo.")
            st.markdown("""
            ### 📋 O que você encontrará no relatório:
            
            - **Informações do Veículo**: Dados completos inseridos
            - **Resumo Executivo**: Métricas principais da inspeção  
            - **Lista de Danos**: Detalhamento de cada dano encontrado
            - **Estimativas de Custo**: Valores por dano e total
            - **Recomendações**: Sugestões priorizadas por urgência
            - **Downloads**: Relatórios em JSON e TXT para arquivo
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>🚗 Carglass - Sistema de Detecção Automática de Danos</strong></p>
        <p>Powered by YOLO + Streamlit | Versão 5.0 | Desenvolvido com IA</p>
        <p><em>Para suporte técnico, entre em contato com a equipe Carglass</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
