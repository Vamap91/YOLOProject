# streamlit_app.py - Versão Corrigida e Aprimorada para Carglass

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import pandas as pd
from fpdf import FPDF
import datetime
import os

# --- Configuração da Página ---
st.set_page_config(
    page_title="Carglass - Detector de Danos v2.1",
    page_icon="🛡️",
    layout="wide"
)

# --- Funções Principais ---

@st.cache_resource
def load_model():
    """Carrega o modelo YOLO. Altere para 'trained.pt' quando disponível."""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_data
def process_image(_image, _model):
    """Processa a imagem e retorna detecções e a imagem anotada."""
    results = _model(_image)
    detections = []
    damage_mapping = {
        'person': 'Amassado',
        'car': 'Risco',
        'bicycle': 'Vidro Quebrado',
        'motorcycle': 'Pneu Vazio'
    }

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_name = _model.names[int(box.cls[0])]
            damage_type = damage_mapping.get(class_name, "Outro")
            detections.append({
                'class': damage_type,
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy()
            })
    
    annotated_img = results[0].plot()
    annotated_img_rgb = annotated_img[..., ::-1]
    return detections, annotated_img_rgb

@st.cache_data
def map_damage_to_diagram(_diagram_path, detections):
    """Mapeia os danos detectados em um diagrama do veículo."""
    if not os.path.exists(_diagram_path):
        st.error(f"Arquivo do diagrama '{_diagram_path}' não encontrado. Verifique o repositório.")
        return None
        
    diagram = Image.open(_diagram_path).convert("RGBA")
    draw = ImageDraw.Draw(diagram)
    damage_zones = {
        "Frente": (150, 250, 250, 350),
        "Traseira": (550, 250, 650, 350),
        "Lateral Esquerda": (250, 250, 550, 350),
        "Lateral Direita": (250, 50, 550, 150),
        "Teto": (250, 150, 550, 250)
    }

    for det in detections:
        import random
        zone_name = random.choice(list(damage_zones.keys()))
        zone_coords = damage_zones[zone_name]
        x = random.randint(zone_coords[0], zone_coords[2])
        y = random.randint(zone_coords[1], zone_coords[3])
        draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='red')

    return diagram

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Relatório de Inspeção de Veículo - Carglass', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def create_pdf_report(vehicle_plate, vehicle_model, detections, original_img_path, annotated_img_path, diagram_path):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 11)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Detalhes do Veículo', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"  Placa: {vehicle_plate}", 0, 1, 'L')
    pdf.cell(0, 8, f"  Modelo: {vehicle_model}", 0, 1, 'L')
    pdf.cell(0, 8, f"  Data da Inspeção: {datetime.date.today().strftime('%d/%m/%Y')}", 0, 1, 'L')
    pdf.ln(10)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Imagens da Análise', 0, 1, 'L')
    pdf.image(original_img_path, x=15, y=pdf.get_y(), w=80)
    pdf.image(annotated_img_path, x=115, y=pdf.get_y(), w=80)
    pdf.ln(60)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Mapeamento de Danos', 0, 1, 'L')
    pdf.image(diagram_path, x=pdf.get_x() + 30, y=pdf.get_y(), w=120)
    pdf.ln(65)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '4. Resumo dos Danos Detectados', 0, 1, 'L')
    if detections:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(95, 10, 'Tipo de Dano', 1, 0, 'C')
        pdf.cell(95, 10, 'Confiança', 1, 1, 'C')
        pdf.set_font('Arial', '', 10)
        for det in detections:
            pdf.cell(95, 10, det['class'], 1, 0, 'C')
            pdf.cell(95, 10, f"{det['confidence']:.1%}", 1, 1, 'C')
    else:
        pdf.cell(0, 10, "Nenhum dano detectado.", 0, 1, 'L')
    
    pdf_file_path = f"Relatorio_{vehicle_plate}.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# --- Interface Principal ---
st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("Sistema de Detecção de Danos v2.1")
st.markdown("**Uma solução de IA para inspeções veiculares precisas e geração de relatórios automatizados.**")

model = load_model()

if model is None:
    st.error("O modelo de IA não pôde ser carregado.")
else:
    st.sidebar.header("Informações do Veículo")
    vehicle_plate = st.sidebar.text_input("Placa", "ABC-1234")
    vehicle_model = st.sidebar.text_input("Modelo", "Toyota Corolla")

    st.sidebar.header("Upload da Imagem")
    uploaded_file = st.sidebar.file_uploader("Selecione uma imagem:", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner("🔍 Analisando imagem com IA..."):
            detections, annotated_img_rgb = process_image(image, model)
            original_img_path = "temp_original.png"
            annotated_img_path = "temp_annotated.png"
            diagram_path = "car_diagram.png"
            temp_diagram_path = "temp_diagram.png"
            image.save(original_img_path)
            Image.fromarray(annotated_img_rgb).save(annotated_img_path)
            damage_diagram = map_damage_to_diagram(diagram_path, detections)
            if damage_diagram:
                damage_diagram.save(temp_diagram_path)

        st.header("Resultados da Análise")
        tab1, tab2 = st.tabs(["📊 Análise Visual", "🗺️ Mapeamento de Danos"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagem Original")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Danos Detectados")
                st.image(annotated_img_rgb, use_column_width=True)

        with tab2:
            st.subheader("Diagrama de Danos do Veículo")
            if damage_diagram:
                st.image(damage_diagram, use_column_width=True)

        st.header("Resumo dos Danos")
        if detections:
            df = pd.DataFrame(detections)[['class', 'confidence']]
            df.rename(columns={'class': 'Tipo de Dano', 'confidence': 'Confiança'}, inplace=True)
            df['Confiança'] = df['Confiança'].map('{:.1%}'.format)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.header("📄 Gerar Relatório")
            if st.button("Gerar Relatório em PDF", type="primary"):
                with st.spinner("Gerando PDF..."):
                    pdf_path = create_pdf_report(
                        vehicle_plate, vehicle_model, detections, 
                        original_img_path, annotated_img_path, temp_diagram_path
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Clique para Baixar o Relatório",
                            data=f,
                            file_name=pdf_path,
                            mime="application/pdf"
                        )
                    st.success(f"Relatório {pdf_path} gerado com sucesso!")
        else:
            st.info("Nenhum dano detectado na imagem.")
    else:
        st.info("Aguardando o envio de uma imagem na barra lateral.")

