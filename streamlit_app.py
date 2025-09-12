import streamlit as st
from PIL import Image
import pandas as pd
import json
import datetime
import requests
import base64
import io
import uuid

st.set_page_config(
    page_title="Carglass - Arya.ai API",
    page_icon="🛡️",
    layout="wide"
)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def call_arya_api(image_base64, manufacturer, model):
    url = "https://ping.arya.ai/api/v1/motor"
    
    headers = {
        'token': 'cb74a998f2626fc3a97be7b61980ae1c',
        'content-type': 'application/json'
    }
    
    payload = {
        "req_id": str(uuid.uuid4()),
        "manufacturer": manufacturer,
        "model": model,
        "doc_base64": [
            {
                "name": "vehicle_image",
                "base64": image_base64
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error_message": f"API Error: {response.status_code} - {response.text}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Connection Error: {str(e)}"
        }

def process_arya_response(api_response):
    if not api_response.get("success", False):
        return None
    
    data = api_response.get("data", {})
    
    processed_damages = []
    total_cost = 0
    
    for part, details in data.items():
        if isinstance(details, dict) and "damage_type" in details:
            damage = {
                "part": part.replace("_", " ").title(),
                "damage_type": details.get("damage_type", "Unknown"),
                "severity": details.get("severity", "Unknown"),
                "confidence": details.get("confidence", 0),
                "repair_cost": details.get("cost", 0),
                "description": f"{details.get('damage_type', 'Damage')} detected on {part.replace('_', ' ')}"
            }
            processed_damages.append(damage)
            total_cost += damage["repair_cost"]
    
    return {
        "damages_detected": processed_damages,
        "total_damages": len(processed_damages),
        "total_cost": total_cost,
        "raw_response": api_response
    }

def create_detailed_report(vehicle_info, processed_data):
    if not processed_data:
        return {"error": "No data to process"}
    
    damages = processed_data["damages_detected"]
    
    report = {
        "inspection_report": {
            "timestamp": datetime.datetime.now().isoformat(),
            "api_provider": "Arya.ai Motor Assessment API",
            "vehicle_info": {
                "manufacturer": vehicle_info["manufacturer"],
                "model": vehicle_info["model"],
                "plate": vehicle_info.get("plate", "N/A"),
                "year": vehicle_info.get("year", "N/A")
            },
            "summary": {
                "total_damages": processed_data["total_damages"],
                "total_repair_cost": f"R$ {processed_data['total_cost']:,.2f}",
                "damage_types": list(set([d["damage_type"] for d in damages])),
                "affected_parts": [d["part"] for d in damages]
            },
            "detailed_damages": []
        }
    }
    
    for i, damage in enumerate(damages, 1):
        damage_detail = {
            "id": i,
            "part": damage["part"],
            "damage_type": damage["damage_type"],
            "severity": damage["severity"],
            "confidence": f"{damage['confidence']:.1%}" if isinstance(damage['confidence'], float) else str(damage['confidence']),
            "repair_cost": f"R$ {damage['repair_cost']:,.2f}",
            "description": damage["description"]
        }
        report["inspection_report"]["detailed_damages"].append(damage_detail)
    
    return report

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Carglass - Arya.ai Integration")
st.markdown("**Detecção real de danos usando API comercial**")

st.success("🔗 **API Ativa:** Arya.ai Motor Assessment API")
st.info("🎯 **Funcionalidade:** Detecção automática de danos com IA comercial")

st.sidebar.header("📋 Informações do Veículo")

manufacturer = st.sidebar.selectbox(
    "Fabricante",
    ["Toyota", "Fiat", "Volkswagen", "Ford", "Chevrolet", "Honda", "Hyundai", "Nissan", "Renault", "Peugeot"]
)

model = st.sidebar.text_input("Modelo", "Corolla")
plate = st.sidebar.text_input("Placa", "ABC-1234")
year = st.sidebar.number_input("Ano", min_value=1990, max_value=2025, value=2020)

vehicle_info = {
    "manufacturer": manufacturer,
    "model": model,
    "plate": plate,
    "year": year
}

st.sidebar.header("📤 Upload da Imagem")
uploaded_file = st.sidebar.file_uploader("Envie a foto do veículo:", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.header("🔍 Análise com Arya.ai API")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Enviada")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🤖 Análise da API")
        
        if st.button("🚀 Analisar com Arya.ai", type="primary", use_container_width=True):
            with st.spinner("🔄 Enviando para API Arya.ai..."):
                image_base64 = image_to_base64(image)
                
                api_response = call_arya_api(image_base64, manufacturer, model)
                
                st.write("**Resposta da API:**")
                st.json(api_response)
            
            if api_response.get("success", False):
                st.success("✅ Análise concluída pela API!")
                
                processed_data = process_arya_response(api_response)
                
                if processed_data and processed_data["total_damages"] > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Danos Detectados", processed_data["total_damages"])
                    
                    with col2:
                        st.metric("Custo Total", f"R$ {processed_data['total_cost']:,.2f}")
                    
                    with col3:
                        avg_confidence = sum([d["confidence"] for d in processed_data["damages_detected"] if isinstance(d["confidence"], float)]) / len(processed_data["damages_detected"])
                        st.metric("Confiança Média", f"{avg_confidence:.1%}")
                    
                    st.subheader("📋 Danos Detectados pela API")
                    
                    for damage in processed_data["damages_detected"]:
                        with st.expander(f"🔧 {damage['damage_type']} - {damage['part']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Peça:** {damage['part']}")
                                st.write(f"**Tipo de Dano:** {damage['damage_type']}")
                                st.write(f"**Severidade:** {damage['severity']}")
                            
                            with col2:
                                st.write(f"**Confiança:** {damage['confidence']}")
                                st.write(f"**Custo:** R$ {damage['repair_cost']:,.2f}")
                            
                            st.write(f"**Descrição:** {damage['description']}")
                    
                    st.header("📄 Relatório Completo")
                    report = create_detailed_report(vehicle_info, processed_data)
                    st.json(report)
                    
                    json_str = json.dumps(report, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Baixar Relatório JSON",
                        data=json_str,
                        file_name=f"relatorio_arya_{plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                else:
                    st.success("✅ **Nenhum dano detectado pela API**")
                    st.info("O veículo parece estar em boas condições segundo a análise da Arya.ai")
            
            else:
                st.error(f"❌ **Erro na API:** {api_response.get('error_message', 'Erro desconhecido')}")
                st.info("💡 **Dica:** Verifique se a imagem está clara e o token da API está correto")

else:
    st.info("👆 **Aguardando:** Envie uma foto do veículo na barra lateral")
    
    st.header("ℹ️ Sobre a Integração Arya.ai")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔗 API Endpoint:**
        `https://ping.arya.ai/api/v1/motor`
        
        **📝 Parâmetros Enviados:**
        - `req_id`: ID único da requisição
        - `manufacturer`: Fabricante do veículo
        - `model`: Modelo do veículo
        - `doc_base64`: Imagem em base64
        """)
    
    with col2:
        st.markdown("""
        **📊 Resposta da API:**
        - `success`: Status da análise
        - `data`: Dados dos danos detectados
        - `req_id`: ID da requisição processada
        
        **🎯 Tipos de Dano Detectados:**
        - Amassados, riscos, rachaduras
        - Vidros quebrados, faróis danificados
        - Estimativas de custo de reparo
        """)

st.markdown("---")
st.markdown("**Carglass Brasil** | Powered by Arya.ai Motor Assessment API")

st.sidebar.markdown("---")
st.sidebar.info("🔑 **API Token:** cb74***eac (configurado)")
st.sidebar.success("🟢 **Status:** API Ativa e Funcional")
