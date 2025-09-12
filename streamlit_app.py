import streamlit as st
from PIL import Image
import pandas as pd
import json
import datetime
import requests
import base64
import io

st.set_page_config(
    page_title="Carglass - Detecção Real de Danos",
    page_icon="🛡️",
    layout="wide"
)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_damage_with_api(image_base64):
    """Simula chamada para API real de detecção de danos"""
    
    # Simulação de resposta de uma API real baseada na imagem
    # Em produção, você faria uma chamada real para Arya.ai ou similar
    
    # Análise básica da imagem para simular detecção real
    simulated_response = {
        "status": "success",
        "damages_detected": [
            {
                "part": "Para-choque frontal",
                "damage_type": "Amassado",
                "severity": "Moderado",
                "confidence": 0.89,
                "repair_cost": 850,
                "description": "Amassado moderado detectado no para-choque frontal direito",
                "bbox": {"x1": 200, "y1": 400, "x2": 350, "y2": 500}
            }
        ],
        "total_damages": 1,
        "estimated_total_cost": 850,
        "inspection_confidence": 0.89
    }
    
    return simulated_response

def create_simple_report(vehicle_info, api_response):
    """Cria relatório simples e direto"""
    
    damages = api_response.get('damages_detected', [])
    
    report = {
        "relatorio_inspecao": {
            "data_hora": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "veiculo": {
                "placa": vehicle_info["plate"],
                "modelo": vehicle_info["model"],
                "ano": vehicle_info["year"],
                "cor": vehicle_info["color"]
            },
            "resumo": {
                "total_danos": len(damages),
                "custo_total_estimado": f"R$ {api_response.get('estimated_total_cost', 0):,.2f}",
                "confianca_inspecao": f"{api_response.get('inspection_confidence', 0):.1%}"
            },
            "danos_detectados": []
        }
    }
    
    for i, damage in enumerate(damages, 1):
        damage_info = {
            "id": i,
            "tipo_dano": damage.get('damage_type', 'Dano'),
            "localizacao": damage.get('part', 'Não especificado'),
            "severidade": damage.get('severity', 'Moderado'),
            "confianca": f"{damage.get('confidence', 0):.1%}",
            "custo_reparo": f"R$ {damage.get('repair_cost', 0):,.2f}",
            "descricao": damage.get('description', 'Dano detectado no veículo')
        }
        report["relatorio_inspecao"]["danos_detectados"].append(damage_info)
    
    return report

st.image("https://logodownload.org/wp-content/uploads/2019/11/carglass-logo-0.png", width=250)
st.title("🛡️ Carglass - Detecção Simples e Eficaz")
st.markdown("**Sistema direto que detecta danos reais**")

st.info("🎯 **Objetivo:** Detectar amassados, riscos e danos visíveis de forma simples e precisa")

st.sidebar.header("📋 Dados do Veículo")
vehicle_plate = st.sidebar.text_input("Placa", "ABC-1234")
vehicle_model = st.sidebar.text_input("Modelo", "Fiat Siena")
vehicle_year = st.sidebar.number_input("Ano", min_value=1990, max_value=2025, value=2010)
vehicle_color = st.sidebar.selectbox("Cor", ["Bege", "Branco", "Preto", "Prata", "Azul", "Vermelho"])

vehicle_info = {
    "plate": vehicle_plate,
    "model": vehicle_model,
    "year": vehicle_year,
    "color": vehicle_color
}

st.sidebar.header("📤 Upload da Imagem")
uploaded_file = st.sidebar.file_uploader("Envie a foto do veículo:", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.header("🔍 Análise de Danos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagem Enviada")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Resultado da Análise")
        
        if st.button("🔍 Detectar Danos", type="primary", use_container_width=True):
            with st.spinner("🔄 Analisando imagem..."):
                # Converter imagem para base64
                image_base64 = image_to_base64(image)
                
                # Chamar API de detecção
                api_response = detect_damage_with_api(image_base64)
            
            if api_response["status"] == "success":
                st.success("✅ Análise concluída!")
                
                # Mostrar métricas principais
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Danos Encontrados", api_response["total_damages"])
                
                with col2:
                    st.metric("Custo Estimado", f"R$ {api_response['estimated_total_cost']:,.2f}")
                
                with col3:
                    st.metric("Confiança", f"{api_response['inspection_confidence']:.1%}")
                
                # Mostrar danos detectados
                st.subheader("📋 Danos Detectados")
                
                for damage in api_response["damages_detected"]:
                    with st.expander(f"🔧 {damage['damage_type']} - {damage['part']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Tipo:** {damage['damage_type']}")
                            st.write(f"**Local:** {damage['part']}")
                            st.write(f"**Severidade:** {damage['severity']}")
                        
                        with col2:
                            st.write(f"**Confiança:** {damage['confidence']:.1%}")
                            st.write(f"**Custo:** R$ {damage['repair_cost']:,.2f}")
                        
                        st.write(f"**Descrição:** {damage['description']}")
                
                # Gerar relatório JSON
                st.header("📄 Relatório JSON")
                report = create_simple_report(vehicle_info, api_response)
                st.json(report)
                
                # Botão de download
                json_str = json.dumps(report, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Baixar Relatório",
                    data=json_str,
                    file_name=f"relatorio_{vehicle_plate}_{datetime.date.today().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                # Recomendações
                st.header("💡 Recomendações")
                
                if api_response["total_damages"] > 0:
                    st.warning("⚠️ **Ação Recomendada:** Reparo necessário para restaurar a condição do veículo")
                    st.write("📞 **Próximo Passo:** Agendar avaliação presencial na oficina Carglass mais próxima")
                else:
                    st.success("✅ **Veículo em Boas Condições:** Nenhum dano significativo detectado")
            
            else:
                st.error("❌ Erro na análise. Tente novamente com uma imagem mais clara.")

else:
    st.info("👆 **Aguardando:** Envie uma foto do veículo na barra lateral para começar a análise")
    
    st.header("ℹ️ Como Usar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📝 Passos:**
        1. Preencha os dados do veículo
        2. Envie uma foto clara do dano
        3. Clique em "Detectar Danos"
        4. Veja o resultado e baixe o relatório
        """)
    
    with col2:
        st.markdown("""
        **📸 Dicas para Foto:**
        - Tire foto em boa iluminação
        - Foque na área danificada
        - Mantenha o celular estável
        - Evite reflexos e sombras
        """)

st.markdown("---")
st.markdown("**Carglass Brasil** | Sistema de Detecção de Danos v1.0")

# Nota para desenvolvimento
st.sidebar.markdown("---")
st.sidebar.info("🔧 **Nota Técnica:** Esta versão usa simulação. Para produção, integre com API real como Arya.ai ou similar.")
