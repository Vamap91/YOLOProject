# 🚗 Detector de Danos Veiculares - Versão 2.0

Esta é uma aplicação web interativa, construída com Streamlit, que utiliza um modelo de Inteligência Artificial (YOLOv8) para detectar, classificar e analisar danos em veículos a partir de imagens. O sistema foi aprimorado para utilizar um modelo especializado, garantindo uma análise precisa e em tempo real.

## 🎯 Funcionalidades Principais

- **Upload de Imagens**: Interface simples para enviar imagens de veículos (JPG, PNG, JPEG).
- **Detecção Precisa de Danos**: Utiliza o modelo `car_damage_best.pt` para identificar 6 tipos de danos:
  - Amassados (`dent`)
  - Riscos (`scratch`)
  - Rachaduras (`crack`)
  - Vidros quebrados (`shattered_glass`)
  - Faróis/lanternas quebradas (`broken_lamp`)
  - Pneus vazios (`flat_tire`)
- **Análise Detalhada**: Cada dano detectado é analisado para determinar:
  - **Severidade**: Leve, Moderado ou Severo.
  - **Localização**: Carroceria, Pintura, Vidros, etc.
  - **Custo Estimado**: Uma faixa de valor para o reparo.
- **Visualização Interativa**: Exibe a imagem original ao lado da imagem com os danos destacados (bounding boxes).
- **Relatórios Completos**: Gera e permite o download de um relatório detalhado em formato JSON, contendo todas as informações da análise.
- **Dashboard de Resultados**: Apresenta métricas-chave como número de danos, custo total estimado e gráficos de confiança.

## 🚀 Como Executar Localmente

Siga os passos abaixo para rodar a aplicação em seu ambiente de desenvolvimento.

### 1. Pré-requisitos

- Python 3.9 ou superior
- `pip` (gerenciador de pacotes)

### 2. Clone o Repositório

```bash
# (Opcional) Se você for clonar de um repositório Git
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_DIRETORIO>
```

### 3. Instale as Dependências

Certifique-se de que o arquivo `requirements.txt` está no diretório do projeto e execute:

```bash
pip install -r requirements.txt
```

### 4. Verifique o Modelo

O modelo treinado `car_damage_best.pt` deve estar na raiz do projeto. Este arquivo é essencial para a detecção de danos.

### 5. Execute a Aplicação Streamlit

No seu terminal, execute o comando:

```bash
streamlit run streamlit_app.py
```

A aplicação será iniciada e um endereço local (geralmente `http://localhost:8501`) será exibido. Abra-o em seu navegador.

## 📦 Estrutura do Projeto

```
/YOLOProject-main
│
├── .streamlit/
│   └── config.toml        # Configurações de tema e servidor do Streamlit
│
├── streamlit_app.py       # Código principal da aplicação Streamlit
├── car_damage_best.pt     # Modelo de IA treinado para detecção de danos
├── requirements.txt       # Dependências Python do projeto
├── test_model.py          # Script para validar o modelo e o ambiente
└── README.md              # Este arquivo de documentação
```

## 🛠️ Tecnologias Utilizadas

- **Backend**: Python 3.11
- **Inteligência Artificial**: Ultralytics YOLOv8, PyTorch
- **Processamento de Imagem**: OpenCV, Pillow
- **Frontend**: Streamlit
- **Visualização de Dados**: Plotly, Pandas

## 🌐 Deploy na Streamlit Cloud

Para fazer o deploy desta aplicação, siga estes passos:

1.  **Envie o projeto para o GitHub**: Certifique-se de que todos os arquivos, incluindo `car_damage_best.pt`, `streamlit_app.py`, `requirements.txt` e `packages.txt` (se necessário), estejam no seu repositório.
    *   **Atenção**: O GitHub tem um limite de 100MB por arquivo. Se o seu modelo `.pt` for maior, use o Git LFS (Large File Storage).
2.  **Conecte sua conta Streamlit Cloud ao GitHub**.
3.  **Crie um novo aplicativo**: No dashboard do Streamlit Cloud, clique em "New app" e selecione o repositório correspondente.
4.  **Configure e faça o deploy**: Verifique se o arquivo principal (`streamlit_app.py`) está correto e clique em "Deploy!".

## 🔮 Próximos Passos e Melhorias

- [ ] Otimizar a velocidade de inferência para análises mais rápidas.
- [ ] Implementar suporte para análise de vídeos.
- [ ] Adicionar um sistema de histórico para salvar e consultar análises passadas.
- [ ] Criar uma API REST para permitir a integração do modelo com outros sistemas.

---

*Este projeto foi aprimorado para fornecer uma solução robusta e precisa para a inspeção de danos veiculares, substituindo a análise simulada por detecções reais e especializadas.*
