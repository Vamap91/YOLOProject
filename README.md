# 🚗 Detector de Danos Veiculares - Streamlit App

Uma aplicação web interativa que utiliza YOLO11m para detectar automaticamente danos em veículos através de imagens.

## 🎯 Funcionalidades

- **Upload de imagens** de veículos
- **Detecção automática** de 6 tipos de danos:
  - Amassados (Dents)
  - Riscos (Scratches)
  - Rachaduras (Cracks)
  - Vidros quebrados (Shattered Glass)
  - Lâmpadas quebradas (Broken Lamps)
  - Pneus vazios (Flat Tires)
- **Visualização interativa** com bounding boxes
- **Relatório detalhado** com níveis de confiança
- **Gráficos** de análise das detecções
- **Recomendações** baseadas nos danos encontrados

## 🚀 Como Executar

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/vehicle-damage-detector-streamlit.git
cd vehicle-damage-detector-streamlit
```

### 2. Instale as Dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe o Modelo Treinado
- Faça download do arquivo `trained.pt` 
- Coloque na raiz do projeto

### 4. Execute a Aplicação
```bash
streamlit run app.py
```

### 5. Acesse no Navegador
A aplicação estará disponível em: `http://localhost:8501`

## 📦 Estrutura do Projeto

```
vehicle-damage-detector-streamlit/
│
├── app.py                 # Aplicação principal Streamlit
├── requirements.txt       # Dependências Python
├── trained.pt            # Modelo YOLO treinado
├── README.md             # Este arquivo
└── examples/             # Imagens de exemplo
    ├── dent_example.jpg
    ├── scratch_example.jpg
    └── glass_example.jpg
```

## 🎮 Como Usar

1. **Abra a aplicação** no navegador
2. **Faça upload** de uma imagem de um veículo
3. **Aguarde** a análise automática
4. **Visualize** os resultados:
   - Imagem original vs imagem com detecções
   - Resumo dos danos encontrados
   - Gráfico de confiança das detecções
   - Tabela detalhada
   - Recomendações de reparo

## 📊 Performance do Modelo

| Tipo de Dano | mAP50 | Precisão |
|--------------|--------|----------|
| Vidros quebrados | 99.4% | Excelente |
| Pneus vazios | 95.9% | Excelente |
| Lâmpadas quebradas | 89.5% | Muito boa |
| Riscos | 90.5% | Muito boa |
| Amassados | 69.2% | Boa |
| Rachaduras | 62.0% | Moderada |

## 🛠️ Tecnologias Utilizadas

- **Streamlit** - Framework web para apps de ML
- **YOLO11m** - Modelo de detecção de objetos
- **OpenCV** - Processamento de imagens
- **Plotly** - Visualizações interativas
- **Pandas** - Manipulação de dados
- **PyTorch** - Framework de deep learning

## 🌐 Deploy

### Streamlit Cloud
1. Fork este repositório
2. Conecte sua conta GitHub ao Streamlit Cloud
3. Deploy direto da interface web

### Heroku
```bash
# Adicione arquivos de config do Heroku
echo "web: streamlit run app.py --server.port \$PORT" > Procfile
echo "python-3.9.0" > runtime.txt

# Deploy
heroku create seu-app-name
git push heroku main
```

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contribuições

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Casos de Uso

- **Concessionárias**: Inspeção pré-entrega e pós-serviço
- **Seguradoras**: Avaliação automática de sinistros
- **Locadoras**: Check-in/out automatizado
- **Frotas**: Auditoria regular de veículos
- **Consumidores**: Avaliação de veículos usados

## 🔮 Melhorias Futuras

- [ ] Suporte a múltiplas imagens
- [ ] Exportação de relatórios em PDF
- [ ] Integração com APIs de seguradoras
- [ ] Modo batch para análise em lote
- [ ] Estimativa de custos de reparo
- [ ] Histórico de análises por usuário
- [ ] API REST para integração externa

## 📝 Licença

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙋‍♂️ Suporte

Tem dúvidas? Abra uma [issue](https://github.com/seu-usuario/vehicle-damage-detector-streamlit/issues) ou entre em contato!

---

**⭐ Se este projeto foi útil, dê uma estrela no GitHub!**
