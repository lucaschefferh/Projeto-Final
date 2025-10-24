# 🧠 Classificação de Doenças Oculares com CNN Customizada e YOLOv8

Este projeto tem como objetivo o **desenvolvimento e comparação de dois modelos de visão computacional** aplicados à **classificação de doenças oculares** em imagens clínicas.  
O estudo busca compreender **quando é mais adequado utilizar um modelo pré-treinado** (como o YOLOv8) e **quando construir uma arquitetura própria (CNN customizada)**, considerando desempenho, interpretabilidade e aplicabilidade.

O trabalho foi desenvolvido como parte do **projeto final do LAMIA**, integrando conhecimentos de redes neurais convolucionais, *transfer learning* e avaliação de desempenho em visão computacional.

---

## 🚀 Tecnologias Utilizadas

- **Python 3.10+**
- **TensorFlow / Keras**
- **Ultralytics YOLOv8**
- **NumPy / Pandas / scikit-learn**
- **Matplotlib / Seaborn**
- **Jupyter Notebook / Google Colab**
- **CUDA 12.8 + cuDNN (GPU NVIDIA)**

---

## 📦 Instalação e Execução

### 1️⃣ Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/Projeto-Final.git
cd Projeto-Final
2️⃣ (Opcional) Crie um ambiente virtual
Linux/macOS

bash
Copiar código
python -m venv venv
source venv/bin/activate
Windows

bash
Copiar código
python -m venv venv
venv\Scripts\activate
3️⃣ Instale as dependências
bash
Copiar código
pip install -r requirements.txt
💡 Dica: Caso utilize GPU NVIDIA, verifique se CUDA e cuDNN estão configurados corretamente.

4️⃣ Execute os notebooks
Abra no Jupyter Notebook ou Google Colab:

modelo_cnn.ipynb → Treinamento e avaliação do modelo customizado

Yolo.ipynb → Treinamento e fine-tuning do YOLOv8n-cls

🗂️ Estrutura do Projeto
text
Copiar código
📂 Projeto-Final/
│
├── modelo_cnn.ipynb               # Notebook com criação, treino e avaliação da CNN
├── Yolo.ipynb                     # Notebook com treino e fine-tuning do YOLOv8n-cls
│
├── best_model.keras               # Modelo customizado salvo (para carregamento direto)
│
├── runs/classify/train5/          # Resultados do YOLOv8 (pesos, logs, gráficos e métricas)
│
├── README.md                      # Descrição completa do projeto e instruções de uso
├── requirements.txt               # Lista das dependências necessárias
│
├── Relatorio_Final.pdf            # Relatório técnico-acadêmico do projeto
└── Apresentacao_Final.pdf         # Slides de apresentação do projeto
🧠 Métodos Utilizados
🟪 Modelo Customizado (CNN)
Arquitetura desenvolvida do zero, com camadas convolucionais, pooling e blocos densos.

Uso de Batch Normalization e Spatial Dropout para estabilidade e regularização.

Data Augmentation leve (flip, rotação, brilho, contraste) para aumentar a diversidade sem distorcer padrões clínicos.

Otimizador AdamW com Cosine Decay Restarts e Early Stopping.

Divisão: 80% treino / 20% validação.

Treinamento de 80 épocas com monitoramento de perda e acurácia.

🟩 YOLOv8n-cls (Pré-treinado)
Modelo da Ultralytics, baseado em arquitetura anchor-free, ajustado via fine-tuning.

Imagens redimensionadas para 224×224 e normalizadas.

Data augmentation leve (RandAugment com flips, rotações e variações de cor).

Otimizador AdamW com warmup e Automatic Mixed Precision (AMP).

Divisão: 90% treino / 10% teste.

Treinamento de 100 épocas, com convergência rápida e estável.

📊 Resultados Obtidos
🟦 Modelo CNN Customizado
Acurácia de validação: 80,2%

F1-score médio: 0,80

Bom desempenho em Cataract e Retinopathy

Vantagem: interpretável e ajustável

Desvantagem: tempo de treino maior e sensível à memória

🟣 YOLOv8n-cls
Acurácia Top-1: 93,4%

Acurácia Top-5: 100%

Treinamento rápido e leve (~18 minutos)

Vantagem: alta generalização, estabilidade e eficiência

Desvantagem: menor interpretabilidade clínica

🧩 Comparativo de Abordagens
Aspecto	YOLOv8n-cls (Pré-treinado)	CNN Customizada (do zero)
Base de treinamento	Fine-tuning sobre modelo pré-treinado	Treino completo do zero
Pré-processamento	Redimensionamento (224×224), normalização 0–1	Igual ao YOLOv8
Data Augmentation	RandAugment leve	Transformações leves e controladas
Regularização	Weight Decay	BatchNorm + Spatial Dropout
Otimização	AdamW + Warmup + AMP	AdamW + Cosine Decay Restarts + EarlyStopping
Épocas / Batch	100 / 16	80 / 32
Acurácia final	93,4%	80,2%
Interpretação	Difícil (modelo fechado)	Alta (camadas visíveis)

🎯 Conclusões
A comparação entre as duas abordagens demonstrou que a escolha entre um modelo pré-treinado e um modelo desenvolvido do zero depende do propósito e do contexto do projeto.

Modelos pré-treinados (como o YOLOv8) são ideais quando há poucos dados, limitação de tempo e foco em desempenho.
São rápidos, eficientes e alcançam alta acurácia com mínimo ajuste.
Entretanto, sua arquitetura complexa reduz a explicabilidade.

Modelos customizados são indicados quando se busca compreensão detalhada, transparência e controle.
Permitem explorar os efeitos de cada camada e técnica de regularização, sendo mais adequados para pesquisa e validação científica, especialmente em domínios sensíveis como o clínico.

➡️ Em síntese:

YOLOv8 → desempenho e eficiência.

CNN própria → controle e interpretabilidade.
As duas abordagens são complementares e contribuem para um entendimento mais profundo de como redes neurais podem ser aplicadas em contextos médicos.

📄 Documentação
📘 Relatório Técnico — Projeto Final

🎤 Apresentação de Slides — Projeto Final

🧩 Dependências
Todas as bibliotecas necessárias estão listadas no arquivo requirements.txt.
Para instalar, execute:

bash
Copiar código
pip install -r requirements.txt
Conteúdo do requirements.txt
txt
Copiar código
tensorflow==2.17.0
tensorflow-addons==0.23.0
ultralytics==8.3.204
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
opencv-python==4.10.0.84
Pillow==10.4.0
tqdm==4.66.5
jupyter==1.1.1
notebook==7.2.2
torch==2.8.0
torchvision==0.19.0
👨‍💻 Autor
Lucas Scheffer
Estudante de Ciência da Computação — UTFPR

📧 E-mail: lshundsdorfer@gmail.com
🔗 LinkedIn: linkedin.com/in/lucas-scheffer-344a36325
💻 GitHub: github.com/lucaschefferh
