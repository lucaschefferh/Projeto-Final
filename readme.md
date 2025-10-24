Classificação de Doenças Oculares com CNN Customizada e YOLOv8

Este projeto desenvolve e compara dois modelos de visão computacional para classificação de doenças oculares em imagens clínicas: um modelo CNN criado do zero e um modelo pré-treinado (YOLOv8) ajustado via fine-tuning.

Projeto final do LAMIA (UTFPR), cobrindo CNNs, transfer learning e avaliação de desempenho.

Tecnologias Utilizadas

Python 3.10+

TensorFlow / Keras

Ultralytics YOLOv8

NumPy / Pandas / scikit-learn

Matplotlib / Seaborn

Jupyter Notebook / Google Colab

CUDA 12.8 + cuDNN (GPU NVIDIA)

Instalação e Execução
1) Clonar o repositório
git clone https://github.com/lucaschefferh/Projeto-Final.git
cd Projeto-Final

2) (Opcional) Criar ambiente virtual

Linux/macOS:

python -m venv venv
source venv/bin/activate


Windows (PowerShell):

python -m venv venv
venv\Scripts\Activate.ps1

3) Instalar dependências
pip install -r requirements.txt


Dica: para usar GPU NVIDIA, verifique instalação e compatibilidade de CUDA e cuDNN.

4) Executar notebooks

Abra no Jupyter Notebook ou Google Colab:

modelo_cnn.ipynb — treinamento e avaliação da CNN customizada

Yolo.ipynb — fine-tuning do YOLOv8n-cls

Estrutura do Projeto
Projeto-Final/
├─ modelo_cnn.ipynb               # Criação, treino e avaliação da CNN
├─ Yolo.ipynb                     # Treinamento e fine-tuning do YOLOv8n-cls
├─ best_model.keras               # Modelo customizado salvo
├─ runs/classify/train5/          # Resultados do YOLOv8 (pesos, logs, métricas)
├─ README.md                      # Este arquivo
├─ requirements.txt               # Dependências
├─ Relatorio_Final.pdf            # Relatório técnico-acadêmico
└─ Apresentacao_Final.pdf         # Slides de apresentação

Métodos Utilizados
CNN Customizada (do zero)

Arquitetura leve com camadas convolucionais, pooling e blocos densos

Batch Normalization e Spatial Dropout para estabilidade e regularização

Data augmentation leve (flip, rotação, brilho, contraste)

Otimizador: AdamW + Cosine Decay Restarts + EarlyStopping

Divisão: 80% treino / 20% validação

80 épocas de treinamento

YOLOv8n-cls (pré-treinado)

Arquitetura anchor-free da Ultralytics ajustada via fine-tuning

Imagens 224×224 normalizadas

Data augmentation leve (RandAugment: flips, rotações, variação de cor)

Otimizador: AdamW + warmup + AMP

Divisão: 90% treino / 10% teste

100 épocas de treinamento

Resultados
CNN Customizada

Acurácia de validação: 80,2%

F1-score médio: 0,80

Bom desempenho em Cataract e Retinopathy

Vantagens: interpretável e ajustável

Limitações: treino mais lento e sensível à memória

YOLOv8n-cls

Acurácia Top-1: 93,4%

Acurácia Top-5: 100%

Treinamento rápido (~18 min)

Vantagens: generalização, estabilidade e eficiência

Limitações: menor interpretabilidade clínica

Comparativo (Resumo)
Aspecto	YOLOv8n-cls (pré-treinado)	CNN do zero
Base	Fine-tuning	Treino completo
Pré-processamento	224×224, normalização 0–1	Igual ao YOLOv8
Augmentation	RandAugment leve	Transformações leves
Regularização	Weight Decay	BatchNorm + Spatial Dropout
Otimização	AdamW + Warmup + AMP	AdamW + Cosine Decay + ES
Épocas / Batch	100 / 16	80 / 32
Acurácia final	93,4%	80,2%
Interpretabilidade	Menor	Maior
Conclusões

Modelos pré-treinados (YOLOv8) são ideais quando há poucos dados, pouco tempo e foco em desempenho.

Modelos customizados (CNN do zero) dão mais controle e transparência, úteis para pesquisa e validação em domínios sensíveis.

Em síntese: YOLOv8 prioriza desempenho/eficiência; a CNN prioriza controle/interpretabilidade. As abordagens são complementares.

Documentação

Relatório técnico: Relatorio_Final.pdf

Apresentação de slides: Apresentacao_Final.pdf

Dependências

Instalação:

pip install -r requirements.txt


Conteúdo de requirements.txt:

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

Autor

Lucas Scheffer — Estudante de Ciência da Computação (UTFPR)
E-mail: lshundsdorfer@gmail.com

LinkedIn: https://linkedin.com/in/lucas-scheffer-344a36325

GitHub: https://github.com/lucaschefferh
