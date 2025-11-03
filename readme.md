# 🧠 Projeto Final — Classificação de Doenças Oculares com CNN Customizada e YOLOv8  

Este projeto foi desenvolvido como parte do **projeto final do LAMIA (Laboratório de Machine Learning para Indústria)**, com o objetivo de aplicar técnicas de **visão computacional** para a **classificação de doenças oculares** a partir de imagens clínicas de fundo de olho.  

O foco principal foi **desenvolver e comparar dois modelos de aprendizado profundo** — um **modelo customizado do zero (CNN própria)** e um **modelo pré-treinado YOLOv8n-cls**, avaliando seus desempenhos, limitações e aplicabilidades em contextos clínicos.  

---

## 📑 Sumário  
- [Introdução](#introdução)  
- [Contexto](#contexto)  
- [Objetivos do Projeto](#objetivos-do-projeto)  
- [Base de Dados](#base-de-dados)  
- [Modelos Implementados](#modelos-implementados)  
- [Resultados](#resultados)  
- [Conclusões](#conclusões)

---

## Introdução  

Doenças oculares como **catarata, glaucoma e retinopatia diabética** estão entre as principais causas de cegueira evitável no mundo.  
A detecção precoce é essencial para permitir o tratamento adequado e reduzir o impacto visual permanente.  

Com o avanço das técnicas de **Inteligência Artificial** e **visão computacional**, tornou-se possível desenvolver modelos capazes de analisar imagens de fundo de olho e realizar diagnósticos automáticos com alta precisão.  

Este projeto explora essa abordagem, utilizando duas estratégias complementares:  
1. A criação de uma **rede neural convolucional (CNN)** desenvolvida do zero.  
2. A utilização do **modelo YOLOv8n-cls** (Ultralytics), ajustado via *fine-tuning*.  

---

## Contexto  

A classificação automática de doenças oculares é um desafio técnico e científico, pois envolve imagens médicas complexas e sensíveis a variações sutis de textura e cor.  
Nesse contexto, **modelos pré-treinados** são vantajosos por já possuírem conhecimento visual amplo, enquanto **modelos customizados** permitem maior controle e interpretabilidade — especialmente relevantes em aplicações clínicas.  

O objetivo foi compreender **em quais cenários cada abordagem é mais apropriada** e como diferentes técnicas de treinamento, regularização e otimização afetam o desempenho final.  

---

## Objetivos do Projeto  

- Aplicar **técnicas de pré-processamento e normalização** das imagens clínicas.  
- Construir um **modelo CNN customizado** para classificação multiclasse (4 categorias).  
- Aplicar o **YOLOv8n-cls** em modo de **fine-tuning**, avaliando desempenho e eficiência.  
- Comparar os modelos com base em métricas clássicas de classificação (acurácia, recall, f1-score).  
- Discutir **quando utilizar um modelo pré-treinado** e **quando desenvolver um modelo próprio**.  

---

## Base de Dados  

Foi utilizada uma base clínica de imagens de fundo de olho, contendo quatro classes principais:  

- **Cataract**  
- **Diabetic Retinopathy**  
- **Glaucoma**  
- **Normal**  

As imagens foram pré-processadas e redimensionadas para **224 × 224 pixels**, com normalização dos valores de pixel entre 0 e 1.  
A divisão dos dados seguiu proporções de **70% para treino**, **15% para validação** e **15% para teste** 
---

## Modelos Implementados  

### 🟪 CNN Customizada  

- Arquitetura construída **do zero**, composta por blocos convolucionais com *Batch Normalization* e *Spatial Dropout*.  
- *Pooling 2D* entre blocos para redução progressiva de dimensionalidade.  
- Camadas densas finais com *Global Average Pooling* e *Dropout*.  
- **Data augmentation leve**, com rotações, zooms e ajustes de brilho e contraste, controlados para preservar as características clínicas.  
- Otimização com **AdamW** e agendamento de *learning rate* via **Cosine Decay Restarts**.  
- Treinamento com **80 épocas** e monitoramento de *early stopping* para evitar sobreajuste.  

### 🟩 YOLOv8n-cls  

- Modelo pré-treinado da **Ultralytics**, baseado em arquitetura **anchor-free**.  
- Treinamento com **100 épocas** e *batch size* de 16.  
- *Data augmentation* controlado (*RandAugment*) para robustez visual.  
- Otimizador **AdamW** com *warmup* de 3 épocas e **Automatic Mixed Precision (AMP)**, acelerando o processo em GPU.  
- Divisão de dados 90% treino e 10% teste.  

---

## Resultados  

### 🟦 CNN Customizada  

- **Acurácia de validação:** 82%  
- **Acurácia de teste:** 77%  
- **F1-score médio:** 0,77  
- **Melhor classe:** *Diabetic Retinopathy* (F1 = 0,88)  
- **Pior classe:** *Glaucoma* (F1 = 0,61)  

### 🟣 YOLOv8n-cls  

- **Acurácia Top-1:** 91,1%  
- **Acurácia de teste:** 90,1%  
- **Acurácia Top-5:** 100%  
- Alta estabilidade e baixo consumo de GPU.  

---

## Conclusões  

A análise comparativa entre os dois modelos mostrou que cada abordagem tem **vantagens e limitações** claras:  

- **YOLOv8n-cls (pré-treinado):**  
  - Excelente desempenho com baixo custo computacional.  
  - Ideal para aplicações rápidas e datasets reduzidos.  
  - Menor interpretabilidade, por se tratar de um modelo fechado e abstrato.  

- **CNN Customizada:**  
  - Permite entender e controlar cada etapa do aprendizado.  
  - Melhor para pesquisas e estudos acadêmicos que exigem explicabilidade.  
  - Mais sensível a hiperparâmetros e limitações de hardware.  

Em síntese:  
- **YOLOv8 → desempenho e eficiência.**  
- **CNN própria → controle e interpretabilidade.**  

Ambas as abordagens são complementares, e juntas evidenciam como o *deep learning* pode ser aplicado de forma flexível à análise de imagens médicas.  

---


