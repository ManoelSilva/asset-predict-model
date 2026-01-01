[Read in English](MODEL_CARD.md)

# Model Card: Modelo LSTM Multi-Task Learning para Previsão de Ativos B3

## Detalhes do Modelo

### Informações do Modelo
- **Nome do Modelo**: Modelo B3 LSTM Multi-Task Learning (MTL)
- **Tipo de Modelo**: Deep Learning - Rede Neural Recorrente (LSTM)
- **Framework**: PyTorch
- **Versão**: 1.0
- **Data**: 01-2026

### Arquitetura do Modelo

**Tipo de Arquitetura**: Long Short-Term Memory (LSTM) com Multi-Task Learning

**Estrutura da Rede**:
```
Camada de Entrada: (batch_size, lookback=32, n_features)
    ↓
Camada LSTM: 
    - Tamanho Oculto: 64 unidades
    - Saída: (batch_size, 32, 64)
    - Usa último timestep: (batch_size, 64)
    ↓
Camada Dropout: 0.4 (taxa de dropout de 40%)
    ↓
    ├─→ Cabeça de Ação (Linear): 64 → 3 (classificação Compra/Venda/Manter)
    └─→ Cabeça de Retorno (Linear): 64 → 1 (regressão de retorno de preço)
```

**Componentes Principais**:
- **Camada LSTM**: LSTM de camada única com 64 unidades ocultas
- **Dropout**: Taxa de 0.4 aplicada após LSTM, antes das cabeças de tarefa
- **Cabeças de Tarefa**: Duas camadas lineares separadas para classificação e regressão
- **Total de Parâmetros**: Aproximadamente 4 × 64 × (64 + n_features) + parâmetros das cabeças de tarefa

## Dados de Treinamento

### Dataset
- **Fonte**: Dados históricos de mercado da B3 (Bolsa de Valores Brasileira)
- **Fonte de Dados**: asset-data-lake
- **Período**: Configurável (padrão: últimos 2 anos)
- **Registros Mínimos**: 1000+ por ativo para previsões confiáveis
- **Frequência de Atualização**: Retreinar quando novos dados estiverem disponíveis

### Pré-processamento de Dados
- **Engenharia de Features**: Indicadores técnicos, movimentos de preço, padrões de volume
- **Normalização**: StandardScaler aplicado às features de entrada
- **Construção de Sequências**: 
  - Janela de lookback: 32 timesteps
  - Horizonte: 2 timesteps à frente
  - Formato: `(n_samples, 32, n_features)`

### Divisão dos Dados
- **Conjunto de Treinamento**: 60% dos dados
- **Conjunto de Validação**: 20% dos dados
- **Conjunto de Teste**: 20% dos dados
- **Método de Divisão**: Divisão temporal (mantém ordem temporal)

### Distribuição de Classes
- **Classe Venda**: Classe majoritária (alta frequência)
- **Classe Compra**: Classe minoritária (baixa frequência)
- **Classe Manter**: Classe minoritária (baixa frequência)
- **Tratamento de Desbalanceamento de Classes**: 
  - Oversampling via WeightedRandomSampler
  - Focal Loss com γ=3.0
  - Pesos de classe opcionais

## Procedimento de Treinamento

### Configuração de Treinamento

**Hiperparâmetros**:
- **Janela de Lookback**: 32 timesteps
- **Horizonte**: 2 timesteps
- **Unidades Ocultas**: 64
- **Taxa de Dropout**: 0.4
- **Taxa de Aprendizado**: 1e-3 (0.001)
- **Tamanho do Lote**: 128
- **Épocas**: 50 (máximo, com early stopping)
- **Otimizador**: Adam
- **Decaimento de Peso (L2)**: 1e-4
- **Corte de Gradiente**: 1.0 (norma máxima)

**Função de Perda**:
```
Perda Total = w_action × FocalLoss + w_return × MSE

Onde:
- w_action = 20.0
- w_return = 50.0
- Focal Loss γ = 3.0
```

**Técnicas de Regularização**:
- **Dropout**: Taxa de 0.4
- **Decaimento de Peso**: 1e-4 (regularização L2)
- **Corte de Gradiente**: Previne gradientes explosivos
- **Early Stopping**: 
  - Paciência: 20 épocas
  - Delta Mínimo: 100.0
  - Monitora perda de validação

**Agendamento da Taxa de Aprendizado**:
- **Agendador**: ReduceLROnPlateau
- **Fator**: 0.5 (reduz pela metade)
- **Paciência**: 5 épocas
- **Taxa de Aprendizado Mínima**: 1e-6

### Processo de Treinamento
1. **Preparação de Dados**: Construção de sequências com janela de lookback
2. **Oversampling**: WeightedRandomSampler para desbalanceamento de classes
3. **Loop de Treinamento**: 
   - Passagem forward através do LSTM
   - Cálculo de perda multi-tarefa
   - Passagem backward com corte de gradiente
   - Atualização de parâmetros via otimizador Adam
4. **Validação**: Monitoramento de perda de validação e métricas
5. **Early Stopping**: Parar se não houver melhoria por 20 épocas
6. **Seleção de Modelo**: Restaurar melhor modelo baseado na perda de validação

## Avaliação

### Métricas de Avaliação

#### Desempenho de Classificação (Previsão de Ação)

**Métricas Gerais**:
- **Acurácia**: 91.3% (vs 33.3% baseline aleatório - melhoria de 2.74×)
- **F1 Score Macro**: 0.411
- **F1 Score Ponderado**: 0.946

**Desempenho por Classe**:

| Classe | Precisão | Recall | F1 Score |
|--------|----------|--------|----------|
| **Compra** | 8.0% | 65.2% | 0.143 |
| **Venda** | 99.7% | 91.5% | 0.954 |
| **Manter** | 7.5% | 77.2% | 0.136 |

**Interpretação**:
- **Classe Venda**: Desempenho excelente (99.7% precisão, 91.5% recall)
- **Classes Compra/Manter**: Baixa precisão devido ao desbalanceamento de classes, mas recall moderado
- **Geral**: Forte desempenho na classe majoritária, desafios com classes minoritárias

#### Desempenho de Regressão (Previsão de Retorno)

**Métricas**:

| Métrica | Validação | Teste |
|---------|-----------|-------|
| **R² Score** | 0.9982 (99.82%) | 0.9983 (99.83%) |
| **MAE** | 1.20 | 1.25 |
| **RMSE** | 5.47 | 5.74 |
| **MAPE** | 8.98% | 8.49% |

**Interpretação**:
- **R² = 99.8%**: Modelo explica 99.8% da variância nos retornos
- **MAE < 1.25**: Erro médio menor que 1.25 unidades de preço
- **MAPE < 9%**: Erro percentual médio abaixo de 9% (excelente para dados financeiros)

### Análise de Overfitting

**Perda de Treinamento vs Validação**:
- **Perda de Treinamento**: 253.35
- **Perda de Validação**: 8,802.48
- **Gap de Overfitting**: 8,549 (diferença de 35×)
- **Melhoria**: 95% de redução do gap inicial de 172,720

**Status**: 
- Melhoria significativa alcançada (95% de redução)
- Gap ainda presente mas gerenciável
- Early stopping e regularização funcionando efetivamente

## Resumo de Desempenho do Modelo

### Pontos Fortes
1. **Alta Acurácia Geral**: 91.3% de acurácia de classificação
2. **Regressão Excelente**: 99.8% R², <9% MAPE
3. **Forte Previsão de Venda**: 99.7% precisão, 91.5% recall
4. **Overfitting Reduzido**: 95% de redução no gap treinamento-validação
5. **Treinamento Estável**: Early stopping e regularização funcionando bem

### Limitações
1. **Precisão Compra/Manter**: Baixa (8.0% e 7.5%) devido ao desbalanceamento de classes
2. **Gap de Overfitting**: Ainda presente (35×), embora muito melhorado
3. **Desbalanceamento de Classes**: Classe Venda domina, afetando desempenho das classes minoritárias
4. **Trade-off**: Maior precisão para Compra/Manter reduziria recall

## Uso Pretendido

### Casos de Uso Principais
1. **Geração de Sinais de Trading**: Prever ações Compra/Venda/Manter para ativos B3
2. **Previsão de Retorno**: Prever retornos de preço para gestão de portfólio
3. **Avaliação de Risco**: Identificar sinais de venda potenciais com alta confiança
4. **Ferramenta de Pesquisa**: Analisar padrões temporais em séries temporais financeiras

### Usos Fora do Escopo
- **Não para**: Trading de alta frequência (projetado para previsões diárias)
- **Não para**: Geração garantida de lucro (previsões são probabilísticas)
- **Não para**: Outros mercados sem retreinar (treinado em dados B3)
- **Não para**: Previsão de longo prazo além de horizonte de 2 dias

## Considerações Éticas

### Viés e Justiça
- **Desbalanceamento de Classes**: Modelo tem melhor desempenho na classe majoritária (Venda)
- **Viés de Mercado**: Reflete padrões históricos de mercado (pode perpetuar vieses)
- **Recomendação**: Monitorar desempenho em diferentes condições de mercado

### Transparência
- **Arquitetura do Modelo**: Totalmente documentada
- **Processo de Treinamento**: Transparente e reproduzível
- **Métricas de Avaliação**: Métricas abrangentes fornecidas
- **Limitações**: Claramente declaradas

### Privacidade de Dados
- **Fonte de Dados**: Dados de mercado públicos (B3)
- **Sem Dados Pessoais**: Modelo usa apenas dados de mercado, sem informações pessoais
- **Conformidade**: Segue políticas de uso de dados

### Avisos de Risco
- **Risco Financeiro**: Previsões do modelo não são aconselhamento financeiro
- **Incerteza**: Mercados financeiros são inerentemente imprevisíveis
- **Validação**: Sempre validar previsões com expertise de domínio
- **Monitoramento**: Monitoramento contínuo necessário para drift do modelo

## Manutenção do Modelo

### Cronograma de Retreinamento
- **Frequência**: Quando novos dados estiverem disponíveis ou desempenho degradar
- **Gatilho**: Mudanças significativas nas condições de mercado
- **Validação**: Monitorar métricas de validação para drift

### Controle de Versão
- **Versionamento do Modelo**: Rastreado via MLflow
- **Armazenamento de Artefatos**: Modelos salvos como arquivos `.pt` (state dicts PyTorch)
- **Armazenamento de Scaler**: Arquivos `.joblib` separados para pré-processamento

### Monitoramento
- **Métricas para Monitorar**:
  - Acurácia de classificação
  - R² e MAPE de regressão
  - Precisão e recall por classe
  - Gap de perda treinamento-validação
- **Limiares de Alerta**: 
  - Queda de acurácia > 5%
  - R² abaixo de 0.95
  - Aumento do gap de overfitting > 50%

## Especificações Técnicas

### Requisitos de Hardware
- **Treinamento**: GPU recomendada (8GB+ VRAM)
- **Inferência**: CPU ou GPU
- **Memória**: 8GB+ RAM recomendado

### Dependências de Software
- **Python**: 3.12+
- **PyTorch**: Versão estável mais recente
- **NumPy**: Para operações numéricas
- **Pandas**: Para manipulação de dados
- **scikit-learn**: Para métricas e utilitários

### Tamanho do Modelo
- **Arquivo do Modelo**: ~500KB - 2MB (state dict PyTorch)
- **Arquivo Scaler**: ~10-50KB (joblib)
- **Total**: < 5MB

### Velocidade de Inferência
- **Tamanho do Lote**: 256 amostras
- **Latência**: < 10ms por amostra (em GPU)
- **Throughput**: ~1000 amostras/segundo (em GPU)

---

**Última Atualização**: 01-2026
**Versão do Modelo**: 1.0
**Status**: Pronto para Produção

