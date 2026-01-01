[Read in English](README.md)

# Asset Predict Model

Este projeto fornece ferramentas e modelos para previsão de preços de ativos, incluindo carregamento de dados, engenharia de features, treinamento de modelos e predição para ativos financeiros como B3.

## Funcionalidades
- Carregamento e pré-processamento de dados para ativos B3
- Engenharia de features e criação de labels
- Treinamento de modelos (Random Forest, etc.)
- Utilitários para predição e plotagem

## Estrutura do Projeto
- `src/` - Código fonte principal
  - `b3/` - Ferramentas para ativos B3
  - `models/` - Modelos pré-treinados
- `requirements.txt` - Dependências Python

## Primeiros Passos
1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore o código:**
   - Explore `src/b3/` para módulos específicos de ativos.

## Requisitos
- Python 3.12+
- Veja `requirements.txt` para todas as dependências

## Visão Geral da Arquitetura

O projeto é construído em torno de uma arquitetura modular de serviços para predição de ativos, com cada etapa do pipeline implementada como uma classe de serviço reutilizável. Essas classes podem ser usadas diretamente em Python ou acessadas via uma API REST.

### Classes de Serviço
- **B3DataLoadingService**: Carrega dados de mercado da B3.
- **B3ModelPreprocessingService**: Realiza pré-processamento, validação de features e geração de labels.
- **B3ModelTrainingService**: Treinamento do modelo e divisão dos dados.
- **B3ModelEvaluationService**: Avaliação e visualização do modelo.
- **B3ModelSavingService**: Persistência do modelo (salvar/carregar modelos).

## Modelos Suportados

O projeto suporta dois modelos de machine learning para previsão de preços de ativos:

### 1. Random Forest (rf)
- **Tipo**: Machine Learning Clássico (Método de Ensemble)
- **Algoritmo**: Random Forest Classifier (scikit-learn)
- **Caso de Uso**: Classificação da direção do preço do ativo (Compra/Venda/Manter)
- **Pontos Fortes**: 
  - Treinamento e inferência rápidos
  - Importância de features interpretável
  - Funciona bem com dados tabulares
  - Não requer sequências
- **Saída**: Predições de ação (classificação)
- **Formato de Armazenamento**: Modelos serializados com Joblib (`.joblib`)
- **Melhor Para**: Predições de ponto único, iterações rápidas, interpretabilidade

### 2. LSTM Multi-Task Learning (lstm/lstm_mtl)
- **Tipo**: Deep Learning (Rede Neural Recorrente)
- **Framework**: PyTorch
- **Arquitetura**: Long Short-Term Memory (LSTM) com Multi-Task Learning
- **Caso de Uso**: 
  - Predição de ação (classificação)
  - Previsão de retorno (regressão)
- **Pontos Fortes**:
  - Captura padrões temporais em séries temporais
  - Multi-task learning melhora a generalização
  - Pode prever ações e retornos de preço
- **Requisitos**: 
  - Sequências históricas (janela de lookback)
  - Dados sequenciais ordenados por tempo
- **Saída**: 
  - Predições de ação (classificação)
  - Predições de retorno (regressão)
  - Predições de preço (derivadas de retornos)
- **Formato de Armazenamento**: Dicionários de estado PyTorch (`.pt`)
- **Melhor Para**: Análise de séries temporais, captura de dependências temporais, predições multi-tarefa

## Modos de Execução

O projeto suporta dois modos de execução para treinamento de modelos:

### 1. Pipeline Completo de Treinamento (End-to-End)

Execute todo o pipeline de treinamento em uma única chamada de API. Este modo gerencia todas as etapas automaticamente:
- Carregamento de dados
- Pré-processamento de dados
- Divisão de dados
- Treinamento do modelo
- Avaliação do modelo
- Salvamento do modelo

**Endpoint**: `POST /api/b3/complete-pipeline`

**Exemplo de Requisição**:
```json
{
  "model_type": "rf",
  "model_dir": "models",
  "n_jobs": 5,
  "test_size": 0.2,
  "val_size": 0.2
}
```

**Para LSTM**:
```json
{
  "model_type": "lstm",
  "model_dir": "models",
  "lookback": 32,
  "horizon": 2,
  "epochs": 50,
  "batch_size": 256,
  "learning_rate": 0.001,
  "units": 64,
  "dropout": 0.4,
  "loss_weight_action": 20.0,
  "loss_weight_return": 50.0,
  "early_stopping_patience": 20,
  "gradient_clip_norm": 1.0,
  "weight_decay": 1e-4,
  "focal_loss_gamma": 3.0,
  "test_size": 0.2,
  "val_size": 0.2
}
```

**Benefícios**:
- Execução simples em uma única chamada
- Gerenciamento automático de estado
- Ideal para deployments em produção
- Gerencia todas as etapas do pipeline sequencialmente

### 2. Endpoints Independentes (Passo a Passo)

Execute cada etapa do pipeline independentemente. Isso fornece controle refinado sobre cada estágio:

**Etapa 1: Carregar Dados**
- **Endpoint**: `POST /api/b3/load-data`
- **Propósito**: Carregar dados de mercado da B3 da fonte de dados
- **Retorna**: Forma dos dados e informações das colunas

**Etapa 2: Pré-processar Dados**
- **Endpoint**: `POST /api/b3/preprocess-data`
- **Propósito**: Engenharia de features, validação e geração de labels alvo
- **Requer**: Dados devem ser carregados primeiro
- **Retorna**: Formas das features e distribuição dos targets

**Etapa 3: Dividir Dados**
- **Endpoint**: `POST /api/b3/split-data`
- **Propósito**: Dividir dados em conjuntos de treino/validação/teste
- **Requer**: Dados pré-processados
- **Parâmetros**: `model_type`, `test_size`, `val_size`
- **Retorna**: Tamanhos das divisões para cada conjunto

**Etapa 4: Treinar Modelo**
- **Endpoint**: `POST /api/b3/train-model`
- **Propósito**: Treinar o modelo selecionado (rf ou lstm)
- **Requer**: Dados divididos
- **Parâmetros**: `model_type`, `n_jobs` (para RF), parâmetros específicos do LSTM
- **Retorna**: Status do treinamento e informações do modelo

**Etapa 5: Avaliar Modelo**
- **Endpoint**: `POST /api/b3/evaluate-model`
- **Propósito**: Avaliar modelo treinado nos conjuntos de validação e teste
- **Requer**: Modelo treinado
- **Retorna**: Métricas de avaliação e caminhos de visualização

**Etapa 6: Salvar Modelo**
- **Endpoint**: `POST /api/b3/save-model`
- **Propósito**: Persistir modelo treinado no armazenamento
- **Requer**: Modelo treinado
- **Parâmetros**: `model_dir`, `model_name`
- **Retorna**: Caminho do arquivo do modelo

**Benefícios**:
- Controle refinado sobre cada etapa
- Capacidade de inspecionar resultados intermediários
- Útil para depuração e experimentação
- Pode modificar dados entre etapas
- Suporta fluxos de trabalho personalizados

**Exemplo de Fluxo de Trabalho**:
```python
import requests

base_url = "http://localhost:5000/api/b3"

# Etapa 1: Carregar dados
requests.post(f"{base_url}/load-data")

# Etapa 2: Pré-processar
requests.post(f"{base_url}/preprocess-data")

# Etapa 3: Dividir (especificar tipo de modelo)
requests.post(f"{base_url}/split-data", json={"model_type": "rf", "test_size": 0.2, "val_size": 0.2})

# Etapa 4: Treinar
requests.post(f"{base_url}/train-model", json={"model_type": "rf", "n_jobs": 5})

# Etapa 5: Avaliar
requests.post(f"{base_url}/evaluate-model")

# Etapa 6: Salvar
requests.post(f"{base_url}/save-model", json={"model_dir": "models"})
```

## API REST (Flask)

Uma API REST baseada em Flask expõe todas as etapas de treinamento e predição como endpoints. A API é documentada com OpenAPI/Swagger (acesse `/swagger` com o servidor rodando).

### Principais Endpoints

#### Pipeline Completo
- `POST /api/b3/complete-pipeline`: Executa o pipeline completo de treinamento (end-to-end)

#### Etapas Individuais
- `POST /api/b3/load-data`: Carrega dados de mercado da B3
- `POST /api/b3/preprocess-data`: Pré-processa os dados carregados
- `POST /api/b3/split-data`: Divide os dados em treino/validação/teste
- `POST /api/b3/train-model`: Treina o modelo (rf ou lstm)
- `POST /api/b3/evaluate-model`: Avalia o modelo treinado
- `POST /api/b3/save-model`: Salva o modelo treinado

#### Predição e Status
- `POST /api/b3/predict`: Faz predições para um ticker específico (suporta rf e lstm)
- `GET /api/b3/pipeline-status`: Consulta o status atual do pipeline
- `POST /api/b3/clear-state`: Limpa o estado do pipeline

### OpenAPI/Swagger
- A API é documentada com Swagger UI, disponível em `/swagger` com o servidor rodando.
- A especificação OpenAPI está em `/swagger/swagger.yml`.

## Exemplos de Uso

### Usando a API Python

```python
from b3.service.pipeline import B3Model

model = B3Model()
model.run(model_dir="models", n_jobs=5, test_size=0.2, val_size=0.2)
predictions = model.predict(new_data, model_dir="models")
```

### Usando Serviços Individuais

```python
from b3.service.data.db.b3_featured.data_loading_service import DataLoadingService
from b3.service.pipeline.model_preprocessing_service import PreprocessingService
from b3.service.pipeline.training.model_training_service import B3ModelTrainingService

data_service = DataLoadingService()
df = data_service.load_data()

preprocessing_service = PreprocessingService()
X, df_processed, y = preprocessing_service.preprocess_data(df)

training_service = B3ModelTrainingService()
X_train, X_val, X_test, y_train, y_val, y_test = training_service.split_data(X, y)
model = training_service.train_model(X_train, y_train)
```

### Usando a REST API
Inicie o servidor da API:
```bash
python src/b3/service/web_api/b3_model_api.py
```

Exemplo de requisição (usando `requests`):
```python
import requests
response = requests.post("http://localhost:5000/api/b3/load-data")
```

Veja a Swagger UI em [http://localhost:5000/swagger](http://localhost:5000/swagger) para documentação completa da API e testes interativos.

## Performance e Avaliação do Modelo

### Métricas do Modelo

#### Random Forest
- **Algoritmo**: Random Forest Classifier
- **Target**: Predição de direção do preço do ativo (Compra/Venda/Manter)
- **Features**: Indicadores técnicos, movimentos de preço, padrões de volume
- **Avaliação**: Validação cruzada com divisões treino/validação/teste
- **Ajuste de Hiperparâmetros**: RandomizedSearchCV com K-fold estratificado

#### LSTM Multi-Task Learning
- **Arquitetura**: LSTM PyTorch com multi-task learning
- **Tarefas**: 
  - Classificação: Predição de ação (Compra/Venda/Manter)
  - Regressão: Previsão de retorno
- **Features**: Indicadores técnicos sequenciais e dados de preço
- **Avaliação**: 
  - Métricas de classificação (acurácia, precisão, recall, F1)
  - Métricas de regressão (MAE, MSE, RMSE, R²)
- **Treinamento**: Otimizador Adam com taxa de aprendizado configurável

### Requisitos de Dados de Treinamento
- **Fonte**: Dados históricos da B3 do asset-data-lake
- **Período**: Configurável (padrão: últimos 2 anos)
- **Registros Mínimos**: 1000+ por ativo para predições confiáveis
- **Frequência de Atualização**: Retreinar quando novos dados estiverem disponíveis

### Resultados de Avaliação do Modelo

#### LSTM Multi-Task Learning - Performance Final

Após otimização sistemática e melhorias, o modelo LSTM alcançou os seguintes resultados:

**Performance de Classificação:**
- **Acurácia Geral**: 91.3% (melhorou de 59.1% - **+54% de melhoria**)
- **F1 Score Macro**: 0.411 (melhorou de 0.28 - **+47% de melhoria**)
- **F1 Score Ponderado**: 0.946 (excelente)
- **Classe Venda**: 
  - Precisão: 99.7% (excelente)
  - Recall: 91.5% (excelente)
  - F1: 0.954 (melhorou de 0.74 - **+29% de melhoria**)
- **Classe Compra**: 
  - Precisão: 8.0% (melhorou de 1.0% - **+700% de melhoria**)
  - Recall: 65.2% (moderado)
  - F1: 0.143
- **Classe Manter**: 
  - Precisão: 7.5% (melhorou de 4.0% - **+88% de melhoria**)
  - Recall: 77.2% (bom)
  - F1: 0.136

**Performance de Regressão:**
- **R² Score**: 0.9982 (99.82% da variância explicada - excelente)
- **Erro Absoluto Médio (MAE)**: 1.20 (validação), 1.25 (teste)
- **Raiz do Erro Quadrático Médio (RMSE)**: 5.47 (validação), 5.74 (teste)
- **Erro Percentual Absoluto Médio (MAPE)**: 8.98% (validação), 8.49% (teste)

**Redução de Overfitting:**
- **Gap de Overfitting**: 8.549 (reduzido de 172.720 - **95% de redução**)
- **Loss de Treinamento**: 253.35 (reduzido de 2.839.81 - **91% de redução**)
- **Loss de Validação**: 8.802.48 (reduzido de 175.560.08 - **95% de redução**)

**Principais Melhorias Alcançadas:**
1. ✅ **95% de redução no overfitting** através de técnicas de regularização
2. ✅ **54% de melhoria na acurácia de classificação** (59% → 91.3%)
3. ✅ **R² score de 99.8%** para regressão (excelente previsão de preços)
4. ✅ **700% de melhoria na precisão de Compra** (1.0% → 8.0%)
5. ✅ **Treinamento estável** com early stopping e taxa de aprendizado adaptativa

**Otimizações Técnicas Aplicadas:**
- Early stopping (patience=20) para prevenir overfitting
- Gradient clipping (max_norm=1.0) para estabilidade do treinamento
- Regularização L2 (weight_decay=1e-4) para reduzir complexidade do modelo
- Regularização Dropout (0.4) para melhorar generalização
- Focal Loss (gamma=3.0) para lidar com desbalanceamento de classes
- Balanceamento de pesos de loss (action:return = 20:50) para multi-task learning
- Agendamento de taxa de aprendizado (ReduceLROnPlateau) para melhor convergência
- Oversampling (WeightedRandomSampler) para classes minoritárias

#### Random Forest
- **Precisão**: Varia por ativo (tipicamente 55-70%)
- **Precision/Recall**: Balanceado para ambas as classes
- **Importância das Features**: Indicadores de momentum e volatilidade de preço mais significativos
- **Validação**: Validação cruzada de séries temporais para prevenir vazamento de dados

## Configuração de Ambiente

### Variáveis de Ambiente Necessárias

```bash
export MOTHERDUCK_TOKEN="seu_token_motherduck"
export environment="AWS"  # ou "LOCAL" para desenvolvimento local
export EC2_HOST="seu_ip_publico_ec2"  # para deploy de produção
```

### Configuração de Desenvolvimento Local

1. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

2. **Definir variáveis de ambiente**
   ```bash
   export MOTHERDUCK_TOKEN="seu_token"
   export environment="LOCAL"
   ```

3. **Executar o servidor da API**
   ```bash
   python src/web_api.py
   ```

## Deploy do Serviço

### Deploy de Produção (AWS EC2)

O serviço é projetado para rodar como um serviço systemd no AWS EC2:

```bash
# Deploy usando o script fornecido
sudo MOTHERDUCK_TOKEN=seu_token EC2_HOST=seu_ip bash deploy_asset_predict_model.sh
```

### Gerenciamento do Serviço

```bash
# Verificar status do serviço
sudo systemctl status asset-predict-pipeline

# Ver logs
sudo journalctl -u asset-predict-pipeline -f

# Reiniciar serviço
sudo systemctl restart asset-predict-pipeline
```

## Versionamento e Persistência do Modelo

### Armazenamento do Modelo
- **Formato**: 
  - Random Forest: Modelos serializados com Joblib (`.joblib`)
  - LSTM: Dicionários de estado PyTorch (`.pt`)
- **Localização**: Diretório `models/`
- **Nomenclatura**: `b3_model.joblib` ou `b3_lstm_mtl.pt`
- **Backup**: Versões anteriores arquivadas antes de atualizações

### Atualizações do Modelo
- **Trigger**: Retreinamento manual ou atualizações agendadas
- **Processo**: Carregar novos dados → Retreinar → Validar → Deploy
- **Rollback**: Versões anteriores do modelo disponíveis para rollback rápido

## Integração da API

### Integração com Frontend
A API do modelo integra com o frontend Angular:
- **Endpoint**: `POST /api/b3/predict`
- **Input**: `{"ticker": "PETR4", "model_type": "rf"}`
- **Output**: Resultado da predição com probabilidades e importância das features

### Integração com Data Lake
- **Fonte de Dados**: Conecta ao asset-data-lake para dados históricos
- **Tempo Real**: Usa os dados mais recentes disponíveis para predições
- **Cache**: Predições do modelo em cache para performance

## Considerações de Performance

### Inferência do Modelo
- **Latência**: < 100ms para predições únicas
- **Throughput**: 100+ predições/segundo
- **Memória**: ~500MB para modelo carregado
- **CPU**: Inferência single-threaded

### Escalabilidade
- **Horizontal**: Múltiplas instâncias atrás de load balancer
- **Vertical**: Instância EC2 t3.large suficiente para carga moderada
- **Cache**: Redis recomendado para cenários de alto tráfego

## Monitoramento e Logging

### Logging
- **Nível**: INFO para operações normais, ERROR para falhas
- **Formato**: Logs estruturados JSON
- **Retenção**: 30 dias (configurável)

## Licença
[MIT License](LICENSE)

---
[Read in English](README.md)
