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
  - `app.py` - Ponto de entrada da aplicação
- `requirements.txt` - Dependências Python

## Primeiros Passos
1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore o código:**
   - Veja `src/app.py` para a lógica principal da aplicação.
   - Explore `src/b3/` para módulos específicos de ativos.

## Requisitos
- Python 3.12+
- Veja `requirements.txt` para todas as dependências

## Uso

Você pode usar a aplicação pela linha de comando (CLI) tanto para treinamento quanto para predição:

### Treinar o Modelo B3

Treine o modelo e especifique o número de jobs paralelos (núcleos de CPU) para o treinamento:

```bash
python src/app.py train --n_jobs 8
```
- `train`: Executa o processo de treinamento.
- `--n_jobs 8`: (Opcional) Número de jobs paralelos para o treinamento. O padrão é 5 se não especificado.

### Predizer Usando o Modelo B3

Faça predições para um ticker específico:

```bash
python src/app.py predict --ticker BTCI11
```
- `predict`: Executa o processo de predição.
- `--ticker BTCI11`: (Obrigatório) O ticker que você deseja prever.

### Ajuda

Para ver todos os comandos e opções disponíveis:

```bash
python src/app.py --help
```

Ou para um comando específico:

```bash
python src/app.py train --help
python src/app.py predict --help
```

## Visão Geral da Arquitetura

O projeto é construído em torno de uma arquitetura modular de serviços para predição de ativos, com cada etapa do pipeline implementada como uma classe de serviço reutilizável. Essas classes podem ser usadas diretamente em Python ou acessadas via uma API REST.

### Classes de Serviço
- **B3DataLoadingService**: Carrega dados de mercado da B3.
- **B3ModelPreprocessingService**: Realiza pré-processamento, validação de features e geração de labels.
- **B3ModelTrainingService**: Treinamento do modelo e divisão dos dados.
- **B3ModelEvaluationService**: Avaliação e visualização do modelo.
- **B3ModelSavingService**: Persistência do modelo (salvar/carregar modelos).

## API REST (Flask)

Uma API REST baseada em Flask expõe todas as etapas de treinamento e predição como endpoints. A API é documentada com OpenAPI/Swagger (acesse `/swagger` com o servidor rodando).

### Principais Endpoints
- `POST /api/b3/load-data`: Carrega dados de mercado da B3
- `POST /api/b3/preprocess-data`: Pré-processa os dados carregados
- `POST /api/b3/split-data`: Divide os dados em treino/validação/teste
- `POST /api/b3/train-model`: Treina o modelo (com ajuste de hiperparâmetros)
- `POST /api/b3/evaluate-model`: Avalia o modelo treinado
- `POST /api/b3/save-model`: Salva o modelo treinado
- `POST /api/b3/complete-training`: Executa todo o pipeline de treinamento
- `POST /api/b3/predict`: Faz predições para um ticker específico
- `GET /api/b3/status`: Consulta o status atual do pipeline
- `GET /api/b3/training-status`: Consulta o status do treinamento
- `POST /api/b3/clear-state`: Limpa o estado do pipeline

### OpenAPI/Swagger
- A API é documentada com Swagger UI, disponível em `/swagger` com o servidor rodando.
- A especificação OpenAPI está em `/swagger/swagger.yml`.

## Exemplos de Uso

### Usando a API Python
```python
from b3.service.model import B3Model

model = B3Model()
model.train(model_dir="models", n_jobs=5, test_size=0.2, val_size=0.2)
predictions = model.predict(new_data, model_dir="models")
```

### Usando Serviços Individuais
```python
from b3.service.data.db.b3_featured.data_loading_service import B3DataLoadingService
from b3.service.model.model_preprocessing_service import B3ModelPreprocessingService
from b3.service.model.model_training_service import B3ModelTrainingService

data_service = B3DataLoadingService()
df = data_service.load_data()

preprocessing_service = B3ModelPreprocessingService()
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
- **Algoritmo**: Random Forest Classifier
- **Target**: Predição de direção do preço do ativo (alta/baixa)
- **Features**: Indicadores técnicos, movimentos de preço, padrões de volume
- **Avaliação**: Validação cruzada com divisões treino/validação/teste

### Requisitos de Dados de Treinamento
- **Fonte**: Dados históricos da B3 do asset-data-lake
- **Período**: Configurável (padrão: últimos 2 anos)
- **Registros Mínimos**: 1000+ por ativo para predições confiáveis
- **Frequência de Atualização**: Retreinar quando novos dados estiverem disponíveis

### Resultados de Avaliação do Modelo
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
sudo systemctl status asset-predict-model

# Ver logs
sudo journalctl -u asset-predict-model -f

# Reiniciar serviço
sudo systemctl restart asset-predict-model
```

## Versionamento e Persistência do Modelo

### Armazenamento do Modelo
- **Formato**: Modelos serializados com Joblib
- **Localização**: Diretório `models/`
- **Nomenclatura**: `b3_model.joblib`
- **Backup**: Versões anteriores arquivadas antes de atualizações

### Atualizações do Modelo
- **Trigger**: Retreinamento manual ou atualizações agendadas
- **Processo**: Carregar novos dados → Retreinar → Validar → Deploy
- **Rollback**: Versões anteriores do modelo disponíveis para rollback rápido

## Integração da API

### Integração com Frontend
A API do modelo integra com o frontend Angular:
- **Endpoint**: `POST /api/b3/predict`
- **Input**: `{"ticker": "PETR4"}`
- **Output**: Resultado da predição com confiança

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

### Health Checks
- **Endpoint**: `GET /api/b3/status`
- **Métricas**: Modelo carregado, conexão de dados, latência de predição
- **Alertas**: Serviço down, falhas de predição

### Logging
- **Nível**: INFO para operações normais, ERROR para falhas
- **Formato**: Logs estruturados JSON
- **Retenção**: 30 dias (configurável)

## Licença
MIT License

---
[Read in English](README.md)
