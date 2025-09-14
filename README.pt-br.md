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

---

## Licença
MIT License

---
[Read in English](README.md)
