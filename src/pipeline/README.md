# ML Pipeline System

Sistema modular de pipeline para o projeto AutoValuePredict ML. O pipeline permite executar todas as etapas do projeto de forma orquestrada e incremental.

## Conceitos

### PipelineStep

Cada etapa do pipeline é uma classe que herda de `PipelineStep` e implementa:

- `execute(context)`: Executa a etapa e retorna o contexto atualizado
- `validate(context)`: Valida se os pré-requisitos estão satisfeitos
- `get_dependencies()`: Retorna lista de etapas que devem ser executadas antes

### MLPipeline

Orquestrador principal que:

- Gerencia a execução das etapas em ordem
- Valida dependências entre etapas
- Salva estado após cada etapa
- Fornece logging detalhado
- Permite executar etapas específicas ou ranges

## Uso Básico

### Executar Pipeline Completo

```python
from pipeline import MLPipeline
from pipeline.steps import LoadDataStep, CleanDataStep, SplitDataStep

pipeline = MLPipeline()
pipeline.add_step(LoadDataStep())
pipeline.add_step(CleanDataStep())
pipeline.add_step(SplitDataStep())

context = pipeline.execute()
```

### Executar Etapas Específicas

```python
# Executar apenas uma etapa específica
context = pipeline.execute(start_from="clean_data", stop_at="split_data")

# Pular etapas específicas
context = pipeline.execute(skip_steps=["validate_data_before"])
```

### Via Script

```bash
# Executar pipeline completo
python scripts/run_pipeline.py

# Listar etapas disponíveis
python scripts/run_pipeline.py --list-steps

# Ver status do pipeline
python scripts/run_pipeline.py --status

# Executar apenas Phase 2 (preprocessing)
python scripts/preprocess_data.py
```

## Estrutura do Contexto

O contexto é um dicionário que passa dados entre etapas:

```python
context = {
    'data': DataFrame,              # Dados principais
    'train_data': DataFrame,        # Dados de treino (após split)
    'val_data': DataFrame,         # Dados de validação
    'test_data': DataFrame,         # Dados de teste
    'X_train': DataFrame,           # Features de treino (após feature engineering)
    'y_train': Series,              # Target de treino
    'X_val': DataFrame,             # Features de validação
    'y_val': Series,                # Target de validação
    'X_test': DataFrame,            # Features de teste
    'y_test': Series,               # Target de teste
    'baseline_models': dict,        # Modelos baseline treinados
    'advanced_models': dict,        # Modelos avançados treinados
    'test_metrics': dict,           # Métricas no test set
    'segment_analysis': DataFrame,  # Análise por segmentos
    'error_analysis': DataFrame,    # Análise de erros
    'config': {},                   # Configurações
    'artifacts': {                  # Artefatos (modelos, validadores, etc.)
        'cleaner': DataCleaner,
        'splitter': DataSplitter,
        'feature_pipeline': FeaturePipeline,
        'split_files': {...}
    },
    'metadata': {                   # Metadados do pipeline
        'pipeline_name': str,
        'created_at': str,
        'steps_executed': [str]
    }
}
```

## Etapas Implementadas

### Phase 2: Data Preprocessing & Cleaning

- ✅ `LoadDataStep`: Carrega datasets enriquecidos
- ✅ `ValidateDataStep`: Valida qualidade dos dados
- ✅ `CleanDataStep`: Limpa e preprocessa dados
- ✅ `SplitDataStep`: Divide dados em train/val/test

### Phase 3: Feature Engineering

- ✅ `FeatureEngineeringStep`: Engenharia de features

### Phase 4: Baseline Models

- ✅ `TrainBaselineModelsStep`: Treina modelos baseline

### Phase 5: Advanced Models

- ✅ `TrainAdvancedModelsStep`: Treina modelos avançados (Random Forest, XGBoost, LightGBM)

### Phase 6: Model Optimization & Validation

- ✅ `EvaluateTestSetStep`: Avalia modelos no test set e compara com validation set
- ✅ `AnalyzeSegmentsAndErrorsStep`: Análise de performance por segmentos e análise de erros

### Phase 7: Model Persistence & Versioning

- ✅ `SaveModelWithVersioningStep`: Salva melhor modelo com versionamento e metadata

## Adicionando Novas Etapas

Para adicionar uma nova etapa:

1. Criar classe que herda de `PipelineStep`:

```python
from pipeline.base import PipelineStep
from typing import Dict, Any, List

class MinhaEtapaStep(PipelineStep):
    def __init__(self):
        super().__init__(name="minha_etapa")
    
    def get_dependencies(self) -> List[str]:
        return ['etapa_anterior']
    
    def validate(self, context: Dict[str, Any]) -> bool:
        # Validar pré-requisitos
        return 'data' in context
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Executar lógica da etapa
        # Atualizar context
        return context
```

2. Adicionar ao pipeline:

```python
pipeline.add_step(MinhaEtapaStep())
```

## Vantagens do Sistema

1. **Modular**: Cada etapa é independente e reutilizável
2. **Extensível**: Fácil adicionar novas etapas
3. **Rastreável**: Estado salvo após cada etapa
4. **Flexível**: Executar etapas específicas ou ranges
5. **Testável**: Cada etapa pode ser testada isoladamente
6. **Reprodutível**: Seeds e configurações centralizadas

## Exemplos de Uso

### Executar apenas limpeza de dados

```python
pipeline = MLPipeline()
pipeline.add_step(LoadDataStep())
pipeline.add_step(CleanDataStep())

context = pipeline.execute()
cleaned_data = context['data']
```

### Executar com validação customizada

```python
pipeline = MLPipeline()
pipeline.add_step(LoadDataStep())
pipeline.add_step(ValidateDataStep())
pipeline.add_step(CleanDataStep())

# Executar até validação
context = pipeline.execute(stop_at="validate_data_after")
```

### Acessar artefatos

```python
context = pipeline.execute()

# Acessar cleaner usado
cleaner = context['artifacts']['cleaner']

# Acessar arquivos gerados
split_files = context['artifacts']['split_files']
train_file = split_files['train']
```

## Estado do Pipeline

O estado do pipeline é salvo automaticamente em `data/processed/pipeline_state.json` após cada etapa, permitindo:

- Retomar execução de onde parou
- Rastrear quais etapas foram executadas
- Verificar configurações usadas

