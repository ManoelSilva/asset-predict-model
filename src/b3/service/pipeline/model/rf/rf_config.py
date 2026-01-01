from b3.service.pipeline.model.config import ModelConfig


class RandomForestConfig(ModelConfig):
    """Configuration class for Random Forest pipeline."""

    def __init__(self, n_jobs: int = 5, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs
        self.random_state = random_state
