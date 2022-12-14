from typing import Union

from rl.experiments.optimization_base import OptimizationExperimentBase


class EMABetaExperimentBase(OptimizationExperimentBase):
    BETA_1: Union[None, float] = None
    BETA_2: Union[None, float] = None

    def __init__(self) -> None:
        super().__init__()

        self.OPTIMIZER_KWARGS["betas"] = (self.BETA_1, self.BETA_2)


class EMABetaConfig1Experiment(EMABetaExperimentBase):
    BETA_1 = 0.85
    BETA_2 = 0.99


class EMABetaConfig2Experiment(EMABetaExperimentBase):
    BETA_1 = 0.95
    BETA_2 = 0.99


class EMABetaConfig3Experiment(EMABetaExperimentBase):
    BETA_1 = 0.9
    BETA_2 = 0.95
