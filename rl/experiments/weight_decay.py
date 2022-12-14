from typing import Union

from rl.experiments.optimization_base import OptimizationExperimentBase


class WeightDecayExperimentBase(OptimizationExperimentBase):
    WEIGHT_DECAY: Union[None, float] = None

    def __init__(self) -> None:
        super().__init__()

        self.OPTIMIZER_KWARGS["weight_decay"] = self.WEIGHT_DECAY


class WeightDecayConfig1Experiment(WeightDecayExperimentBase):
    WEIGHT_DECAY = 0.0  # 0%


class WeightDecayConfig2Experiment(WeightDecayExperimentBase):
    WEIGHT_DECAY = 1e-3  # 0.1%


class WeightDecayConfig3Experiment(WeightDecayExperimentBase):
    WEIGHT_DECAY = 2e-2  # 2%


class WeightDecayConfig4Experiment(WeightDecayExperimentBase):
    WEIGHT_DECAY = 5e-2  # 5%
