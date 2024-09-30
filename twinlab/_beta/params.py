from typeguard import typechecked

from ..params import TrainParams

from typeguard import typechecked
from ..params import TrainParams


@typechecked
class TrainParamsBeta(TrainParams):
    def __init__(self, warp_inputs: bool = False, **kwargs):  # New parameter!
        super().__init__(**kwargs)
        self.warp_inputs = warp_inputs

    def unpack_parameters(self):
        emulator_params, training_params = super().unpack_parameters()
        emulator_params["warp_inputs"] = self.warp_inputs
        return emulator_params, training_params
