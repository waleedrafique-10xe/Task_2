from ..base import Algorithm


class Qat(Algorithm):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError
