import sys

import tqdm as tqdm_module

_OriginalTqdm = tqdm_module.tqdm


class PatchedTqdm(_OriginalTqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ncols", 100)
        super().__init__(*args, **kwargs)

    def close(self):
        super().close()
        # Force a newline after the bar finishes
        # sys.stderr.write("\n")
        sys.stderr.flush()


tqdm_module.tqdm = PatchedTqdm
