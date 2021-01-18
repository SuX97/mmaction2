from .bsn import TEM, PEM
from pdb import set_trace as st
import numpy as np
import torch
import torch.nn as nn

class SnippetTEM(TEM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                raw_feature,
                label_action=None,
                label_start=None,
                label_end=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(raw_feature,
                                      label_action,
                                      label_start,
                                      label_end)

        return self.forward_test(raw_feature, video_meta)