from .trunet_dataset import TruNetDataset
import mmcv
from pdb import set_trace as st
import copy
import random.shuffle as shuf
from math import floor, ceil
import numpy as np



class SnippetDataset(TruNetDataset):
    def __init__(self, snippet_length=7, pos_neg_ratio=1., *args, **kwargs):
        self.snippet_length = snippet_length
        self.pos_neg_ratio = pos_neg_ratio
        super().__init__(*args, **kwargs)
        self.snippet_infos = self.load_snippet_annotations()
        self.filter_neg()

    def load_snippet_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        snippet_infos = []
        for v_id, v_info in self.video_infos.items():
            for i in range(v_info['duration_seconds'] - self.snippet_length):
                snippet_infos[f'{v_id}_{i}'] = self._assign(i, i + self.snippet_length, v_info)

        return snippet_infos

    def filter_neg(self):
        '''Filter out too many negative snippets.
        '''
        self.pos_snippets = []
        self.neg_snippets = []
        self.snippet_infos = shuf(self.snippet_infos)
        for snippet in self.snippet_infos.items():
            if snippet['neg']:
                self.neg_snippets.append(snippet)
            else:
                self.pos_snippets.append(snippet)

        self.neg_snippets = shuf(self.neg_snippets)[:int(len(self.pos_snippets) * self.pos_neg_ratio)]

        self.snippet_infos = shuf(self.neg_snippets + self.pos_snippets)
            
def _assign(self, start, end, video_info):
    label = {
            'action'=np.zeros(self.snippet_length),
            'start': np.zeros(self.snippet_length),
            'end': np.zeros(self.snippet_length),
            'neg': True
            }
    for segment in video_info['annotations']:
        if max(start, segment[0]) <= min(end, segment[1]):
            # intersecting
            start_idx = int(ceil(max(start, segment[0])))
            end_idx = int(floor(min(end, segment[1])))
            label['start'][start_idx] = 1.0
            label['end'][end_idx] = 1.0
            label['action'][start_idx + 1 : end_idx] = 1.0
            label['neg'] = False
            return label
    return label

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.snippet_infos[idx])
        results['data_prefix'] = self.data_prefix
        results['snippet_length'] = self.snippet_length

        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.snippet_infos[idx])
        results['data_prefix'] = self.data_prefix
        results['snippet_length'] = self.snippet_length
        return self.pipeline(results)