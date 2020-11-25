import os.path as osp

from .rawframe_dataset import RawframeDataset
from .registry import DATASETS


@DATASETS.register_module
class AudioFeatureRawframeDataset(RawframeDataset):
    """Dataset that read both audio and visual, supporting both rawframes and
    videos."""

    def __init__(self, ann_file, pipeline, audio_prefix, **kwargs):
        self.audio_prefix = audio_prefix
        self.video_prefix = kwargs.pop('video_prefix', None)
        super().__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                if self.audio_prefix is not None:
                    audio_path = osp.join(self.audio_prefix,
                                          frame_dir) + '.npy'
                if self.video_prefix:
                    video_path = osp.join(self.video_prefix,
                                          frame_dir) + '.mp4'

                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                video_info['audio_path'] = audio_path
                video_info['filename'] = video_path
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    video_info['total_frames'] = int(line_split[idx])
                    idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert len(label), f'missing label in line: {line}'
                if self.multi_class:
                    assert self.num_classes is not None
                    video_info['label'] = label
                else:
                    assert len(label) == 1
                    video_info['label'] = label[0]
                video_infos.append(video_info)
        return video_infos
