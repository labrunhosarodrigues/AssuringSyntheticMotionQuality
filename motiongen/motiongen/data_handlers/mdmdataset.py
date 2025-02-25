from mobert.dataset.primary_evaluator_dataset import HumanML3DDataset
from mobert.dataset.primary_evaluator_dataset import process_file
import numpy as np
import os
import torch


class MDMGeneratedDataset(HumanML3DDataset):
    """
    Dataset implementation specific for generated samples
    from Motion diffusion Model.
    """

    def load_motion_data(self, path, params):
        feature_data = []
        _, _, files = next(os.walk(path))
        for file in files:
            if '.npy' not in file:
                continue

            file_data = np.load(
                os.path.join(path, file),
                allow_pickle=True)[()]

            motions = file_data.get('motion', [])
            texts = file_data.get('text', [])
            i = 0
            for rep in range(file_data['num_repetitions']):
                for sample in range(file_data['num_samples']):
                    positions = np.transpose(motions[i], (2, 0, 1))
                    text = texts[i]
                    i += 1

                    key = f"{file}_sample{sample:02d}_rep{rep:02d}"
                    processed_data, _, _, _ = process_file(
                        positions, 0.002, **params
                    )
                    feature_data.append((key, processed_data, text))

        return feature_data

    def __len__(self):
        return len(self.feat_data)

    def __getitem__(self, idx):
        _, data, text = self.feat_data[idx]
        data_len = len(data)
        motion_chunks = []
        for i in range(0, data_len, self.chunk_size):
            if (len(motion_chunks) == self.pad_len):
                continue
            motion_chunk = data[i: i + self.chunk_size + self.overlap].flatten()  # noqa : E501
            if (i + self.chunk_size + self.overlap >= len(data)):
                continue
            motion_chunks.append(
                motion_chunk.reshape(1, self.chunk_size + self.overlap, -1)
            )

        motion_chunks = np.concatenate(motion_chunks, axis=0)
        motion_chunks = torch.from_numpy(
            (motion_chunks - self.mean) / self.std
        )

        motion_masks = torch.zeros(self.pad_len)
        motion_masks[:len(motion_chunks)] = 1
        return {
            "motion_chunks": torch.concatenate([
                motion_chunks,
                torch.zeros((
                    self.pad_len - len(motion_chunks),
                    self.chunk_size + self.overlap, motion_chunks.shape[2]))],
                dim=0),
            "motion_masks": motion_masks,
            "texts": text
        }
