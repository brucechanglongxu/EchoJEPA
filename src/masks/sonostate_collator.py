import torch
from src.masks.multiseq_multiblock3d import MaskCollator


class SonoStateCollator:
    """Collator for SonoState training with consecutive clip pairs.

    Expects each dataset sample to contain 2 clips (num_clips=2).
    Separates clip_t (for JEPA masking) and clip_{t+1} (for state target),
    applies JEPA masking to clip_t, and batches clip_{t+1} separately.
    """

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=224,
        patch_size=16,
        tubelet_size=2,
    ):
        self.mask_collator = MaskCollator(
            cfgs_mask=cfgs_mask,
            dataset_fpcs=dataset_fpcs,
            crop_size=crop_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )
        self.mask_generators = self.mask_collator.mask_generators

    def step(self):
        self.mask_collator.step()

    def __call__(self, batch):
        """
        Args:
            batch: list of ([clip_t, clip_{t+1}], label, [indices_t, indices_{t+1}])
                   where each clip is (C, T, H, W) after transform.

        Returns:
            list of (jepa_collation, clip_next_tensor) per fpc group, where:
                jepa_collation = (collated_batch, masks_enc, masks_pred) for clip_t
                clip_next_tensor = (B, C, T, H, W) stacked clip_{t+1}
        """
        filtered = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            clips, label, clip_indices = sample
            fpc = len(clip_indices[-1])
            filtered[fpc].append(sample)

        fpc_collations = []
        for fpc in filtered:
            fpc_batch = filtered[fpc]
            if len(fpc_batch) == 0:
                continue
            batch_size = len(fpc_batch)

            clip_t_batch = []
            clip_next_batch = []
            for sample in fpc_batch:
                clips, label, clip_indices = sample
                clip_t_batch.append(([clips[0]], label, [clip_indices[0]]))
                clip_next_batch.append(clips[1])

            collated_t = torch.utils.data.default_collate(clip_t_batch)

            collated_masks_pred, collated_masks_enc = [], []
            for mask_generator in self.mask_generators[fpc]:
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)

            clip_next_tensor = torch.stack(clip_next_batch)

            fpc_collations.append(
                (collated_t, collated_masks_enc, collated_masks_pred, clip_next_tensor)
            )

        return fpc_collations
