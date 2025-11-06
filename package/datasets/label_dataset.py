from torchgeo.datasets import RasterDataset
import matplotlib.pyplot as plt


class LabelDataset(RasterDataset):
    """
    Initialize with a path or list of paths to a directory containing the
    masks.
    """

    filename_glob = "T*.tif"
    filename_regex = r"^.{6}"
    is_image = False

    def plot(self, sample, ax=None):
        # Reorder and rescale the image
        mask = sample['mask']

        # Plot the image
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(mask, cmap='Blues')

        if ax is None:
            return fig


# # How to use:
# img = Sentinel2(argv[1])
# mask = LabelDataset(argv[2])
# ds = img & mask
# g = torch.Generator().manual_seed(3)
# sampler = RandomGeoSampler(ds, size=256, length=3, generator=g)
# dataloader = DataLoader(ds, sampler=sampler, collate_fn=stack_samples)
