# Proof of Concept

- `jp2_vis.py`: visualize and annotate a .jp2 image
- `gen_subimages.py`: crop .jp2 image into many smaller ones
- `gen_multispectral_subimages.py`: combine several spectra and crop into subimages
- `gen_dataset.py`: save subimages as new .jp2 images, conserving their geotags
- `gen_labels_csv.py`: generate `labels.csv` with filename and True/False label for images based on how much of the mask
is painted. If a very small area of the image fits the label, it is true, otherwise it is false. This file can be used
in a custom dataset.

## TODO
- add truth layer, i.e. paint areas to label
- binary label generation from truth layer, e.g. if coverage > some percentage then yes else no
- train small classification model based on that generated data, from just one or a small number of images
    - [x] create dataset on hardcoded downloaded images
    - [ ] create dataset downloading images
