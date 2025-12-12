## Model Checkpoints

This directory contains trained model checkpoints used in our experiments.

- Files prefixed with **`own_net`** correspond to **Light-U-Net**, our proposed model.
- Files prefixed with **`U_Net`** correspond to the baseline **U-Net** architecture.

### File Mapping
- `own_net_center.hdf5`
- `own_net_regions_50.hdf5`  
  → Light-U-Net (our model)

- `U_Net_center*.hdf5`
- `U_Net_regions*.hdf5`  
  → Standard U-Net