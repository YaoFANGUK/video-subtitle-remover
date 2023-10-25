python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
celeba-hq-dataset/val_source_256/ \
celeba-hq-dataset/val_256/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
celeba-hq-dataset/val_source_256/ \
celeba-hq-dataset/val_256/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
celeba-hq-dataset/val_source_256/ \
celeba-hq-dataset/val_256/random_medium_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
celeba-hq-dataset/visual_test_source_256/ \
celeba-hq-dataset/visual_test_256/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
celeba-hq-dataset/visual_test_source_256/ \
celeba-hq-dataset/visual_test_256/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
celeba-hq-dataset/visual_test_source_256/ \
celeba-hq-dataset/visual_test_256/random_medium_256/
