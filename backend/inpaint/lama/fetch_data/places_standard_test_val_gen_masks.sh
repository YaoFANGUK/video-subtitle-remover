mkdir -p places_standard_dataset/val/
mkdir -p places_standard_dataset/visual_test/


python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_512.yaml \
places_standard_dataset/val_hires/ \
places_standard_dataset/val/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_512.yaml \
places_standard_dataset/visual_test_hires/ \
places_standard_dataset/visual_test/