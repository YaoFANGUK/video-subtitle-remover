# 0. folder preparation
mkdir -p places_standard_dataset/evaluation/hires/
mkdir -p places_standard_dataset/evaluation/random_thick_512/
mkdir -p places_standard_dataset/evaluation/random_thin_512/
mkdir -p places_standard_dataset/evaluation/random_medium_512/
mkdir -p places_standard_dataset/evaluation/random_thick_256/
mkdir -p places_standard_dataset/evaluation/random_thin_256/
mkdir -p places_standard_dataset/evaluation/random_medium_256/

# 1. sample 30000 new images
OUT=$(python3 fetch_data/eval_sampler.py)
echo ${OUT}

FILELIST=$(cat places_standard_dataset/original/eval_random_files.txt)
for i in $FILELIST
do
    $(cp ${i} places_standard_dataset/evaluation/hires/)
done


# 2. generate all kinds of masks

# all 512
python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_512.yaml \
places_standard_dataset/evaluation/hires \
places_standard_dataset/evaluation/random_thick_512/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_512.yaml \
places_standard_dataset/evaluation/hires \
places_standard_dataset/evaluation/random_thin_512/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_512.yaml \
places_standard_dataset/evaluation/hires \
places_standard_dataset/evaluation/random_medium_512/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
places_standard_dataset/evaluation/hires \
places_standard_dataset/evaluation/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
places_standard_dataset/evaluation/hires \
places_standard_dataset/evaluation/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
places_standard_dataset/evaluation/hires \
places_standard_dataset/evaluation/random_medium_256/
