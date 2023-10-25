mkdir -p places_standard_dataset/val_hires/
mkdir -p places_standard_dataset/visual_test_hires/


# randomly sample images for test and vis
OUT=$(python3 fetch_data/sampler.py)
echo ${OUT}

FILELIST=$(cat places_standard_dataset/original/test_random_files.txt)

for i in $FILELIST
do
    $(cp ${i} places_standard_dataset/val_hires/)
done

FILELIST=$(cat places_standard_dataset/original/val_random_files.txt)

for i in $FILELIST
do
    $(cp ${i} places_standard_dataset/visual_test_hires/)
done

