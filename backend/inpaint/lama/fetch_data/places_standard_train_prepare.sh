mkdir -p places_standard_dataset/train

# untar without folder structure
tar -xvf train_large_places365standard.tar -C places_standard_dataset/train

# create location config places.yaml
PWD=$(pwd)
DATASET=${PWD}/places_standard_dataset
PLACES=${PWD}/configs/training/location/places_standard.yaml

touch $PLACES
echo "# @package _group_" >> $PLACES
echo "data_root_dir: ${DATASET}/" >> $PLACES
echo "out_root_dir: ${PWD}/experiments/" >> $PLACES
echo "tb_dir: ${PWD}/tb_logs/" >> $PLACES
echo "pretrained_models: ${PWD}/" >> $PLACES
