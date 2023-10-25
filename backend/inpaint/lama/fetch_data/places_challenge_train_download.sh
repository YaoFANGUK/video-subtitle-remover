mkdir places_challenge_dataset


declare -a TARPARTS
for i in {a..z}
do
    TARPARTS[${#TARPARTS[@]}]="http://data.csail.mit.edu/places/places365/train_large_split/${i}.tar"
done    
ls
printf "%s\n" "${TARPARTS[@]}" > places_challenge_dataset/places365_train.txt

cd places_challenge_dataset/
xargs -a places365_train.txt -n 1 -P 8 wget [...]
ls *.tar | xargs -i tar xvf {}
