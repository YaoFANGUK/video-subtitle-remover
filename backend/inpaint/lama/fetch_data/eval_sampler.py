import os
import random

val_files_path           = os.path.abspath('') + '/places_standard_dataset/original/val/'
list_of_random_val_files = os.path.abspath('') + '/places_standard_dataset/original/eval_random_files.txt'
val_files      = [val_files_path + image for image in os.listdir(val_files_path)]

print(f'Sampling 30000 images out of {len(val_files)} images in {val_files_path}' + \
      f'and put their paths to {list_of_random_val_files}')

print('In our paper we evaluate trained models on these 30k sampled (mask,image) pairs in our paper (check Sup. mat.)')

random.shuffle(val_files)
val_files_random = val_files[0:30000]

with open(list_of_random_val_files, 'w') as fw:
    for filename in val_files_random:
        fw.write(filename+'\n')
print('...done')      

