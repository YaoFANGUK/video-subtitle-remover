import os
import random

test_files_path           = os.path.abspath('') + '/places_standard_dataset/original/test/'
list_of_random_test_files = os.path.abspath('') + '/places_standard_dataset/original/test_random_files.txt'

test_files = [
    test_files_path + image for image in os.listdir(test_files_path)
]

print(f'Sampling 2000 images out of {len(test_files)} images in {test_files_path}' + \
      f'and put their paths to {list_of_random_test_files}')
print('Our training procedure will pick best checkpoints according to metrics, computed on these images.')

random.shuffle(test_files)
test_files_random = test_files[0:2000]
with open(list_of_random_test_files, 'w') as fw:
    for filename in test_files_random:
        fw.write(filename+'\n')
print('...done')


# --------------------------------

val_files_path           = os.path.abspath('') + '/places_standard_dataset/original/val/'
list_of_random_val_files = os.path.abspath('') + '/places_standard_dataset/original/val_random_files.txt'

val_files = [
    val_files_path + image for image in os.listdir(val_files_path)
]

print(f'Sampling 100 images out of {len(val_files)} in {val_files_path} ' + \
      f'and put their paths to {list_of_random_val_files}')
print('We use these images for visual check up of evolution of inpainting algorithm epoch to epoch' )

random.shuffle(val_files)
val_files_random = val_files[0:100]
with open(list_of_random_val_files, 'w') as fw:
    for filename in val_files_random:
        fw.write(filename+'\n')
print('...done')      

