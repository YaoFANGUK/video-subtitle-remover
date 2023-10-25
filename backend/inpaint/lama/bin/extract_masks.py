import PIL.Image as Image
import numpy as np
import os


def main(args):
    if not args.indir.endswith('/'):
        args.indir += '/'
    os.makedirs(args.outdir, exist_ok=True)

    src_images = [
        args.indir+fname for fname in  os.listdir(args.indir)]

    tgt_masks = [
        args.outdir+fname[:-4] + f'_mask000.png' 
            for fname in  os.listdir(args.indir)]

    for img_name, msk_name in zip(src_images, tgt_masks):
        #print(img)
        #print(msk)

        image = Image.open(img_name).convert('RGB')
        image = np.transpose(np.array(image), (2, 0, 1))

        mask = (image == 255).astype(int)

        print(mask.dtype, mask.shape)


        Image.fromarray(
            np.clip(mask[0,:,:] * 255, 0, 255).astype('uint8'),mode='L'
        ).save(msk_name)




    '''
    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            mask = 

            Image.fromarray(
                np.clip(
                    cur_mask * 255, 0, 255).astype('uint8'),
                    mode='L'
                ).save(cur_basename + f'_mask{i:03d}.png')
    '''



if __name__ == '__main__':
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--indir', type=str, help='Path to folder with images')
    aparser.add_argument('--outdir', type=str, help='Path to folder to store aligned images and masks to')

    main(aparser.parse_args())
