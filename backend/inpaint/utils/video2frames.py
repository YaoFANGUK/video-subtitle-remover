import cv2
import imageio 
import os 
import shutil
from tqdm import tqdm
from glob import glob 
from os import path as osp

def video2frames(video_path, frame_path):
    video = cv2.VideoCapture(video_path)
    os.makedirs(frame_path, exist_ok=True)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    initial_img = None
    for idx in tqdm(range(frame_num), 'Extract frames'):
        success, image = video.read()
        if idx == 0: initial_img = image.copy()
        assert success, 'extract the {}th frame in video {} failed!'.format(idx, video_path)
        cv2.imwrite("{}/{:05d}.jpg".format(frame_path, idx), image)
    return fps, initial_img

if __name__ == '__main__':
    video_path = './example/remove-anything-video/breakdance-flare/original_video.mp4'
    frame_path = './example/remove-anything-video/breakdance-flare/frames/'
    fps, initial_img = video2frames(video_path, frame_path)