import cv2
import imageio 
import os 
import shutil
from tqdm import tqdm
from glob import glob 
from os import path as osp

def frames2video(frames_list, video_path, fps=30, remove_tmp=False):
    # frames_list: frames dir or list of images.
    if isinstance(frames_list, str):
        frames_list = glob(f'{frames_list}/*.jpg')
    video_dir = os.path.dirname(video_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    # writer = imageio.get_writer(video_path, fps=fps)
    writer = imageio.get_writer(video_dir, fps=fps, plugin='ffmpeg')
    for frame in tqdm(frames_list, 'Export video'):
        if isinstance(frame, str):
            frame = imageio.imread(frame)
        else:
            # convert cv2 (rgb) to PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imageio.core.util.Array(frame)
        writer.append_data(frame)
    writer.close()
    print(f'find video at {video_path}.')
    if remove_tmp and isinstance(frames_list, str):
        shutil.rmtree(frames_list)

if __name__ == '__main__':
    video_path = './demo/soccerball/original_video.mp4'
    frame_path = '/data0/datasets/davis/JPEGImages/480p/soccerball'
    fps = 30
    frames2video(frame_path, video_path, fps, True)