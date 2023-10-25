import os
import imageio
from imageio_ffmpeg import write_frames

def write_frames(frame_path, fps, size, codec='libx264', quality=8):
    for filename in sorted(os.listdir(frame_path)):
        if not filename.endswith('.jpg'):
            continue
        yield imageio.imread(os.path.join(frame_path, filename))

def frames2video(frame_path, video_path, fps, show_progress=False, codec='libx264', quality=8):
    sample_frame = imageio.imread(os.path.join(frame_path, os.listdir(frame_path)[0]))
    height, width, _ = sample_frame.shape
    size = (width, height)
    video_dir = os.path.dirname(video_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    writer = imageio.get_writer(video_path, fps=fps, codec=codec, quality=quality)

    for frame in write_frames(frame_path, fps=fps, size=size, codec=codec, quality=quality):
        writer.append_data(frame)
        if show_progress:
            print(f'Frame {writer.get_length()} written')

    writer.close()

if __name__ == '__main__':
    frame_path = "/data0/datasets/davis/JPEGImages/480p/dog-gooses/"
    video_path = "./demo/dog-gooses/original_video.mp4"
    fps = 30
    frames2video(frame_path, video_path, fps, False)
