import cv2


def merge_video(video_input_path0, video_input_path1, video_output_path):
    """
    将两个视频文件安装水平方向合并
    """
    input_video_cap0 = cv2.VideoCapture(video_input_path0)
    input_video_cap1 = cv2.VideoCapture(video_input_path1)
    fps = input_video_cap1.get(cv2.CAP_PROP_FPS)
    size = (int(input_video_cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_video_cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2)
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    while True:
        ret0, frame0 = input_video_cap0.read()
        ret1, frame1 = input_video_cap1.read()
        if not ret1 and not ret0:
            break
        else:
            show = cv2.vconcat([frame0, frame1])
            video_writer.write(show)
    video_writer.release()


if __name__ == '__main__':
    v0_path = '../../test/test4.mp4'
    v1_path = '../../test/test4_no_sub(1).mp4'
    video_out_path = '../../test/demo.mp4'
    merge_video(v0_path, v1_path, video_out_path)
    # ffmpeg 命令 mp4转gif
    # ffmpeg -i demo3.mp4 -vf "scale=w=720:h=-1,fps=15,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 -r 15 -f gif output.gif
    # 宽度固定400，高度成比例：
    # ffmpeg - i input.avi -vf scale=400:-2
