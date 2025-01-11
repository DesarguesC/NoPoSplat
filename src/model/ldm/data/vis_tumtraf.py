import numpy as np
import cv2
import os, pdb

if __name__ == '__main__':
    pdb.set_trace()

    TYPE = 'train'

    base_path = f'./{TYPE.lower()}/images'
    video_base_path = f'./{TYPE.upper()}-video'
    if not os.path.exists(video_base_path): os.mkdir(video_base_path)

    folder = os.listdir(base_path)
    folder = [f for f in folder if '.' not in f]
    print(f'image forlders: {folder}')
    video_names = [f'{i}.mp4' for i in range(len(folder))]

    video_folder = [f'./{TYPE.upper()}-video/{i}' for i in range(len(folder))] # video names
    for x in video_folder:
        if not os.path.exists(x): os.mkdir(x)

    image_folders = [os.path.join(base_path, path) for path in folder]


    fps = 7  # 帧率

    # 获取图片列表并排序

    assert len(folder) == len(video_names) and len(folder) == len(image_folders)

    for i in range(len(folder)):
        now_video_path = video_folder[i]
        image_folder = image_folders[i]
        images = os.listdir(image_folder)
        images = [img for img in images if img.endswith((".png", ".jpg", ".jpeg"))]
        images.sort()
        # video_name = video_names[i]

        # 截取多个视频

        start_img, end_img = [0], []
        idx = 0
        img_idx_list = [int(x.split('_')[0]) for x in images]
        img_idx_list.sort()
        # pdb.set_trace()
        while idx < len(img_idx_list):
            f = img_idx_list[idx]
            idx += 1
            if idx == len(img_idx_list): continue
            f_ = img_idx_list[idx]
            if f_ - f <= 1: continue
            else:
                end_img.append(idx-1)
                start_img.append(idx)
        end_img.append(idx-1)
        if len(start_img) != len(end_img): pdb.set_trace()

        video_names_now = [f'{u}.mp4' for u in range(len(start_img))]
        for j in range(len(start_img)):
            v_now = video_names_now[j]
            # pdb.set_trace()
            images_frame = [images[q] for q in range(start_img[j], end_img[j]+1) if images[q].endswith((".png", ".jpg", ".jpeg"))]
            images_frame.sort()  # 确保按照文件名顺序合成视频

            # 获取第一张图片的尺寸
            first_image_path = os.path.join(image_folder, images_frame[0])
            frame = cv2.imread(first_image_path)
            height, width, layers = frame.shape

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式，可用 'MP4V' 或 'XVID'
            video = cv2.VideoWriter(f'{now_video_path}/{v_now}', fourcc, fps, (width, height))

            # 将每张图片写入视频
            for image in images_frame:
                img_path = os.path.join(image_folder, image)
                frame = cv2.imread(img_path)
                video.write(frame)

            # 释放资源
            video.release()
            cv2.destroyAllWindows()

            print(f"视频已保存为 {now_video_path}/{v_now}")
