import random
import numpy as np
import os
import cv2
from PIL import Image

def create_noise(create_file_num, file_path):
    num_augmented_images = 4*create_file_num

    file_names = os.listdir(file_path)
    total_origin_image_num = len(file_names)
    cnt = 1

    for i in range(1, num_augmented_images):
        change_picture_index = random.randrange(1, total_origin_image_num-1)
        print(change_picture_index)
        print(file_names[change_picture_index])
        file_name = file_names[change_picture_index]

        origin_image_path = file_path + file_name
        print(origin_image_path)
        image = Image.open(origin_image_path)
        random_augment = random.randrange(1, 5)

        # 좌우 반전
        if random_augment == 1:
            print("Left-Right-invert")
            LR_inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            save_path = os.path.join(file_path, f'LR_inverted_{cnt}.png')
            LR_inverted_image.save(save_path)
        # 상하 반전
        elif random_augment == 2:
            print("Up-Down-invert")
            UD_inverted_imgae = image.transpose(Image.FLIP_TOP_BOTTOM)
            save_path = os.path.join(file_path, f'UD_inverted_{cnt}.png')
            UD_inverted_imgae.save(save_path)

        # 기울이기
        elif random_augment == 3:
            print("rotate")
            rotated_image = image.rotate(random.randint(-20, 20))
            save_path = os.path.join(file_path, f'rotated_{cnt}.png')
            rotated_image.save(save_path)

        elif random_augment == 4:
            img = cv2.imread(origin_image_path)
            print("noise")
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var**0.5 # 제곱근
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy_array = img + gauss
            noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
            save_path = os.path.join(file_path, f'noiseAdded_{cnt}.png')
            noisy_image.save(save_path)

        cnt += 1

# 각원본 개수만큼 진행
create_noise(329, 'train/dog/')
create_noise(205, 'train/elephant/')
create_noise(235, 'train/giraffe/')
create_noise(134, 'train/guitar/')
create_noise(151, 'train/horse/')
create_noise(245, 'train/house/')
create_noise(399, 'train/person/')