import cv2 as cv
import numpy as np
import os
import re
import shutil
import random

def process_raw_data(raw_data_path, train_path, test_path, member_names, label_names, test_ratio):
    train_images_path = os.path.join(train_path, "input/images/")
    train_masks_path = os.path.join(train_path, "output/masks/")

    test_images_path = os.path.join(test_path, "input/images/")
    test_masks_path = os.path.join(test_path, "output/masks/")

    if os.path.exists(train_images_path):
        for file in os.listdir(train_images_path):
            os.remove(os.path.join(train_images_path, file))
    else:
        os.makedirs(train_images_path)
    
    if os.path.exists(train_masks_path):
        for file in os.listdir(train_masks_path):
            os.remove(os.path.join(train_masks_path, file))
    else:
        os.makedirs(train_masks_path)

    if os.path.exists(test_images_path):
        for file in os.listdir(test_images_path):
            os.remove(os.path.join(test_images_path, file))
    else:
        os.makedirs(test_images_path)
    
    if os.path.exists(test_masks_path):
        for file in os.listdir(test_masks_path):
            os.remove(os.path.join(test_masks_path, file))
    else:
        os.makedirs(test_masks_path)
    
    sample_num = 0

    for member in member_names:
        member_images_path = os.path.join(raw_data_path, member, "images")
        member_masks_path = os.path.join(raw_data_path, member, "masks")
        
        for img_name in os.listdir(member_images_path):
            img_num = int(img_name.split('.')[0])
            new_img_name = f"img_{sample_num:05}.jpg"
            
            shutil.copyfile(
                os.path.join(member_images_path, img_name),
                os.path.join(train_images_path, new_img_name) 
            )
            
            mask_list = []

            for label in label_names:
                temp_list = []

                for mask_name in os.listdir(member_masks_path):
                    mask_info = re.split("-|\.", mask_name)

                    if int(mask_info[1]) == img_num + 1 and mask_info[7] == label:
                        temp = np.load(os.path.join(member_masks_path, mask_name))
                        temp[temp > 0] = 255
                        temp = np.expand_dims(temp, axis=2)
                        temp_list.append(temp)

                if len(temp_list) > 1:
                    temp_list = np.concatenate(temp_list, axis=2)
                elif len(temp_list) == 1:
                    temp_list = temp_list[0]
                else:
                    temp = np.load(os.path.join(member_masks_path, mask_name))
                    temp = np.expand_dims(temp, axis=2)
                    temp_list = np.zeros_like(temp)
                
                temp_list = np.expand_dims(np.max(temp_list, axis=2), axis=2)
                mask_list.append(temp_list)
                        
            mask_list[1][mask_list[0] > 0] = 0

            mask_B = np.zeros_like(mask_list[0])
            mask_B[(mask_list[0] + mask_list[1]) == 0] = 255
            mask_list.append(mask_B)
            mask_list.reverse()
            mask = np.concatenate(mask_list, axis=2)

            # np.save(os.path.join(train_masks_path, f"mask_{sample_num:05}.npy"), mask)
            cv.imwrite(os.path.join(train_masks_path, f"mask_{sample_num:05}.jpg"), mask)

            sample_num += 1

    test_size = int(test_ratio*sample_num)

    test_inds = random.sample(range(sample_num), test_size)
    train_ind = 0
    test_ind = 0

    for ind in range(sample_num):
        img_name = "img_" + f"{ind:05}.jpg"
        img_path = os.path.join(train_images_path, img_name)

        mask_name = "mask_" + f"{ind:05}.jpg"
        mask_path = os.path.join(train_masks_path, mask_name)

        if ind in test_inds:
            shutil.move(img_path, os.path.join(test_images_path, f"img_{test_ind:05}.jpg"))
            shutil.move(mask_path, os.path.join(test_masks_path, f"mask_{test_ind:05}.jpg"))
            test_ind += 1
        else:
            shutil.move(img_path, os.path.join(train_images_path, f"img_{train_ind:05}.jpg"))
            shutil.move(mask_path, os.path.join(train_masks_path, f"mask_{train_ind:05}.jpg"))
            train_ind += 1