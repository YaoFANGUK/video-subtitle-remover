import cv2
import numpy as np

def paste_object(source, source_mask, target, target_coords, resize_scale=1):
    assert target_coords[0] < target.shape[1] and target_coords[1] < target.shape[0]
    # Find the bounding box of the source_mask
    x, y, w, h = cv2.boundingRect(source_mask)
    assert h < source.shape[0] and w < source.shape[1]
    obj = source[y:y+h, x:x+w]
    obj_msk = source_mask[y:y+h, x:x+w]
    if resize_scale != 1:
        obj = cv2.resize(obj, (0,0), fx=resize_scale, fy=resize_scale)
        obj_msk = cv2.resize(obj_msk, (0,0), fx=resize_scale, fy=resize_scale)
        _, _, w, h = cv2.boundingRect(obj_msk)

    xt = max(0, target_coords[0]-w//2)
    yt = max(0, target_coords[1]-h//2)
    if target_coords[0]-w//2 < 0:
        obj = obj[:, w//2-target_coords[0]:]
        obj_msk = obj_msk[:, w//2-target_coords[0]:]
    if target_coords[0]+w//2 > target.shape[1]:
        obj = obj[:, :target.shape[1]-target_coords[0]+w//2]
        obj_msk = obj_msk[:, :target.shape[1]-target_coords[0]+w//2]
    if target_coords[1]-h//2 < 0:
        obj = obj[h//2-target_coords[1]:, :]
        obj_msk = obj_msk[h//2-target_coords[1]:, :]
    if target_coords[1]+h//2 > target.shape[0]:
        obj = obj[:target.shape[0]-target_coords[1]+h//2, :]
        obj_msk = obj_msk[:target.shape[0]-target_coords[1]+h//2, :]
    _, _, w, h = cv2.boundingRect(obj_msk)

    target[yt:yt+h, xt:xt+w][obj_msk==255] = obj[obj_msk==255]
    target_mask = np.zeros_like(target)
    target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
    target_mask[yt:yt+h, xt:xt+w][obj_msk==255] = 255
    
    return target, target_mask

if __name__ == '__main__':
    source = cv2.imread('example/boat.jpg')
    source_mask = cv2.imread('example/boat_mask_1.png', 0)
    target = cv2.imread('example/hippopotamus.jpg')
    print(source.shape, source_mask.shape, target.shape)
    
    target_coords = (700, 400)  # (x, y)
    resize_scale = 1
    target, target_mask = paste_object(source, source_mask, target, target_coords, resize_scale)
    cv2.imwrite('target_pasted.png', target)
    cv2.imwrite('target_mask.png', target_mask)
    print(target.shape, target_mask.shape)