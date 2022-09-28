import cv2
import numpy as np
from quantize import quantize
from range_filter import make_mask

MIN_MATCH_COUNT = 2



def get_matching(temp_img, map_img):
    # find the keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(temp_img, None)
    kp2, des2 = sift.detectAndCompute(map_img, None)

    # akaze
    # akaze = cv2.AKAZE_create()
    # kp1, des1 = akaze.detectAndCompute(temp_img, None)
    # kp2, des2 = akaze.detectAndCompute(map_img, None)
    # des1 = np.float32(des1)
    # des2 = np.float32(des2)

    # orb
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(temp_img, None)
    # kp2, des2 = orb.detectAndCompute(map_img, None)
    # des1 = np.float32(des1)
    # des2 = np.float32(des2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find matches by knn which calculates point distance in 128 dim
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # find homography
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        h, w = temp_img.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)  # matched coordinates

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))

    out = map_img.copy()
    out = cv2.warpPerspective(out, M, (1024, 1024))
    return dst, M


def crop_and_fix_map(template, img):
    # read images
    template_color = cv2.imread(template)
    img_color = cv2.imread(img)

    template_img_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./template-gray.png", template_img_gray)

    # TODO: crop map area automatically (if possible)
    cropped_img_color = img_color[0:666, 444:1080, :]

    # # if you want to quantize
    # quantized_cropped_img_color = quantize(cropped_img_color, clusters=32)
    # quantized_cropped_img_gray = cv2.cvtColor(quantized_cropped_img_color, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./cropped-gray.png", quantized_cropped_img_gray)

    # TODO: map name
    mask = make_mask(cropped_img_color, 'Ascent')
    masked_cropped_img_color = cv2.bitwise_and(cropped_img_color, cropped_img_color, mask=mask)


    masked_cropped_img_gray = cv2.cvtColor(masked_cropped_img_color, cv2.COLOR_BGR2GRAY)

    # equalize histograms
    template_img_eq = cv2.equalizeHist(template_img_gray)
    img_eq = cv2.equalizeHist(masked_cropped_img_gray)

    _, M = get_matching(template_img_eq, img_eq)

    out = cv2.warpPerspective(cropped_img_color, M, (1024, 1024))
    return out


if __name__ == '__main__':
    import os
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--template', required=True, help='path to template image')
    ap.add_argument('-i', '--image_dir', required=True, help='path to input image directory')
    ap.add_argument('-o', '--output_dir', required=True, help='path to output directory')
    args = vars(ap.parse_args())

    template = args['template'] 
    img_dir = args['image_dir']
    out_dir = args['output_dir']

    for img in sorted(os.listdir(img_dir)):
        imgpath = os.path.join(img_dir, img)
        print(imgpath)
        m_img = crop_and_fix_map(template, imgpath)
        out = os.path.join(out_dir, img)
        cv2.imwrite(out, m_img)