import cv2
import argparse


filters = {
    'Ascent': {
        'upper': (255, 48, 185),
        'lower': (0, 0, 123)
    }
}

def make_mask(image, filter):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, filters[filter]['lower'], filters[filter]['upper'])
    return mask


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='path to input')
    ap.add_argument('-m', '--map_name', required=True, help='map name')
    ap.add_argument('-o', '--output', required=True, help='path to output')
    args = vars(ap.parse_args())

    img = cv2.imread(args['image'])

    mask = make_mask(img, args['map_name'])
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(args['output'], masked_img)