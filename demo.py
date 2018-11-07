import torch
import argparse
import numpy as np
import skimage.io as sio
from scipy.ndimage import zoom
import matplotlib.pylab as plt

from models.docs import DOCSNet

def load_image(filename, rgb_mean, input_size=512):
    im = sio.imread(filename)

    h, w = im.shape[:2]
    if h>=w and h>input_size:
        im=zoom(im,(input_size/h,input_size/h,1))
        h, w = im.shape[:2]
    elif w>=h and w>input_size:
        im=zoom(im,(input_size/w,input_size/w,1))
        h, w = im.shape[:2]
    
    pad_top = (input_size - h)//2
    pad_lef = (input_size - w )//2
    pad_bottom = input_size - h - pad_top
    pad_right  = input_size - w  - pad_lef
    pad = ((pad_top, pad_bottom), (pad_lef, pad_right), (0,0))
    im_padded = np.pad(im, pad, 'constant', constant_values=0)
    im_padded = im_padded.astype(np.float32)
    im_padded -= rgb_mean
    im_padded = torch.from_numpy(im_padded.transpose((2,0,1))).unsqueeze(0)

    return im, im_padded, pad

def remove_pad(a, pad):
    return a[pad[0][0]:a.shape[0]-pad[0][1],pad[1][0]:a.shape[1]-pad[1][1]]

def parse_args():
    parser = argparse.ArgumentParser(description='Deep Object Co-Segmentation (DOCS) Demo: '
						 'Given two input images, segments the common objects within two images.')
    parser.add_argument('gpu', metavar='GPU', type=int, help='gpu-id')
    parser.add_argument('image_a_path', metavar='IMG_A_PATH', help='path to first image.')
    parser.add_argument('image_b_path', metavar='IMG_B_PATH', help='path to second image.')
    parser.add_argument('snapshot', metavar='SNAPSHOT_PATH', help='paht to model\'s snapshot.')
    return parser.parse_args()

def main():
    args = parse_args()

    rgb_means = [122.67892, 116.66877, 104.00699]

    # set the device
    if not torch.cuda.is_available():
        raise RuntimeError('You need gpu for running this demo.')
    device = torch.device('cuda:%d'%args.gpu)
    print('Device:', device)

    print('Setting up the network...')
    state = torch.load(args.snapshot, map_location='cpu')
    net = DOCSNet(init_weights=False)
    net.load_state_dict(state['net_params'])
    net.eval()
    net.to(device)

    # load img_a 
    img_a, img_a_padded, pad_a= load_image(args.image_a_path, rgb_means)
        
    # load img_b
    img_b, img_b_padded, pad_b= load_image(args.image_b_path, rgb_means)

    img_a_padded = img_a_padded.to(device)
    img_b_padded = img_b_padded.to(device)
    out_a, out_b = net.forward(img_a_padded, img_b_padded, softmax_out=True)

    result_a = remove_pad(out_a[0,1].cpu().detach().numpy(), pad_a)>0.5
    result_b = remove_pad(out_b[0,1].cpu().detach().numpy(), pad_b)>0.5

    filtered_img_a = img_a * np.tile(result_a,(3,1,1)).transpose((1,2,0))
    filtered_img_b = img_b * np.tile(result_b,(3,1,1)).transpose((1,2,0))

    plt.subplot(2,2,1)
    plt.imshow(img_a)
    plt.subplot(2,2,2)
    plt.imshow(img_b)
    plt.subplot(2,2,3)
    plt.imshow(filtered_img_a)
    plt.subplot(2,2,4)
    plt.imshow(filtered_img_b)
    plt.show()

if __name__ == '__main__':
    main()
