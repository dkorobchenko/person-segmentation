
import imageio
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

import model

CKPT_DIR = 'logs/person_coco_aspp/ckpt'
INPUT_IMG_FPATH = 'img/me_512.png'
OUTPUT_IMG_FPATH = 'img/me_512_seg.png'

net = model.Model()

ckpt = tf.train.Checkpoint(model=net)
ckpt.restore(tf.train.latest_checkpoint(CKPT_DIR))

img = imageio.imread(INPUT_IMG_FPATH)
img = img[None, ...].astype(np.float32) / np.float32(255.)

logits = net(img, is_training=False)
img_out = tf.sigmoid(logits).numpy()
img_out = np.round(img_out[0, ...] * 255.).astype(np.uint8)

imageio.imsave(OUTPUT_IMG_FPATH, img_out)
