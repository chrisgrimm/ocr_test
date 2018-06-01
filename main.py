from text_finder import TextFinder
from utils import horz_stack_images
import numpy as np
import cv2
import os

image_size = 128

def preprocess_images(images):
    processed_images = []
    for image in images:
        processed = cv2.resize(image, (image_size, image_size)) / 255.
        processed_images.append(processed)
    return np.array(processed_images)

def load_images(image_dir):
    files = os.listdir(image_dir)
    masks = sorted([f for f in files if 'mask' in f])
    texts = sorted([f for f in files if 'text' in f])
    text_images = preprocess_images([cv2.imread(os.path.join(image_dir, f)) for f in texts])
    mask_images = preprocess_images([cv2.imread(os.path.join(image_dir, f)) for f in masks])
    return text_images, mask_images

def load_images_test(image_dir):
    files = os.listdir(image_dir)
    texts = sorted([f for f in files if 'text' in f])
    text_images = preprocess_images([cv2.imread(os.path.join(image_dir, f)) for f in texts])
    return text_images

def make_batcher(image_dir):
    text_images, mask_images = load_images(image_dir)
    def get_batch(batch_size):
        idx = np.random.randint(0, len(text_images), size=batch_size)
        text_batch, mask_batch = text_images[idx, :, :, :], mask_images[idx, :, :, :]
        return text_batch, mask_batch
    return get_batch

image_dir = './images'
test_dir = './test_images'
def run(batch_size=32, disp_interval=10, save_interval=1000):
    i = 1
    finder = TextFinder()
    get_batch = make_batcher(image_dir)
    while True:
        text_images, mask_images = get_batch(batch_size)
        loss = finder.train(text_images, mask_images)
        print(i, loss)

        if i % disp_interval == 0:
            mask_image = finder.produce_mask([text_images[0]])[0]
            print(text_images[0].shape, mask_image.shape)
            image = 255*horz_stack_images(text_images[0], mask_image, background_color=(255,0,0))
            cv2.imwrite('./disp.png', image)

        if i % save_interval == 0:
            finder.save('./finder.ckpt')

        i += 1

#run()

def test(result_dir='./result_dir'):
    finder = TextFinder()
    finder.restore('./finder.ckpt')
    text_images = load_images_test(test_dir)
    print(text_images)
    mask_images = finder.produce_mask(text_images)
    for i, mask_image in enumerate(mask_images):
        cv2.imwrite(os.path.join(result_dir, '%s_mask.png' % i), 255*mask_image)
        result = 255*horz_stack_images(text_images[i], mask_image, background_color=(255, 0, 0))
        cv2.imwrite(os.path.join(result_dir, str(i))+'.png', result)

test()