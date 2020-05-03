import os
import pickle
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
from datetime import datetime
import cv2
import glob
import shutil

# number of create StyleGAN image file
IMAGE_NUM = 30
# number of split frame two GAN files for changing image
SPLIT_NUM = 19
# image size
IMG_SIZE = 512
FILENAME_PREFIX = datetime.now().strftime("%Y%m%d%H%M%S")

def generate_image():
    """Generate GAN Image """
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    # 2019-03-08-stylegan-animefaces-network-02051-021980.pkl (https://www.gwern.net/Faces#anime-faces)
    with open("2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl", "rb") as f:
        _, _, Gs = pickle.load(f)
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState()

    # create latents stacks because of changing several latents vectors.
    for i in range(IMAGE_NUM):
        latents = rnd.randn(1, Gs.input_shape[1])
        if i == 0:
            stacked_latents = latents
        else:
            stacked_latents = np.vstack([stacked_latents, latents])

    for i in range(len(stacked_latents) - 1):
        latents_before = stacked_latents[i].reshape(1, -1)
        latents_after = stacked_latents[i + 1].reshape(1, -1)
        for j in range(SPLIT_NUM + 1):
            latents = latents_before + (latents_after - latents_before) * j / SPLIT_NUM
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
            os.makedirs("results", exist_ok=True)
            png_filename = os.path.join("results", FILENAME_PREFIX + "-{0:04d}-{1:04d}".format(i, j + 1) + ".png")
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


def create_mp4():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter('success_animation_' + FILENAME_PREFIX + '.mp4', fourcc, 10, (IMG_SIZE, IMG_SIZE))
    files = sorted(glob.glob('results/*.png'))
    for file in files:
        img = cv2.imread(file)
        if img is None:
            break
        print(file)
        video.write(img)
    video.release()


def clear_results():
    shutil.rmtree("results", ignore_errors=True)


if __name__ == "__main__":
    clear_results()
    generate_image()
    create_mp4()
