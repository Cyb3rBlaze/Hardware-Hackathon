from PIL import Image
import torch
import os

import time

from clip_text_decoder.model import ImageCaptionInferenceModel


def current_milli_time():
    return round(time.time() * 1000)

files = os.listdir("images/")


model = ImageCaptionInferenceModel.download_pretrained("clip/model.pt")
model.to("cpu")

start = current_milli_time()
for file in files:
    image_path = "images/" + file
    image = Image.open(image_path)
    jpg_image = image.convert('RGB')
    # if img.shape > 3:
    #     image = images[0]
    # The beam_size argument is optional. Larger beam_size is slower, but has
    # slightly higher accuracy. Recommend using beam_size <= 3.
    caption = model(jpg_image, beam_size=1)
    print(file[:-4] + ": " + caption)
    print("Time elapsed: " + str(int((current_milli_time()-start)/1000)))