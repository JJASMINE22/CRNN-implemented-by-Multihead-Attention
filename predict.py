# -*- coding: UTF-8 -*-
'''
@Project ：CRNN
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import re
import torch
import numpy as np
import config as cfg
from torch import nn
from PIL import Image, ImageFont, ImageDraw
from net.networks import CRNN
from chars import characters
from _utils.generate import Generator
from _utils.utils import letterbox, remove_duplicate


if __name__ == '__main__':

    model = CRNN()

    try:
        ckpt = torch.load(os.path.join(cfg.ckpt_path, "模型参数路径"))
    except:
        raise ("Params' path wrong!")
    else:
        model.load_state_dict(ckpt['state_dict'])
        print("model successfully loaded, ctc loss {:.3f}".format(ckpt['loss']))

    while True:
        file_path = input('Input image filename:')
        try:
            image = Image.open(file_path)
        except FileNotFoundError:
            print('Can not open {:s}, try again !'.format(file_path))
            continue
        else:
            coordinates = file_path.split('/')[-1].split('-')[2]
            coordinates = map(int, re.sub(r"[^a-zA-Z0-9 ]", r" ", coordinates).split())
            # letterbox(image.crop((coordinates)), cfg.cropped_size)
            cropped_image = image.crop((coordinates)).resize(cfg.cropped_size, Image.BICUBIC)

            cropped_image = np.array(cropped_image)/127.5 - 1.
            cropped_image = np.clip(cropped_image, -1., 1.)
            cropped_image = np.transpose(cropped_image, [2, 0, 1])[np.newaxis, ...]

            logits = model(torch.tensor(cropped_image, dtype=torch.float))
            logits = logits.detach().permute(1, 0, 2).argmax(dim=-1).numpy()
            logits = remove_duplicate(logits, cfg.target_size)

            text = str()
            for _ in np.squeeze(logits):
                text += characters[_]

            font = ImageFont.truetype(font=cfg.font_path,
                                      size=np.floor(5e-2 * image.size[1] + 0.5).astype('int'))
            draw = ImageDraw.Draw(image)

            draw.text(np.array([image.size[0] // 2, image.size[1] // 2]),
                      text, fill=(255, 0, 0), font=font)

            del draw
            image.show()
