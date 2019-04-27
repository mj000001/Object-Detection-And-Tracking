from imagehash import phash
import numpy as np
from PIL import Image

def identify_obj(examplar_path, candidate_path, targets):
    # get the closest obj
    examplar = Image.open(examplar_path)
    candidate = Image.open(candidate_path)
    scores = []
    p_examplar = phash(examplar)

    for i, target in enumerate(targets):
        target = tuple(int(x) for x in target)
        img_cropped = candidate.crop(target)
        img_cropped.resize(examplar.size)
        p_img_cropped = phash(img_cropped)
        scores.append(p_img_cropped - p_examplar)

    scores.sort()
    return targets[np.argsort(scores)[0]]
