import numpy as np

def run_mdnet(img_list, init_box, gt = None, save_fig='', display=False):

    # init box
    target_bbox = np.array(img_list)
    result = np.zeros(len(img_list), 4)
    rest_bb = np.zeros(len(img_list), 4)
    result[0] = target_bbox
    rest_bb[0] = target_bbox

    # init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda