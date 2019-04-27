import numpy as np
from PIL import Image, ImageDraw
import os
import time
import matplotlib.pyplot as plt
import matplotlib
import tracking.siamrpn as siamrpn
matplotlib.use("TKAgg")

def tracking_obj(img_path, net_path, box, display=True, pause=0.001):
    img_files = os.listdir(img_path)
    img_files.sort()
    tracker = siamrpn.TrackerSiamRPN(net_path = net_path)
    frame_nums = len(img_files)
    boxs = np.zeros((frame_nums, 4))
    boxs[0] = box
    times = np.zeros(frame_nums)
    standard_size = (640, 360)
    for frame, img_name in enumerate(img_files):
        img = Image.open(os.path.join(img_path, img_name))
        if not img.mode == 'RGB':
            img = img.convert('RGB')

        start_time = time.time()
        standard_hight = 15
        if frame == 0:
            tracker.init(img, boxs[0])
            if display:
                dpi = 80.0
                figsize = (img.size[0]/dpi, img.size[1]/dpi)
                alpha = standard_hight / (img.size[0]/dpi)
                figsize = (standard_hight, img.size[1]/dpi * alpha)
                (fig_x, fig_y, fig_w, fig_h) = (100,500, figsize[0], figsize[1])
                fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
                thismanager = plt.get_current_fig_manager()
                thismanager.window.wm_geometry("+350+150")
                # thismanager.window.setGeometry(fig_x, fig_y, fig_w, fig_h)
                # fig.canvas.manager.window.Move(100,400)
                ax= plt.Axes(fig, [0,0,1,1])
                ax.set_axis_off()
                fig.add_axes(ax)
                im = ax.imshow(img, aspect='auto')

                # show first image
                gt_rect = plt.Rectangle(tuple(box[0:2]), box[2],box[3],
                                        linewidth=3, zorder=1, fill=False, color='red')
                ax.add_patch(gt_rect)

        else:
            boxs[frame] = tracker.update(img)
        end_time = time.time()
        times[frame] = end_time - start_time
        if display:
            im.set_data(img)
            gt_rect.set_xy(boxs[frame,:2])
            gt_rect.set_width(boxs[frame,2])
            gt_rect.set_height(boxs[frame,3])
            plt.pause(pause)
            plt.draw()