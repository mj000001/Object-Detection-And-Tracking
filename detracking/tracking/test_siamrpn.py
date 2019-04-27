import siamrpn
import numpy as np
from PIL import Image, ImageDraw
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TKAgg")

def start_track(img_path, net_path, box, display=True, pause=0.001):
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


if __name__ == '__main__':
    video_set = {
        'CarScale':'/home/s07/wcx/zhengzhou/data/OTB/CarScale/img/',
        'BlueBody':'/home/s07/wcx/zhengzhou/data/OTB/BlurBody/img/',
        'Car1':'/home/s07/wcx/zhengzhou/data/OTB/Car1/img/',
        'MotorRolling':'/home/s07/wcx/zhengzhou/data/OTB/MotorRolling/img/',
        'Skiing':'/home/s07/wcx/zhengzhou/data/OTB/Skiing/img/',
        'Tiger1':'/home/s07/wcx/zhengzhou/data/OTB/Tiger1/img/',
        'Ironman':'/home/s07/wcx/zhengzhou/data/OTB/Ironman/img/',
        'Matrix': '/home/s07/wcx/zhengzhou/data/OTB/Matrix/img/',
        'Bolt': '/home/s07/wcx/zhengzhou/data/OTB/Bolt/img/',
        'ant1':'/home/s07/wcx/ants/ants1_img/',
        'ant3': '/home/s07/wcx/ants/ants3_img/',
        'Freeman4': '/home/s07/wcx/zhengzhou/data/OTB/Freeman4/img/',
        'boat1':'/home/s07/wcx/vot-tire/boat1/img/',
        'quadrocopter':'/home/s07/wcx/vot-tire/quadrocopter/img/',
        'Basketball':'/home/s07/wcx/zhengzhou/data/OTB/Basketball/img/',
        'Biker':'/home/s07/wcx/zhengzhou/data/OTB/Biker/img/',
        'Bird1':'/home/s07/wcx/zhengzhou/data/OTB/Bird1/img/',
        'Bird2':'/home/s07/wcx/zhengzhou/data/OTB/Bird2/img/',
        'BlurCar1':'/home/s07/wcx/zhengzhou/data/OTB/BlurCar1/img/',
        'BlurCar2': '/home/s07/wcx/zhengzhou/data/OTB/BlurCar2/img/',
        'BlurCar3': '/home/s07/wcx/zhengzhou/data/OTB/BlurCar3/img/',
        'BlurCar4': '/home/s07/wcx/zhengzhou/data/OTB/BlurCar4/img/',
        'BlurFace': '/home/s07/wcx/zhengzhou/data/OTB/BlurFace/img/',
        'BlurOwl':'/home/s07/wcx/zhengzhou/data/OTB/BlurOwl/img/',
        'Board':'/home/s07/wcx/zhengzhou/data/OTB/Board/img/',
        'Bolt2':'/home/s07/wcx/zhengzhou/data/OTB/Bolt2/img/',
        'Box':'/home/s07/wcx/zhengzhou/data/OTB/Box/img/',
        'Boy':'/home/s07/wcx/zhengzhou/data/OTB/Boy/img/',
        'Car24':'/home/s07/wcx/zhengzhou/data/OTB/Car24/img/',
        'Car2': '/home/s07/wcx/zhengzhou/data/OTB/Car2/img/',
        'Car4': '/home/s07/wcx/zhengzhou/data/OTB/Car4/img/',
        'CarDark': '/home/s07/wcx/zhengzhou/data/OTB/CarDark/img/',
        'ClifBar': '/home/s07/wcx/zhengzhou/data/OTB/ClifBar/img/',
        'Coupon':'/home/s07/wcx/zhengzhou/data/OTB/Coupon/img/',
        'Shaking':'/home/s07/wcx/zhengzhou/data/OTB/Shaking/img/',
        'Surfer':'/home/s07/wcx/zhengzhou/data/OTB/Surfer/img/',
        'Rubik':'/home/s07/wcx/zhengzhou/data/OTB/Rubik/img/',
        'Woman':'/home/s07/wcx/zhengzhou/data/OTB/Woman/img/',
        'Crowds':'/home/s07/wcx/zhengzhou/data/OTB/Crowds/img/',
        'Doll':'/home/s07/wcx/zhengzhou/data/OTB/Doll/img/',
        'Dog':'/home/s07/wcx/zhengzhou/data/OTB/Dog/img/',
        'FaceOcc1':'/home/s07/wcx/zhengzhou/data/OTB/FaceOcc1/img/',
        'FaceOcc2': '/home/s07/wcx/zhengzhou/data/OTB/FaceOcc2/img/',
        'Dancer':'/home/s07/wcx/zhengzhou/data/OTB/Dancer/img/',
        'Human2': '/home/s07/wcx/zhengzhou/data/OTB/Human2/img/',
        'Human3': '/home/s07/wcx/zhengzhou/data/OTB/Human3/img/',
        'Human8':'/home/s07/wcx/zhengzhou/data/OTB/Human8/img/',
        'Vase':'/home/s07/wcx/zhengzhou/data/OTB/Vase/img/',
        'Fish':'/home/s07/wcx/zhengzhou/data/OTB/Fish/img/',
        'MountainBike':'/home/s07/wcx/zhengzhou/data/OTB/MountainBike/img/',
        'Skating2':'/home/s07/wcx/zhengzhou/data/OTB/Skating2/img/',
        'Soccer':'/home/s07/wcx/zhengzhou/data/OTB/Soccer/img/',
        'Dudek':'/home/s07/wcx/zhengzhou/data/OTB/Dudek/img/',
        'Human7':'/home/s07/wcx/zhengzhou/data/OTB/Human7/img/',
        'Subway':'/home/s07/wcx/zhengzhou/data/OTB/Subway/img/',
        'DragonBaby':'/home/s07/wcx/zhengzhou/data/OTB/DragonBaby/img/',
        'Singer2':'/home/s07/wcx/zhengzhou/data/OTB/Singer2/img/',
        'Trellis':'/home/s07/wcx/zhengzhou/data/OTB/Trellis/img/',
        'Freeman1':'/home/s07/wcx/zhengzhou/data/OTB/Freeman1/img/',
        'Freeman3': '/home/s07/wcx/zhengzhou/data/OTB/Freeman3/img/',
        'Diving':'/home/s07/wcx/zhengzhou/data/OTB/Diving/img/',
        'RedTeam':'/home/s07/wcx/zhengzhou/data/OTB/RedTeam/img/',
        'Liquor':'/home/s07/wcx/zhengzhou/data/OTB/Liquor/img/',
        'Couple':'/home/s07/wcx/zhengzhou/data/OTB/Couple/img/',
        'Lemming':'/home/s07/wcx/zhengzhou/data/OTB/Lemming/img/',
        'Mhyang':'/home/s07/wcx/zhengzhou/data/OTB/Mhyang/img/',
        'Jogging':'/home/s07/wcx/zhengzhou/data/OTB/Jogging/img/',
        'Jump':'/home/s07/wcx/zhengzhou/data/OTB/Jump/img/',
        'Panda':'/home/s07/wcx/zhengzhou/data/OTB/Panda/img/',
        'Sylvester':'/home/s07/wcx/zhengzhou/data/OTB/Sylvester/img/',
        'Twinnings':'/home/s07/wcx/zhengzhou/data/OTB/Twinnings/img/',
        'Human4': '/home/s07/wcx/zhengzhou/data/OTB/Human4/img/',
        'KiteSurf': '/home/s07/wcx/zhengzhou/data/OTB/KiteSurf/img/',
        'Crossing': '/home/s07/wcx/zhengzhou/data/OTB/Crossing/img/',
        'Football1': '/home/s07/wcx/zhengzhou/data/OTB/Football1/img/',
        'Human6': '/home/s07/wcx/zhengzhou/data/OTB/Human6/img/',
        'BlurBody': '/home/s07/wcx/zhengzhou/data/OTB/BlurBody/img/',
        'David': '/home/s07/wcx/zhengzhou/data/OTB/David/img/',
        'Football': '/home/s07/wcx/zhengzhou/data/OTB/Football/img/',
        'Skater2': '/home/s07/wcx/zhengzhou/data/OTB/Skater2/img/',
        'Jumping': '/home/s07/wcx/zhengzhou/data/OTB/Jumping/img/',
        'David2': '/home/s07/wcx/zhengzhou/data/OTB/David2/img/',
        'Gym': '/home/s07/wcx/zhengzhou/data/OTB/Gym/img/',
        'Walking': '/home/s07/wcx/zhengzhou/data/OTB/Walking/img/',
        'Trans': '/home/s07/wcx/zhengzhou/data/OTB/Trans/img/',
        'Girl2': '/home/s07/wcx/zhengzhou/data/OTB/Girl2/img/',
        'Man': '/home/s07/wcx/zhengzhou/data/OTB/Man/img/',
        'Skating1': '/home/s07/wcx/zhengzhou/data/OTB/Skating1/img/',
        'Human9': '/home/s07/wcx/zhengzhou/data/OTB/Human9/img/',
        'FleetFace': '/home/s07/wcx/zhengzhou/data/OTB/FleetFace/img/',
        'Human5': '/home/s07/wcx/zhengzhou/data/OTB/Human5/img/',
        'Dancer2': '/home/s07/wcx/zhengzhou/data/OTB/Dancer2/img/',
        'Warking2': '/home/s07/wcx/zhengzhou/data/OTB/Warking2/img/',
        'Girl': '/home/s07/wcx/zhengzhou/data/OTB/Girl/img/',
        'Tiger2': '/home/s07/wcx/zhengzhou/data/OTB/Tiger2/img/',
        'David3': '/home/s07/wcx/zhengzhou/data/OTB/David3/img/',
        'Singer1': '/home/s07/wcx/zhengzhou/data/OTB/Singer1/img/',
        'Toy': '/home/s07/wcx/zhengzhou/data/OTB/Toy/img/',
        'Coke': '/home/s07/wcx/zhengzhou/data/OTB/Coke/img/',
        'Deer': '/home/s07/wcx/zhengzhou/data/OTB/Deer/img/',
        'Dog1': '/home/s07/wcx/zhengzhou/data/OTB/Dog1/img/',
        'Suv': '/home/s07/wcx/zhengzhou/data/OTB/Suv/img/',
        'Skater': '/home/s07/wcx/zhengzhou/data/OTB/Skater/img/',
        'hiding':'/home/s07/vot-tire/hiding/img',
        'crouching':'/home/s07/vot-tire/crouching/img',
        'car1':'/home/s07/vot-tire/car1/img',
        'crowd':'/home/s07/vot-tire/crowd/img',
        'boat2':'/home/s07/vot-tire/boat2/img',
        'car2':'/home/s07/vot-tire/car2/img',
        'birds':'/home/s07/vot-tire/birds/img',
        'quadrocopter2':'/home/s07/vot-tire/quadrocopter2/img',
        'jacket':'/home/s07/vot-tire/jacket/img',
        'tree2':'/home/s07/vot-tir2016/tree2/img',
        'street': '/home/s07/vot-tir2016/street/img',
        'ragged': '/home/s07/vot-tir2016/ragged/img',
        'tree1': '/home/s07/vot-tir2016/tree1/img',
        'excavator':'/home/s07/vot-tir2016/excavator/img',
        'saturated':'/home/s07/vot-tir2016/saturated/img'
    }
    start_pos = {
        'CarScale':[6,166,42,26],
        'BlueBody':[400,48,87,319],
        'Car1':[23,88,66,55],
        'tree1':[43, 212, 20,56],
        'MotorRolling':[117,68,122,125],
        'Skiing':[446,181,29,26],
        'Tiger1':[232,88,76,84],
        'Ironman':[206,85,49,57],
        'Matrix':[331,39,38,42],
        'Bolt':[336,165,26,61],
        'ant1':[125,458,15,68],
        'ant3': [625,676,5,15],
        'Freeman4':[125,86,15,16],
        'boat1':[255,44,21,20],
        'quadrocopter':[226,319,86,27],
        'Basketball':[198,214,34,81],
        'Biker':[262,94,16,26],
        'Bird1':[450,91,31,37],
        'Bird2': [82,218,69,73],
        'BlurCar1':[250,168,106,105],
        'BlurCar2':[227,207,122,99],
        'BlurCar3':[228,236,80,64],
        'BlurCar4':[197,203,170,149],
        'BlurFace':[246,226,94,114 ],
        'BlurOwl':[352,197,56,100],
        'Board':[57,156,198,173],
        'Bolt2': [269,75,34,64],
        'Box':[478,143,80,111],
        'Boy':[288,143,35,42],
        'Car24':[164,121,27,24],
        'Car2':[76,79,64,52],
        'Car4':[70,51,107,87],
        'CarDark':[73,126,29,23],
        'ClifBar':[143,125,30,54],
        'Coupon':[144,63,57,89],
        'Shaking':[225,135,61,71] ,
        'Surfer':[275,137,23,26],
        'Rubik':[276,161,73,74],
        'Woman':[213,121,21,95],
        'Crowds':[561,311,22,51],
        'Doll':[146,150,32,73],
        'Dog':[74,86,56,48],
        'FaceOcc1':[118,69,114,162],
        'FaceOcc2':[118,57,82,98],
        'Dancer':[176,75,47,102],
        'Human2':[198,249,95,325],
        'Human3':[264,311,37,69],
        'Human8':[110,101,30,91],
        'Vase': [139,90,45,59],
        'Fish':[134,55,60,88],
        'MountainBike':[319,185,67,56],
        'Skating2':[289,67,64,236],
        'Soccer':[302,135,67,81],
        'Dudek':[123,87,132,176],
        'Human7':[110,111,37,116],
        'Subway':[16,88,19,51],
        'DragonBaby':[160,83,56,65],
        'Singer2':[298,149,67,122],
        'Trellis':[146,54,68,101],
        'Freeman1':[253,66,23,28],
        'Freeman3':[245,64,12,13],
        'Diving':[177,51,21,129],
        'RedTeam':[197,87,38,18],
        'Liquor':[256,152,73,210],
        'Couple':[51,47,25,62],
        'Lemming':[40,199,61,103],
        'Mhyang':[84,53,62,70],
        'Jogging':[111,98,25,101],
        'Jump':[136,35,52,182],
        'Panda':[58,100,28,23],
        'Sylvester':[122,51,51,61],
        'Twinnings':[125,162,74,55],
        'Human4':[99,237,27,82],
        'KiteSurf':[204,41,23,30],
        'Crossing':[205,151,17,50],
        'Football1':[153,105,26,43],
        'Human6':[340,358,18,55],
        'saturated':[287, 175, 34, 88],
        'BlurBody':[400,48,87,319],
        'David':[129,80,64,78],
        'Football':[310,102,39,50],
        'Skater2':[163,44,47,164],
        'Jumping':[147,110,34,33],
        'David2':[141,73,27,34],
        'Gym':[167,69,24,127],
        'Walking':[692,439,24,79],
        'Trans':[196,51,139,194],
        'Girl2':[294,135,44,171],
        'Man':[69,48,26,39],
        'Skating1':[162,188,34,84],
        'Human9':[93,113,34,109],
        'FleetFace':[405,256,122,148],
        'Human5':[326,414,15,42],
        'Dancer2':[150,53,40,148],
        'Warking2':[130,132,31,115],
        'Girl':[57,21,31,45],
        'Tiger2':[32,60,68,78],
        'David3':[83,200,35,131],
        'Singer1':[51,53,87,290],
        'Toy':[152,102,40,67],
        'Coke':[298,160,48,80],
        'Deer': [306,5,95,65],
        'Dog1':[139,112,51,36],
        'Suv':[142,125,91,40],
        'Skater':[138,57,39,137],
        'hiding':[110,94, 27,90],
        'crouching':[400,47,33, 112],
        'car1':[464,50,147,70],
        'crowd':[497, 52, 51, 147],
        'boat2':[414, 42, 21, 9],
        'car2':[78,160,82,41],
        'birds':[337, 130, 44, 159],
        'quadrocopter2': [434,258,22,11],
        'jacket':[330,302,31,91],
        'street':[356, 243, 16, 49],
        'tree2':[210, 111, 5, 9],
        'ragged':[42,359, 62, 26],
        'excavator':[560, 237, 36, 32]
    }
    img_path = video_set['excavator']
    net_path = 'pretrained/siamrpn/model.pth'
    img_files = os.listdir(img_path)
    box = start_pos['excavator']
    start_track(img_path, net_path, box)
# skiing fail
# ironman fail
# matrix fail
# ant3 fail
# Freeman4 fail
# Basketball fail
# Skiing fail
# hiding fail