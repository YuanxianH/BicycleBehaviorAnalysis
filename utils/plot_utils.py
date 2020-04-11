from PIL import Image,ImageFont,ImageDraw
import colorsys
import numpy as np
import matplotlib.font_manager as fm # to create font
from .data_utils import Camera
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from yolo3_deepsort.deep_sort.track import MotionState,SecurityState,Track

# 不同状态的显示颜色
state_color = { "Pending": None,
                "Dangerous": (128,0,0),
                "Caution": (128,128,0),
                "Safe": (0,128,0)}
# state_color = {
#                 "Pending": (0,0,0),
#                 "Turning": (0,0,128),
#                 "Speed up": (128,0,128),
#                 "Slow down": (0,128,128),
#                 "Uniform Linear": (0,0,0),
#
#                 "Dangerous": (128,0,0),
#                 "Caution": (128,128,0),
#                 "Safe": (0,128,0)
#                 }

def draw_one_box(img,bbox,id,cat,color,fontsize=10,state=None):
    '''
    img: PIL format
    bbox: (left,top,right,bottom)
    id: identity of the object
    cat: catagory,string
    color: (r,g,b)
    '''
    thickness = (img.size[0] + img.size[1]) // 600
    left,top,right,bottom = bbox
    top = max(0,np.floor(top+0.5).astype('int32'))
    left = max(0,np.floor(left+0.5).astype('int32'))
    bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(img.size[0], np.floor(right + 0.5).astype('int32'))

    label = 'No.{}:{}'.format(id,cat)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),fontsize)
    # font = ImageFont.truetype("arial.pil",size = fontsize)
    # font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', fontsize)
    draw = ImageDraw.Draw(img)
    label_size = draw.textsize(label,font)

    if top-label_size[1]>=0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for j in range(thickness):
        draw.rectangle([left + j, top + j, right - j, bottom - j],
                        outline=color)
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
          fill=tuple(color))
    draw.text(list(text_origin),label,font=font, fill=(0,0,0))

    if state is not None:
        mask1 = Image.new("RGB",(right-left,bottom-top),"rgb"+str(state_color[state]))
        box_blend = Image.blend(img.crop((left,top,right,bottom)),mask1,0.5)
        img.paste(box_blend,(left,top))

    return img


class Canvas:
    """
    创建画布，用于显示跟踪结果

    Attributes:
    ==========
    line_space: 行距
    h: 高
    w1: 信息区的宽
    w2: 图像区的宽
    separator: 不同栏目的分隔符
    x0,y0: 第一个字符的坐标
    font: 字体
    font_color: 字体颜色
    """
    _defaults = {
        "line_space": 15,
        "h": 9*70,
        "w1": 200,
        "w2": 16*70,
        "separator": "-"*20,
        "x0": 10,
        "y0": 10,
        "font_color": (0,0,0),
        "font": None
    }
    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

        self.init_canvas()

    def init_canvas(self):
        """
        初始化信息区
        """
        self.num_line = 0
        self.info_area = Image.new("RGB",(self.w1,self.h),"rgb(255,255,255)")
        self.draw_info = ImageDraw.Draw(self.info_area)
        self.draw_info.rectangle([(0,0),(self.w1,self.h)],outline=(0,0,0),width=5)#边框
        self.add_text("Infomation Area:\n=============")

        self.canvas = Image.new("RGB",(self.w1 + self.w2, self.h))
        self.canvas.paste(self.info_area,(0,0))

    def update_img(self,img):
        """
        更新图像区
        ==========
        Parameters:
            img:图像(pil)
        """
        img = img.resize((self.w2,self.h))
        self.canvas.paste(img,(self.w1,0))

    def update(self,img,frame,unix_time,camera,tracks=[]):
        """
        更新信息区
        """
        self.init_canvas()
        self.update_img(img)
        self.update_time(frame,unix_time)
        self.update_camerainfo(camera)
        if len(tracks)>=3:
            tracks = tracks[:3]
        for track in tracks:
            self.update_objinfo(track)

        self.canvas.paste(self.info_area,(0,0))


    def update_time(self,frame=0, unix_time=0):
        """
        更新基本信息
        ===========
        Parameters:
            frame: 当前帧数
            unix_time: 当前unix时间
        """
        self.add_text("Frame:"+str(frame))
        self.add_text("UNIX time:"+str(unix_time))

    def update_camerainfo(self,camera):
        """
        更新相机信息
        """
        x,y,z = camera.XYZ
        self.add_text("\n"+self.separator)
        self.add_text("Camera:",(0,0,255))
        self.add_text("XYZ: ( %.3f, %.3f, %.3f )"%(x,y,z))
        self.add_text("Direction: %.3f ° " % (camera.direction))
        self.add_text("Velocity: %.3f m/s " % (camera.velocity))
        self.add_text("Acceleration: %.3f m/s2 " % (camera.acceleration))
        self.add_text("Curvature: %.5f " % (camera.curve))
        # self.add_text("Motion State: %s "%(camera.motion_state))
        # self.add_text("Security State: %s "%(camera.security_state))

    def update_objinfo(self,track):
        """
        更新目标信息
        """

        self.add_text("\n" + self.separator)
        self.add_text("Object " + str(track.track_id) + ":",(0,0,255))
        self.add_text("Category: " + str(track.object_class))

        self.add_text("ImageCoord: ( %.3f, %.3f )" % (track.mean[0],track.mean[1]))
        # track.XYZ_array = np.resize(track.XYZ_array,(3,track.XYZ_array.size//3))
        if track.XYZ_array.size < 3:
            X,Y,Z = 0,0,0
        else:
            X,Y,Z = track.XYZ_array[:,-1]
        self.add_text("WorldCoord: ( %.3f, %.3f, %.3f )" % (X,Y,Z))
        self.add_text("Distance: %.3f m " % np.mean(track.distance))
        self.add_text("Velocity: %.3f m/s " % np.mean(track.velocity))
        self.add_text("Acceleration: %.3f m/s2 " % np.mean(track.acceleration))
        self.add_text("Curvature: %.5f " % np.mean(track.curvature))
        # self.add_text("Motion State: %s " % (track.motion_state))
        self.add_text("Security State: %s " % (track.security_state),
                                            fill=state_color[track.security_state])

    def add_text(self,text,fill=(0,0,0)):
        """
        添加文字信息
        ==========
        Parameters:
            text: 文字内容，string
            num_line: 要添加的行数
        """
        self.draw_info.text((self.x0,self.y0 + self.num_line*self.line_space),
                        text,fill=fill,font=self.font)

        self.num_line += text.count("\n")+1
