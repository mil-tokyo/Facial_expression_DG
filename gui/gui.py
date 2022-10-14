import os
import wx
import numpy as np
import cv2
import torch
import sys
sys.path.append('../')
from train.eval import eval_demo
from util.util import *
import threading
from PIL import Image
from torchvision import transforms
import argparse

cap = cv2.VideoCapture(0)

class FacialExpressionDemo:
    def __init__(self, model_name, train, num_class, num_domains, device, path):
        self.best_model = get_model(model_name, train)(num_classes=num_class,
                                                     num_domains=num_domains, pretrained=False)
        self.best_model.load_state_dict(torch.load(os.path.join(
            path, 'models',
            "model_best.pt"), map_location=torch.device('cpu')))
        self.best_model = self.best_model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])
        self.device = device

    def evaluation(self, image):
        image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')
        if image.size != (256, 256):
            image = image.resize((256, 256))
        image = self.transform(image)
        image = torch.cat([image, image, image], dim=0)
        image = image.unsqueeze(0)
        output = eval_demo(self.best_model, image, device=self.device)

        print(output)
        _, pred = torch.max(output, dim=1)
        labels = ['Neutral', 'Happy', 'Anger', 'Sad', 'Disgust', 'Fear', 'Surprise']
        print(labels[pred])
        return output


class VideoView(wx.StaticBitmap):
    def __init__(self, parent, id, image):
        super(VideoView, self).__init__(parent, id, image)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.image = image

    def set_image(self, image):
        self.image = image
        self.SetBitmap(image)
        self.Refresh(False)

    def on_size(self, event):
        event.Skip()
        self.Refresh(False)

    def on_paint(self, event):
        if not self.image:
            return

        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()
        dc.DrawBitmap(self.image, 0, 0, True)

class Result_graph(wx.Panel):
    def __init__(self, parent, id):
        super(Result_graph, self).__init__(parent, id, pos=(360, 0), size=(120, 120))
        self.SetBackgroundColour('WHITE')
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.SetPen(wx.Pen('blue'))
        dc.SetBrush(wx.Brush('blue'))
        dc.DrawRectangle(20, 20, 260, 40)
        dc.SetPen(wx.Pen('yellow'))
        dc.SetBrush(wx.Brush('yellow'))
        dc.DrawRectangle(20, 60, 200, 40)
        dc.SetPen(wx.Pen('red'))
        dc.SetBrush(wx.Brush('red'))
        dc.DrawRectangle(20, 100, 160, 40)


class MyFrame(wx.Frame):
    def __init__(self, parent, title, model_name, train, num_domains, path):
        super(MyFrame, self).__init__(parent=parent, title=title)
        self.Bind(wx.EVT_CLOSE, self.frame_close)
        self.stopFlag = False
        self.image_size = (320, 240)
        self.demo = FacialExpressionDemo(model_name=model_name, train=train, num_class=7, num_domains=num_domains, device=0, path=path)
        panel = wx.Panel(self, wx.ID_ANY)
        img = wx.Image(self.image_size[0], self.image_size[1])
        self.video_frame = VideoView(panel, wx.ID_ANY, wx.Bitmap(img))
        self.graph = Result_graph(self, wx.ID_ANY)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.video_frame)
        self.Fit()

        # # Read 1st frame
        # ret, image = cap.read()
        # image = cv2.flip(image, 1)
        # cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # output = demo.evaluation(image)
        #
        # self.wximage = wx.Image(cv2image.shape[1], cv2image.shape[0], cv2image)
        # bitmap = self.wximage.ConvertToBitmap()
        # self.stbmp = wx.StaticBitmap(self, -1, bitmap, (0, 0), self.GetClientSize())
        # # ビットマップを表示
        # self.SetSize(self.wximage.GetSize())
        # # フレームの大きさを画像サイズに合わせる
        self.stream()

    def stream(self):
        ret, image = cap.read()
        self.video_frame.set_image(self.create_wx_bitmap_from_cv2_image(image))
        wx.CallLater(30, self.stream)

        # def updateFrame():
        #     if self.stopFlag:
        #         return
        #     ret, image = cap.read()
        #     image = cv2.flip(image, 1)
        #     cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     output = self.demo.evaluation(cv2image)
        #     self.wximage = wx.Image(cv2image.shape[1], cv2image.shape[0], cv2image)
        #     bitmap = self.wximage.ConvertToBitmap()
        #     self.stbmp.SetBitmap(bitmap)


        # def updateFrame():
        #     if self.stopFlag:
        #         return
        #
        #     image = cv2.flip(image, 1)
        #     cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     self.wximage = wx.Image(cv2image.shape[1], cv2image.shape[0], cv2image)
        #     bitmap = self.wximage.ConvertToBitmap()
        #     self.stbmp.SetBitmap(bitmap)
        #
        #
        #     # bitmapの再描画
        #     wx.CallLater(33, updateFrame)  # フレーム・レートの調整
        # updateFrame()

    def create_wx_bitmap_from_cv2_image(self, cv2_image):
        cv2_image = cv2.flip(cv2_image, 1)
        cv2_rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        resize_image = cv2.resize(cv2_rgb_image, self.image_size, cv2.INTER_LINEAR)
        output = self.demo.evaluation(resize_image)
        return wx.Bitmap.FromBuffer(self.image_size[0], self.image_size[1], resize_image)

    def frame_close(self, event):
        self.stopFlag = True
        wx.CallLater(100, self.Destroy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--train', default='general')
    parser.add_argument('--num_domains', default=10)
    parser.add_argument('--model_path', default='202210110317_general_10cluster')
    args = parser.parse_args()

    args.model_path = os.path.join('/home/yusuke/data/Facial_expression_detection_DG', args.model_path)
    app = wx.App()
    frame = MyFrame(None, 'Facial_expression_prediction', model_name=args.model_name, train=args.train, num_domains=args.num_domains, path=args.model_path)
    # frame = wx.Frame(parent=None, id=-1, title="wxPython", size=(400, 400))
    frame.Show()
    # app.SetTopWindow(frame)
    app.MainLoop()
