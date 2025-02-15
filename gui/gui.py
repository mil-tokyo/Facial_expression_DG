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
        prob, pred = torch.max(output, dim=1)
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
        super(Result_graph, self).__init__(parent, id, size=(640, 640))
        self.SetBackgroundColour('#F5F5F5')
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        label1 = wx.StaticText(self, -1, "Neutral", (20, 35))
        label2 = wx.StaticText(self, -1, "Happy", (20, 85))
        label3 = wx.StaticText(self, -1, "Anger", (20, 135))
        label4 = wx.StaticText(self, -1, "Sad", (20, 185))
        label5 = wx.StaticText(self, -1, "Disgust", (20, 235))
        label6 = wx.StaticText(self, -1, "Fear", (20, 285))
        label7 = wx.StaticText(self, -1, "Surprise", (20, 335))

        self.preds = [0.0] * 7

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 20, 500 * self.preds[0], 40)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 70, 500 * self.preds[1], 40)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 120, 500 * self.preds[2], 40)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 170, 500 * self.preds[3], 40)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 220, 500 * self.preds[4], 40)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 270, 500 * self.preds[5], 40)
        dc.SetPen(wx.Pen('#FFA500'))
        dc.SetBrush(wx.Brush('#FFA500'))
        dc.DrawRectangle(100, 320, 500 * self.preds[6], 40)


class MyFrame(wx.Frame):
    def __init__(self, parent, title, model_name, train, num_domains, path):
        super(MyFrame, self).__init__(parent=parent, title=title, size=(1400, 500))
        self.Bind(wx.EVT_CLOSE, self.frame_close)
        self.stopFlag = False
        self.image_size = (640, 480)
        self.demo = FacialExpressionDemo(model_name=model_name, train=train, num_class=7, num_domains=num_domains, device=0, path=path)
        self.cascade = cv2.CascadeClassifier('/opt/conda/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        panel = wx.Panel(self, wx.ID_ANY)
        img = wx.Image(self.image_size[0], self.image_size[1])
        self.video_frame = VideoView(panel, wx.ID_ANY, wx.Bitmap(img))
        self.graph = Result_graph(panel, wx.ID_ANY)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.video_frame)
        sizer.Add(self.graph)
        panel.SetSizer(sizer)
        #self.Fit()

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
        face_detect_image = cv2.resize(cv2_rgb_image, (320, 240), cv2.INTER_LINEAR)
        image_gray = cv2.cvtColor(face_detect_image, cv2.COLOR_BGR2GRAY)
        face = self.cascade.detectMultiScale(image_gray)
        if len(face) == 1:
            x, y, w, h = face[0]
            output = self.demo.evaluation(face_detect_image[x:x+w, y:y+h])
            self.graph.preds = output.numpy()[0]
        elif len(face) == 2:
            if face[0][0] < face[1][0]:
                x, y, w, h = face[1]
            else:
                x, y, w, h = face[0]
            output = self.demo.evaluation(face_detect_image[x:x + w, y:y + h])
            self.graph.preds = output.numpy()[0]

        else:
            self.graph.preds = [0.0] * 7
        self.graph.Refresh()
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
