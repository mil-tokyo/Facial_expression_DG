import wx
import numpy as np
import cv2
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
            "model_best.pt"), map_location=device))
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

        output = eval_demo(image, device=self.device)

        print(output)
        return output


class MyFrame(wx.Frame):
    def __init__(self, parent, title, model_name, train, num_domains, path):
        super(MyFrame, self).__init__(parent=parent, title=title)
        self.Bind(wx.EVT_CLOSE, self.frame_close)
        self.stopFlag = False

        demo = FacialExpressionDemo(model_name=model_name, train=train, num_class=7, num_domains=num_domains, device=0, path=path)


        # Read 1st frame
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = demo.evaluation(image)

        self.wximage = wx.Image(cv2image.shape[1], cv2image.shape[0], cv2image)
        bitmap = self.wximage.ConvertToBitmap()
        self.stbmp = wx.StaticBitmap(self, -1, bitmap, (0, 0), self.GetClientSize())
        # ビットマップを表示
        self.SetSize(self.wximage.GetSize())
        # フレームの大きさを画像サイズに合わせる
        self.stream()

    def stream(self):

        def updateFrame():
            if self.stopFlag:
                return
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.wximage = wx.Image(cv2image.shape[1], cv2image.shape[0], cv2image)
            bitmap = self.wximage.ConvertToBitmap()
            self.stbmp.SetBitmap(bitmap)


            # bitmapの再描画
            wx.CallLater(33, updateFrame)  # フレーム・レートの調整
        updateFrame()

    def frame_close(self, event):
        self.stopFlag = True
        wx.CallLater(100, self.Destroy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--train', default='general')
    parser.add_argument('--num_domains', default=10)
    parser.add_argument('--model_path', default='')
    args = parser.parse_args()

    app = wx.App()
    frame = MyFrame(None, 'Facial_expression_prediction', model_name=args.model_name, train=args.train, num_domains=args.num_domains, path=args.model_path)
    # frame = wx.Frame(parent=None, id=-1, title="wxPython", size=(400, 400))
    frame.Show()
    # app.SetTopWindow(frame)
    app.MainLoop()
