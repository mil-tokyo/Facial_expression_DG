import wx
import numpy as np
import cv2

cap = cv2.VideoCapture(0)


class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent=parent, title=title)
        self.Bind(wx.EVT_CLOSE, self.frame_close)
        self.stopFlag = False

        # Read 1st frame
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(cv2image.shape)
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
    app = wx.App()
    frame = MyFrame(None, 'Facial_expression_prediction')
    #frame = wx.Frame(parent=None, id=-1, title="wxPython", size=(400, 400))
    frame.Show()
    #app.SetTopWindow(frame)
    app.MainLoop()
