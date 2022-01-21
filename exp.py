import numpy as np
import MR
import os
import cv2
from random import *

import wx
from wx.core import FileDialog, MessageBox
import wx.xrc


#图像的各个数据
filepath = ''
image = []
output = []
saveName = 1


class Image_show ( wx.Panel ):
	def __init__( self, parent ):
		wx.Panel.__init__ ( self, parent, id = wx.ID_ANY, pos = wx.DefaultPosition, size = wx.Size( 500,347 ), style = wx.TAB_TRAVERSAL )
		# Connect Events
		self.Bind( wx.EVT_PAINT, self.OnPaint )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def OnPaint( self, event, image ):
		wxbmp = wx.Bitmap.FromBuffer(self.Size[0],self.Size[1],image)	#设置为bitmap输出显示
		wx.StaticBitmap(parent=self,bitmap=wxbmp)	#显示图像


class ImageProcess ( wx.Frame ):
	

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"图像处理", pos = wx.DefaultPosition, size = wx.Size( 582,407 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		self.main_menu = wx.MenuBar( 0 )
		self.file = wx.Menu()
		self.ReadFile = wx.MenuItem( self.file, wx.ID_ANY, u"读取", wx.EmptyString, wx.ITEM_NORMAL )
		self.file.AppendItem( self.ReadFile )
		
		self.SaveFile = wx.MenuItem( self.file, wx.ID_ANY, u"保存", wx.EmptyString, wx.ITEM_NORMAL )
		self.file.AppendItem( self.SaveFile )
		
		self.main_menu.Append( self.file, u"文件" ) 




		self.solution = wx.Menu()
		self.AddNoise = wx.Menu()
		self.SpNoise = wx.MenuItem( self.AddNoise, wx.ID_ANY, u"椒盐噪声", wx.EmptyString, wx.ITEM_NORMAL )
		self.AddNoise.AppendItem( self.SpNoise )
		
		self.GaussNoise = wx.MenuItem( self.AddNoise, wx.ID_ANY, u"高斯噪声", wx.EmptyString, wx.ITEM_NORMAL )
		self.AddNoise.AppendItem( self.GaussNoise )

		self.solution.AppendSubMenu( self.AddNoise, u"添加噪声" )



		self.Fliter = wx.Menu()
		self.MeanFliter = wx.MenuItem( self.Fliter, wx.ID_ANY, u"均值滤波", wx.EmptyString, wx.ITEM_NORMAL )
		self.Fliter.AppendItem( self.MeanFliter )
		
		self.MaxMinFliter = wx.MenuItem( self.Fliter, wx.ID_ANY, u"最大最小值滤波", wx.EmptyString, wx.ITEM_NORMAL )
		self.Fliter.AppendItem( self.MaxMinFliter )
		
		self.MedianFliter = wx.MenuItem( self.Fliter, wx.ID_ANY, u"中值滤波", wx.EmptyString, wx.ITEM_NORMAL )
		self.Fliter.AppendItem( self.MedianFliter )
		
		self.RankFliter = wx.MenuItem( self.Fliter, wx.ID_ANY, u"统计滤波", wx.EmptyString, wx.ITEM_NORMAL )
		self.Fliter.AppendItem( self.RankFliter )
		
		self.FrequencyFliter = wx.MenuItem( self.Fliter, wx.ID_ANY, u"频率域滤波", wx.EmptyString, wx.ITEM_NORMAL )
		self.Fliter.AppendItem( self.FrequencyFliter )
		
		self.solution.AppendSubMenu( self.Fliter, u"滤波" )
		
		self.ImageDetection = wx.MenuItem( self.solution, wx.ID_ANY, u"显著性检测", wx.EmptyString, wx.ITEM_NORMAL )
		self.solution.AppendItem( self.ImageDetection )
		
		self.ImageSegment = wx.MenuItem( self.solution, wx.ID_ANY, u"图像分割", wx.EmptyString, wx.ITEM_NORMAL )
		self.solution.AppendItem( self.ImageSegment )
		
		self.Face = wx.MenuItem( self.solution, wx.ID_ANY, u"人脸检测", wx.EmptyString, wx.ITEM_NORMAL )
		self.solution.AppendItem( self.Face )
		
		self.main_menu.Append( self.solution, u"图像处理" ) 
		
		self.SetMenuBar( self.main_menu )
		
		
		self.Centre( wx.BOTH )
		

		self.showImage = Image_show(self)


		# Connect Events
		self.Bind( wx.EVT_MENU, self.ReadMyFile, id = self.ReadFile.GetId() )	#读取文件
		self.Bind( wx.EVT_MENU, self.SaveMyFile, id = self.SaveFile.GetId() )	#保存文件
		self.Bind( wx.EVT_MENU, self.AddSpNoise, id = self.SpNoise.GetId() )	#添加椒盐噪声
		self.Bind( wx.EVT_MENU, self.AddGaussNoise, id = self.GaussNoise.GetId() )	#添加高斯噪声
		self.Bind( wx.EVT_MENU, self.MeanF, id = self.MeanFliter.GetId() )	#均值滤波
		self.Bind( wx.EVT_MENU, self.MaxMinF, id = self.MaxMinFliter.GetId() )		#最大最小值滤波
		self.Bind( wx.EVT_MENU, self.MedianF, id = self.MedianFliter.GetId() )	#中值滤波
		self.Bind( wx.EVT_MENU, self.FrequencyF, id = self.FrequencyFliter.GetId() )	#频率域滤波
		self.Bind( wx.EVT_MENU, self.Detection, id = self.ImageDetection.GetId() )	#显著性检测
		self.Bind( wx.EVT_MENU, self.Segment, id = self.ImageSegment.GetId() )	#图像分割
		self.Bind( wx.EVT_MENU, self.face, id = self.Face.GetId() )	#人脸检测	



	# Virtual event handlers, overide them in your derived class
	def ReadMyFile( self, event ):
		global filepath
		global image
		global output
		#读取图片路径
		dlg = wx.FileDialog(self,message=u"选择文件",defaultDir=os.getcwd(),defaultFile="",style=wx.FD_OPEN)
		if (dlg.ShowModal() == wx.ID_OK):
			paths = dlg.GetPaths()
			filepath = paths[0]
		
		image = cv2.imread(filepath,cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		output = np.copy(image)
		h,w = image.shape[:2];	self.Size  = w+16,h+60;	self.showImage.Size = w,h	#调整窗体大小
		self.showImage.OnPaint(self.showImage,image)	#窗口绘图

		event.Skip()
	
	def SaveMyFile( self, event ):
		global saveName
		cv2.imwrite('saveImage/'+str(saveName)+'.jpg',output)	#保存图像
		saveName +=1
		MessageBox("图像已保存！请在文件夹中查看","INFO")
		event.Skip()

	def AddSpNoise( self, event ):
		global output
		prob = 0.05
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		new_output = np.copy(output)
		thres = 1-prob
		for i in range(new_output.shape[0]):
			for j in range(new_output.shape[1]):
				randomNum = random()
				if randomNum <prob:
					new_output[i][j] = 0
				elif randomNum > thres:
					new_output[i][j] = 255
		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图

		event.Skip()

	def AddGaussNoise( self, event ):
		global output
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		mean=0;	var=0.001
		new_output = np.copy(output/255)
		noise = np.random.normal(mean, var ** 0.5, output.shape)
		new_output = new_output + noise
		new_output = np.clip(new_output,0.,1.0)
		new_output = np.uint8(new_output*255)

		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图
		event.Skip()

	def MeanF( self, event ):
		global output
		size = 5	#滤波矩阵大小
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		new_output = np.copy(output)
		fliter_size = np.ones((size,size))
		padding_num = int((size-1)/2)
		new_output = np.pad(new_output, (padding_num, padding_num), mode="constant", constant_values=0)
		w,h = new_output.shape[0],new_output.shape[1]
		new_output = np.copy(new_output)
		for i in range(padding_num,w-padding_num):
			for j in range(padding_num,h-padding_num):
				new_output[i,j]=np.sum(fliter_size * new_output[i-padding_num:i+padding_num+1,j-padding_num:j+padding_num+1])/size**2
		new_output = new_output[padding_num:w - padding_num, padding_num:h - padding_num]
		
		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图

		event.Skip()
	
	def MaxMinF( self, event ):
		global output
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		new_output = np.copy(output)
		
		w,h = new_output.shape[0],new_output.shape[1]
		for i in range(0,w-3):
			for j in range(0,h-3):
				Matrix3_3 = np.copy(new_output[i:i+3,j:j+3])
				Matrix3_3[1][1] = 1000; minNum = np.min(Matrix3_3)
				Matrix3_3[1][1] = -100; maxNum = np.max(Matrix3_3)
				if new_output[i+1][j+1] < minNum:
					new_output[i+1][j+1] = minNum
				elif new_output[i+1][j+1] >maxNum:
					new_output[i+1][j+1] = maxNum

		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图
		event.Skip()
	
	def MedianF( self, event ):
		global output
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		new_output = np.copy(output)

		w,h = new_output.shape[0],new_output.shape[1]
		for i in range(0,w-3):
			for j in range(0,h-3):
				Mid = np.median(new_output[i:i+3,j:j+3])
				new_output[i+1][j+1]=Mid

		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图
		event.Skip()
	
	def FrequencyF( self, event ):
		#巴特沃斯低通滤波器
		global output
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		img = np.copy(output)
		h,w = img.shape
		fft = np.fft.fftshift(np.fft.fft2(img))#傅里叶变换
		mask0 = np.zeros(img.shape,np.float32)
		R0 = 35	#截止频率
		n = 1	#阶数
		for i in range(0,h):
			for j in range(0,w):
				Rxy = ((i-h/2)**2+(j-w/2)**2)**(1/2)
				mask0[i,j] = 1/(1+(Rxy/R0)**(2*n))
		p0 = fft*mask0
		new_output = np.abs(np.fft.ifft2(np.fft.ifftshift(p0)))	#反变换
		new_output = new_output.astype(np.uint8)
		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图

		event.Skip()

	def Detection( self, event ):
		global output
		mr = MR.MR_saliency()
		output = mr.saliency(filepath)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图
		event.Skip()
	
	def Segment( self, event ):
		global output
		output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		#new_output = cv2.Canny(new_output,50,150)	cv库
		#利用sobel算子实现
		h, w = output.shape
		new_output = np.copy(output);new_outputX = np.copy(output);new_outputY = np.copy(output)
		GX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
		GY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
		for i in range(h-2):
			for j in range(w-2):
				new_outputX[i+1, j+1] = abs(np.sum(output[i:i+3, j:j+3] * GX))
				new_outputY[i+1, j+1] = abs(np.sum(output[i:i+3, j:j+3] * GY))
				new_output[i+1, j+1] = (int(new_outputX[i+1, j+1])**2 + int(new_outputY[i+1, j+1])**2)**0.5
		
		output = cv2.cvtColor(new_output, cv2.COLOR_GRAY2RGB)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图
		event.Skip()
	
	def face( self, event ):
		global output
		gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
		classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
		faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
		if len(faceRects)>0:
			for faceRect in faceRects:
				x,y,w,h = faceRect
				cv2.rectangle(output, (x - 10, y - 10), (x + w + 10, y + h + 10), (255,0,0), 3)
		self.showImage.OnPaint(self.showImage,output)	#窗口绘图
		
		event.Skip()


app = wx.App()   
f= ImageProcess(None)  
f.Show()
app.MainLoop()
