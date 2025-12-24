# Micro-Expression-Datasets
This repository contains a unified and extensible codebase for preprocessing and modeling micro-expression datasets, including SMIC, SAMM, CASME II, CASME III, and DFME.

上述文件缺少两个权重：Flownet2.0的权重文件与dlib库的人脸68点位图。
FlowNet2_checkpoint.pth.tar & shape_predictor_68_face_landmarks.dat
请自行下载，或者联系创建者

CAMSE2数据集
5-class：
	other（99）
	disgust（63）
	happiness（32）
	repression（27）
	surprise（25）
3-class：
	positive：happiness（32）
	negative：disgust（63),repression（27）
	surprise：surprise（25）
	
CAMSE3数据集
4-Class：
	negative：disgust（250），fear（86），anger（64），sad（57）
	positive：happy（57）
	surprise：surprise（187）
	others：others（161）
3-Class：
	negative：disgust（250），fear（86），anger（64），sad（57）
	positive：happy（57）
	surprise：surprise（187）
	
SMIC数据集
3-class：
	positive：Happiness（26）
	negative：Sadness，Disgust，Contempt，Fear，Anger（92）
	Surprise：Surprise（15）
5-class：
	Anger：（57）
	Contempt：（12）
	Happiness：（26）
	Surprise：（15）
	Other：（26）
	
SAMM数据集
3-class：
	positive：Happiness（26）
	negative：Sadness，Disgust，Contempt，Fear，Anger（92）
	Surprise：Surprise（15）

5-class：
	Anger：（57）
	Contempt：（12）
	Happiness：（26）
	Surprise：（15）
	Other：（26）
