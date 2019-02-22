# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:16:05 2019

@author: zhxing
this code can draw position precision of tracking result in vot challenge
"""

import math
import matplotlib.pyplot as plt
import numpy as np

#存放文件的路径以及各种文件的路径
path="results//txt2015//"
savepath="results//png2015//"
res_ground="_res_ground.txt"
res_kcf="_res_kcf.txt"
res_kcf_interpolation="_res_kcf_interpolation.txt"
res_kcf_lab="_res_kcf_lab.txt"
res_kcf_interpolation_lab="_res_kcf_interpolation_lab.txt"


file=open(path+"list.txt")
lines=file.readlines()





#calculate IOU,rect1 and rect2 are rectangles(x,y,width,height)
def calculateIOU(rect1,rect2):
    #calculate the area
    area1=rect1[2]*rect1[3]  
    area2=rect2[2]*rect2[3]
    #calculate the sum area
    area=area1+area2
    
    #calculate the edge line of every rect
    top1=rect1[1]
    left1=rect1[0]
    bottom1=rect1[1]+rect1[3]
    right1=rect1[0]+rect1[2]
    
    top2=rect2[1]
    left2=rect2[0]
    bottom2=rect2[1]+rect2[3]
    right2=rect2[0]+rect2[2]
    
    #calculate the intersect rectangle
    top=max(top1,top2)
    left=max(left1,left2)
    bottom=min(bottom1,bottom2)
    right=min(right1,right2)
    
    #if no intersect
    if top>=bottom or right<=left:
        return 0
    else:
        intersectArea=(bottom-top)*(right-left)
        return intersectArea/(area-intersectArea)
    
def GetIOUList(ResGroundLines,ResLines):
    num_of_frame=len(ResGroundLines)-2
    IOU=[]
    for index in range(1,(num_of_frame+1)):
        #每一行拿出来，第一列是分别是 frame	x	y	width	height,分离出来并转换成数字
        GroundPos=(ResGroundLines[index]).split('\t')
        ResPos=(ResLines[index]).split('\t')
       
        #a line,trans to num
        GroundPos=list(map(float,GroundPos))
        ResPos=list(map(float,ResPos))
        
        #get the rectangle and calculate the IOU
        rect1=GroundPos[1:5]
        rect2=ResPos[1:5]
        iou=calculateIOU(rect1,rect2)
        IOU.append(iou)
    return IOU
        

#calculate the Pre,CLE is CENTOR LOCATION ERROR,and it is a list
def calculatePre(CLE):
    res=[]
    for thresh in range(1,100):
        tmp=np.array(CLE)  #get the temporary variable
        tmp[tmp<=thresh]=1
        tmp[tmp>thresh]=0
        num=sum(tmp)
        rate=float(num)/float(tmp.size)
        res.append(rate)
    return res


#定义画中心位置误差图像的函数   
def GetCLE(ground_pos,res_pos):
    
    #get the pos
    PG=[ground_pos[1]+ground_pos[3]/2,ground_pos[2]+ground_pos[4]/2]
    PR=[res_pos[1]+res_pos[3]/2,res_pos[2]+res_pos[4]/2]
    
    #calculate cle
    CLE=math.sqrt((PG[0]-PR[0])**2+(PG[1]-PR[1])**2)

    return CLE

        

def GetCLElist(title,ResGroundLines,ResLines):
   
    num_of_frame=len(ResGroundLines)-2        #帧数，去掉表头和最后一帧（主要是我结果好像少写了一帧）
    #list,numype
    CLE=[]  
    for index in range(1,(num_of_frame+1)):
        #每一行拿出来，第一列是分别是 frame	x	y	width	height,分离出来并转换成数字
        GroundPos=(ResGroundLines[index]).split('\t')
        ResPos=(ResLines[index]).split('\t')
       
        #a line,trans to num
        GroundPos=list(map(float,GroundPos))
        ResPos=list(map(float,ResPos))
        
        #append to the list
        cletmp=GetCLE(GroundPos,ResPos)
        CLE.append(cletmp) 
    return CLE


#draw fps
def calculateFPS(ResGroundLines,ResLines):
    num_of_frame=len(ResGroundLines)-2        #帧数，去掉表头和最后一帧（主要是我结果好像少写了一帧）
    #list,numype
    area=0
    fps=0
    for index in range(1,(num_of_frame+1)):
        #每一行拿出来，第一列是分别是 frame	x	y	width	height,分离出来并转换成数字
        GroundPos=(ResGroundLines[index]).split('\t')
        ResPos=(ResLines[index]).split('\t')
       
        #a line,trans to num(float)
        GroundPos=list(map(float,GroundPos))
        ResPos=list(map(float,ResPos))
        
        area=area+GroundPos[3]*GroundPos[4]   # calculate the area
        fps=fps+ResPos[5]         #calculate the fps
    ave_area=area/num_of_frame
    ave_fps=fps/num_of_frame
    return ave_area,ave_fps
    

def dreaFPS(area,fps):
    area_fps=np.array([area,fps])  #transform to matrix
    
    
    
    

AREA=[]
FPS=[]
for target in lines:
    print("this is the:\t"+target)
    #target有个回车，这里需要把这个回车给去掉,然后下面把当前target下的文件读取
    ResGround=open(path+target[:-1]+res_ground)
    ResKcf=open(path+target[:-1]+res_kcf)
    ResKcfI=open(path+target[:-1]+res_kcf_interpolation)
    ResKcf_lab=open(path+target[:-1]+res_kcf_lab)
    ResKcfI_lab=open(path+target[:-1]+res_kcf_interpolation_lab)
    #open the txt
    
    #read lines,this is string list
    ResGroundLines=ResGround.readlines()
    ResKcfLines=ResKcf.readlines()
    ResKcfILines=ResKcfI.readlines()
    ResKcf_lablines=ResKcf_lab.readlines()
    ResKcfI_lablines=ResKcfI_lab.readlines()
    
    CLE_KCF=GetCLElist(target,ResGroundLines,ResKcfLines)
    CLE_KCFI=GetCLElist(target,ResGroundLines,ResKcfILines)
    CLE_KCF_LAB=GetCLElist(target,ResGroundLines,ResKcf_lablines)
    CLE_KCFI_LAB=GetCLElist(target,ResGroundLines,ResKcfI_lablines)
    
    
    IOU_KCF=GetIOUList(ResGroundLines,ResKcfLines)
    IOU_KCFI=GetIOUList(ResGroundLines,ResKcfILines)
    IOU_KCF_LAB=GetIOUList(ResGroundLines,ResKcf_lablines)
    IOU_KCFI_LAB=GetIOUList(ResGroundLines,ResKcfI_lablines)
    
    area,fps=calculateFPS(ResGroundLines,ResKcfLines)
    AREA.append(area)
    FPS.append(fps)
  
'''
    #draw the CLE
    plt.figure()
    plt.title(str(target+"  CLE"))
    plt.plot(CLE_KCF,color='red',label='CLE_KCF',LineWidth=0.5)
    plt.plot(CLE_KCFI,color='green',label='CLE_KCFI',LineWidth=0.5)
    plt.plot(CLE_KCF_LAB,color='blue',label='CLE_KCF_LAB',LineWidth=0.5)
    plt.plot(CLE_KCFI_LAB,color='black',label='CLE_KCFI_LAB',LineWidth=0.5)
    plt.legend()
    plt.savefig(savepath+target+"_cle.png",dpi=600)
    
    #draw the preplot
    plt.figure()
    plt.title(str(target+"  Pre"))
    plt.plot(calculatePre(CLE_KCF),color='red',label='CLE_KCF',LineWidth=0.5)
    plt.plot(calculatePre(CLE_KCFI),color='green',label='CLE_KCFI',LineWidth=0.5)
    plt.plot(calculatePre(CLE_KCF_LAB),color='blue',label='CLE_KCF_LAB',LineWidth=0.5)
    plt.plot(calculatePre(CLE_KCFI_LAB),color='black',label='CLE_KCFI_LAB',LineWidth=0.5)
    plt.legend()
    plt.savefig(savepath+target+"_Pre.png",dpi=600)
    
    
    #draw the IOU
    plt.figure()
    plt.title(str(target+" IOU"))
    plt.plot(IOU_KCF,color='red',label='IOU_KCF',LineWidth=0.5)
    plt.plot(IOU_KCFI,color='green',label='IOU_KCFI',LineWidth=0.5)
    plt.plot(IOU_KCF_LAB,color='blue',label='IOU_KCF_LAB',LineWidth=0.5)
    plt.plot(IOU_KCFI_LAB,color='black',label='IOU_KCFI_LAB',LineWidth=0.5)
    plt.legend()
    plt.savefig(savepath+target+"_iou.png",dpi=600)
'''
plt.figure()
plt.scatter(AREA,FPS) 
plt.xlim(0,20000)   #set x range
plt.xlabel("area")
plt.ylabel("fps")
plt.title("FPS_AREA")
        
        
    
    
    
    
    



