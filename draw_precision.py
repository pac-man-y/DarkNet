# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:16:05 2019

@author: zhxing
this code can draw position precision of tracking result in vot challenge
"""

import math
import matplotlib.pyplot as plt
import numpy

#存放文件的路径以及各种文件的路径
path="results//txt2014//"
ave_fps_kcf="_ave_fps_kcf.txt"
ave_fps_kcf_inter="_ave_fps_kcf_inter.txt"
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
        GroundPos=list(map(int,GroundPos))
        ResPos=list(map(int,ResPos))
        
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
        tmp=numpy.array(CLE)  #get the temporary variable
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
        GroundPos=list(map(int,GroundPos))
        ResPos=list(map(int,ResPos))
        
        #append to the list
        cletmp=GetCLE(GroundPos,ResPos)
        CLE.append(cletmp) 
    return CLE


for target in lines:
    print("this is the:\t"+target)
    #target有个回车，这里需要把这个回车给去掉,然后下面把当前target下的文件读取
    AveFpsKcf=open(path+target[:-1]+ave_fps_kcf)
    AveFpsKcfI=open(path+target[:-1]+ave_fps_kcf_inter)
    ResGround=open(path+target[:-1]+res_ground)
    ResKcf=open(path+target[:-1]+res_kcf)
    ResKcfI=open(path+target[:-1]+res_kcf_interpolation)
    ResKcf_lab=open(path+target[:-1]+res_kcf_lab)
    ResKcfI_lab=open(path+target[:-1]+res_kcf_interpolation_lab)
    #open the txt
    
    #read lines,this is string list
    AveFpsKcfLines=AveFpsKcf.readlines()
    AveFpsKcfILines=AveFpsKcfI.readlines()
    ResGroundLines=ResGround.readlines()
    ResKcfLines=ResKcf.readlines()
    ResKcfILines=ResKcfI.readlines()
    ResKcf_lablines=ResKcf_lab.readlines()
    ResKcfI_lablines=ResKcfI_lab.readlines()
    
    CLE_KCF=GetCLElist(target,ResGroundLines,ResKcfLines)
    CLE_KCFI=GetCLElist(target,ResGroundLines,ResKcfILines)
    CLE_KCF_LAB=GetCLElist(target,ResGroundLines,ResKcf_lablines)
    CLE_KCFI_LAB=GetCLElist(target,ResGroundLines,ResKcfI_lablines)
    
    
    CLE_KCF=GetCLElist(ResGroundLines,ResKcfLines)
    CLE_KCFI=GetCLElist(ResGroundLines,ResKcfILines)
    CLE_KCF_LAB=GetCLElist(ResGroundLines,ResKcf_lablines)
    CLE_KCFI_LAB=GetCLElist(ResGroundLines,ResKcfI_lablines)
    
    
    #draw the CLE
    plt.figure()
    plt.title(target)
    plt.plot(CLE_KCF,color='red',label='CLE_KCF',LineWidth=1)
    plt.plot(CLE_KCFI,color='green',label='CLE_KCFI',LineWidth=1)
    plt.plot(CLE_KCF_LAB,color='blue',label='CLE_KCF_LAB',LineWidth=1)
    plt.plot(CLE_KCFI_LAB,color='black',label='CLE_KCFI_LAB',LineWidth=1)
    plt.legend()
    plt.savefig("results//png2014//"+target+"_Pre.png",dpi=600)
    
    #draw the preplot
    plt.figure()
    plt.title(target)
    plt.plot(calculatePre(CLE_KCF),color='red',label='CLE_KCF',LineWidth=1)
    plt.plot(calculatePre(CLE_KCFI),color='green',label='CLE_KCFI',LineWidth=1)
    plt.plot(calculatePre(CLE_KCF_LAB),color='blue',label='CLE_KCF_LAB',LineWidth=1)
    plt.plot(calculatePre(CLE_KCFI_LAB),color='black',label='CLE_KCFI_LAB',LineWidth=1)
    plt.legend()
    plt.savefig("results//png2014//"+target+"_Pre.png",dpi=600)
    
    
    
    
    
    
    
    #drawCLE(target,ResGroundLines,ResKcfLines,ResKcfILines)
        
        
    
    
    
    
    



