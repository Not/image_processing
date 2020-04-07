# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:06:41 2020

@author: dq
"""
from operator import sub
import math
from queue import Queue
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt





def filter_object(img,min_size:int)->list:
    """funcja wyszukuje fragmenty obrazu o niezerowej wartosci i zwraca listę
    obiektów zawierającyh maskę fragmentu oraz jego rozmiar"""
    print(f"Finding objects of size>{min_size}")
    img=img.copy()
    shapes=[]
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            
            shape=select_object(img,(j,i),1)
            if shape!=None:
                if shape['size']>0:
                    img-=shape["mask_img"]
                if shape['size']>min_size:
                    shapes.append(shape)
    return shapes

def select_object(img,position:tuple,value=None)->np.array:
    """funkcja zaznacza cały jednolity obszar (o takiej samej wartosci-value )
    wartosc docelowa moze zostac ustalona z punktu początkowego lub zadana
    przez użytkownika. Funkcja zwraca maskę danego obszaru lub None w przypadku
    wybrania piksela o wartosci innej niz docelowej"""
    
    object_info={'mask_img':None,'size':0}
    dst=np.zeros(img.shape,dtype=np.uint8)
    def recursive_fill(x,y,value):
    
        #color=img[start_point]
        # if thresh1[point]!=value:
        h,w=img.shape[0],img.shape[1]
      
        if img[y,x] !=value or dst[y,x]==1: 
            
            return

        dst[y,x]+=1;
        object_info['size']+=1;
        
        if x<w-1:
            recursive_fill(x+1,y,value)
        if y<h-1:
            recursive_fill(x,y+1,value)
        if x>0:
            recursive_fill(x-1,y,value)
        if y>0:
            recursive_fill(x,y-1,value)
            
    def fill(img, position,old):
        """funkcja pomocnicza - jest potrzebna aby możliwe było rekurencyjne 
        wywołanie odpowiednika tej wersji - powyżej. Jednak python miał z nią
        problem"""
        x,y=position
        h,w=img.shape[0],img.shape[1]
        if(img[y,x]!=old):
            return None
        
        dst=np.zeros(img.shape,dtype=np.uint8)
        q=Queue()
        q.put(position)
        while(q.qsize()):
            
            object_info['size']+=1;
            pos=q.get()
            #print(pos)
            x,y=pos
            if x<w-1 and img[y,x+1]==old and dst[y,x+1] !=1:
                q.put((x+1,y))
                dst[y,x+1]=1 
            if y<h-1 and img[y+1,x]==old and   dst[y+1,x]!=1:
                q.put((x,y+1))
                dst[y+1,x]=1
            if x>0 and img[y,x-1]==old and   dst[y,x-1]!=1:
                q.put((x-1,y))
                dst[y,x-1]=1
            if y>0 and img[y-1,x]==old and  dst[y-1,x]!=1:
                q.put((x,y-1))
                dst[y-1,x]=1        
        return dst     
    
    x,y=position
    if value==None:
        value=img[y,x]
    #recursive_fill(x,y,value)
    object_info["mask_img"]=fill(img,(x,y),value)
    return object_info

def center(img):
    """funkcja znajduje srodek ciężkosci danego obrazu - wartosci pikseli- waga"""
    #start = timer()
    h,w=img.shape
    yy,xx=np.mgrid[:h,:w];
    x_centers=np.array([np.sum(img*xx,axis=1)/np.sum(img,axis=1),
                        np.sum(img,axis=1)]) #srodki ciezkosci rzedow, masa
    np.nan_to_num(x_centers,copy=False)
    y_center=np.sum(x_centers[1]*range(h))/np.sum(x_centers[1])
    x_center=np.average(x_centers[0],weights=x_centers[1])
    
   # end = timer()
   # print(end - start) # 
    return np.array([x_center,y_center])

def line_transform(img,size:tuple)->np.array:
    """implementacja transforamcji Hougha, zwraca reprezentacje transformaty 
    w postaci obrazu o zadanym rozmiarze"""
    print("line transform in progress")
    h,w=img.shape
    alfa_size=size[0] #zakres docelowej transformaty
    r_size=size[0];
    dst=np.zeros((alfa_size,r_size),dtype=(np.int16))
    
    for y in range(h):
        for x in range(w):
            if not img[y,x]:
                continue
            for alfa in range(alfa_size):
                real_alfa=(alfa/alfa_size)*math.pi
                real_r=x*math.cos(real_alfa)+y*math.sin(real_alfa)
                r=(real_r/math.sqrt(w**2 + h**2))*r_size/2
                dst[int(alfa),int(r+r_size/2)]+=1
    return dst

def get_lines(line_img, line_count:int, threshold: int,dst_size)->list:
    """funkcja na podstawie transformaty znajduje prawdopodobne proste i zwraca
    wynik w postaci słownika zawierajacego listy prostych - w postaci współczynników
    a i b postaci kierunkowej oraz współrzędnych na obrazie transformaty - a i r"""
    point_obj=[]
    
    while len(point_obj)<line_count:
        lines_bin=thresh(line_img,threshold)
        #lines_bin=operation(lines_bin,get_circle(2),"dilate")
        point_obj=filter_object(lines_bin,0)
        point_obj=sorted(point_obj,key=lambda i:i['size'],reverse=True)[:line_count]
        print(f"thresholded with {threshold}")
        threshold-=15;

    #points_ar=[center(obj["mask_img"]) for obj in point_obj]
    points_ar=np.flip([np.unravel_index((obj["mask_img"]*line_img).argmax(),
                                        lines_bin.shape) for obj in point_obj],axis=1)    
    points_ab=[]
    for p_ar in points_ar:
        real_alfa=p_ar[1]/line_img.shape[0]*math.pi;
        real_r=(2*p_ar[0]/line_img.shape[1]-1)*math.sqrt(dst_size[0]**2+dst_size[1]**2)
        a=-1/math.tan(real_alfa)
        b=real_r/math.sin(real_alfa)
        points_ab.append((a,b))   
    result={"ar":points_ar,"ab":points_ab}    
    show(lines_bin,'Maksima')
    return result;

def get_circle(radius):
    """Funkcja zwraca element strukturalny w kształcie koła o danym promieniu"""
    xx, yy = np.mgrid[:radius*2-1, :radius*2-1]
    circle = (xx - radius+1) ** 2 + (yy - radius+1) ** 2
    return (circle<(radius)**2).astype(np.uint8)

def operation(img, kernel, mode,a_p=None):
    """funkcja wykonująca operacje z wykorzystaniem kernela,
    parametr mode okresla rodzaj operacji:
    -dilate
    -erode
    -convolution
    -median
    parametr a_p okresla punkt srodkowy jądra, niewymagany """
    
    print(f"{mode} in progress")  
    h,w=img.shape
    m,n=kernel.shape
    #Jesli nie podamy Anchor point, to musi istniec srodek kernela
    assert (m%2==1 and n%2==1) or a_p!=None
    if a_p==None:
        a_p=(int(n/2),int(m/2))
    margins=((a_p[1],m-a_p[1]-1),(a_p[0],n-a_p[0]-1))
    img=img.copy()
    
    img=np.pad(img,margins,'constant',constant_values=0)
    dst=np.zeros((h,w),dtype=np.int16)
    
    modes={"dilate":lambda x,y:np.any(x*y),
           "erode":lambda x,y: not np.any(y-x*y),
           "convolution":lambda x,y:np.sum(x*y),
           "median":lambda x,y:np.median(x)}
    funkcja=modes[mode];        
    
    for y in range(h):
        for x in range(w):
            roi=img[y:y+m,x:x+n];
            dst[y,x]=funkcja(roi,kernel)
    return dst

def thresh(img, min_value=None, max_value=None):
    """Binaryzacja """
    print("thresholding in progress")
   
    if min_value==None and max_value==None:
        return np.clip(img,0,1,).astype(np.uint8)
    if min_value==None:
        return np.clip((img<=max_value)*img,0,1)
    if max_value==None:
        return np.clip((img>=min_value)*img,0,1)

def find_corners(list_ab):
    """funkcja wyszukuje narożniki kart mając dane współczynniki a i b 4 prostych i
    zwraca listę współrzędnych punktów wpołrzędnych zgodnie ze wskazówkami zegara,
    tak aby punkty 0 i 1 wyznaczały krótszą krawędź"""

    c = list(combinations(list_ab, 2))
    intersections=[]
    for pair in c:
        
        a1=pair[0][0]
        a2=pair[1][0]
        b1=pair[0][1]
        b2=pair[1][1]
        Wg=a2-a1
        Wx=b1-b2
        Wy=a2*b1-a1*b2
        if abs(Wg/(a1*a2+1))>1:
            intersections.append((Wx/Wg,Wy/Wg))
                 
    x=sum(i[0] for i in intersections)/len(intersections)
    y=sum(i[1] for i in intersections)/len(intersections)
    intersections=sorted(intersections,key=lambda i:math.atan2(y-i[1],x-i[0]))
    distances=np.array(np.sum(list(map(lambda i : np.sqrt(i * i),
                                       map(sub,intersections,
                                           np.roll(intersections,1
                                                   ,axis=0)))),axis=1))
    if np.argmax(distances)%2:
        intersections=np.roll(intersections,1,axis=0)
    
    return intersections

def show(img,descr):
    plt.figure(figsize=(20,10))
    plt.title(descr)
    if len(img.shape)==3:
       # plt.imshow(cv.img,'gray')  
       pass
    plt.imshow(img,'gray')
    plt.show()
    
def sort_lines(intersections):
    """funkcja znając narożniki karty zwraca proste zawierające boki w odpowiedniej
    kolejnosci  - zaczynajac od dłuższych krawędzi"""
    c=intersections
    #punkty 0 i 3 oraz 1 i 2 wyznaczaja dłuższe boki to trzeba zmienic
    a1=(c[0][1]-c[3][1])/(c[0][0]-c[3][0]) #y2-y1/x2-x1
    a2=(c[1][1]-c[2][1])/(c[1][0]-c[2][0])
    b1=c[0][1]-a1*c[0][0]
    b2=c[1][1]-a2*c[1][0]
    a3=(c[0][1]-c[1][1])/(c[0][0]-c[1][0]) #y2-y1/x2-x1
    a4=(c[2][1]-c[3][1])/(c[2][0]-c[3][0])
    b3=c[0][1]-a3*c[0][0]
    b4=c[1][1]-a3*c[1][0]
    
    #angle=math.degrees(math.atan(2/(a1+a2)))
    return np.array([[a1,b1],[a2,b2],[a3,b3],[a4,b4]])
    
    # print("\033[1mProste przechodzące przez dłuższe boki:\n",
    #       f"y={a1:.2f}x+{b1:.2f} oraz y={a2:.2f}x+{b2:.2f}\n",
    #       f"Kąt nachylenia karty względem osi Y: {angle}")

    
    