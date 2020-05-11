
import cv2 as cv
import numpy as np
from timeit import default_timer as timer
import card_finder as cf
import math
import matplotlib.pyplot as plt
import sys
 # %% WCZYTANIE OBRAZU
print("Start")
start_p=timer()

plik='tt2.jpg'#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<WYBÓR PLIKU

img_full_display = cv.imread(plik)
if img_full_display is None:
    print("No such picture")
    sys.exit()
    
    

img_full_display=cv.cvtColor(img_full_display,cv.COLOR_BGR2RGB)
img_full=cv.cvtColor(img_full_display,cv.COLOR_RGB2GRAY)
# 013.jpg, 013_blur.jpg, 013_gradient.jpg 013_salt.jpg
img_full_color=cv.imread(plik,1)

upper=100000
org_size=img_full.size
quality = 4
scale=np.clip(math.sqrt(org_size/(quality*(upper/10))),0,100)

print(scale)
# %%

org_size=img_full.shape[1],img_full.shape[0]
new_size = int(img_full.shape[1]/scale), int(img_full.shape[0]/scale)
img = cv.resize(img_full,new_size)
cf.show(img_full_display,"obraz oryginalny")
cf.show(img,"obraz wejsciowy")

"""
 1-128x96 - 12000 36
 2-192*144 -27000 24
 3 256*192 -50000 18
 
 10- 250 000

"""

# %% ZNAJDOWANIE KART

#img=operation(img,get_circle(2),'median') #USUWA SZUM, NIEPOTRZEBNE

thresh1=cf.thresh(img,130)### BINARYZACJA
cf.show(thresh1,"Po binaryzacji")
good=False;

background=cf.select_object(thresh1,(0,0),0) #ZAZNACZENIE CAŁEGO TŁA
if background['mask_img'] is  None:
    print("Error")
    sys.exit()
        
        
        
cards_mask=np.logical_not(background["mask_img"]).astype(np.uint8) #KARTY

size_factor =0.025 #Rozpoznawane będą obiekty powyżej 3.5% obrazu
size_pix=thresh1.size*size_factor
cards_list=cf.filter_object(cards_mask,size_pix) #OPERACJA SEGMENTACJI


for  i,card in enumerate(cards_list): #TYLKO DO ZAZNACZENIA KART
    ctr=tuple(map(int,cf.center(card["mask_img"])))
    card["center"]=ctr
    card["index"]=i
    cv.circle(cards_mask,ctr,3,(0,0,0))
    cv.putText(cards_mask,str(i),ctr,cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0))

cf.show(cards_mask,'Wszystkie potencjalne karty wyczyszczone')
cards_mask_full=cv.resize(cards_mask,org_size) 

## znalezione karty

# %% OPERACJA DLA DANEJ KARTY
for card in cards_list:
 
    cf.show(card["mask_img"],f'maska')
    
    kernel=np.array([[1,1,1], 
                     [1,-8,1],
                     [1,1,1]])
        
    card_border1=cf.operation(card["mask_img"],kernel,"convolution")  #WYKRYCIE KRAWEDZI
    card_border1=cf.thresh(abs(card_border1))
    #show(card_border1,f"Krawędź karty Laplace {nr}") #ALBO TAK:
    #card_border=operation(karta2,get_circle(3),"dilate")-karta2
    cf.show(card_border1,f"Krawędź karty")
    card_border1_full=cv.resize(card_border1,org_size) 
    card_border_org=img_full_display
    card_border_org[:,:,0]+=card_border1_full*255
    
    
    lines=cf.line_transform(card_border1,(400,400)) ##TRANSFROMACJA
    cf.show(lines,"Hough")
    
    points=cf.get_lines(lines,4,75,card_border1.shape)
    #FUNKJA ZNAJDUJE DANA ILOSC MAKSIMOW LOKALNYCH W OBRAZIE TRANSFORMATY
    #points zawiera listę puntów reprezentującyh proste na obrazie 
    
    corners=cf.find_corners(points["ab"]) 
    if len(corners) is not 4:
        print("This is not a card")
        cards_list.remove(card)
        continue
    
    
    card["corners"]=corners
    card["eges"]=cf.sort_lines(corners)
    card["angle"]=math.degrees(math.atan(2/(card["eges"][0][0]+
                                            card["eges"][1][0])))
     
    img_draw=img.copy()  #WIZUALIZACJA NAROZNIKOW NA KARCE
    for i, p in enumerate(corners):
       # cv.putText(img_draw,str(i), (int(p[0]),int(p[1])),
       #            cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
        cv.circle(img_draw, (int(p[0]),int(p[1])),4, (255,0,0))
    img_draw_full=img_full_color.copy()
    
    for i in range(4): #OBRAMOWANIE KARY NA CAŁYM OBRAZIE
        p1=(int(corners[i][0]*scale),int(corners[i][1]*scale))
        p2=(int(corners[(i+1)%4][0]*scale),int(corners[(i+1)%4][1]*scale))
        cv.line(img_draw_full,p1,p2,(255,0,0),10)
    
    cf.show(img_draw_full)
       
    for point_ar in points["ar"]: #WIZUALIZACJA PROSTYCH W TRANSFORMACIE
        cv.circle(lines, tuple(map(int,point_ar)),4, (255,0,0))
    for point_ab in points["ab"]: #WIZUALIZACJA PROSTYCH NA OBRAZIE
        a,b=point_ab
        h,w=img.shape
        cv.line(img_draw,(0,int(b)),(int(w),int(a*w+b)),(255,0,0))

    
    cf.show(lines,"Wykryte proste")
    cf.show(img_draw,"Wykryte boki karty")
   # input("Wcisnij dowolny klawisz...")

# %%  ANALIZA OBRAZU DANEJ KARTY
for card in cards_list:
    print(f"Rozpoznawanie karty nr {card['index']}")
    width,height=130,200 
    warped=np.zeros((height,width))
    rectangle=np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)
    source=np.array(card['corners'],dtype=np.float32)*scale #narozniki na glownym obrazie
    
    matrix=cv.getPerspectiveTransform(source, rectangle)
    warped=cv.warpPerspective(img_full, matrix, (width,height))
    cf.show(warped,"karta bez perspetywy")
    
    border=20
    roi=warped[border:-border,border:-border]
    median=np.median(warped)
    
   # roi_t=cf.thresh(roi,0.65*median) #binaryzacja jednej karty
    roi_t=cf.thresh(roi,cf.find_thresh_otsu(warped)) #binaryzacja jednej karty
    roi_t=cf.operation(roi_t,cf.get_circle(2),"median")#flitr medianowy
    cf.show(roi_t,"Tu szukam symboli")
    cv.imwrite("krol.png",roi_t*255)
    break
    
    size_factor =0.02 #Rozpoznawane będą obiekty powyżej 3.5% obrazu
    size_pix=roi.size*size_factor
    symbols=cf.filter_object(np.logical_not(roi_t).astype(np.uint8), size_pix) #SEGMEMTACJA SYMBOLI 
    for symbol in symbols:
       symbol["center"]=cf.center(symbol["mask_img"]) 
       
    patterns=["kier","pik","trefl","karo"]
    corr=dict.fromkeys(patterns,0)
    for pattern in patterns:
        
        img=np.logical_not(cv.imread(pattern+".png",0))*2-1
        card_img=np.logical_not(roi_t)
        result=cf.operation(card_img,img,'convolution')
        result_flip=cf.operation(card_img,np.flip(img),'convolution')#ZBADANIE KORELACJI
        #cf.show(result, f"Porownanie z {pattern}")
        corr[pattern]=max(result.max(),result_flip.max())
    
    type_of_card=max(corr.keys(),key=lambda x:corr[x])
    print(f"Ta karta to {len(symbols)} {type_of_card}")
    card['suit']=type_of_card
    card['number']=len(symbols)
    
    matrix_inv=np.linalg.inv(matrix)
    
    roi_t_drw=roi_t.copy()
    img_full_color=cv.cvtColor(img_full,cv.COLOR_GRAY2RGB)
    for symbol in symbols:
         center=np.dot(matrix_inv,np.append((symbol['center']+np.array([border,border])),1))
         cv.circle(img_full_color,tuple(map(int,center[:2]/center[2])),20,(255,0,0),thickness=15)
        # cv.circle(warped,tuple(map(int,symbol['center']+np.array([border,border]))),1,255,thickness=2)
    cf.show(warped,"wynik")
    #cf.show(img_full_color,'Wynik')  
     