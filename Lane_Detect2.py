import cv2
import numpy as np
image=cv2.imread('./lane_images/Road_lane_10102.jpg')

height,width=image.shape[0],image.shape[1]
print(width,height)
# cv2.imshow('image',image)

def grayscale(img):
	"""
	图像转灰度
	"""
	return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_image=grayscale(image)

# cv2.imshow('gray_image',gray_image)


def Canny(img,low=50,high=150):
	"""
	使用 canny 进行二值化   low 100 high 150
	"""
	return cv2.Canny(img,low,high)

canny_image=Canny(gray_image)
# cv2.imshow('canny_image',canny_image)

def ROI(img):
	"""
	图像坐标    0,0     852,0
	           0,480   852，482
	分割出一个三角形区域
	"""
	height,width=img.shape[0],img.shape[1]
	roi_vertices=[(0,height),(width/2,height/2),(width,height)]
	# print(roi_vertices)
	mask=np.zeros_like(img)
	match_mask_color=255
	#### ROi正常  其余置零
	cv2.fillPoly(mask,np.array([roi_vertices],np.int32),match_mask_color)
	mask_image=cv2.bitwise_and(img,mask)
	return mask_image

roi=ROI(canny_image)
# cv2.imshow('roi',roi)

def hough_lines(img,rho=4,theta=np.pi/180,
				threshold=160,min_line_len=60,max_line_gap=25):
	"""
	霍夫概率
	"""
	lines=cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),
		minLineLength=min_line_len,maxLineGap=max_line_gap)
	return lines

def draw_lines(img,lines,color=(0,255,0),thickness=8):
	if lines is None:
		return 
	img=np.copy(img)
	img_channels=img.shape[2]
	line_img=np.zeros_like(img)
	for line in lines:
		for x,y,x1,y1 in line:
			cv2.line(line_img,(x,y),(x1,y1),color,thickness)
	# 原图和划线的图像融合
	img=cv2.addWeighted(img,0.8,line_img,1,0)
	return img

lines=hough_lines(roi)
line_imgs=draw_lines(image,lines)

# cv2.imshow('line_img',line_imgs)

def group_lines_and_draw(img,lines):
        """
        根据斜率，将所有的线分为左右两组,斜率绝对值小于0.5的舍去（影响不显著）
        （因为图像的原点在左上角，slope<0是left lines，slope>0是right lines)
        设定min_y作为left和right的top线,max_y为bottom线，求出四个x值即可确定直线：
        将left和right的点分别线性拟合，拟合方程根据y值，求出x值，画出lane
        """
        left_x,left_y,right_x,right_y=[],[],[],[]
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope=(y2-y1)/(x2-x1)
                # if abs(slope)<0.5: continue
                if slope<0:
                    left_x.extend([x1,x2])
                    left_y.extend([y1,y2])
                if slope>0:
                    right_x.extend([x1,x2])
                    right_y.extend([y1,y2])
        #设定top 和 bottom的y值，left和right的y值都一样
        min_y=int(img.shape[0]*(0.7))
        max_y=int(img.shape[0])
        
        #对left的所有点进行线性拟合
        poly_left = np.poly1d(np.polyfit(left_y,left_x,deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        #对right的所有点进行线性拟合
        poly_right = np.poly1d(np.polyfit(right_y,right_x,deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        mid_left_x=(left_x_start+left_x_end)//2
        mid_left_y=(max_y-min_y)//2+min_y
        mid_right_x=(right_x_start+right_x_end)//2
        mid_right_y=(max_y-min_y)//2+min_y
        judge_mid_x=(mid_left_x+mid_right_x)//2
        line_image=draw_lines(img,[[
                [left_x_start,max_y,left_x_end,min_y],
                [right_x_start,max_y,right_x_end,min_y],          
                ]],thickness=8)
        cv2.circle(line_image, (mid_left_x, mid_left_y), 2, (0,0,255), 6)
        cv2.circle(line_image, (mid_right_x, mid_right_y), 2, (0,0,255), 6)

        cv2.line(line_image,(mid_left_x, mid_left_y-25),(mid_left_x, mid_left_y+25),(0,128,255),6)
        cv2.line(line_image,(mid_right_x, mid_right_y-25),(mid_right_x, mid_right_y+25),(0,128,255),6)
        cv2.line(line_image,(judge_mid_x, mid_left_y-50),(judge_mid_x, mid_right_y+50),(0,128,255),6)
        cv2.line(line_image,(mid_left_x, mid_left_y),(mid_right_x, mid_right_y),(0,128,255),6)
        white_line_x=img.shape[1]//2
        cv2.line(line_image,(white_line_x, mid_left_y-60),(white_line_x, mid_right_y+60),(255,255,255),8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if (judge_mid_x-white_line_x)>0:
        	print("turned left")
        	line_image = cv2.putText(line_image, 'turned left', (50, 50), font, 1, (255, 255, 255), 2)
        elif (judge_mid_x-white_line_x)<0:
        	print("turned right")
        	line_image = cv2.putText(line_image, 'turned right', (50, 50), font, 1, (255, 255, 255), 2)
        else:
        	print("go stright!")
        	line_image = cv2.putText(line_image, 'go stright!', (50, 50), font, 1, (255, 255, 255), 2)
        return line_image

final_image=group_lines_and_draw(image,lines)
# cv2.imshow('final_image',final_image)
# cv2.waitKey(5000)



cap = cv2.VideoCapture('./lane.flv')  
while(cap.isOpened()):  
	
    ret, frame = cap.read()  
    gray_image=grayscale(frame)
    canny_image=Canny(gray_image)
    roi=ROI(canny_image)
    lines=hough_lines(roi)
    final_image=group_lines_and_draw(frame,lines)
    cv2.imshow('image', final_image)  
    k = cv2.waitKey(20)  
    #q键退出
    if (k & 0xff == ord('q')):  
        break  

cap.release()  
cv2.destroyAllWindows()