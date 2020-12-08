import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from yolop.msg import twoimgs

class YOLOP(object):
	def __init__(self):
		self.image = None
		self.yoloresult = twoimgs()
		self.cvb = CvBridge()

		self.pub = rospy.Publisher('/masks', twoimgs,queue_size=10)

		rospy.Subscriber("/imgs",Image,self.callback)

	def callback(self, msg):
		rospy.loginfo('Image received...')
		try:
		    self.image = self.cvb.imgmsg_to_cv2(msg, "bgr8")
            	except CvBridgeError as e:
                    print(e)
           	width = self.image.shape[0]
            	height = self.image.shape[1]
           	yolo_idmask = np.ones((width,height),np.uint8)
            	yolo_idmask = yolo_idmask *255;
            	yolo_scoremask = np.ones((width,height),np.uint8)
            	yolo_scoremask = yolo_scoremask *255;
		
		###################################################################
		## put your yolo process code here                               ##
		## the input of yolo is self.image                               ##
		## put the output of yolo id mask into yolo_idmask               ##
		## put the output of yolo id mask into yolo_scoremask            ##
		## 1. find the corner location of the bounding box (x1,y1), (x2, y2), ...
		## 2. find the area inside the bounding box, colored them with id and score
		###################################################################
		self.yoloresult.rawimg = self.cvb.cv2_to_imgmsg(self.image)
		self.yoloresult.idmask = self.cvb.cv2_to_imgmsg(yolo_idmask)
		self.yoloresult.scoremask = self.cvb.cv2_to_imgmsg(yolo_scoremask)
		self.pub.publish(self.yoloresult)
		

if __name__ == '__main__':
	rospy.init_node("yolop", anonymous=True)
	yolop = YOLOP()
	rospy.spin()
