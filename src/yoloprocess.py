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
		self.image = self.cvb.imgmsg_to_cv2(msg)
		yolo_idmask = self.image #this should be your first result
		yolo_scoremask = self.image #this should be your second result
		###################################################################
		## put your yolo process code here                               ##
		## the input of yolo is self.image                               ##
		## put the output of yolo id mask into yolo_idmask               ##
		## put the output of yolo id mask into yolo_scoremask            ##
		###################################################################
		self.yoloresult.idmask = self.cvb.cv2_to_imgmsg(yolo_idmask)
		self.yoloresult.scoremask = self.cvb.cv2_to_imgmsg(yolo_scoremask)
		self.pub.publish(self.yoloresult)
		



if __name__ == '__main__':
	rospy.init_node("yolop", anonymous=True)
	yolop = YOLOP()
	rospy.spin()
