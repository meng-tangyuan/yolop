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

		self.pub_raw = rospy.Publisher('/Yolo/Row', Image,queue_size=1)
		self.pub_id = rospy.Publisher('/Yolo/Id_masks', Image,queue_size=1)
		self.pub_score = rospy.Publisher('/Yolo/Score_masks', Image,queue_size=1)

		rospy.Subscriber("/kitti/camera_gray_left/image_raw",Image,self.callback,queue_size=1)

		parser = argparse.ArgumentParser()
		parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
		parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
		parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
		parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
		parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
		parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
		parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
		parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
		parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
		parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
		opt = parser.parse_args()

   		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

		if opt.weights_path.endswith(".weights"):
			# Load darknet weights
			model.load_darknet_weights(opt.weights_path)
		else:
			# Load checkpoint weights
			model.load_state_dict(torch.load(opt.weights_path))

		model.eval()  # Set in evaluation mode

		dataloader = DataLoader(
			ImageFolder(opt.image_folder, img_size=opt.img_size),
			batch_size=opt.batch_size,
			shuffle=False,
			num_workers=opt.n_cpu,
		)

		classes = load_classes(opt.class_path)  # Extracts class labels from file

    	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
		img_detections = []

	def callback(self, msg):
		rospy.loginfo('Image received...')
		try:
		    self.image = self.cvb.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		# YOLO Detection		
		###################################################################
		input_imgs = self.image
		# Configure input
		input_imgs = Variable(input_imgs.type(Tensor))

		# Get detections
		with torch.no_grad():
			detections = model(input_imgs)
			detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        global yolo_idmask = np.zeros((input_imgs.shape[0],input_imgs.shape[1],1),np.uint8)
        cv2.rectangle(yolo_idmask,(0,0),(input_imgs.shape[1],input_imgs.shape[0]),(255),-1)  

        global yolo_scoremask = np.zeros((input_imgs.shape[0],input_imgs.shape[1],1),np.uint8)
        cv2.rectangle(yolo_scoremask,(0,0),(input_imgs.shape[1],input_imgs.shape[0]),(255),-1)  
        
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, input_imgs.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                grey= int(classes[int(cls_pred)])
                cv2.rectangle(yolo_idmask,(x1,y1),(x2,y2),(grey),-1)  
     
                score1= cls_conf.item()
                score=score1*1000//4
                print(score)
                cv2.rectangle(yolo_scoremask,(x1,y1),(x2,y2),(score),-1)  
                
		###################################################################
		try:
			self.yoloresult.rawimg = self.cvb.cv2_to_imgmsg(self.image,encoding = "mono8)
			self.yoloresult.idmask = self.cvb.cv2_to_imgmsg(yolo_idmask,encoding = "mono8)
			self.yoloresult.scoremask = self.cvb.cv2_to_imgmsg(yolo_scoremask,encoding = "mono8)
		except CvBridgeError as e:
			print(e)
			
		self.pub_raw.publish(self.yoloresult.rawimg)
		self.pub_id.publish(self.yoloresult.idmask)
		self.pub_score.publish(self.yoloresult.scoremask)
		

if __name__ == '__main__':
	rospy.init_node("yolop", anonymous=True)
	yolop = YOLOP()
	rospy.spin()
