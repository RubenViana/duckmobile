#!/usr/bin/env python3
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import BoolStamped, Pixel, LanePose
from duckietown_msgs.srv import ChangePattern
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Float32, Float64MultiArray, String
from image_geometry import PinholeCameraModel
import cv2
import numpy as np
import rospy
import yaml
from duckietown_utils import (logger, get_duckiefleet_root, load_camera_intrinsics, load_homography)
from dynamic_obstacle_avoidance.msg import dynamic_obstacle

from duckietown.dtros import DTROS, NodeType, TopicType


class DuckieDetectionNode(DTROS):

    def __init__(self, node_name):
        super(DuckieDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.bridge = CvBridge()
        self.active = True
        self.config = self.setupParam("~config", "baseline")

        self.publish_duration = rospy.Duration.from_sec(1.0/2.0)
        self.last_stamp = rospy.Time.now()

        self.lane_width = 0.1145 #half of lane
        self.crop_factor = 0.3 #percentage of image cropped from the top
        self.d = 0.0
        self.phi = 0.0
        self.duckie_pos_arr_prev = []

        self.pcm = PinholeCameraModel()
        # self.H = load_homography(rospy.get_namespace().strip("/"))  # load_homography not working! so hardcoded
        self.H = np.array([
            [-0.00015389980014132867, 0.00033474595519600086, 0.3198466654067245],
            [-0.0013999151846608236, -1.316249553467081e-05, 0.43310887263768993],
            [-0.00039075225726160825, 0.010959818387184139, -1.5504152912606086]
        ])

        self.resolution=np.array([0,0])
        self.resolution[0]=rospy.get_param('/%s/camera_node/res_w' %rospy.get_namespace().strip("/"))
        self.resolution[1]=rospy.get_param('/%s/camera_node/res_h' %rospy.get_namespace().strip("/"))

        #hsv range for yellow duckies
        self.yellow_low = np.array([14, 100, 80]) # np.array([103, 140, 140])
        self.yellow_high = np.array([35, 255, 255]) # np.array([123, 190, 190])


        self.sub_image = rospy.Subscriber("~image", CompressedImage, self.processImage,
                                            buff_size=921600, queue_size=1)

        self.sub_switch = rospy.Subscriber("~switch", BoolStamped,
                                           self.cbSwitch, queue_size=1)

        self.sub_lane_pose = rospy.Subscriber("~lane_pose", LanePose, self.cbLanePose, queue_size=1)

        self.sub_info = rospy.Subscriber(
            "~camera_info", CameraInfo, self.cb_process_camera_info, queue_size=1
        )

        self.pub_boundingbox_image = rospy.Publisher("~duckiedetected_image/compressed",
                                                       CompressedImage, queue_size=1)

        self.pub_mask_image = rospy.Publisher("~duckiedetected_mask/compressed",
                                                      CompressedImage, queue_size=1)
        self.pub_time_elapsed = rospy.Publisher("~detection_time",
                                                Float32, queue_size=1)

        self.pub_duckie = rospy.Publisher("~detected_duckie",
                                                dynamic_obstacle, queue_size=1)

        rospy.loginfo("[%s] Initialization completed" % (self.node_name))

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)
        rospy.loginfo("[%s] %s = %s " % (self.node_name, param_name, value))
        return value

    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data

    def cb_process_camera_info(self, msg):
        self.pcm.fromCameraInfo(msg)

    def processImage(self, image_msg):
        if not self.active:
            return

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        start = rospy.Time.now()

        #crop upper percentage and lower 1/4 of image
        cv_image_crop = cv_image[int(cv_image.shape[0]*self.crop_factor):int(cv_image.shape[0]*3/4)][:]
        hsv_img = cv2.cvtColor(cv_image_crop, cv2.COLOR_BGR2HSV)

        #create a yellow mask
        mask = cv2.inRange(hsv_img, self.yellow_low, self.yellow_high)

        # TODO: improve duckie detection

        #initialize arrays
        duckiefound = False
        duckie_loc_pix = Pixel()
        duckie_msg = dynamic_obstacle()
        duckie_pos_arr = []
        duckie_pos_arr_new = []
        duckie_state_arr = []
        keypoints=[]
        prev_detected=[]
        i=0

        #initalize parameters for blob detector
        #minInertiaRatio is especially important, it filters out the elongated lane segments
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 60
        params.filterByInertia = True
        params.minInertiaRatio = 0.5 #if high ratio: only compact blobs are detected, elongated blobs are filtered out
        params.filterByConvexity = False
        params.maxConvexity = 0.99
        params.filterByCircularity = False
        params.minCircularity = 0.5
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs and draw them in image and in mask
        keypoints = detector.detect(mask)
        t = cv2.drawKeypoints(cv_image_crop, keypoints, cv_image_crop, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        t = cv2.drawKeypoints(mask, keypoints, mask, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #loop to check location of duckies to see if they are in the lane
        if keypoints:
            duckiefound = True
            for key in keypoints:
                duckie_loc_pix.u=key.pt[0]
                duckie_loc_pix.v=key.pt[1]+key.size/2+float(self.resolution[1])*self.crop_factor #to compensate for crop

                #project duckie location to ground
                duckie_loc_world = self.pixel2ground(duckie_loc_pix)
                duckie_pos_arr_new.append([duckie_loc_world.x,duckie_loc_world.y])

                #calculate distance of duckie to the middle line
                duckie_side = np.cos(self.phi)*(duckie_loc_world.y+self.d)+np.sin(self.phi)*duckie_loc_world.x

                # rospy.loginfo("[%s] ------- " % (self.node_name))
                # rospy.loginfo("[%s] Duckie detected at (%f,%f) in pixel coordinates" % (self.node_name,duckie_loc_pix.u,duckie_loc_pix.v))
                # rospy.loginfo("[%s] Duckie detected at (%f,%f) in world coordinates" % (self.node_name,duckie_loc_world.x,duckie_loc_world.y))
                # rospy.loginfo("[%s] Duckie at p(%f,%f)   w[%f,%f]   %fm" % (self.node_name,duckie_loc_pix.u,duckie_loc_pix.v,duckie_loc_world.x,duckie_loc_world.y,duckie_side))

                #tracking for outlier prevention: add duckie only if it has been detected already previously close by (within a range of 10cm)
                # for a in range(len(self.duckie_pos_arr_prev)):
                #     dist = np.sqrt((duckie_loc_world.x-self.duckie_pos_arr_prev[a][0])**2+(duckie_loc_world.y-self.duckie_pos_arr_prev[a][1])**2)
                #     prev_detected.append([dist<0.1])

                #check if duckie is on the left or right lane
                if abs(duckie_side)<self.lane_width: #right lane
                    duckie_state_arr.append([1])
                    duckie_pos_arr.append([duckie_loc_world.x,duckie_loc_world.y])
                    rospy.loginfo("[%s] Duckie on right lane at %fm" % (self.node_name, duckie_loc_world.x))
                elif duckie_side>self.lane_width and duckie_side<self.lane_width*2: #left lane
                    duckie_state_arr.append([2])
                    duckie_pos_arr.append([duckie_loc_world.x,duckie_loc_world.y])
                    rospy.loginfo("[%s] Duckie on left lane at %fm" % (self.node_name, duckie_loc_world.x))

                i=i+1

        self.duckie_pos_arr_prev = duckie_pos_arr_new

        if not duckie_state_arr:
            duckie_state_arr.append([0]) #write zero if no duckie detected

        # fill information into the duckie_msg and publish it
        duckie_msg.pos=np.array(duckie_pos_arr).ravel()
        duckie_msg.state=np.array(duckie_state_arr).ravel()
        duckie_msg.header.stamp = rospy.Time.now()
        self.pub_duckie.publish(duckie_msg)



        # image_msg_out = self.bridge.cv2_to_imgmsg(cv_image_crop,"bgr8")
        # image_mask_msg_out = self.bridge.cv2_to_imgmsg(mask,"passthrough")
        compress_image_msg_out = self.bridge.cv2_to_compressed_imgmsg(cv_image_crop)
        compress_image_mask_msg_out = self.bridge.cv2_to_compressed_imgmsg(mask)

        self.pub_boundingbox_image.publish(compress_image_msg_out)
        self.pub_mask_image.publish(compress_image_mask_msg_out)

        elapsed_time = (rospy.Time.now() - start).to_sec()
        self.pub_time_elapsed.publish(elapsed_time)

    def cbLanePose(self,lane_pose_msg): #callback to lane pose
        self.d = lane_pose_msg.d
        self.phi = lane_pose_msg.phi
        # rospy.loginfo("[%s] Lane pose received" % (self.node_name))

    def pixel2ground(self,pixel):
        uv_raw = np.array([pixel.u, pixel.v])
        uv_raw = np.append(uv_raw, np.array([1]))
        ground_point = np.dot(self.H, uv_raw)
        point = Point()
        x = ground_point[0]
        y = ground_point[1]
        z = ground_point[2]
        point.x = x/z
        point.y = y/z
        point.z = 0.0
        return point     

if __name__ == '__main__':
    duckie_detection_node = DuckieDetectionNode(node_name='duckie_detection_node')

    rospy.spin()
