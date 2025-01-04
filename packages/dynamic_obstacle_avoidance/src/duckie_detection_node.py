#!/usr/bin/env python3
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import BoolStamped, Pixel, LanePose
from duckietown_msgs.srv import ChangePattern
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Float64MultiArray, String
import cv2
import numpy as np
import os
import rospkg
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

        self.publish_debugimg = True #boolean to publish debug image (boundingbox and mask)
        self.lane_width = 0.1145 #half of lane
        self.crop_factor = 0.3 #percentage of image cropped from the top
        self.d = 0.0
        self.phi = 0.0
        self.duckie_pos_arr_prev = []


        self.resolution=np.array([0,0])
        self.resolution[0]=rospy.get_param('/%s/camera_node/res_w' %rospy.get_namespace().strip("/"))
        self.resolution[1]=rospy.get_param('/%s/camera_node/res_h' %rospy.get_namespace().strip("/"))

        try:
            self.intrinsics = load_camera_intrinsics(rospy.get_namespace().strip("/"))
            self.fx = self.intrinsics['K'][0][0]
            self.fy = self.intrinsics['K'][1][1]
        except KeyError as e:
            rospy.logerr(f"Missing key in camera intrinsics: {e}")
            self.intrinsics = {'K': [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 'D': [0, 0, 0, 0, 0]}
            self.fx = self.fy = 1.0
        try:
            self.H = load_homography(rospy.get_namespace().strip("/"))
        except KeyError as e:
            rospy.logerr(f"Missing key in homography: {e}")
            self.H = np.eye(3)
        self.Hinv = np.linalg.inv(self.H)

        #hsv range for yellow duckies
        self.yellow_low = np.array([14, 100, 80])
        self.yellow_high = np.array([35, 255, 255])

        self.sub_image = rospy.Subscriber("~image", CompressedImage, self.processImage,
                                            buff_size=921600, queue_size=1)

        self.sub_switch = rospy.Subscriber("~switch", BoolStamped,
                                           self.cbSwitch, queue_size=1)

        self.sub_lane_pose = rospy.Subscriber("~lane_pose", LanePose, self.cbLanePose, queue_size=1)

        self.pub_boundingbox_image = rospy.Publisher("~duckiedetected_image/compressed",
                                                       CompressedImage, queue_size=1)

        self.pub_mask_image = rospy.Publisher("~duckiedetected_mask/compressed",
                                                      CompressedImage, queue_size=1)
        self.pub_time_elapsed = rospy.Publisher("~detection_time",
                                                Float32, queue_size=1)

        self.pub_duckie = rospy.Publisher("~detected_duckie",
                                                dynamic_obstacle, queue_size=1)

        # self.changePattern = rospy.ServiceProxy("", ChangePattern)
        # self.last_led_state = None

        rospy.loginfo("[%s] Initialization completed" % (self.node_name))

    def setupParam(self, param_name, default_value):
        value = rospy.get_param(param_name, default_value)
        rospy.set_param(param_name, value)
        rospy.loginfo("[%s] %s = %s " % (self.node_name, param_name, value))
        return value

    def cbSwitch(self, switch_msg):
        self.active = switch_msg.data

    def processImage(self, image_msg):
        if not self.active:
            return

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        start = rospy.Time.now()

        # hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # yellow_mask = cv2.inRange(hsv_img, self.yellow_low, self.yellow_high)

        # #publish debug images
        # self.pub_mask_image.publish(self.bridge.cv2_to_imgmsg(yellow_mask, "passthrough"))

        # rospy.loginfo("[%s] Image processed" % (self.node_name))

        # elapsed_time = (rospy.Time.now() - start).to_sec()
        # self.pub_time_elapsed.publish(elapsed_time)

        #crop upper percentage and lower 1/4 of image
        cv_image_crop = cv_image[int(cv_image.shape[0]*self.crop_factor):int(cv_image.shape[0]*3/4)][:]
        hsv_img = cv2.cvtColor(cv_image_crop, cv2.COLOR_BGR2HSV)

        #create a yellow mask
        mask = cv2.inRange(hsv_img, self.yellow_low, self.yellow_high)

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
        params.minArea = 50
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
                # rospy.loginfo("[%s] Duckie detected at %f m to the side of the middle line" % (self.node_name,duckie_side))

                #tracking for outlier prevention: add duckie only if it has been detected already previously close by (within a range of 10cm)
                # for a in range(len(self.duckie_pos_arr_prev)):
                #     dist = np.sqrt((duckie_loc_world.x-self.duckie_pos_arr_prev[a][0])**2+(duckie_loc_world.y-self.duckie_pos_arr_prev[a][1])**2)
                #     prev_detected.append([dist<0.1])

                #check if duckie is on the left or right lane
                if duckie_loc_pix.u >= 250: #right lane
                    duckie_state_arr.append([1])
                    duckie_pos_arr.append([duckie_loc_world.x,duckie_loc_world.y])
                    rospy.loginfo("[%s] Duckie detected on right lane" % (self.node_name))
                elif duckie_loc_pix.u < 250: #left lane
                    duckie_state_arr.append([2])
                    duckie_pos_arr.append([duckie_loc_world.x,duckie_loc_world.y])
                    rospy.loginfo("[%s] Duckie detected on left lane" % (self.node_name))

                i=i+1

        self.duckie_pos_arr_prev = duckie_pos_arr_new

        if not duckie_state_arr:
            duckie_state_arr.append([0]) #write zero if no duckie detected

        # fill information into the duckie_msg and publish it
        duckie_msg.pos=np.array(duckie_pos_arr).ravel()
        duckie_msg.state=np.array(duckie_state_arr).ravel()
        duckie_msg.header.stamp = rospy.Time.now()
        self.pub_duckie.publish(duckie_msg)

        #if boolean is on True: publish debug images (mask and hsv image)
        if self.publish_debugimg:
            image_msg_out = self.bridge.cv2_to_imgmsg(cv_image_crop,"bgr8")
            image_mask_msg_out = self.bridge.cv2_to_imgmsg(mask,"passthrough")

            compress_image_msg_out = self.bridge.cv2_to_compressed_imgmsg(cv_image_crop)
            compress_image_mask_msg_out = self.bridge.cv2_to_compressed_imgmsg(mask)

            self.pub_boundingbox_image.publish(compress_image_msg_out)
            self.pub_mask_image.publish(compress_image_mask_msg_out)

        elapsed_time = (rospy.Time.now() - start).to_sec()
        self.pub_time_elapsed.publish(elapsed_time)

        # try:
        #     self.trigger_led_hazard_light(
        #         duckiefound, False
        #     )
        # except Exception:
        #     self.trigger_led_hazard_light(duckiefound, False)

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
    
    # def trigger_led_hazard_light(self, detection, stopped):
    #     """
    #     Publish a service message to trigger the hazard light at the back of the robot
    #     """
    #     msg = String()

        # if stopped:
        #     msg.data = "OBSTACLE_STOPPED"
        #     if msg.data != self.last_led_state:
        #         self.changePattern(msg)
        #     self.last_led_state = msg.data
        # elif detection:
        #     msg.data = "OBSTACLE_ALERT"
        #     if msg.data != self.last_led_state:
        #         self.changePattern(msg)
        #     self.last_led_state = msg.data
        # else:
        #     if self.state == "LANE_FOLLOWING":
        #         msg.data = "CAR_DRIVING"
        #         if msg.data != self.last_led_state:
        #             self.changePattern(msg)
        #         self.last_led_state = "CAR_DRIVING"
        #     elif self.state == "NORMAL_JOYSTICK_CONTROL":
        #         msg.data = "WHITE"
        #         if msg.data != self.last_led_state:
        #             self.changePattern(msg)
        #         self.last_led_state = msg.data

        # if detection:
        #     msg.data = "OBSTACLE_ALERT"
        #     if msg.data != self.last_led_state:
        #         self.changePattern(msg)
        #     self.last_led_state = msg.data
        # else:
        #     msg.data = "CAR_DRIVING"
        #     if msg.data != self.last_led_state:
        #         self.changePattern(msg)
        #     self.last_led_state = msg.data

if __name__ == '__main__':
    duckie_detection_node = DuckieDetectionNode(node_name='duckie_detection_node')

    rospy.spin()
