#!/usr/bin/env python3
import os
import rospy
import numpy as np
import time
import math
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String, Float64, Float32, Float64MultiArray
from geometry_msgs.msg import Point32, Point
from duckietown_msgs.msg import BoolStamped, Twist2DStamped, LEDPattern, FSMState
from duckietown_msgs.srv import SetCustomLEDPattern
from dynamic_obstacle_avoidance.msg import dynamic_obstacle


class Dynamic_Controller(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(Dynamic_Controller, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")


        self.nr_steps = 4   #nr of steps to split the d_offset during transition from one lane to another
        self.transition_time = 4.0  #sec, time for transition from one lane to another
        self.lanewidth = 0.22   #m, width of one lane
        self.leftlane_time = 3  #sec, time spent on left lane during overtaking


        # variable definition and initialisation
        self.stop = False
        self.stop_prev = False
        self.overtaking = False

        self.duckie_left_state = False
        self.duckie_right_state = False
        self.duckie_left_pose = 0
        self.duckie_right_pose = 0

        self.d_offset = 0.0
        self.gain_calib = rospy.get_param("/%s/kinematics_node/gain" %self.veh_name)
        self.gain = self.gain_calib
        self.gain_overtaking = 1.3 * self.gain_calib
        self.dist_max = 0.7

        self.fsm_state = "NORMAL_JOYSTICK_CONTROL"

        # construct publisher
        self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)
        self.sub_duckie = rospy.Subscriber('/%s/duckie_detection_node/detected_duckie' %self.veh_name, dynamic_obstacle, self.cbDuckie, queue_size=1)
        self.sub_car_cmd = rospy.Subscriber("~car_cmd", Twist2DStamped, self.cbCarCmd, queue_size=1)

        self.car_cmd_pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size = 1)
        self.fsm_mode_pub = rospy.Publisher("~fsm_mode", FSMState, queue_size = 1)

        # changing LED to two white in the front and two red in the back
        self.led_pattern_srv = rospy.ServiceProxy('/%s/led_emitter_node/set_custom_pattern' %self.veh_name, SetCustomLEDPattern)
        self.led_pattern = LEDPattern()
        self.led_pattern.color_list = ['white', 'switchedoff', 'white', 'yellow', 'yellow'] # FL, !, FR, RR, RL
        self.led_pattern.color_mask = [1,1,1,1,1]
        self.led_pattern.frequency = 0.0
        self.led_pattern.frequency_mask = [1,1,1,1,1]

        self.led_state = "drive" # drive, reverse, signal_left, signal_right, stop, emergency_stop

        self.led_pattern_srv(self.led_pattern)

        self.last_stop_time = rospy.Time.now()  

        self.led_states = ["drive", "signal_left", "signal_right", "stop", "emergency_stop"]
        self.current_index = 0

        rospy.loginfo("[%s] initialized" %self.node_name)

    def cbMode(self,fsm_state_msg):
        self.fsm_state = fsm_state_msg.state    # String of current FSM state
        rospy.loginfo("[%s] fsm_state changed to %s  " % (self.node_name, self.fsm_state))

    def cbDuckie(self,msg):
        state = np.array(msg.state)
        self.duckie_right_state = any(state==1)
        self.duckie_left_state = any(state==2)

        if self.duckie_right_state:
            self.duckie_right_pose = msg.pos[2*np.argwhere(state==1)[0][0]] # only x, only take first detected duckie
            # rospy.loginfo("[%s] duckie right detected" %self.node_name)
        else:
            self.duckie_right_pose = 0

        if self.duckie_left_state:
            self.duckie_left_pose = msg.pos[2*np.argwhere(state==2)[0][0]] # only x, only take first detected duckie
            # rospy.loginfo("[%s] duckie left detected" %self.node_name)
        else:
            self.duckie_left_pose = 0

    # velocity and omega of car_cmd are set to zero if there is a stop
    def cbCarCmd(self, car_cmd_msg):
        car_cmd_msg_current = Twist2DStamped()
        car_cmd_msg_current = car_cmd_msg
        car_cmd_msg_current.header.stamp = rospy.Time.now()
        if self.stop or self.stop_prev:
            car_cmd_msg_current.omega = 0
            car_cmd_msg_current.v = 0
        self.stop_prev = self.stop
        # if self.stop and (rospy.Time.now() - self.last_stop_time).to_sec() < 2:
        #     car_cmd_msg_current.omega = 0
        #     car_cmd_msg_current.v = 0
        self.car_cmd_pub.publish(car_cmd_msg_current)
        # rospy.loginfo("[%s] car_cmd published: %f" % (self.node_name, car_cmd_msg_current.v))

    # function which decides when to overtake or stop
    def overwatch(self):

        # self.set_led_state(self.led_states[self.current_index])
        # self.current_index = (self.current_index + 1) % len(self.led_states)
        # rospy.sleep(5)

        if rospy.Time.now() - self.last_stop_time > rospy.Duration(2):
            self.stop = False
            self.set_led_state("drive")

        # self.stop = False
        # self.set_led_state("drive")

        # if self.fsm_state == "LANE_FOLLOWING":
        #     if self.duckie_right_state and self.duckie_left_state:
        #         rospy.logerr("[%s] Stop!" % self.node_name)
        #         self.stop = True
        #         self.stop_prev = True
        #         self.last_stop_time = rospy.Time.now()
        #         self.set_led_state("emergency_stop")

        if self.fsm_state == "LANE_FOLLOWING" and not self.overtaking:
            if self.duckie_right_state: # checking for duckies on right lane
                if not (self.duckie_left_state): # checking for duckies on left lane
                    self.leftlane_time = 2
                    self.dist_max = 1.0
                    if (self.duckie_right_pose > 0.20 and self.duckie_right_pose < self.dist_max): # checking if duckie in overtaking range
                        self.overtake()
                # if self.duckie_left_state: # MAYBE IMPLEMENT 'inversao de marcha'!   
                if (self.duckie_right_pose < 0.20 and self.duckie_right_state): # checking if duckie to close
                    rospy.logerr("[%s] Stop!" % self.node_name)
                    self.set_led_state("stop")
                    self.stop = True
                    # self.last_stop_time = rospy.Time.now()
                else:
                    self.stop = False
                    self.set_led_state("drive")

    # overtaking by adjusting d_offset
    def overtake(self):
        self.overtaking = True
        rospy.loginfo("[%s] overtaking started" % self.node_name)
        rospy.loginfo("[%s] going to the left" % self.node_name)
        self.set_led_state("signal_left")

        for i in range(1,self.nr_steps+1): # changing d_offset as a descrete sine from 0 to pi/2
            self.d_offset = math.sin(i*math.pi/(self.nr_steps*2)) * self.lanewidth
            rospy.set_param("/%s/lane_controller_node/d_offset" %self.veh_name, self.d_offset) #sets d_offset parameter
            rospy.sleep(self.transition_time/float(self.nr_steps))

        # increase speed, atm set_param disabled because lanefollowing cant cope with it
        self.gain = self.gain_overtaking
        # rospy.set_param("/%s/kinematics_node/gain" %self.veh_name, self.gain)

        # checking for duckiebot head while leftlane_time (on the left lane); detected --> emergency stop
        t_start = rospy.get_rostime().secs
        while (t_start + self.leftlane_time) > rospy.get_rostime().secs:
            if self.duckie_left_state: #stop if duckie are facing us on left lane
                rospy.logerr("[%s] Emergency stop while overtaking!" % self.node_name)
                self.set_led_state("emergency_stop")
                self.last_stop_time = rospy.Time.now()
                self.stop = True
                while self.stop and (rospy.Time.now() - self.last_stop_time).to_sec() < 2:
                    if self.duckie_left_state:
                        self.stop = True
                        self.last_stop_time = rospy.Time.now()
                    else:
                        self.stop = False
                    rospy.sleep(0.5)

            #as soon as left lane is free again, we exit stop and continue driving on left lane
            rospy.sleep(0.1)

        # decrease speed, atm set_param disabled because lanefollowing cant cope with it
        self.gain = self.gain_calib     #decrease to normal speed
        # rospy.set_param("/%s/kinematics_node/gain" %self.veh_name, self.gain)

        rospy.loginfo("[%s] going back to the right" % self.node_name)
        self.set_led_state("signal_right")

        for i in range(1,self.nr_steps+1):   # changing d_offset back
            self.d_offset = (1 - math.sin(i*math.pi/(self.nr_steps*2)) )* self.lanewidth
            rospy.set_param("/%s/lane_controller_node/d_offset" %self.veh_name, self.d_offset)
            rospy.sleep(self.transition_time/float(self.nr_steps))
        self.d_offset = 0.0

        rospy.loginfo("[%s] overtaking done" % self.node_name)
        self.overtaking = False

    # Setting the led state has some delay!
    def set_led_state(self, new_led_state):
        if new_led_state != self.led_state:
            if new_led_state == "drive":
                self.led_pattern.color_list = ['white', 'switchedoff', 'white', 'yellow', 'yellow']
                self.led_pattern.color_mask = [1,1,1,1,1]
                self.led_pattern.frequency = 0.0
                self.led_pattern.frequency_mask = [1,1,1,1,1]
                self.led_pattern_srv(self.led_pattern)
                self.led_state = "drive"
            elif new_led_state == "reverse":
                self.led_pattern.color_list = ['white', 'switchedoff', 'white', 'pink', 'pink']
                self.led_pattern.color_mask = [1,1,1,1,1]
                self.led_pattern.frequency = 0.0
                self.led_pattern.frequency_mask = [1,1,1,1,1]
                self.led_pattern_srv(self.led_pattern)
                self.led_state = "reverse"
            elif new_led_state == "signal_left":
                self.led_pattern.color_list = ['yellow', 'switchedoff', 'white', 'yellow', 'yellow']
                self.led_pattern.color_mask = [1,1,1,1,1]
                self.led_pattern.frequency = 1.0
                self.led_pattern.frequency_mask = [1,0,0,0,1]
                self.led_pattern_srv(self.led_pattern)
                self.led_state = "signal_left"
            elif new_led_state == "signal_right":
                self.led_pattern.color_list = ['white', 'switchedoff', 'yellow', 'yellow', 'yellow']
                self.led_pattern.color_mask = [1,1,1,1,1]
                self.led_pattern.frequency = 1.0
                self.led_pattern.frequency_mask = [0,0,1,1,0]
                self.led_pattern_srv(self.led_pattern)
                self.led_state = "signal_right"
            elif new_led_state == "stop":
                self.led_pattern.color_list = ['white', 'switchedoff', 'white', 'red', 'red']
                self.led_pattern.color_mask = [1,1,1,1,1]
                self.led_pattern.frequency = 0.0
                self.led_pattern.frequency_mask = [1,1,1,1,1]
                self.led_pattern_srv(self.led_pattern)
                self.led_state = "stop"
            elif new_led_state == "emergency_stop":
                self.led_pattern.color_list = ['red', 'switchedoff', 'red', 'red', 'red']
                self.led_pattern.color_mask = [1,1,1,1,1]
                self.led_pattern.frequency = 1.0
                self.led_pattern.frequency_mask = [1,0,1,1,1]
                self.led_pattern_srv(self.led_pattern)
                self.led_state = "emergency_stop"


    def run(self):
        # run at 50Hz
        rate = rospy.Rate(50) 
        while not rospy.is_shutdown():
            self.overwatch()    # execute overwatch
            # rospy.loginfo("FSM STATE: %s" %self.fsm_state)

if __name__ == '__main__':
    # create the node
    node = Dynamic_Controller(node_name='dynamic_controller_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
