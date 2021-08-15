# -*- coding: utf-8 -*-

import rospy
import mavros
from geometry_msgs.msg import PoseStamped,Point
from mavros_msgs.msg import State 
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import LaserScan
import numpy as np
import tensorflow as tf

#from keras.models import load_model
from tensorflow.keras.models import load_model


# callback method for state sub
current_state = State() 
offb_set_mode = SetMode

choques = [0,0,0,0,0,0]

lidar0 = 0.00
lidar1 = 0.00
lidar2 = 0.00
lidar3 = 0.00
lidar4 = 0.00
lidar5 = 0.00

poses = Point()

mission = PoseStamped()

missao = Point()

model = load_model('/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dritk_qualificacao.h5')


def state_cb(state):
    global current_state
    current_state = state

############## CALL BACKS ##############

def pose_callback(pose):
    global poses
    poses = pose.pose.position

def lidar0_callback(sensors):
    global lidar0
    lidar00  = np.array(sensors.ranges)
    lidar0 = lidar00[0]
def lidar1_callback(sensors):
    global lidar1
    lidar11  = np.array(sensors.ranges)
    lidar1 = lidar11[0]
def lidar2_callback(sensors):
    global lidar2
    lidar22  = np.array(sensors.ranges)
    lidar2 = lidar22[0]
def lidar3_callback(sensors):
    global lidar3
    lidar33  = np.array(sensors.ranges)
    lidar3 = lidar33[0]
def lidar4_callback(sensors):
    global lidar4
    lidar44  = np.array(sensors.ranges)
    lidar4 = lidar44[0]
def lidar5_callback(sensors):
    global lidar5
    lidar55  = np.array(sensors.ranges)
    lidar5 = lidar55[0]

########## funções auxialiares ###########

def choque(ranges):
    
    choques[0] = ranges[0]
    choques[1] = ranges[1]
    choques[2] = ranges[2]
    choques[3] = ranges[3]  
    choques[4] = ranges[4]
    choques[5] = ranges[5]  
    
    for i in range(len(choques)):
        teste = float(choques[i])
        if choques[i] == float('inf') : 
            choques[i] =0 
        else:
            choques[i]=1
    
    return choques

def entrada(pose,target,sensores):
    entrada = np.zeros((1,12))
    entrada[0,0]=pose.x
    entrada[0,1]=pose.y
    entrada[0,2]=pose.z
    entrada[0,3]=target[0]
    entrada[0,4]=target[1]
    entrada[0,5]=target[2]
    entrada[0,6]=sensores[0]
    entrada[0,7]=sensores[1]
    entrada[0,8]=sensores[2]
    entrada[0,9]=sensores[3]
    entrada[0,10]=sensores[4]
    entrada[0,11]=sensores[5]
   #entrada[0,12]=sensores[6]
    
    #entrada[0,7]=pose.y
    #entrada[0,8]=pose.z
    #entrada[0,6]=pose.x
    #entrada[0,7]=pose.y
    #entrada[0,8]=pose.z
    #entrada[0,3:6] = target
    #entrada[0,6:12] = sensores
    
    return entrada

def acao(output):
    global poses
    global mission
    vencedor = 0
    # ve qual e o maior output
    for i in range(6):
        if output[0,i]>vencedor:
            vencedor = output[0,i]
            acao = i
            
    #print('acao : ', acao)
    if acao ==0:
        mission.pose.position.x =poses.x + 0
        mission.pose.position.y =poses.y + 0
        mission.pose.position.z =poses.z + 1 
    if acao ==1:
        mission.pose.position.x =poses.x + 0 
        mission.pose.position.y =poses.y + 0
        mission.pose.position.z =poses.z - 1 
    if acao ==2:
        mission.pose.position.x =poses.x + 0
        mission.pose.position.y =poses.y + 1
        mission.pose.position.z =poses.z + 0 
    if acao ==3:
        mission.pose.position.x =poses.x + 0
        mission.pose.position.y =poses.y - 1
        mission.pose.position.z =poses.z + 0 
    if acao ==4:
        mission.pose.position.x =poses.x + 1
        mission.pose.position.y =poses.y + 0
        mission.pose.position.z =poses.z + 0 
    if acao ==5:
        mission.pose.position.x =poses.x - 1
        mission.pose.position.y =poses.y + 0
        mission.pose.position.z =poses.z + 0
    
    return mission

########## MAVROS ##################

local_pos_pub = rospy.Publisher("mavros/setpoint_position/local",PoseStamped, queue_size = 10)
rospy.Subscriber('mavros/local_position/pose',PoseStamped , pose_callback)
state_sub = rospy.Subscriber('/mavros/state', State , state_cb)
arming_client = rospy.ServiceProxy("mavros/cmd/arming",CommandBool)
set_mode_client = rospy.ServiceProxy("mavros/set_mode",SetMode) 
rospy.Subscriber('/drone/lidar0/scan',LaserScan , lidar0_callback)
rospy.Subscriber('/drone/lidar1/scan',LaserScan , lidar1_callback)
rospy.Subscriber('/drone/lidar2/scan',LaserScan , lidar2_callback)
rospy.Subscriber('/drone/lidar3/scan',LaserScan , lidar3_callback)
rospy.Subscriber('/drone/lidar4/scan',LaserScan , lidar4_callback)
rospy.Subscriber('/drone/lidar5/scan',LaserScan , lidar5_callback)

########## MAVROS ##################




pose = PoseStamped()
missao = PoseStamped()
pose.pose.position.x = 0
pose.pose.position.y = 0
pose.pose.position.z = 2


def position_control():
    rospy.init_node('offb_node', anonymous=True)
    prev_state = current_state
    rate = rospy.Rate(20.0) # MUST be more then 2Hz

    # send a few setpoints before starting
    for i in range(100):
        local_pos_pub.publish(pose)
        print("Aguardando")
        rate.sleep()
    
    # wait for FCU connection
    while not current_state.connected:
        rate.sleep()

    last_request = rospy.get_rostime()
    while not rospy.is_shutdown():
        now = rospy.get_rostime()
        if current_state.mode != "OFFBOARD" and (now - last_request > rospy.Duration(5.)):
            set_mode_client(base_mode=0, custom_mode="OFFBOARD")
            last_request = now 
        else:
            if not current_state.armed and (now - last_request > rospy.Duration(5.)):
               arming_client(True)
               last_request = now 

        # older versions of PX4 always return success==True, so better to check Status instead
        if prev_state.armed != current_state.armed:
            rospy.loginfo("Vehicle armed: %r" % current_state.armed)
        if prev_state.mode != current_state.mode: 
            rospy.loginfo("Current mode: %s" % current_state.mode)
        prev_state = current_state

        # Update timestamp and publish pose 
        
        ranges=[lidar0,lidar1,lidar2,lidar3,lidar4,lidar5]
        #print('ranges :',ranges)
        choqui = choque(ranges)
        
        poses.x = round(poses.x)
        poses.y = round(poses.y)
        poses.z = round(poses.z)

        input = entrada(poses,[5,5,5],choqui)

        #print('entrada :',input)
        output = model.predict(input)
        #print(output)
        missao = acao(output)
        #print(missao)
        missao.header.stamp = rospy.Time.now()
        print('posicao -->',poses)
        if poses.x != 5 or poses.y != 5 or poses.z != 5:
            local_pos_pub.publish(missao)
            rate.sleep()
        

if __name__ == '__main__':
    try:
        position_control()
    except rospy.ROSInterruptException:
        pass
