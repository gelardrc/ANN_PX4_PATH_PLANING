#!/usr/bin/env python2
#***************************************************************************
#
#   Copyright (c) 2015 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#***************************************************************************/

#
# @author Andreas Antener <andreas@uaventure.com>
#
# The shebang of this file is currently Python2 because some
# dependencies such as pymavlink don't play well with Python3 yet.
from __future__ import division

PKG = 'px4'

import rospy
from geometry_msgs.msg import Quaternion, Vector3, Point,TwistStamped
from mavros_msgs.msg import AttitudeTarget,PositionTarget
from mavros_test_common import MavrosTestCommon
from pymavlink import mavutil
from six.moves import xrange
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from threading import Thread
from tf.transformations import quaternion_from_euler


class MavrosOffboardAttctlTest(MavrosTestCommon):
    """
    Tests flying in offboard control by sending attitude and thrust setpoints
    via MAVROS.

    For the test to be successful it needs to cross a certain boundary in time.
    """

    def setUp(self):
        super(MavrosOffboardAttctlTest, self).setUp()

        self.att = AttitudeTarget()

        self.jesus = PositionTarget()
        self.jesus.header = Header()
        self.jesus.position = Point()
        self.jesus.velocity = Vector3()
        self.jesus.acceleration_or_force = Vector3()

        self.att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)

        self.god = rospy.Publisher('mavros/setpoint_raw/local',PositionTarget,queue_size=10)


        # send setpoints in seperate thread to better prevent failsafe
        self.att_thread = Thread(target=self.send_att, args=())
        self.att_thread.daemon = True
        self.att_thread.start()

    def tearDown(self):
        super(MavrosOffboardAttctlTest, self).tearDown()

    #
    # Helper methods
    #
    def send_att(self):
        rate = rospy.Rate(10)  # Hz
        self.att.body_rate = Vector3()
        self.att.header = Header()
        self.att.header.frame_id = "base_footprint"
        self.att.orientation = Quaternion(*quaternion_from_euler(-0.25, 0.15,
                                                                 0))
        self.att.thrust = 0.7
        self.att.type_mask = 7  # ignore body rate



        self.jesus.header.frame_id = "drone"
        self.jesus.type_mask = 3576  # tudo mesmo o pose x y z
        self.jesus.coordinate_frame = 9  # off borard mode algo assim
        self.jesus.position.x = 1
        self.jesus.position.y = 2
        self.jesus.position.z = 3
        #self.jesus.velocity.x = 1
        #self.jesus.velocity.y = 2
        #self.jesus.velocity.z = 3
        #self.jesus.acceleration_or_force.x = 1
        #self.jesus.acceleration_or_force.y = 2
        #self.jesus.acceleration_or_force.z = 3
        #self.jesus.yaw = 1


        while not rospy.is_shutdown():
            self.att.header.stamp = rospy.Time.now()
            self.jesus.header.stamp = rospy.Time.now()
            #self.att_setpoint_pub.publish(self.att)
            self.god.publish(self.jesus)
	    try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    #
    # Test method
    #
    def test_attctl(self):
        """Test offboard attitude control"""
        # boundary to cross
        boundary_x = 200
        boundary_y = 100
        boundary_z = 20

        # make sure the simulation is ready to start the mission
        self.wait_for_topics(60)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   10, -1)

        self.log_topic_vars()
        self.set_mode("OFFBOARD", 5)
        self.set_arm(True, 5)

        rospy.loginfo("run mission")
        rospy.loginfo("attempting to cross boundary | x: {0}, y: {1}, z: {2}".
                      format(boundary_x, boundary_y, boundary_z))
        # does it cross expected boundaries in 'timeout' seconds?
        timeout = 90  # (int) seconds
        loop_freq = 2  # Hz
        rate = rospy.Rate(loop_freq)
        crossed = False
        for i in xrange(timeout * loop_freq):
            if (self.local_position.pose.position.x > boundary_x and
                    self.local_position.pose.position.y > boundary_y and
                    self.local_position.pose.position.z > boundary_z):
                rospy.loginfo("boundary crossed | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                crossed = True
                break

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

        self.assertTrue(crossed, (
            "took too long to cross boundaries | current position x: {0:.2f}, y: {1:.2f}, z: {2:.2f} | timeout(seconds): {3}".
            format(self.local_position.pose.position.x,
                   self.local_position.pose.position.y,
                   self.local_position.pose.position.z, timeout)))

        self.set_mode("AUTO.LAND", 5)
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,
                                   90, 0)
        self.set_arm(False, 5)


if __name__ == '__main__':
    import rostest
    rospy.init_node('test_node', anonymous=True)

    rostest.rosrun(PKG, 'mavros_offboard_attctl_test',
                   MavrosOffboardAttctlTest)
