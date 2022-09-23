#!/usr/bin/env python

from math import pi, sqrt, atan2, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D


class Controller:
    def __init__(self, P=0.0, D=0.0, Derivator=0):
        self.Kp = P
        self.Kd = D
        self.Derivator = Derivator
        self.set_point = 0.0
        self.error = 0.0

    def update(self, current_value):
        # calculate P_value and D_value; u = P_value + D_value
	# current_value = y; set_point = r
	# r - y = error
	# NOTE: u is omega (angular vel)
	errorOld = self.error
	self.error = self.set_point - current_value
	if self.error > pi:
              self.error = self.error - 2*pi
        elif self.error < -pi:
              self.error = self.error + 2*pi

	self.Derivator = (self.error - errorOld)/0.1 #0.1 Seconds Time Interval corresponding to messaging rate
	P_value = self.Kp * self.error
	D_value = self.Kd * self.Derivator
	u = P_value + D_value
        return u

    def setPoint(self, set_point):
        self.set_point = set_point
        self.Derivator = 0
    
    def setPD(self, set_P, set_D): #MANUALLY TUNE
        self.Kp = set_P
        self.Kd = set_D

class Turtlebot():
    def __init__(self):
        rospy.init_node("turtlebot_move")
        rospy.loginfo("Press Ctrl + C to terminate")
        self.vel_pub = rospy.Publisher("cmd_vel_mux/input/navi", Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        #Global variable intialization to save position and velocity
        self.previousp_endx = 0
        self.previousp_endy = 0.5
        self.previousv_endx = 0
        self.previousv_endy = 0

        # reset odometry to zero
        self.reset_pub = rospy.Publisher("mobile_base/commands/reset_odometry", Empty, queue_size=10)
        for i in range(10):
            self.reset_pub.publish(Empty())
            self.rate.sleep()
        
        # subscribe to odometry
        self.pose = Pose2D()
        self.logging_counter = 0
        self.trajectory = list()
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

        try:
            self.run()
        except rospy.ROSInterruptException:
            rospy.loginfo("Action terminated.")
        finally:
            # save trajectory into csv file
            np.savetxt('trajectory.csv', np.array(self.trajectory), fmt='%f', delimiter=',')


    def run(self):
        start = (0, 0.5)
        goal = (4.5, 0.5)
        obstacles = [(-0.5, 0), (-0.5, -0.5), (-0.5, 0.5), (-0.5, 1.0), (-0.5, -1.0), (-0.5, 1.5), (-0.5, 2.0), (0.0, 2.0), (0.5, 2.0),\
                      (1.0, 2.0), (1.5, 2.0), (2.0, 2.0), (2.5, 2.0), (3.0, 2.0), (3.5, 2.0), (4.0, 2.0), (4.5, 2.0), (5.0, 2.0), (5.0, 1.5),\
                      (5.0, 1.0), (5.0, 0.5), (5.0, 0.0), (5.0, -0.5), (5.0, -1.0), (4.5, -1.0), (4.0, -1.0), (3.5, -1.0), (3.0, -1.0), (2.5, -1.0),\
                      (2.0, -1.0), (1.5, -1.0), (1.0, -1.0), (0.5, -1.0), (0.0, -1.0), (1.5, 0.0), (1.5, 0.5), (1.5, -0.5), (3.5, 0.5), (3.5, 1.0),\
                      (3.5, 1.5)]
        waypoints = get_path_from_A_star(start, goal, obstacles)
        for i in range(len(waypoints)-1):
            self.move_to_point(waypoints[i], waypoints[i+1])


    def move_to_point(self, current_waypoint, next_waypoint):
        # Generate polynomial trajectory and move to current_waypoint
        # Next_waypoint is to help determine the velocity to pass current_waypoint

	#Vector intialization of t, position and velocity:
        t = np.arange(0.0,3.1,0.1)
        x = np.empty(31, dtype=float)
        x_dot = np.empty(31, dtype=float)
        y = np.empty(31, dtype=float)
        y_dot = np.empty(31, dtype=float)
        m_dot = np.empty(31, dtype=float)

        previousp = np.empty(2, dtype=float)

	#Intialization of start and end positions:
        x0 = self.previousp_endx
        x1 = current_waypoint[0]
        y0 = self.previousp_endy
        y1 = current_waypoint[1]

        previousp[0] = x0
        previousp[1] = y0

        xv_s = self.previousv_endx
        yv_s = self.previousv_endy

	#Intialization of iteration variables:
        i = 0
        indx = 0

        #Calculation of End_v
        v_vec = np.subtract(next_waypoint,previousp)
        norm_v = (v_vec/np.linalg.norm(v_vec))*0.1


	#Intialization of Controller and PD:
        vel = Twist()
        pd = Controller()
        pd.setPD(7,0)
        vel.angular.z = pd.update(self.pose.theta)
        self.vel_pub.publish(vel)

	# Call Time Scaling Function:
	ax = self.polynomial_time_scaling_3rd_order(x0,xv_s,x1,norm_v[0],3)
        a0x = ax[3]
        a1x = ax[2]
        a2x = ax[1]
        a3x = ax[0]

        ay = self.polynomial_time_scaling_3rd_order(y0,yv_s,y1,norm_v[1],3)
        a0y = ay[3]
        a1y = ay[2]
        a2y = ay[1]
        a3y = ay[0]

	#Calculation of trajectory and velocities for each moment of t given coefficents:
        for i in t:
           x[indx] = a0x + a1x*i + a2x*(i**2) + a3x*(i**3)
           x_dot[indx] = a1x + 2*a2x*i + 3*a3x*(i**2)
           y[indx] = a0y + a1y*i + a2y*(i**2) + a3y*(i**3)
           y_dot[indx] = a1y + 2*a2y*i + 3*a3y*(i**2)
           m_dot[indx] = sqrt((x_dot[indx]**2)+(y_dot[indx]**2)) 
  
           #Send linear and angular velocity commands
           vel.linear.x = m_dot[indx]
           angle = atan2(y_dot[indx], x_dot[indx])
           pd.setPoint(angle)
           vel.angular.z = pd.update(self.pose.theta)
           self.vel_pub.publish(vel)
           self.rate.sleep()
           indx += 1



 	#Saving ending position and velocity
        self.previousp_endx = current_waypoint[0]
        self.previousp_endy = current_waypoint[1]
        self.previousv_endx = norm_v[0]
        self.previousv_endy = norm_v[1]

        pass
           
        
    def polynomial_time_scaling_3rd_order(self, p_start, v_start, p_end, v_end, T):
        # Input: p,v: position and velocity of start/end point
        #        T: the desired time to complete this segment of trajectory (in second)
        # Output: the coefficients of this polynomial
        i0 = p_start
	iT = p_end
	i_dot0 = v_start
	i_dotT = v_end

        T_matrix = np.array([[0,0,0,1],[T**3,T**2,T,1],[0,0,1,0],[3*T**2,2*T,1,0]])
	i_vector = np.array([[i0],[iT],[i_dot0],[i_dotT]])

	# Coefficient Calculation:
	ai = np.dot(np.linalg.inv(T_matrix),i_vector)

        return [ai[0], ai[1], ai[2], ai[3]]

    def h_cost(node, goal):
        val = 0
        val = abs(goal[0]-node[0]) + abs(goal[1]-node[1])
        return val

    def get_path_from_A_star(start, goal, obstacles):
        # input  start: integer 2-tuple of the current grid, e.g., (0, 0)
        #        goal: integer 2-tuple  of the goal grid, e.g., (5, 1)
        #        obstacles: a list of grids marked as obstacles, e.g., [(2, -1), (2, 0), ...]
        # output path: a list of grids connecting start to goal, e.g., [(1, 0), (1, 1), ...]
        #   note that the path should contain the goal but not the start
        #   e.g., the path from (0, 0) to (2, 2) should be [(1, 0), (1, 1), (2, 1), (2, 2)]

        #Intialize Open Dict:
        S_Open = dict()
        S_Open.update({start : 0})

        #Intialize Closed list:
        Closed = list()

        #Intialize Parent:
        parent = dict()

        #Intialize past_cost:
        past_cost = dict()
        for x in range(30):
            for y in range(30):
                past_cost[x*0.5,y*0.5] = 100
                past_cost[-x*0.5,-y*0.5] = 100
                past_cost[-x*0.5,y*0.5] = 100
                past_cost[x*0.5,-y*0.5] = 100


        past_cost[start] = 0

        #Intialize total_cost:
        total_cost = dict()
        for x in range(30):
            for y in range(30):
                total_cost[x*0.5,y*0.5] = None
                total_cost[-x*0.5,-y*0.5] = None
                total_cost[-x*0.5,y*0.5] = None
                total_cost[x*0.5,-y*0.5] = None

        #Intialize Neighbor:
        nbr = []
    
        #Intialize path:
        path = []

        while S_Open:
            #Extract node from Open:
            current = min(S_Open, key=S_Open.get)
            S_Open.pop(current)

            #Add current node to closed list:
            Closed.append(current)

            #Calculate neighboring nodes (4-connect):
            n1 = (current[0] + 0.5, current[1])
            n2 = (current[0] - 0.5, current[1])
            n3 = (current[0], current[1] + 0.5)
            n4 = (current[0], current[1] - 0.5)
            nbr = (n1, n2, n3, n4)

            if current == goal:
                temp = current
                while parent:
                    path.append(temp)
                    temp = parent[current]
                    del parent[current]
                    current = temp
                    if current == start:
                       path.reverse()
                       return path                  
                return path

            for i in range(len(nbr)):
                if nbr[i] not in Closed and nbr[i] not in obstacles:
                   tpc = past_cost[current] + h_cost(current,nbr[i])
                   if tpc < past_cost[nbr[i]]:
                      past_cost[nbr[i]] = tpc
                      parent[nbr[i]] = current
                      total_cost[nbr[i]] = past_cost[nbr[i]] + h_cost(nbr[i], goal)
                      S_Open.update({nbr[i] : total_cost[nbr[i]]})
    
        return

    def odom_callback(self, msg):
        # get pose = (x, y, theta) from odometry topic
        quarternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,\
                    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quarternion)
        self.pose.theta = yaw
        self.pose.x = msg.pose.pose.position.x
        self.pose.y = msg.pose.pose.position.y

        # logging once every 100 times (Gazebo runs at 1000Hz; we save it at 10Hz)
        self.logging_counter += 1
        if self.logging_counter == 100:
            self.logging_counter = 0
            self.trajectory.append([self.pose.x, self.pose.y])  # save trajectory
            rospy.loginfo("odom: x=" + str(self.pose.x) +\
                ";  y=" + str(self.pose.y) + ";  theta=" + str(yaw))

if __name__ == '__main__':
    whatever = Turtlebot()
