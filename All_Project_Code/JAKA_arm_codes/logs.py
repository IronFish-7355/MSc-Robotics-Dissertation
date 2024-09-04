# D:\Python_Virtual_Enviroment\Python_3.8\python.exe D:\YOLOV5\yolov5-7.0\yolov5-7.0\JAKA_arm_codes\JAKA_hand_eye_grasp_traj_V1.1.py
# Attempting to connect to robot, IP address: 10.5.5.100
# Starting login...
# login succeeded
# Powering on the robot...
# power on succeeded
# Enabling the robot...
# enable succeeded
# Retrieving joint angles...
# retrieve joint angles succeeded
# Current joint angles are: [180.00208960876103, 70.72678353295078, -90.56370299113053, 180.00224178095092, 90.78200804394717, 0.00010640204534865608]
# Retrieving end effector position...
# retrieve end effector position succeeded
# Current end effector position is: [-195.4064646996781, 5.979399078580832, 332.3668816188086, 0.3598681729070412, -1.5015442172120722e-05, -1.5707237054245935]
# Initializing robot arm to start pose
# moving to start pose succeeded
# Robot is at start pose
# Key pressed: 115
# Selection mode activated.
# Selected object index: 1
# Key pressed: 115
# Selection mode activated.
# Selected object index: 2
# Key pressed: 99
# Median Coordinates: [-335.29943284  343.23047148  137.72974253]
# Confirmed coordinates: [-335.29943284  343.23047148  137.72974253]
# Final coords: [-335.29943284  343.23047148  137.72974253]
# Object rz angle: -175.7 degrees
# Confirmation mode activated. Press 'g' to start robot movement.
# Key pressed: 103
# Calculated Arm_stop_pre: (-331.52410774993143, 293.3732060060689, 137.72974253201988, 0.0, 0.0, -175.66967475646553)
# Starting robot_grasp_pre_non_blocking...
# Retrieving joint angles...
# Current joint angles: [180.0, 70.0, -90.0, 180.0, 90.0, 0.0]
# Target Arm_stop_pre: (-331.52410774993143, 293.3732060060689, 137.72974253201988, 0.0, 0.0, -175.66967475646553)
# Calculating inverse kinematics...
# Failed to calculate inverse kinematics. Error code: -4
# This usually means the target position is unreachable.
# Check if the Arm_stop_pre coordinates are within the robot's workspace.
# Failed to set target. Movement not started.
# Failed to start robot movement.

import jkrc
import time
import math

def check_ret(ret, fun_name):
    if ret[0] == 0:
        print(f"{fun_name}成功")
    else:
        print(f"{fun_name}失败,错误码:{ret[0]}")
        print("程序退出")
        exit()

def deg_to_rad(degrees):
    return [d * math.pi / 180 for d in degrees]

# 修改机器人控制器的IP地址
robot_ip = "10.5.5.100"
use_real_robot = True

# 定义五个固定位置的关节角度（转换为弧度）
position1 = deg_to_rad([-63, 34, 47, -1, 106, -115])
position2 = deg_to_rad([-5, 24, 47, -1, 106, -115])
position3 = deg_to_rad([46, 34, 47, -1, 106, -115])
position4 = deg_to_rad([30, 45, 37, -1, 106, -115])
position5 = deg_to_rad([-15, 30, 37, -1, 106, -115])

print(f"尝试连接机器人,IP地址:{robot_ip}")
robot = jkrc.RC(robot_ip)

print("开始登录...")
ret = robot.login()
check_ret(ret, "登录")

if use_real_robot:
    print("机器人上电中...")
    ret = robot.power_on()
    check_ret(ret, "上电")

    print("机器人使能中...")
    ret = robot.enable_robot()
    check_ret(ret, "使能")

print("获取关节角度中...")
angles = robot.get_joint_position()
check_ret(angles, "获取关节角度")
print(f"当前关节角度为:{angles[1]}")

print("获取末端位姿中...")
pose = robot.get_tcp_position()
check_ret(pose, "获取末端位姿")
print(f"当前机器人末端位姿为:{pose[1]}")

print("移动到位置1...")
ret = robot.joint_move(position1, 0, True, 30)
check_ret(ret, "移动到位置1")
print("移动到位置1完成")

print("移动到位置2...")
ret = robot.joint_move(position2, 0, True, 30)
check_ret(ret, "移动到位置2")
print("移动到位置2完成")

print("移动到位置3...")
ret = robot.joint_move(position3, 0, True, 30)
check_ret(ret, "移动到位置3")
print("移动到位置3完成")

print("移动到位置4...")
ret = robot.joint_move(position4, 0, True, 30)
check_ret(ret, "移动到位置4")
print("移动到位置4完成")

print("移动到位置5...")
ret = robot.joint_move(position5, 0, True, 30)
check_ret(ret, "移动到位置5")
print("移动到位置5完成")

print("退出登录中...")
ret = robot.logout()
check_ret(ret, "退出登录")

print("程序运行完毕")

