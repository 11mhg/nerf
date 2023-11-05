import os, sys
import math
import numpy as np

def euler_to_rot_mat(yaw, pitch, roll, radians = False):
    if not radians:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

    yaw = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    return yaw @ pitch @ roll


def euler_from_rot_mat(R):
    """
    R = [
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33],
    ]
    """

    R = R[:3, :3].reshape(3, 3)
    
    if math.fabs(R[2, 0]) != 1. :
        pitch_1 = -1 * np.arcsin(R[2, 0])
        #pitch_2 = np.pi - pitch_1

        roll_1 = np.arctan2(R[2, 1] / np.cos(pitch_1), R[2, 2] / np.cos(pitch_1))
        #roll_2 = np.arctan2(R[2, 1] / np.cos(pitch_2), R[2, 2] / np.cos(pitch_2))
        
        yaw_1 = np.arctan2(R[1, 0] / np.cos(pitch_1), R[0, 0] / np.cos(pitch_1))
        #yaw_2 = np.arctan2(R[1, 0] / np.cos(pitch_2), R[0, 0] / np.cos(pitch_2))

        pitch = pitch_1
        roll = roll_1
        yaw = yaw_1
    else:
        yaw = 0
        if R[2, 0] == -1:
            pitch = np.pi / 2.
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2.
            roll = (-1*yaw) + np.arctan2(-1*R[0, 1], -1*R[0, 2])

    roll = roll * 180.0 / np.pi
    pitch = pitch * 180.0 / np.pi
    yaw = yaw * 180.0 / np.pi

    rxyz_deg = np.array([roll, pitch, yaw])
    return rxyz_deg
