import configparser
import numpy as np
import cv2
import threading

cf = configparser.ConfigParser()
cf.read("SN23076.conf")


def get_conf(option, section):
    item = cf.get(option, section)
    return item


def get_matrix(opt):
    fx = float(get_conf(opt, 'fx'))
    fy = float(get_conf(opt, 'fy'))
    cx = float(get_conf(opt, 'cx'))
    cy = float(get_conf(opt, 'cy'))
    k1 = float(get_conf(opt, 'k1'))
    k2 = float(get_conf(opt, 'k2'))
    p1 = float(get_conf(opt, 'p1'))
    p2 = float(get_conf(opt, 'p2'))
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distCoef = np.array([k1, k2, p1, p1])
    # distCoef = np.array([k1, k2, 0, 0])
    return mtx, distCoef


def get_baseline():
    Baseline = float(get_conf("STEREO", 'BaseLine'))
    return Baseline


def undistortion(image, camera_matrix, dist_coef):
    h, w = image.shape[:2]
    # Get new camera matrix with roi
    newCamMtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coef, (w, h), 0, (w, h))
    # Rectify frame
    rec_img = cv2.undistort(image, camera_matrix, dist_coef, None, newCamMtx)
    x, y, w, h = roi
    rec_img = rec_img[y:y + h, x:x + w]
    return rec_img


def frame_crop(frame):
    left_view = frame[0:frame.shape[0], 0:int(frame.shape[1] / 2)]
    right_view = frame[0:frame.shape[0], int(frame.shape[1] / 2) + 1:frame.shape[1]]
    return left_view, right_view


def frame_undistort(camMtx1, camMtx2, distCoef1, distCoef2, frame):
    # Crop full view frame into left and right view
    left_view, right_view = frame_crop(frame)
    # Undistort frame
    rec_left = undistortion(left_view, camMtx1, distCoef1)
    rec_right = undistortion(right_view, camMtx2, distCoef2)
    # Array concatenate using Numpy
    rec_img = np.concatenate([rec_left, rec_right], axis=1)

    return rec_img


def get_distance(video_mode, camMtx1, u1, v1, u2, co_format):
    fx1 = camMtx1[0][0]
    # fx_r = mtx_r[0][0]
    (w, h) = get_resolution(video_mode)
    u = sorted([u1, u2])
    u[1] = u[1] - (w / 2)
    doffs = u[0] - u[1]
    baseline = get_baseline()

    z = fx1 * baseline / doffs
    x = u[0] * z / doffs
    y = v1 * z / doffs

    if co_format is 'uv':
        return u[0], v1, z
    if co_format is 'xy':
        return x, y, z


def get_resolution(opt):
    video_mode = {'CAM_2K': (4416, 1242), 'CAM_FHD': (3840, 1080), 'CAM_HD': (2560, 720), 'CAM_VGA': (1344, 376)}
    (w, h) = video_mode.get(opt)
    return w, h


def set_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def people_3d_coord(ppl, ppl_num, video_mode, camera_matrix):
    coordinates_u = []
    coordinates_v = []
    dists = []
    keypoint_order = {'Nose': 0,
                      'LShoulder': 5, 'RShoulder': 6,
                      'LWrist': 9, 'RWrist': 10,
                      # 'LHip': 11, 'RHip': 12,
                      'LAnkle': 15, 'RAnkle': 16}
    if ppl_num is 2:
        for i in range(ppl_num):
            kpt_num1 = ppl[i]['keypoints'].numpy().shape[0]
            kpt_num2 = ppl[i + 1]['keypoints'].numpy().shape[0]
            if kpt_num1 == kpt_num2:
                for j in keypoint_order.values():
                    x1 = ppl[i]['keypoints'].numpy()[j][0]
                    y1 = ppl[i]['keypoints'].numpy()[j][1]
                    x2 = ppl[i + 1]['keypoints'].numpy()[j][0]
                    # y2 = result['result'][i+1]['keypoints'].numpy()[j][1]
                    u, v, z = get_distance(video_mode, camera_matrix, x1, y1, x2, 'uv')
                    coordinates_u.append(u)
                    coordinates_v.append(v)
                    dists.append(z)
            if i + 2 >= ppl_num:
                break

    return coordinates_u, coordinates_v, dists
