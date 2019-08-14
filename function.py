import configparser
import numpy as np
import cv2
import threading
import pyzed.sl as sl

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


def get_distance(video_mode, camMtx1, x1, y1, x2):
    fx1 = camMtx1[0][0]
    # fx_r = mtx_r[0][0]
    (w, h) = get_resolution(video_mode)
    if x1 <= x2:
        x2 = x2 - (w/2)
    else:
        t = x1
        x1 = x2
        x2 = t - (w/2)

    doffs = x1 - x2
    baseline = get_baseline()

    z = fx1 * baseline / doffs
    # x = x1 * z / doffs
    # y = y1 * z / doffs

    # z_right = baseline * fx_r / doffs
    # x2 = z_right * u_right / doffs
    # y2 = z_right * v_right / doffs

    return x1, y1, z


def get_resolution(opt):
    video_mode = {'CAM_2K': (4416, 1242), 'CAM_FHD': (3840, 1080), 'CAM_HD': (2560, 720), 'CAM_VGA': (1344, 376)}
    (w, h) = video_mode.get(opt)
    return w, h


def set_camera(capture):
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class zed_videocapture:
    def __init__(self):
        self.frame = []
        self.image = sl.Mat()
        self.status = False
        self.isStop = False

        # Connect webcam
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_VGA
        self.init_params.camera_fps = 30
        self.err = self.zed.open(self.init_params)

    def start(self):
        # Deamon = True: threading is close along with main
        t = threading.Thread(target=self.get_frame, args=())
        t.daemon = True
        t.start()
        print('Camera started')

    def get_frame(self):
        if self.err is not sl.ERROR_CODE.SUCCESS:
            self.zed.close()
        if self.zed.grab(sl.RuntimeParameters()) is sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.VIEW_SIDE_BY_SIDE, sl.MEM.MEM_CPU)
            self.frame = self.image.get_data()
            self.status = True
        return self.status, self.frame

    def get_fps(self):
        return self.zed.get_camera_fps()

    def get_frameSize(self):
        return self.zed.get_resolution()

