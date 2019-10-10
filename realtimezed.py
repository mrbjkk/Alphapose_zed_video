import numpy as np
import pyzed.sl as sl
import cv2

camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

def main():
    print("Running...")
    init = sl.InitParameters()
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    depth_map = sl.Mat()
    depth_fordisp = sl.Mat()
    disparity_map = sl.Mat()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            matshow = mat.get_data()
            cv2.imshow("test", matshow)
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)

    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
