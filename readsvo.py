import sys
import pyzed.sl as sl
import cv2


def main():

#     if len(sys.argv) != 2:
#         print("Please specify path to .svo file.")
#         exit()

#     filepath = sys.argv[1]
    filepath = '/home/yurik/Documents/Program/Alphapose_zed_video/testdata/high_accuracy/HD720_30.svo'
    print("Reading SVO file: {0}".format(filepath))

    init = sl.InitParameters(svo_input_filename=filepath,svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_QUALITY
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    lena = cv2.imread('/home/yurik/Pictures/Lena.png')

    key = ''
    print("  Save the current image:     s")
    print("  Quit the video reading:     q\n")
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_DEPTH)
            depthmap = mat.get_data()
            depthmap = cv2.cvtColor(depthmap, cv2.COLOR_RGBA2RGB)
            depthmap = cv2.applyColorMap(depthmap, cv2.COLORMAP_JET)
#             print(depthmap[100,100,2])
#             depthmap[depthmap < 255] == 255
            cv2.imshow("ZED", depthmap)
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    cam.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
