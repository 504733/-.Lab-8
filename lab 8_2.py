import cv2
import numpy as np


def track_circle_quadrants():
    # ----------------------- 自动读取同文件夹下的苍蝇图 -----------------------
    fly_bytes = np.fromfile("fly64.png", dtype=np.uint8)
    fly_img = cv2.imdecode(fly_bytes, cv2.IMREAD_UNCHANGED)

    if fly_img is None:
        print("Error: 找不到 fly64.png")
        return

    fly_bgr = fly_img[:, :, :3]
    fly_alpha = fly_img[:, :, 3]
    fly_h, fly_w = fly_img.shape[:2]

    # ----------------------- ArUco 标记追踪（你能跑的稳定版本） -----------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 摄像头打不开")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测 ArUco 标记
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            # 找标记中心
            marker_corners = corners[0][0]
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))

            # 画标记
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"X: {center_x}, Y: {center_y}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ----------------------- 贴苍蝇（中心对齐标记） -----------------------
            x1 = center_x - fly_w // 2
            y1 = center_y - fly_h // 2
            x2 = x1 + fly_w
            y2 = y1 + fly_h

            fh, fw = frame.shape[:2]
            x_start = max(0, x1)
            y_start = max(0, y1)
            x_end = min(fw, x2)
            y_end = min(fh, y2)

            if x_end > x_start and y_end > y_start:
                frame_roi = frame[y_start:y_end, x_start:x_end]
                fx = x_start - x1
                fy = y_start - y1

                fly_roi = fly_bgr[fy:fy + (y_end - y_start), fx:fx + (x_end - x_start)]
                alpha_roi = fly_alpha[fy:fy + (y_end - y_start), fx:fx + (x_end - x_start)]
                alpha_mask = alpha_roi[:, :, np.newaxis] / 255.0

                frame_roi[:] = frame_roi * (1 - alpha_mask) + fly_roi * alpha_mask

        # 显示画面
        cv2.imshow("Cuadro con mosca", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


track_circle_quadrants()