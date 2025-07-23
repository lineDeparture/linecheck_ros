import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

import time
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional
from pathlib import Path
from line_check_msg.msg import LineCheckMSG
from line_checker.publisher.MSG_publisher import MSG_Line_Check
from line_checker.publisher.video_publisher import VideoPublisher
import pkg_resources
import rclpy

# --- 설정값 ---
CONF_THRESHOLD = 0.3
DIST_THRESHOLD = 1200  # cm
FOCAL_LENGTH = 400
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

KNOWN_HEIGHTS = {
    0: 160,  # 사람
    2: 150,  # 자동차
    3: 100,  # 오토바이
    5: 350,  # 버스
    7: 350   # 트럭
}
CLASS_COLORS = {
    0: (0, 255, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    5: (255, 255, 0),
    7: (255, 0, 255)
}
VALID_CLASS_IDS = list(KNOWN_HEIGHTS.keys())
resource_path = Path(__file__).parent / "LC_resource"

def get_resource_path(rel_path):
    return str(Path(pkg_resources.resource_filename('line_checker', rel_path)))

MODEL_PATH = get_resource_path('LC_resource/best.pt')
WARNING_BANNER_PATH = get_resource_path('LC_resource/warning_banner.png')
WARNING_ICON_PATH = get_resource_path('LC_resource/warning_icon.png')

# --- 텍스트 배경 그리기 함수 ---
def draw_text_with_background(img, text, org, font, scale, color, thickness):
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - base), (x + tw + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, org, font, scale, color, thickness)

#경고 이미지 배너 오버레이 함수
def overlay_warning_banner(frame, banner_img, x, y):
    bh, bw = banner_img.shape[:2]
    fh, fw = frame.shape[:2]
    if x >= fw or x + bw <= 0 or y >= fh or y + bh <= 0:
        return
    x1_frame = max(x, 0)
    y1_frame = max(y, 0)
    x1_banner = max(0, -x)
    y1_banner = max(0, -y)
    x2_frame = min(fw, x + bw)
    y2_frame = min(fh, y + bh)
    x2_banner = x2_frame - x
    y2_banner = y2_frame - y
    if banner_img.shape[2] == 4:
        alpha = banner_img[y1_banner:y2_banner, x1_banner:x2_banner, 3] / 255.0
        for c in range(3):
            frame[y1_frame:y2_frame, x1_frame:x2_frame, c] = (
                frame[y1_frame:y2_frame, x1_frame:x2_frame, c] * (1 - alpha) +
                banner_img[y1_banner:y2_banner, x1_banner:x2_banner, c] * alpha
            ).astype(np.uint8)
    else:
        frame[y1_frame:y2_frame, x1_frame:x2_frame] = banner_img[y1_banner:y2_banner, x1_banner:x2_banner]

# --- 비디오 스레드 클래스 ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal()

    def __init__(self, module_name: str, video_path: str, line_check_msg: LineCheckMSG, video_publisher: VideoPublisher):
        super().__init__()
        self.line_check_msg = line_check_msg
        """
        #self.socket_client = SocketClient()
        #self.socket_client.socket_connet()
        #self.socket_client.start()
        """

        self.module_name = module_name
        self.video_path = video_path
        self.running = True

        # YOLO 모델, 경고 리소스 로드 (경로는 본인 환경에 맞게)
        self.model = YOLO(MODEL_PATH)

        self.warning_banner = cv2.imread(WARNING_BANNER_PATH, cv2.IMREAD_UNCHANGED)
        if self.warning_banner is not None:
            self.warning_banner = cv2.resize(self.warning_banner, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        self.warning_icon = cv2.imread(WARNING_ICON_PATH, cv2.IMREAD_UNCHANGED)
        if self.warning_icon is not None:
            self.warning_icon = cv2.resize(self.warning_icon, (60, 60), interpolation=cv2.INTER_AREA)

        self.video_publisher = VideoPublisher()

    # --- 객체 검출 후 거리 계산 및 경고 표시 ---
    def process_detections(self, results, lane_polygon, M, frame_shape, annotated_frame):
        collision_warning = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf.item())
            class_id = int(box.cls.item())
            pixel_height = y2 - y1

            if pixel_height < 5:
                continue

            center_y = y2 - 0.2 * (y2 - y1)
            center = np.array([[(x1 + x2) / 2, center_y]], dtype=np.float32)
            # polygon 내부인지 여부
            is_inside = cv2.pointPolygonTest(lane_polygon, tuple(center[0]), False)

            if class_id in VALID_CLASS_IDS and conf > CONF_THRESHOLD and pixel_height > 20:
                center_warped = cv2.perspectiveTransform(np.array([center]), M)[0][0]
                known_height = KNOWN_HEIGHTS.get(class_id, 170)
                dist_pixel = (known_height * FOCAL_LENGTH) / pixel_height
                warped_y = center_warped[1]
                dist_y = DIST_THRESHOLD * (1 - warped_y / frame_shape[0])
                dist_y = max(50, dist_y)
                distance_cm = dist_pixel * 0.7 + dist_y * 0.3
                distance_m = distance_cm / 100

                if distance_cm < DIST_THRESHOLD:
                    collision_warning = True
                    box_color = (0, 0, 255)
                    thickness = 3
                    # 경고 아이콘 오버레이
                    if self.warning_icon is not None:
                        icon_x = x1
                        icon_y = y1 - self.warning_icon.shape[0] - 10
                        overlay_warning_banner(annotated_frame, self.warning_icon, icon_x, icon_y)
                    # self.socket_client.set_data(class_id,  distance_cm, annotated_frame)
                    #self.socket_client.set_data(class_id,  distance_cm, frame_shape[0])

                    self.line_check_msg.publish_MSG_Line_Check(class_id,  distance_cm, frame_shape[0])
                else:
                    box_color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    thickness = 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                dist_label = f"{distance_m:.1f}m"
                draw_text_with_background(annotated_frame, dist_label, (x1, y2 + 25),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.circle(annotated_frame, (int(center_warped[0]), int(center_warped[1])), 5, (255, 0, 0), -1)
        return annotated_frame, collision_warning

# --- 비디오 스레드 ---
    def run(self):

        import line_checker.line_check_frame as line_check_frame

        line_check_module = line_check_frame
        # 동적 모듈 로딩
        if self.module_name == "line_check":
            line_check_func = line_check_module.line_check
            
        elif self.module_name == "line_check_sobel":
            line_check_func = line_check_module.line_check_sobel
        else:
            print(f"Unknown module: {self.module_name}")
            return

        LaneTracker = line_check_module.LaneTracker
        

        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output.mp4", fourcc, 30, (RESIZE_WIDTH, RESIZE_HEIGHT))

        src = np.float32([
            [RESIZE_WIDTH * 0.45, RESIZE_HEIGHT * 0.57],
            [RESIZE_WIDTH * 0.55, RESIZE_HEIGHT * 0.57],
            [RESIZE_WIDTH * 0.9, RESIZE_HEIGHT],
            [RESIZE_WIDTH * 0.1, RESIZE_HEIGHT]
        ])
        dst = np.float32([
            [RESIZE_WIDTH * 0.3, 0],
            [RESIZE_WIDTH * 0.7, 0],
            [RESIZE_WIDTH * 0.7, RESIZE_HEIGHT],
            [RESIZE_WIDTH * 0.3, RESIZE_HEIGHT]
        ])

        M = line_check_module.warp_M(src, dst)
        Minv = line_check_module.Re_warp(src, dst)

        LT = LaneTracker(nwindows=9, margin=50, minimum=30)

        warning_counter = 0

        while cap.isOpened() and self.running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            h, w = frame.shape[:2]
            # 동적 원근 행렬/폴리곤 계산
            lane_polygon = np.array([[
                (w * 0.2, h), (w * 0.8, h), (w * 0.6, h * 0.6), (w * 0.4, h * 0.6)
            ]], dtype=np.int32)

            # 차선 시각화
            lane_result = line_check_func(frame, M, Minv, LT)
            # YOLO 검출
            results = self.model(frame, conf=CONF_THRESHOLD, iou=0.5)
            # 객체+경고 표시 (lane_result 위에 그림)
            annotated_frame, collision_warning = self.process_detections(
                results, lane_polygon[0], M, frame.shape, lane_result)

            warning_counter = min(warning_counter + 5, 30) if collision_warning else max(warning_counter - 1, 0)
            # 경고 카운터가 있을 시, 경로상 경고 배너 이미지가 존재할 시 아래 로직 실행 
            if warning_counter > 0 and self.warning_banner is not None:
                banner_width = self.warning_banner.shape[1]
                x_pos = int((RESIZE_WIDTH - banner_width) / 2)
                y_pos = -90
                overlay_warning_banner(annotated_frame, self.warning_banner, x_pos, y_pos)
            

    
            
            # FPS 계산 및 표시
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            out.write(annotated_frame)
            self.video_publisher.publish_video(annotated_frame)
            self.change_pixmap_signal.emit(annotated_frame)

        # 비디오 종료 후 리소스 정리    
        cap.release()
        # 비디오 파일 저장
        out.release()
    
        self.finished_signal.emit()
        
    

#  --- 스레드 중지 함수 ---
    def stop(self):
        self.running = False
        #self.socket_client.stop()

        



# --- 메인 윈도우 클래스 ---
class MainWindow(QMainWindow):
    def __init__(self, line_check_msg: LineCheckMSG, video_publisher: VideoPublisher):
        super().__init__()
        self.line_check_msg = line_check_msg
        self.thread: Optional[VideoThread] = None
        self.init_ui()
        self.video_publisher = VideoPublisher

    def get_mp4_files(self, folder_path):
        import os 
        mp4_files = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".mp4") or file_name.endswith(".avi"):
                mp4_files.append(file_name)
        return mp4_files



# --- UI 초기화 ---
    def init_ui(self):
        self.setWindowTitle("Lane Detection + YOLO + Warning")
        self.setGeometry(100, 100, 1400, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        control_layout = QHBoxLayout()

        self.module_combo = QComboBox()
        self.module_combo.addItems(["line_check", "line_check_sobel"])
        self.module_combo.setCurrentText("line_check")
        control_layout.addWidget(QLabel("Module:"))
        control_layout.addWidget(self.module_combo)

        self.video_combo = QComboBox()
        # self.video_combo.addItems(["project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"])
        files = self.get_mp4_files(get_resource_path("LC_resource/test_video"))
        self.video_combo.addItems(files)
        self.video_combo.setCurrentText(files[0] )
        control_layout.addWidget(QLabel("Video:"))
        control_layout.addWidget(self.video_combo)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        layout.addLayout(control_layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1280, 720)
        self.video_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.video_label)


# --- 비디오 시작 및 중지 함수 ---
    def start_video(self):
        if self.thread is None or not self.thread.running:
            module_name = self.module_combo.currentText()
            video_file = self.video_combo.currentText()
            video_path = get_resource_path(f"LC_resource/test_video/{video_file}")
            self.thread = VideoThread(module_name, video_path, self.line_check_msg, self.video_publisher)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.finished_signal.connect(self.video_finished)
            self.thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.module_combo.setEnabled(False)
            self.video_combo.setEnabled(False)

# --- 비디오 중지 함수 ---
    def stop_video(self):
        if self.thread and self.thread.running:
            
            self.thread.stop()
            self.thread.wait()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.module_combo.setEnabled(True)
            self.video_combo.setEnabled(True)

# --- 비디오 종료 후 처리 ---
    def video_finished(self):
        if self.thread is None:
            return
        if self.thread.running:
            self.thread.stop()
            self.thread.wait()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.module_combo.setEnabled(True)
            self.video_combo.setEnabled(True)
            self.send_video_data()
    
# --- 비디오 데이터 전송 함수 ---
    def send_video_data(self):
        """
        socket_client = SocketClient()
        socket_client.socket_connet(isVideoSocket=True)
        socket_client.start(isVideoSocket= True)
        """
        # socket_client.set_video_data()  # 비디오 종료 신호 전송

# --- 이미지 업데이트 함수 ---
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

# --- 메인 함수 ---
def main():
    rclpy.init()

    line_check_msg = MSG_Line_Check()
    video_publisher = VideoPublisher()
    app = QApplication(sys.argv)
    window = MainWindow(line_check_msg, video_publisher)
    window.show()
    sys.exit(app.exec_())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
