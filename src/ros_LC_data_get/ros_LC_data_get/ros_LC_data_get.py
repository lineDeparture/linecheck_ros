import rclpy
from rclpy.node import Node
from line_check_msg.msg import LineCheckMSG
import os
from datetime import datetime
from threading import Lock
import json

log_mutex = Lock()

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def create_log_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def save_log(dir_path, section, message):
    with log_mutex:
        create_log_dir(dir_path)
        date = get_current_date()
        file_path = os.path.join(dir_path, f"logs_{date}.ini")

        # 섹션 존재 여부 확인
        section_exists = False
        if os.path.exists(file_path):
            with open(file_path, "r") as in_file:
                for line in in_file:
                    if line.strip() == f"[{section}]":
                        section_exists = True
                        break

        with open(file_path, "a") as out_file:
            if not section_exists:
                out_file.write(f"[{section}]\n")
            timestamp = get_current_time()
            out_file.write(f"{timestamp} = {message}\n")

class LogSubscriber(Node):
    def __init__(self):
        super().__init__('log_subscriber')
        self.subscription = self.create_subscription(
            LineCheckMSG,
            'MSG_Line_Check',  # 퍼블리시하는 토픽 이름과 일치해야 함
            self.listener_callback,
            10
        )
        self.log_dir = "log_dir"  # 원하는 로그 디렉토리명

    def listener_callback(self, msg):
        # 로그 메시지를 JSON 형식으로 생성
        now_str = get_current_time()
        log_dict = {
            "timestamp": now_str,
            "label": msg.id,
            "distance": msg.cm,
            "frame": msg.height
        }
        section = "ClientLog"
        message = json.dumps(log_dict, ensure_ascii=False)
        save_log(self.log_dir, section, message)
        self.get_logger().info(f"Logged: [{section}] {message}")

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LogSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
