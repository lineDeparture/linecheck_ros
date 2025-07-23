import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value

        QOS_RKL10V = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability = QoSDurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(Image, 'video_topic', QOS_RKL10V)
        self.bridge = CvBridge()

    def publish_video(self, frame):
            # OpenCV BGR 이미지를 ROS2 Image 메시지로 변환
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(msg)
        # 프레임 속도 조절 (예: 30fps)
        #rclpy.spin_once(self, timeout_sec=1/30)
