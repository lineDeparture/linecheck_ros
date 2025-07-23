from line_check_msg.msg import LineCheckMSG
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.logging import LoggingSeverity

class MSG_Line_Check(Node):
    def __init__(self):
        super().__init__("MSG_Line_Check")
        self.get_logger().info("MSG_Line_Check node has been started")

        self.declare_parameter('qos_depth', 10)
        qos_depth = self.get_parameter('qos_depth').value
        self.get_logger().info(f'qos_depth set {qos_depth}')


        QOS_RKL10V = QoSProfile(
            reliability = QoSReliabilityPolicy.RELIABLE,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability = QoSDurabilityPolicy.VOLATILE
        )


        self.MSG_Line_Check_publisher = self.create_publisher(LineCheckMSG, "MSG_Line_Check", QOS_RKL10V)

    def publish_MSG_Line_Check(self, id, cm, height):
        msg = LineCheckMSG()
        msg.stamp = self.get_clock().now().to_msg()
        msg.id = id
        msg.cm = float(cm) 
        msg.height = height
        self.MSG_Line_Check_publisher.publish(msg)



