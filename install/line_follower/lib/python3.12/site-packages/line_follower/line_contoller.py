import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class LineContorller(Node):
    def __init__(self):
        super().__init__('line_controller')

        self.declare_parameter('base_linear_speed', 0.12)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('min_confidence', 0.25)
        self.declare_parameter('line_lost_timeout', 0.5)

        self.base_linear_speed = self.get_parameter('base_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.line_lost_timeout = self.get_parameter('line_lost_timeout').value

        self.latest_error = 0.0
        self.latest_confidence = 0.0
        self.previous_error = 0.0
        self.last_error_time = None

        self.error_sub = self.create_subscription(
            Float32,
            '/line_follower/center_error',
            self.error_callback,
            10,
        )

        self.confidence_sub = self.create_subscription(
            Float32,
            '/line_follower/confidence',
            self.confidence_callback,
            10,
        )
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.timer = self.create_timer(0.05, self.control_timer_callback)

        self.get_logger().info('Line controller started')

        def error_callback(self, msg):
            """Receive center error from image processor"""
            self.latest_error = msg.data
            self.last_error_time = self.get_clock().now()
    
    
        def confidence_callback(self, msg):
            """Receive confidence from image processor"""
            self.latest_confidence = msg.data
    
    
        def control_timer_callback(self):
            """Main control loop - runs at 20 Hz"""
            if self.last_error_time is None:
                self.get_logger().warn('No error received yet')
                self.stop_robot()
                return
            time_since_error = (self.get_clock().now() - self.last_error_time).nanoseconds / 1e9
            if time_since_error > self.line_lost_timeout:
                self.get_logger().warn('Line lost (error timeout)')
                self.stop_robot()
                return
            error = self.latest_error
            angular_z = -self.kp * error
            if self.kd > 0:
                error_rate = (error - self.previous_error) / 0.05  # dt = 0.05s
                angular_z += -self.kd * error_rate
            angular_z = float(np.clip(angular_z, -self.max_angular_speed, self.max_angular_speed))
            speed_scale = 1.0 - 0.7 * abs(error)
            speed_scale = float(np.clip(speed_scale, 0.25, 1.0))
            linear_x = self.base_linear_speed * speed_scale
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.cmd_pub.publish(twist)
            self.previous_error = error
            
        def stop_robot(self):
            """Publish zero velocity"""
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LineContorller()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
