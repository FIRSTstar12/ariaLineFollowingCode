from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

class LineImageProcessor(Node):
    def __init__(self):
        super().__init__('line_image_processor')

        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('debug_image_topic', '/line_follower/debug_image')

        image_topic = self.get_parameter('image_topic').value
        debug_image_topic = self.get_parameter('debug_image_topic').value

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.debug_image_pub = self.create_publisher(
            Image,
            debug_image_topic,
            10,
        )

        self.error_pub = self.create_publisher(
            Float32,
            '/line_follower/center_error',
            10,
        )

        self.confidence_pub = self.create_publisher(
            Float32,
            '/line_follower/confidence',
            10,
        )

        self.get_logger().info(f'Subscribed to image topic: {image_topic}')
        self.get_logger().info(f'Publishing debug images on: {debug_image_topic}')


    def image_callback(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        except Exception as error:
            self.get_logger().error(f'cv_bridge conversion failed: {error}')
            return
    
        debug_image, center_error, confidence = self.process_image(image)

        error_msg = Float32()
        error_msg.data = float(center_error)
        self.error_pub.publish(error_msg)

        confidence_msg = Float32()
        confidence_msg.data = float(confidence)
        self.confidence_pub.publish(confidence_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)  

    def process_image(self, image):
        height, width = image.shape[:2]
    
    # Convert to HSV and create mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        lower_blue = np.array([90, 80, 50])
        upper_blue = np.array([130, 255, 255])
    
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        # Use ROI
        roi_y_start = height // 2
        roi_mask = mask[roi_y_start:height, 0:width]
    
    # Find contours
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find best contour
        min_area = 500.0
        best_contour = None
        best_area = 0.0
    
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > best_area:
                best_area = area
                best_contour = contour
    
    # Initialize placeholders
        center_error = 0.0
        confidence = 0.0
        debug = image.copy()
    
    # Draw ROI boundary and image center
        cv2.line(debug, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)
        cv2.line(debug, (0, roi_y_start), (width, roi_y_start), (0, 255, 255), 2)
    
        if best_contour is not None and best_area >= min_area:
            moments = cv2.moments(best_contour)
        
            if moments["m00"] != 0:
                roi_cx = int(moments["m10"] / moments["m00"])
                roi_cy = int(moments["m01"] / moments["m00"])
            
                full_cx = roi_cx
                full_cy = roi_cy + roi_y_start
            
                image_center_x = width / 2.0
                center_error_px = full_cx - image_center_x
                center_error = center_error_px / (width / 2.0)
                center_error = float(np.clip(center_error, -1.0, 1.0))
            
            # Compute confidence
                roi_height = height - roi_y_start
                roi_area = width * roi_height
                area_ratio = best_area / float(roi_area)
                expected_area_ratio = 0.05
                confidence = area_ratio / expected_area_ratio
                confidence = float(np.clip(confidence, 0.0, 1.0))
            
            # Draw contour and centroid
                shifted_contour = best_contour.copy()
                shifted_contour[:, :, 1] += roi_y_start
                cv2.drawContours(debug, [shifted_contour], -1, (0, 255, 0), 2)
                cv2.circle(debug, (full_cx, full_cy), 8, (0, 0, 255), thickness=-1)
    
        return debug, center_error, confidence


    
    
def main(args=None):
    rclpy.init(args=args)
    
    node = LineImageProcessor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()