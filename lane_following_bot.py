#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.bridge = CvBridge()

        self.linear_speed = 0.4
        self.angular_gain = 0.015

        self.filtered_error = 0.0
        self.alpha = 0.7

        self.obstacle_detected = False
        self.red_line_detected = False
        self.stop_start_time = None  
        self.red_line_cooldown = None  # Cooldown to ignore red line for some time

        self.get_logger().info("Lane Follower node has been started.")

    def image_callback(self, msg):
        self.get_logger().info("Camera callback received.")

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.detect_large_obstacle(cv_image):
            self.obstacle_detected = True
        else:
            self.obstacle_detected = False

        blur = cv2.blur(cv_image, (3, 3))
        edge = cv2.Canny(blur, 160, 180)

        # Red line detection with cooldown
        current_time = time.time()
        
        if self.red_line_cooldown and (current_time - self.red_line_cooldown < 3.0):
            self.get_logger().info(f"Ignoring red line due to cooldown: {3.0 - (current_time - self.red_line_cooldown):.1f} seconds left.")
        else:
            if self.detect_red_line(cv_image) and not self.red_line_detected:
                self.red_line_detected = True
                self.stop_start_time = current_time  # Start stop timer
                self.get_logger().info("Red line detected, stopping for 5 seconds.")

        if self.red_line_detected:
            elapsed_time = current_time - self.stop_start_time
            if elapsed_time < 5.0:
                twist_msg = Twist()
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(twist_msg)

                self.get_logger().info(f"Stopping for red sign: {5.0 - elapsed_time:.1f} seconds left...")
                return
            else:
                self.red_line_detected = False
                self.red_line_cooldown = current_time  # Set cooldown timer
                self.get_logger().info("Resuming movement after red line.")

        if self.obstacle_detected:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg)

            self.get_logger().info("Obstacle detected! Stopping the robot.")
            return

        # Normal lane following
        centroid_x = self.freespace(edge, cv_image)

        if centroid_x >= 0:
            width = cv_image.shape[1]
            center_x = width // 2
            error = centroid_x - center_x

            self.filtered_error = (self.alpha * self.filtered_error) + ((1 - self.alpha) * error)

            angular_z = -self.filtered_error * self.angular_gain

            twist_msg = Twist()
            twist_msg.linear.x = self.linear_speed
            twist_msg.angular.z = angular_z
            self.cmd_vel_pub.publish(twist_msg)
        else:
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(twist_msg)

        cv2.imshow("Camera", cv_image)
        cv2.imshow("Edge", edge)
        cv2.waitKey(1)

    def freespace(self, canny_frame, img):
        height, width = canny_frame.shape

        DreaptaLim = width // 2
        StangaLim = width // 2

        for i in range(width // 2, width-1):
            if canny_frame[height - 10, i]:
                DreaptaLim = i
                break

        for i in range(width // 2):
            if canny_frame[height - 10, width // 2 - i]:
                StangaLim = width // 2 - i
                break

        if StangaLim == width // 2:
            StangaLim = 1
        if DreaptaLim == width // 2:
            DreaptaLim = width - 1

        contour = []
        contour.append((StangaLim, height - 10))
        for j in range(StangaLim, DreaptaLim + 1, 10):
            for i in range(height - 10, 9, -1):
                if canny_frame[i, j]:
                    contour.append((j, i))
                    break
                if i == 10:
                    contour.append((j, i))
        contour.append((DreaptaLim, height - 10))

        contours = [np.array(contour)]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, contours, 0, 255, cv2.FILLED)

        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

            cv2.arrowedLine(
                img, 
                (width // 2, height - 10), 
                (centroid_x, centroid_y), 
                (60, 90, 255), 
                4
            )
            return centroid_x
        else:
            return -1

    def detect_red_line(self, cv_image):
        lower_part = cv_image[int(cv_image.shape[0] * 2 / 3):, :]
        hsv = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 70, 50])
        upper_red = np.array([180, 255, 255])

        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(lower_part, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.get_logger().info("Red line detected!")
                return True
        return False

    def detect_large_obstacle(self, cv_image):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        h, w = thresh.shape
        top = 0
        bottom = int(h * 0.25)
        left = int(w * 0.3)
        right = int(w * 0.7)

        roi = thresh[top:bottom, left:right]

        white_pixels = cv2.countNonZero(roi)

        return white_pixels > 2000

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
