from rclpy.node import Node
import rclpy
import torch
import cv2
from cv_bridge import CvBridge

import sensor_msgs.msg as sensor_msgs_types
from vision_msgs.srv import Sam2Inference, Sam2Inference_Request, Sam2Inference_Response

from .utils import Camera
import matplotlib.pyplot as plt

class Sam2ServiceSelector(Node):
    def __init__(self):
        super().__init__("sam2_service_selector")
        self.get_logger().info("Initializing Sam2ServiceSelector Node")

        # Initialize camera
        self.camera = Camera()
        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            sensor_msgs_types.Image,
            "/camera/rgb/image_raw", 
            self.camera_callback_rgb,
            10
        )
        self._recent_rgb = None

        self.get_logger().info("Waiting for first RGB image...")

        while self._recent_rgb is None:
            rclpy.spin_once(self)  
        
        self.get_logger().info("Received first RGB image, launching selection window.")

        selection = self.get_click_or_bbox(self._recent_rgb)
        self.get_logger().info(f"Selection result: {selection}")

    def camera_callback_rgb(self, msg: sensor_msgs_types.Image):

        self._recent_rgb = msg
        self.get_logger().debug("RGB image received.")


    def get_click_or_bbox(self, img):
        """
        Let user click on image to get either:
        - a single point (x, y)
        - or a bounding box (x_min, y_min, x_max, y_max)
        Works headless-friendly (matplotlib inline or virtual display)
        """

        coords = []

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                coords.append((int(event.xdata), int(event.ydata)))
                print(f"Clicked at: {coords[-1]}")
                if len(coords) == 2:  # two points for bounding box
                    plt.close()

        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()  # works in headless if you have Xvfb or Jupyter
        fig.canvas.mpl_disconnect(cid)

        if len(coords) == 1:
            return {"x": coords[0][0], "y": coords[0][1]}
        elif len(coords) == 2:
            x_min = min(coords[0][0], coords[1][0])
            y_min = min(coords[0][1], coords[1][1])
            x_max = max(coords[0][0], coords[1][0])
            y_max = max(coords[0][1], coords[1][1])
            return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
        else:
            return None

def main():
    
    try:
         rclpy.init()
         node = Sam2ServiceSelector()
    
         rclpy.spin(node)  # Keep the node alive until Ctrl+C
    except KeyboardInterrupt:
        print("Ctrl+C pressed, shutting down...")
    finally:
        node.destroy_node()  # Cleanly destroy the node
        rclpy.shutdown()     # Shutdown ROS client library
