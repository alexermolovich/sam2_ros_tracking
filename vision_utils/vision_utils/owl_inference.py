from dataclasses import dataclass
from rclpy.node import Node
import rclpy
from vision_msgs.srv import OwlInference, OwlInference_Request, OwlInference_Response
from vision_msgs.srv import OwlInferenceM, OwlInferenceM_Request, OwlInferenceM_Event
from sensor_msgs.msg import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, logging
from cv_bridge import CvBridge
from PIL import Image as _pil_image
import torch
import inspect
import time
import cv2

logging.set_verbosity_error()

def log_info(self, msg):
    self.get_logger().info(msg)

@dataclass
class Camera:
    height: int
    width: int
    _recent_rgb: _pil_image = None
    _recent_depth: Image = None

class OwlService(Node):
    @dataclass
    class OwlServiceLog:
        running_time: int = 0

    @dataclass
    class OwlServiceData:
        log: "OwlService.OwlServiceLog"
        rgb_topic: str
        depth_topic: str

    camera = Camera(height=0, width=0)
    bridge = CvBridge()

    def _declare_parameters(self):
        self.declare_parameter("rgb_topic", "/camera/rgb/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image")

    def __init__(self):
        super().__init__("owl_service")
        self._declare_parameters()

        # Initialize OWL processor and model
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.eval()

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        # Services
        self.srv_message = self.create_service(
            OwlInferenceM,
            "owl_infernce_message",
            self.owl_inference_message
        )

        self.srv_topic = self.create_service(
            OwlInference,
            "owl_infernce_topic",
            self.owl_inference_topic
        )

        self.get_logger().warn("OwlService node initialized and ready.")

    # ----------------------------
    # Callbacks
    # ----------------------------
    def rgb_callback(self, msg: Image):
        # Convert ROS Image to PIL (RGB)
        cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.camera._recent_rgb = _pil_image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def depth_callback(self, msg: Image):
        self.camera._recent_depth = msg

    # ----------------------------
    # Services
    # ----------------------------
    def owl_inference_topic(self, request: OwlInference_Request, response: OwlInference_Response):
        self.get_logger().info("Received /owl_infernce_topic service call.")

        target_class_label = request.target_class
        if not target_class_label:
            response.answer = "No target class provided."
            log_info(self, response.answer)
            return response

        text_labels = [[f"a photo of {target_class_label}"]]

        # Wait for RGB image (non-blocking with timeout)
        timeout = 5.0
        start_time = time.time()
        while self.camera._recent_rgb is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                response.answer = "Timeout: No RGB image received."
                log_info(self, response.answer)
                return response

        # Run OWL detection
        inputs = self.processor(
            text=text_labels,
            images=[self.camera._recent_rgb],
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(self.camera._recent_rgb.height, self.camera._recent_rgb.width)])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1,
            text_labels=text_labels
        )

        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"]

        if boxes.numel() == 0:
            response.answer = f"No instances of '{target_class_label}' detected."
            log_info(self, response.answer)
            return response

        best_idx = torch.argmax(scores)
        best_box = boxes[best_idx]
        best_box_formatted = [round(v, 2) for v in best_box.tolist()]

        response.answer = str(best_box_formatted)
        log_info(self, response.answer)
        return response

    def owl_inference_message(self, request: OwlInferenceM_Request, response: OwlInference_Response):
        self.get_logger().info("Received /owl_infernce_message service call.")

        target_class_label = request.target_class
        if not target_class_label:
            response.answer = "No target class provided."
            return response

        if request.rgb is None:
            response.answer = "No image provided."
            return response

        text_labels = [[f"a photo of {target_class_label}"]]
        inputs = self.processor(
            text=text_labels,
            images=request.rgb,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(request.rgb.height, request.rgb.width)])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1,
            text_labels=text_labels
        )

        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"]

        if boxes.numel() == 0:
            response.answer = f"No instances of '{target_class_label}' detected."
            return response

        best_idx = torch.argmax(scores)
        best_box = boxes[best_idx]
        best_score = scores[best_idx].item()
        best_label = labels[best_idx]
        best_box_formatted = [round(v, 2) for v in best_box.tolist()]

        response.answer = (
            f"Detected '{best_label}' "
            f"(confidence {best_score:.3f}) "
            f"at {best_box_formatted}"
        )

        return response

# ----------------------------
# Main
# ----------------------------
def main(args=None):
    rclpy.init(args=args)
    owl_service_node = OwlService()
    try:
        rclpy.spin(owl_service_node)
    except KeyboardInterrupt:
        owl_service_node.get_logger().info("Shutting down OwlService node...")
    finally:
        owl_service_node.destroy_node()
        rclpy.shutdown()
