from typing import List
from .utils import Camera
from rclpy.node import Node
import inspect
from transformers import *
from vision_msgs.srv import Sam2Inference_Request, Sam2Inference_Response, Sam2Inference
from sensor_msgs.msg import Image
from transformers import Sam2Processor, Sam2Model
from accelerate import Accelerator
import torch
import PIL.Image as pil_image
import cv2
from cv_bridge import CvBridge
from dataclasses import dataclass, field
from enum import Enum
import time
import rclpy
import logging
import numpy as np
import matplotlib.cm as cm
import asyncio
import random
import threading
from torchvision.ops import masks_to_boxes
# Ensure logging level is set
logging.getLogger().setLevel(logging.DEBUG)
bridge = CvBridge()
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from cv_bridge import CvBridge
import threading
import torchvision.ops as ops
from std_msgs.msg import String

import torch
from torchvision.ops import masks_to_boxes

def masks_batched_to_boxes(masks: torch.Tensor):
    """
    masks: (B, C, H, W) tensor (bool or 0/1)
    returns:
        boxes: (B, C, 4) in (x1, y1, x2, y2) format
    """
    B, C, H, W = masks.shape

    # Ensure boolean
    masks = masks.bool()

    # Flatten batch and channels
    flat_masks = masks.view(B * C, H, W)

    # Convert masks -> boxes
    boxes = masks_to_boxes(flat_masks)  # (B*C, 4)

    # Reshape back
    boxes = boxes.view(B, C, 4)

    return boxes



def ltrb_to_tlwh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]
    
@dataclass
class Log:
    node_name: str
    method_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = None
    additional_info: str = ""

    def finalize(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        return f"[{self.node_name}::{self.method_name}] Duration: {duration:.3f}s | Info: {self.additional_info}"


class Sam2InferenceType(Enum):
    VIDEO = 1
    IMAGE = 2


class Sam2Service(Node):

    def __init__(self):
       
        super().__init__("sam2_service")
        self.tracking_type = Sam2InferenceType.IMAGE
       
        self.tracker_counter = 0        
        
        if self.tracking_type is None:
            self.get_logger().error("Runtime Error: Model type is node defined")
        
        self.device = Accelerator().device
        self._declare_parameters()

        # Initialize camera
        self.camera = Camera(height=0, width=0, _recent_rgb=None, _recent_depth=None)

        # Load model
        if self.tracking_type == Sam2InferenceType.IMAGE:  
            self.model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(self.device)
            self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

        # ROS subscriptions & services
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value

        self.rgb_sub = self.create_subscription(Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.srv = self.create_service(Sam2Inference, "sam2_inference_image", self.sam2_inference_create_trackable)

        self.model.eval()
        self.get_logger().debug("Model initialization completed, ready to accept requests")
        self.active_trackables = []


    def _declare_parameters(self):
        self.declare_parameter("rgb_topic", "/camera/rgb/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")

    def _inference_sam2(self, _image: pil_image, bbox):

        inputs = self.processor(
            images=_image,
            input_boxes=[[[bbox[0], bbox[1], bbox[2], bbox[3]]]],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    
    def sam2_inference_image(self, request: Sam2Inference_Request, response: Sam2Inference_Response):
        log_entry = Log(node_name=self.get_name(), method_name=inspect.currentframe().f_code.co_name)
        self.mutex = threading.Lock(
        
        )
        curr_time = self.get_clock().now().to_msg().sec 
        self.get_logger().debug(
            f"Starting awaiting mutex in sam2_inference_image, time started: "
            f"{self.get_clock().now().to_msg().sec}"
        )
        
        with self.mutex: 
            try:
                self.get_logger().info("Starting image-based SAM2 inference")

                bbox_target = request.bounding_box
                if bbox_target is None:
                    response.output = "No target bbox provided."
                    return response

                if self.camera._recent_rgb is None:
                    response.output = "No recent RGB image available."
                    return response

                # Convert image to PIL format
                img = cv2.cvtColor(self.camera._recent_rgb, cv2.COLOR_BGR2RGB)
                rgb_pil: pil_image.Image = pil_image.fromarray(img)

                masks = self._inference_sam2(rgb_pil, bbox=bbox_target)

                final_image: pil_image.Image = overlay_masks_on_image(original_image=rgb_pil, masks=masks)
                
                #response.output = f"Detected masks: {masks.shape[0]} objects"

                #return response
            except Exception as e:
                if e is KeyboardInterrupt:
                   response.output = "Keyboard interrupt is called, exiting the function" 
                   self.get_logger().warn("Warn: Exception recieved keyboard interrupt, closing the function ")
                   return response
                else:
                    response.output = f"Function call threw an exception,  {e}"
                    self.get_logger().warn(f"Function call threw an exception, {e}") 
                    return response
     
            finally:
                log_message = log_entry.finalize()
                if self.get_logger().is_enabled_for(logging.DEBUG):
                    self.get_logger().debug(log_message)

                self.get_logger().debug(
                    f"Finished mutex awaiting, function completed in:"
                    f"{curr_time - self.get_clock().now().to_msg().sec}"
                )
    def get_unique_random(self):
        self._used_numbers = set()
        self._lock_number_gen = threading.Lock()

        with self._lock_number_gen:
            if len(self._used_numbers) >= 100:
                raise RuntimeError("All numbers from 0 and 99 have been used")

            while True:
                n = random.randint(0, 99)
                if n not in self._used_numbers:
                    self._used_numbers.add(n)
                    return n

    def mask_to_bbox(mask):
        """
        mask: (H, W) binary mask of an object
        Returns: [x1, y1, x2, y2]
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None  # no object
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return [x1, y1, x2, y2]
    
    #function logic description
    # start and get first detection from the inference sam2
    # launcing async tracker that is gonna be publshing trackers output in async
    # when async is done we are gonna go and check the assigned id in the container of all of them 
    # it should let us know the next action to be peformed with the tracker 
    #   if yes:
    #       take the last tracked bouding box 
    #       and re run the detection on it 
    #   if reset: no reset the tracker is just gonna be removed if done 
    #   if no just remove the tracker and free the resources  
    
    def sam2_update_trackable(self, trackable):
        
        log_entry = Log(node_name=self.get_name(), method_name=inspect.currentframe().f_code.co_name)
        curr_time = self.get_clock().now().to_msg().sec
        self.get_logger().debug(f"Updating trackable {trackable._id} with SAM2 inference at time: {curr_time}")

        if trackable._most_recent_bbox is None:
            self.get_logger().warn(f"Trackable {trackable._id} has no bounding box to run SAM2 on.")
            return

        if self.camera._recent_rgb is None:
            self.get_logger().warn("No recent RGB frame available for SAM2 inference.")
            return

        try:
            # Convert latest frame to PIL image
            img = cv2.cvtColor(self.camera._recent_rgb, cv2.COLOR_BGR2RGB)
            rgb_pil: pil_image.Image = pil_image.fromarray(img)

            # Run SAM2 on the trackable's last bbox
            masks = self._inference_sam2(rgb_pil, bbox=trackable._most_recent_bbox)

            if masks.shape[0] == 0:
                self.get_logger().warn(f"SAM2 returned no masks for trackable {trackable._id}")
                return

            # Get refined bbox from the first mask (you could handle multiple masks if needed)
            refined_bbox = self.mask_to_bbox(masks[0])
            if refined_bbox is not None:
                # Update the Trackable object with the refined bbox trackable.update_bbox(refined_bbox)
                # Also update the frame so tracker can use it
                trackable._recent_frame = self.camera._recent_rgb

                self.get_logger().info(f"Trackable {trackable._id} updated with new bbox: {refined_bbox}")
        
        except Exception as e:
            if e is KeyboardInterrupt:
                self.get_logger().warn("Warn: Exception recieved keyboard interrupt, closing the function ")
                return 
            else:
                self.get_logger().warn(f"Function call threw an exception, {e}") 
                return
        finally:
            log_message = log_entry.finalize()
            if self.get_logger().is_enabled_for(logging.DEBUG):
                self.get_logger().debug(log_message)

    def sam2_inference_create_trackable(self, request: Sam2Inference_Request, response: Sam2Inference_Response):
        self.mutex = threading.Lock
    
        log_entry = Log(node_name=self.get_name(), method_name=inspect.currentframe().f_code.co_name)

        curr_time = self.get_clock().now().to_msg().sec 
        self.get_logger().debug(
            f"Starting awaiting mutex in sam2_inference_image, time started: "
            f"{self.get_clock().now().to_msg().sec}"
        )

        try:
                #self.get_logger().info("Starting image-based SAM2 inference")

                bbox_target = request.bounding_box
                if bbox_target is None:
                    response.output = "No target bbox provided."
                    return response

                if self.camera._recent_rgb is None:
                    response.output = "No recent RGB image available."
                    return response

                # Convert image to PIL format
                img = cv2.cvtColor(self.camera._recent_rgb, cv2.COLOR_BGR2RGB)
                rgb_pil: pil_image.Image = pil_image.fromarray(img)
                
                inputs = self.processor(
                    images=rgb_pil,
                    input_boxes=[[[bbox_target[0], bbox_target[1], bbox_target[2], bbox_target[3]]]],
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
               # response.output = str(masks) 
               # return response
                self.active_trackables.append( Trackable(self, 1, 10000, 20, init_masks=masks))
               
        except Exception as e:
            if e is KeyboardInterrupt:
                response.output = "Keyboard interrupt is called, exiting the function" 
                self.get_logger().warn("Warn: Exception recieved keyboard interrupt, closing the function ")
                return response
            else:
                response.output = f"Function call threw an exception,  {e}"
                self.get_logger().warn(f"Function call threw an exception, {e}") 
                return response

        log_message = log_entry.finalize()
    
        if self.get_logger().is_enabled_for(logging.DEBUG):
            self.get_logger().debug(log_message)
        
        self.get_logger().debug(
                f"Finished mutex awaiting, function completed in:"
                f"{curr_time - self.get_clock().now().to_msg().sec}"
            )
        return response

    def depth_callback(self, msg: Image):
        self.camera._recent_depth = msg
        if self.get_logger().is_enabled_for(logging.DEBUG):
            self.get_logger().debug(f"Received new depth frame")

    def rgb_callback(self, msg: Image):
        self.camera._recent_rgb = bridge.imgmsg_to_cv2(msg, "rgb8")
        if self.get_logger().is_enabled_for(logging.DEBUG):
            self.get_logger().debug(f"Received new RGB frame")
        for trackable in self.active_trackables:
            trackable.update_frame_and_bbox(
                frame=self.camera._recent_rgb)

def overlay_masks_on_image(original_image, masks, alpha=0.5):
    image_np = np.array(original_image)
    overlay = image_np.copy()
    num_objects = masks.shape[0]
    colors = cm.rainbow(np.linspace(0, 1, num_objects))

    for i in range(num_objects):
        mask = masks[i, 0] if len(masks.shape) == 4 else masks[i]
        binary_mask = mask > 0.5
        color = (colors[i][:3] * 255).astype(np.uint8)
        overlay[binary_mask] = (overlay[binary_mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    return pil_image.fromarray(overlay)

from typing import List, Optional


class Trackable:
    def __init__(
        self,
        node: Node,
        update_rate: float,
        time_alive: int,
        id: int,
        init_masks,
        pub_topic: Optional[str] = None,
    ):
        self._node = node
        self._update_rate = update_rate
        self._time_alive = time_alive
        self._id = id
        self._pub_topic = pub_topic or f"/trackable/id_{id}"

        self._counter = 0
        self._active = True

        # Frame + bbox inputs
        self._recent_frame = None
        self._recent_bboxes = []

        # Tracking state
        self._tracked_objects = []
        self._init_masks = init_masks
        self._sam_request_pending = False

        # Publisher
        self.tracking_pub = self._node.create_publisher(
            String, self._pub_topic, 10
        )

        # Tracker
        self.tracker = DeepSort(max_age=30)

        self._node.get_logger().info(
            f"Trackable {id} initialized, publishing to {self._pub_topic}"
        )

        # Timer
        self._timer = self._node.create_timer(
            self._update_rate, self._update_tracker
        )

        # Lazy initialization (safer)
        self._initialized = False

    # ------------------------------------------------------------------

    def update_frame_and_bbox(self, frame):
        self._recent_frame = frame

    # ------------------------------------------------------------------

    def _update_tracker(self):
        if not self._active:
            return

        if self._counter >= self._time_alive:
            self._cleanup()
            return

        # Initialize tracker when first frame arrives
        if not self._initialized and self._recent_frame is not None:
            self._init_tracker(self._recent_frame, self._init_masks)
            self._initialized = True

        # Trigger SAM if tracking is weak (with cooldown)
        if (
            self._counter >= 500000
            and not self._sam_request_pending
            and len(self._tracked_objects) == 0
        ):
            self._sam_request_pending = True
            self._cleanup()
        if self._recent_frame is None:
            return

        # Update tracker
        self._tracked_objects = self.update(
            self._recent_frame, self._recent_bboxes
        )
        if len(self._tracked_objects) >= 1:
            tracked_info = self._tracked_objects[0].to_ltrb()
        else:
            tracked_info = 0

        # Publish results
        msg = String()
        msg.data = f"Trackable {self._id}: {tracked_info}"
        self.tracking_pub.publish(msg)

        self._node.get_logger().debug(msg.data)

        self._counter += 1

    # ------------------------------------------------------------------

    def _cleanup(self):
        self._active = False
        self._node.destroy_timer(self._timer)

        self._recent_frame = None
        self._recent_bboxes = []
        self._tracked_objects = []

        self._node.get_logger().info(
            f"Trackable {self._id} expired and cleaned up."
        )

    # ------------------------------------------------------------------

    def _init_tracker(self, frame, masks):
        detections = []

        for mask in masks:
            bbox = masks_batched_to_boxes(mask)
            if bbox is None:
                continue
            bbox_ = bbox.detach().cpu().numpy()[0, 0]

            tlwh = ltrb_to_tlwh(bbox=bbox_)
            detections.append((tlwh, 1.0, "object"))

        self._tracked_objects = self.tracker.update_tracks(detections, frame=frame)
        self._recent_bboxes.append(self._tracked_objects[0].to_ltrb())
        self._node.get_logger().debug(f"Tracker Intialization at: {self._node.get_clock().now().seconds_nanoseconds()}")
        self._node.get_logger().debug(f"Detections found: {self._tracked_objects[0].to_ltrb()}")
         
        
        return
    # ------------------------------------------------------------------

    def update(self, frame, bboxes: List):
        detections = []

        for bbox in bboxes:
            tlwh = ltrb_to_tlwh(bbox)
            detections.append((tlwh, 1.0, "object"))

        tracks = self.tracker.update_tracks(detections, frame=frame)


        return tracks 

def main(args=None):
    
    rclpy.init(args=args)

    sam2_service_node = Sam2Service()

    try:
        rclpy.spin(sam2_service_node)
    except KeyboardInterrupt:
        sam2_service_node.get_logger().info("Shutting down Sam2Service node...")
    finally:
        sam2_service_node.destroy_node()
        rclpy.shutdown()
