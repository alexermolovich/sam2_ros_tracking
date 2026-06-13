# SAM2 ROS Tracking

ROS 2 prototype for open-vocabulary object detection, SAM2 segmentation, and
lightweight object tracking from camera images.

The repository contains two ROS 2 packages:

- `vision_msgs`: custom ROS messages and services.
- `vision_utils`: Python nodes for OWLv2 inference, SAM2 inference, bounding-box
  selection, and tracking.

At a high level, the intended flow is:

1. Subscribe to RGB/depth camera topics.
2. Use OWLv2 to find a text-prompted object and return its bounding box.
3. Send that bounding box to SAM2 to create an object mask.
4. Initialize a DeepSORT tracker and publish trackable updates.

## Repository Layout

```text
.
|-- docker/
|   |-- build.sh                  # Builds the CUDA/PyTorch container image
|   |-- run.sh                    # Runs the container with GPU, host networking, and X11
|   |-- owl.docker_container      # Container definition
|   `-- ros2_setup.sh             # ROS 2 Jazzy setup helper for the container
|-- vision_msgs/
|   |-- msg/OwlInfo.msg
|   `-- srv/
|       |-- OwlInference.srv      # Text prompt -> best OWL bounding box
|       |-- OwlInferenceM.srv     # Image message + text prompt -> OWL result
|       `-- Sam2Inference.srv     # Bounding box -> SAM2/tracking request
`-- vision_utils/
    |-- launch/
    |   |-- owl_infer.launch.py
    |   `-- config/owl_config.yaml
    `-- vision_utils/
        |-- owl_inference.py      # OWLv2 ROS service node
        |-- sam2_inference.py     # SAM2 + DeepSORT service/tracking node
        |-- sam2_bbox_selector.py # Experimental matplotlib bbox selector
        `-- utils.py
```

## Requirements

The Docker scripts target Ubuntu 24.04, ROS 2 Jazzy, CUDA, and PyTorch. A GPU is
strongly recommended because both OWLv2 and SAM2 are large vision models.

Main runtime dependencies used by the nodes:

- ROS 2 Jazzy with `colcon`
- `sensor_msgs`, `std_msgs`, `geometry_msgs`, `cv_bridge`
- `torch`, `torchvision`, `transformers`, `accelerate`
- `opencv-python`, `pillow`, `matplotlib`, `numpy`
- `deep-sort-realtime`, `pycocotools`

The first model run will download Hugging Face checkpoints:

- `google/owlv2-base-patch16-ensemble`
- `facebook/sam2.1-hiera-large`

## Build Locally

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash

# Install ROS package dependencies where rosdep rules exist.
rosdep install --from-paths . --ignore-src -r -y

# Install Python ML dependencies not currently declared in package.xml.
python3 -m pip install --upgrade \
  transformers accelerate opencv-python pillow matplotlib numpy \
  deep-sort-realtime pycocotools

# Install torch/torchvision for your CUDA or CPU environment.
# Example only; choose the PyTorch command that matches your machine.
python3 -m pip install --upgrade torch torchvision

colcon build --symlink-install
source install/setup.bash
```

## Docker Workflow

Build and enter the container:

```bash
./docker/build.sh
./docker/run.sh
```

The container mounts this repository at `/workspace`, uses host networking, and
passes through the GPU and display environment.

Inside the container, install/source ROS 2 as needed, then build:

```bash
bash docker/ros2_setup.sh
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Camera Topics

Default topic settings live in
`vision_utils/launch/config/owl_config.yaml`:

```yaml
/**:
  ros__parameters:
    rgb_topic: "/camera/rgb/image_raw"
    depth_topic: "/camera/depth/image_raw"
```

The OWL node respects these parameters. The SAM2 node declares the same
parameters, but its RGB subscription is currently hard-coded to
`/camera/rgb/image_raw`.

## Running OWLv2 Inference

Launch the OWL service with the included launch file:

```bash
ros2 launch vision_utils owl_infer.launch.py
```

Or run it directly with parameter overrides:

```bash
ros2 run vision_utils owl_inference --ros-args \
  -p rgb_topic:=/camera/rgb/image_raw \
  -p depth_topic:=/camera/depth/image_raw
```

Call the text-prompt service:

```bash
ros2 service call /owl_inference/owl_infernce_topic vision_msgs/srv/OwlInference \
  "{target_class: 'mug'}"
```

When run without the launch namespace, the service is:

```bash
ros2 service call /owl_infernce_topic vision_msgs/srv/OwlInference \
  "{target_class: 'mug'}"
```

Note: the service names are currently spelled `infernce` in the code.

## Running SAM2 Tracking

Start the SAM2 service:

```bash
ros2 run vision_utils sam2_service_inference
```

Create a trackable from a bounding box:

```bash
ros2 service call /sam2_inference_image vision_msgs/srv/Sam2Inference \
  "{settings: '', bounding_box: [100, 120, 320, 360], inferce_type: 'image', inference_state: 'continue'}"
```

The current implementation creates a trackable with ID `20` and publishes its
latest tracked box as a string:

```bash
ros2 topic echo /trackable/id_20
```

## Bounding Box Selector

An experimental selector node is available:

```bash
ros2 run vision_utils sam2_bbox_selector
```

It listens to `/camera/rgb/image_raw` and opens a matplotlib window for selecting
one point or two corners of a bounding box. This node requires a working display
or virtual display.

## ROS Interfaces

### Services

| Service name | Type | Purpose |
| --- | --- | --- |
| `/owl_infernce_topic` | `vision_msgs/srv/OwlInference` | Uses the latest subscribed RGB frame and a text class prompt to return the best OWLv2 bounding box as a string. |
| `/owl_infernce_message` | `vision_msgs/srv/OwlInferenceM` | Intended to run OWLv2 on an image provided in the service request. |
| `/sam2_inference_image` | `vision_msgs/srv/Sam2Inference` | Accepts a bounding box and initializes a SAM2/DeepSORT trackable. |

### Topics

| Topic | Type | Direction | Notes |
| --- | --- | --- | --- |
| `/camera/rgb/image_raw` | `sensor_msgs/msg/Image` | Subscribe | Default RGB stream. |
| `/camera/depth/image_raw` | `sensor_msgs/msg/Image` | Subscribe | Default depth stream in config/SAM2. |
| `/trackable/id_20` | `std_msgs/msg/String` | Publish | Current hard-coded trackable output topic. |

## Development

Run package tests after building:

```bash
colcon test --packages-select vision_msgs vision_utils
colcon test-result --verbose
```

Useful service definition checks:

```bash
ros2 interface show vision_msgs/srv/OwlInference
ros2 interface show vision_msgs/srv/OwlInferenceM
ros2 interface show vision_msgs/srv/Sam2Inference
```

## Current Caveats

This repository is still in prototype shape. A few implementation details to keep
in mind while using or extending it:

- `owl_infernce_topic` and `owl_infernce_message` are misspelled service names.
- `Sam2Inference.srv` has a misspelled `inferce_type` field.
- `OwlInferenceM.srv` returns `output`, but `owl_inference.py` currently writes
  to `answer` in the message-service callback.
- `vision_utils/package.xml` does not yet declare every runtime dependency used
  by the Python nodes.
- The SAM2 tracking service currently hard-codes trackable ID `20`.
