from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.utilities import prefix_namespace
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import os 

def generate_launch_description():
    grasp_gen_config = os.path.join(
        get_package_share_directory('vision_utils'),
        'launch',
        'config',
        'owl_config.yaml'
    )
    GraspGenNode = Node(
        package='vision_utils',            # package name
        namespace='owl_inference',   # optional, usually for grouping
        executable='owl_inference',  # matches setup.py entry point
        name='owl_inference_node',   # actual node name
        parameters=[grasp_gen_config]     
    ) 
    
    return LaunchDescription([GraspGenNode])     
