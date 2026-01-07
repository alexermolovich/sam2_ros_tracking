from dataclasses import dataclass
from sensor_msgs.msg import Image
# ========= Data Classes defintions =======#

@dataclass 
class Camera:
        height: int = 640
        width : int = 640 
        _recent_rgb:    Image = None
        _recent_depth:  Image = None 
