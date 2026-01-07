apt update && apt install -y ros-dev-tools

sudo apt update

apt update && \
    apt upgrade -yq && \
    apt install -yq ros-jazzy-desktop

echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
	&& /bin/bash -c "source /root/.bashrc"

echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> \ 
bashrc

echo 'eval "$(register-python-argcomplete ros2)"' >> ~/.bashrc
echo 'eval "$(register-python-argcomplete colcon)"' >> ~/.bashrc
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
apt-get install -y ros-jazzy-rmw-cyclonedds-cpp
