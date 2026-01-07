cmake_minimum_required(VERSION 3.8)
project(vision_msgs)

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =====================
# Dependencies
# =====================
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# =====================
# Message / Service generation
# =====================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OwlInfo.msg"
  "srv/OwlInference.srv"
  "srv/OwlInferenceM.srv"
  "srv/Sam2Inference.srv"
  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
)

# Export runtime dependency for downstream packages
ament_export_dependencies(rosidl_default_runtime)

# =====================
# (Optional) C++ node
# =====================
add_executable(${PROJECT_NAME}_node include/main.cpp)

target_include_directories(${PROJECT_NAME}_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME}_node PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(${PROJECT_NAME}_node
  geometry_msgs
  sensor_msgs
)

# IMPORTANT: link generated message types
rosidl_target_interfaces(${PROJECT_NAME}_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# =====================
# Testing
# =====================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

