std_msgs/Header header

Description2D[] descriptions

sensor_msgs/Image image_set_of_marks
sensor_msgs/Image image_rgb #image in which 2D descriptions where observed (optional)
sensor_msgs/Image image_depth #depth image to possibly pass from Description to Description3D (needs camera_info too) (optional)
sensor_msgs/CameraInfo camera_info #camera info to transform from Description to Description3D (needs image_depth too) (optional)
sensor_msgs/PointCloud2 points #point cloud to transform from Description to Description3D (nothing is needed with it) (optional)