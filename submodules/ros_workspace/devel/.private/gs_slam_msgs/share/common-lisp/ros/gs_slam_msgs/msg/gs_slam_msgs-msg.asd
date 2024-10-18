
(cl:in-package :asdf)

(defsystem "gs_slam_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "visual_merged_msg" :depends-on ("_package_visual_merged_msg"))
    (:file "_package_visual_merged_msg" :depends-on ("_package"))
  ))