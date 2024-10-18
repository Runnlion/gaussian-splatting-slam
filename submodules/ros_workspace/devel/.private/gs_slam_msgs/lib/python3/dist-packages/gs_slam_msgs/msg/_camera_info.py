# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from gs_slam_msgs/camera_info.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg

class camera_info(genpy.Message):
  _md5sum = "ca223b75c34f358f39ca28c66f6f8425"
  _type = "gs_slam_msgs/camera_info"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """sensor_msgs/Image Image
sensor_msgs/CameraInfo CameraInfo
geometry_msgs/TransformStamped CameraPose
sensor_msgs/PointCloud2 Local_Map
================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: sensor_msgs/CameraInfo
# This message defines meta information for a camera. It should be in a
# camera namespace on topic "camera_info" and accompanied by up to five
# image topics named:
#
#   image_raw - raw data from the camera driver, possibly Bayer encoded
#   image            - monochrome, distorted
#   image_color      - color, distorted
#   image_rect       - monochrome, rectified
#   image_rect_color - color, rectified
#
# The image_pipeline contains packages (image_proc, stereo_image_proc)
# for producing the four processed image topics from image_raw and
# camera_info. The meaning of the camera parameters are described in
# detail at http://www.ros.org/wiki/image_pipeline/CameraInfo.
#
# The image_geometry package provides a user-friendly interface to
# common operations using this meta information. If you want to, e.g.,
# project a 3d point into image coordinates, we strongly recommend
# using image_geometry.
#
# If the camera is uncalibrated, the matrices D, K, R, P should be left
# zeroed out. In particular, clients may assume that K[0] == 0.0
# indicates an uncalibrated camera.

#######################################################################
#                     Image acquisition info                          #
#######################################################################

# Time of image acquisition, camera coordinate frame ID
Header header    # Header timestamp should be acquisition time of image
                 # Header frame_id should be optical frame of camera
                 # origin of frame should be optical center of camera
                 # +x should point to the right in the image
                 # +y should point down in the image
                 # +z should point into the plane of the image


#######################################################################
#                      Calibration Parameters                         #
#######################################################################
# These are fixed during camera calibration. Their values will be the #
# same in all messages until the camera is recalibrated. Note that    #
# self-calibrating systems may "recalibrate" frequently.              #
#                                                                     #
# The internal parameters can be used to warp a raw (distorted) image #
# to:                                                                 #
#   1. An undistorted image (requires D and K)                        #
#   2. A rectified image (requires D, K, R)                           #
# The projection matrix P projects 3D points into the rectified image.#
#######################################################################

# The image dimensions with which the camera was calibrated. Normally
# this will be the full camera resolution in pixels.
uint32 height
uint32 width

# The distortion model used. Supported models are listed in
# sensor_msgs/distortion_models.h. For most cameras, "plumb_bob" - a
# simple model of radial and tangential distortion - is sufficient.
string distortion_model

# The distortion parameters, size depending on the distortion model.
# For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
float64[] D

# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
# Projects 3D points in the camera coordinate frame to 2D pixel
# coordinates using the focal lengths (fx, fy) and principal point
# (cx, cy).
float64[9]  K # 3x3 row-major matrix

# Rectification matrix (stereo cameras only)
# A rotation matrix aligning the camera coordinate system to the ideal
# stereo image plane so that epipolar lines in both stereo images are
# parallel.
float64[9]  R # 3x3 row-major matrix

# Projection/camera matrix
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
# By convention, this matrix specifies the intrinsic (camera) matrix
#  of the processed (rectified) image. That is, the left 3x3 portion
#  is the normal camera intrinsic matrix for the rectified image.
# It projects 3D points in the camera coordinate frame to 2D pixel
#  coordinates using the focal lengths (fx', fy') and principal point
#  (cx', cy') - these may differ from the values in K.
# For monocular cameras, Tx = Ty = 0. Normally, monocular cameras will
#  also have R = the identity and P[1:3,1:3] = K.
# For a stereo pair, the fourth column [Tx Ty 0]' is related to the
#  position of the optical center of the second camera in the first
#  camera's frame. We assume Tz = 0 so both cameras are in the same
#  stereo image plane. The first camera always has Tx = Ty = 0. For
#  the right (second) camera of a horizontal stereo pair, Ty = 0 and
#  Tx = -fx' * B, where B is the baseline between the cameras.
# Given a 3D point [X Y Z]', the projection (x, y) of the point onto
#  the rectified image is given by:
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w
#  This holds for both images of a stereo pair.
float64[12] P # 3x4 row-major matrix


#######################################################################
#                      Operational Parameters                         #
#######################################################################
# These define the image region actually captured by the camera       #
# driver. Although they affect the geometry of the output image, they #
# may be changed freely without recalibrating the camera.             #
#######################################################################

# Binning refers here to any camera setting which combines rectangular
#  neighborhoods of pixels into larger "super-pixels." It reduces the
#  resolution of the output image to
#  (width / binning_x) x (height / binning_y).
# The default values binning_x = binning_y = 0 is considered the same
#  as binning_x = binning_y = 1 (no subsampling).
uint32 binning_x
uint32 binning_y

# Region of interest (subwindow of full camera resolution), given in
#  full resolution (unbinned) image coordinates. A particular ROI
#  always denotes the same window of pixels on the camera sensor,
#  regardless of binning settings.
# The default setting of roi (all values 0) is considered the same as
#  full resolution (roi.width = width, roi.height = height).
RegionOfInterest roi

================================================================================
MSG: sensor_msgs/RegionOfInterest
# This message is used to specify a region of interest within an image.
#
# When used to specify the ROI setting of the camera when the image was
# taken, the height and width fields should either match the height and
# width fields for the associated image; or height = width = 0
# indicates that the full resolution image was captured.

uint32 x_offset  # Leftmost pixel of the ROI
                 # (0 if the ROI includes the left edge of the image)
uint32 y_offset  # Topmost pixel of the ROI
                 # (0 if the ROI includes the top edge of the image)
uint32 height    # Height of ROI
uint32 width     # Width of ROI

# True if a distinct rectified ROI should be calculated from the "raw"
# ROI in this message. Typically this should be False if the full image
# is captured (ROI not used), and True if a subwindow is captured (ROI
# used).
bool do_rectify

================================================================================
MSG: geometry_msgs/TransformStamped
# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id
#
# This message is mostly used by the 
# <a href="http://wiki.ros.org/tf">tf</a> package. 
# See its documentation for more information.

Header header
string child_frame_id # the frame id of the child frame
Transform transform

================================================================================
MSG: geometry_msgs/Transform
# This represents the transform between two coordinate frames in free space.

Vector3 translation
Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: sensor_msgs/PointCloud2
# This message holds a collection of N-dimensional points, which may
# contain additional information such as normals, intensity, etc. The
# point data is stored as a binary blob, its layout described by the
# contents of the "fields" array.

# The point cloud data may be organized 2d (image-like) or 1d
# (unordered). Point clouds organized as 2d images may be produced by
# camera depth sensors such as stereo or time-of-flight.

# Time of sensor data acquisition, and the coordinate frame ID (for 3d
# points).
Header header

# 2D structure of the point cloud. If the cloud is unordered, height is
# 1 and width is the length of the point cloud.
uint32 height
uint32 width

# Describes the channels and their layout in the binary data blob.
PointField[] fields

bool    is_bigendian # Is this data bigendian?
uint32  point_step   # Length of a point in bytes
uint32  row_step     # Length of a row in bytes
uint8[] data         # Actual point data, size is (row_step*height)

bool is_dense        # True if there are no invalid points

================================================================================
MSG: sensor_msgs/PointField
# This message holds the description of one point entry in the
# PointCloud2 message format.
uint8 INT8    = 1
uint8 UINT8   = 2
uint8 INT16   = 3
uint8 UINT16  = 4
uint8 INT32   = 5
uint8 UINT32  = 6
uint8 FLOAT32 = 7
uint8 FLOAT64 = 8

string name      # Name of field
uint32 offset    # Offset from start of point struct
uint8  datatype  # Datatype enumeration, see above
uint32 count     # How many elements in the field
"""
  __slots__ = ['Image','CameraInfo','CameraPose','Local_Map']
  _slot_types = ['sensor_msgs/Image','sensor_msgs/CameraInfo','geometry_msgs/TransformStamped','sensor_msgs/PointCloud2']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       Image,CameraInfo,CameraPose,Local_Map

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(camera_info, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.Image is None:
        self.Image = sensor_msgs.msg.Image()
      if self.CameraInfo is None:
        self.CameraInfo = sensor_msgs.msg.CameraInfo()
      if self.CameraPose is None:
        self.CameraPose = geometry_msgs.msg.TransformStamped()
      if self.Local_Map is None:
        self.Local_Map = sensor_msgs.msg.PointCloud2()
    else:
      self.Image = sensor_msgs.msg.Image()
      self.CameraInfo = sensor_msgs.msg.CameraInfo()
      self.CameraPose = geometry_msgs.msg.TransformStamped()
      self.Local_Map = sensor_msgs.msg.PointCloud2()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.Image.header.seq, _x.Image.header.stamp.secs, _x.Image.header.stamp.nsecs))
      _x = self.Image.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.Image.height, _x.Image.width))
      _x = self.Image.encoding
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_BI().pack(_x.Image.is_bigendian, _x.Image.step))
      _x = self.Image.data
      length = len(_x)
      # - if encoded as a list instead, serialize as bytes instead of string
      if type(_x) in [list, tuple]:
        buff.write(struct.Struct('<I%sB'%length).pack(length, *_x))
      else:
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.CameraInfo.header.seq, _x.CameraInfo.header.stamp.secs, _x.CameraInfo.header.stamp.nsecs))
      _x = self.CameraInfo.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.CameraInfo.height, _x.CameraInfo.width))
      _x = self.CameraInfo.distortion_model
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      length = len(self.CameraInfo.D)
      buff.write(_struct_I.pack(length))
      pattern = '<%sd'%length
      buff.write(struct.Struct(pattern).pack(*self.CameraInfo.D))
      buff.write(_get_struct_9d().pack(*self.CameraInfo.K))
      buff.write(_get_struct_9d().pack(*self.CameraInfo.R))
      buff.write(_get_struct_12d().pack(*self.CameraInfo.P))
      _x = self
      buff.write(_get_struct_6IB3I().pack(_x.CameraInfo.binning_x, _x.CameraInfo.binning_y, _x.CameraInfo.roi.x_offset, _x.CameraInfo.roi.y_offset, _x.CameraInfo.roi.height, _x.CameraInfo.roi.width, _x.CameraInfo.roi.do_rectify, _x.CameraPose.header.seq, _x.CameraPose.header.stamp.secs, _x.CameraPose.header.stamp.nsecs))
      _x = self.CameraPose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.CameraPose.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_7d3I().pack(_x.CameraPose.transform.translation.x, _x.CameraPose.transform.translation.y, _x.CameraPose.transform.translation.z, _x.CameraPose.transform.rotation.x, _x.CameraPose.transform.rotation.y, _x.CameraPose.transform.rotation.z, _x.CameraPose.transform.rotation.w, _x.Local_Map.header.seq, _x.Local_Map.header.stamp.secs, _x.Local_Map.header.stamp.nsecs))
      _x = self.Local_Map.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.Local_Map.height, _x.Local_Map.width))
      length = len(self.Local_Map.fields)
      buff.write(_struct_I.pack(length))
      for val1 in self.Local_Map.fields:
        _x = val1.name
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = val1
        buff.write(_get_struct_IBI().pack(_x.offset, _x.datatype, _x.count))
      _x = self
      buff.write(_get_struct_B2I().pack(_x.Local_Map.is_bigendian, _x.Local_Map.point_step, _x.Local_Map.row_step))
      _x = self.Local_Map.data
      length = len(_x)
      # - if encoded as a list instead, serialize as bytes instead of string
      if type(_x) in [list, tuple]:
        buff.write(struct.Struct('<I%sB'%length).pack(length, *_x))
      else:
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.Local_Map.is_dense
      buff.write(_get_struct_B().pack(_x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.Image is None:
        self.Image = sensor_msgs.msg.Image()
      if self.CameraInfo is None:
        self.CameraInfo = sensor_msgs.msg.CameraInfo()
      if self.CameraPose is None:
        self.CameraPose = geometry_msgs.msg.TransformStamped()
      if self.Local_Map is None:
        self.Local_Map = sensor_msgs.msg.PointCloud2()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.Image.header.seq, _x.Image.header.stamp.secs, _x.Image.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.Image.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.Image.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.Image.height, _x.Image.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.Image.encoding = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.Image.encoding = str[start:end]
      _x = self
      start = end
      end += 5
      (_x.Image.is_bigendian, _x.Image.step,) = _get_struct_BI().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.Image.data = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.CameraInfo.header.seq, _x.CameraInfo.header.stamp.secs, _x.CameraInfo.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraInfo.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraInfo.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.CameraInfo.height, _x.CameraInfo.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraInfo.distortion_model = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraInfo.distortion_model = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sd'%length
      start = end
      s = struct.Struct(pattern)
      end += s.size
      self.CameraInfo.D = s.unpack(str[start:end])
      start = end
      end += 72
      self.CameraInfo.K = _get_struct_9d().unpack(str[start:end])
      start = end
      end += 72
      self.CameraInfo.R = _get_struct_9d().unpack(str[start:end])
      start = end
      end += 96
      self.CameraInfo.P = _get_struct_12d().unpack(str[start:end])
      _x = self
      start = end
      end += 37
      (_x.CameraInfo.binning_x, _x.CameraInfo.binning_y, _x.CameraInfo.roi.x_offset, _x.CameraInfo.roi.y_offset, _x.CameraInfo.roi.height, _x.CameraInfo.roi.width, _x.CameraInfo.roi.do_rectify, _x.CameraPose.header.seq, _x.CameraPose.header.stamp.secs, _x.CameraPose.header.stamp.nsecs,) = _get_struct_6IB3I().unpack(str[start:end])
      self.CameraInfo.roi.do_rectify = bool(self.CameraInfo.roi.do_rectify)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraPose.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraPose.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraPose.child_frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraPose.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 68
      (_x.CameraPose.transform.translation.x, _x.CameraPose.transform.translation.y, _x.CameraPose.transform.translation.z, _x.CameraPose.transform.rotation.x, _x.CameraPose.transform.rotation.y, _x.CameraPose.transform.rotation.z, _x.CameraPose.transform.rotation.w, _x.Local_Map.header.seq, _x.Local_Map.header.stamp.secs, _x.Local_Map.header.stamp.nsecs,) = _get_struct_7d3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.Local_Map.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.Local_Map.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.Local_Map.height, _x.Local_Map.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.Local_Map.fields = []
      for i in range(0, length):
        val1 = sensor_msgs.msg.PointField()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.name = str[start:end].decode('utf-8', 'rosmsg')
        else:
          val1.name = str[start:end]
        _x = val1
        start = end
        end += 9
        (_x.offset, _x.datatype, _x.count,) = _get_struct_IBI().unpack(str[start:end])
        self.Local_Map.fields.append(val1)
      _x = self
      start = end
      end += 9
      (_x.Local_Map.is_bigendian, _x.Local_Map.point_step, _x.Local_Map.row_step,) = _get_struct_B2I().unpack(str[start:end])
      self.Local_Map.is_bigendian = bool(self.Local_Map.is_bigendian)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.Local_Map.data = str[start:end]
      start = end
      end += 1
      (self.Local_Map.is_dense,) = _get_struct_B().unpack(str[start:end])
      self.Local_Map.is_dense = bool(self.Local_Map.is_dense)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.Image.header.seq, _x.Image.header.stamp.secs, _x.Image.header.stamp.nsecs))
      _x = self.Image.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.Image.height, _x.Image.width))
      _x = self.Image.encoding
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_BI().pack(_x.Image.is_bigendian, _x.Image.step))
      _x = self.Image.data
      length = len(_x)
      # - if encoded as a list instead, serialize as bytes instead of string
      if type(_x) in [list, tuple]:
        buff.write(struct.Struct('<I%sB'%length).pack(length, *_x))
      else:
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_3I().pack(_x.CameraInfo.header.seq, _x.CameraInfo.header.stamp.secs, _x.CameraInfo.header.stamp.nsecs))
      _x = self.CameraInfo.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.CameraInfo.height, _x.CameraInfo.width))
      _x = self.CameraInfo.distortion_model
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      length = len(self.CameraInfo.D)
      buff.write(_struct_I.pack(length))
      pattern = '<%sd'%length
      buff.write(self.CameraInfo.D.tostring())
      buff.write(self.CameraInfo.K.tostring())
      buff.write(self.CameraInfo.R.tostring())
      buff.write(self.CameraInfo.P.tostring())
      _x = self
      buff.write(_get_struct_6IB3I().pack(_x.CameraInfo.binning_x, _x.CameraInfo.binning_y, _x.CameraInfo.roi.x_offset, _x.CameraInfo.roi.y_offset, _x.CameraInfo.roi.height, _x.CameraInfo.roi.width, _x.CameraInfo.roi.do_rectify, _x.CameraPose.header.seq, _x.CameraPose.header.stamp.secs, _x.CameraPose.header.stamp.nsecs))
      _x = self.CameraPose.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.CameraPose.child_frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_7d3I().pack(_x.CameraPose.transform.translation.x, _x.CameraPose.transform.translation.y, _x.CameraPose.transform.translation.z, _x.CameraPose.transform.rotation.x, _x.CameraPose.transform.rotation.y, _x.CameraPose.transform.rotation.z, _x.CameraPose.transform.rotation.w, _x.Local_Map.header.seq, _x.Local_Map.header.stamp.secs, _x.Local_Map.header.stamp.nsecs))
      _x = self.Local_Map.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self
      buff.write(_get_struct_2I().pack(_x.Local_Map.height, _x.Local_Map.width))
      length = len(self.Local_Map.fields)
      buff.write(_struct_I.pack(length))
      for val1 in self.Local_Map.fields:
        _x = val1.name
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = val1
        buff.write(_get_struct_IBI().pack(_x.offset, _x.datatype, _x.count))
      _x = self
      buff.write(_get_struct_B2I().pack(_x.Local_Map.is_bigendian, _x.Local_Map.point_step, _x.Local_Map.row_step))
      _x = self.Local_Map.data
      length = len(_x)
      # - if encoded as a list instead, serialize as bytes instead of string
      if type(_x) in [list, tuple]:
        buff.write(struct.Struct('<I%sB'%length).pack(length, *_x))
      else:
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      _x = self.Local_Map.is_dense
      buff.write(_get_struct_B().pack(_x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.Image is None:
        self.Image = sensor_msgs.msg.Image()
      if self.CameraInfo is None:
        self.CameraInfo = sensor_msgs.msg.CameraInfo()
      if self.CameraPose is None:
        self.CameraPose = geometry_msgs.msg.TransformStamped()
      if self.Local_Map is None:
        self.Local_Map = sensor_msgs.msg.PointCloud2()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.Image.header.seq, _x.Image.header.stamp.secs, _x.Image.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.Image.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.Image.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.Image.height, _x.Image.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.Image.encoding = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.Image.encoding = str[start:end]
      _x = self
      start = end
      end += 5
      (_x.Image.is_bigendian, _x.Image.step,) = _get_struct_BI().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.Image.data = str[start:end]
      _x = self
      start = end
      end += 12
      (_x.CameraInfo.header.seq, _x.CameraInfo.header.stamp.secs, _x.CameraInfo.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraInfo.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraInfo.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.CameraInfo.height, _x.CameraInfo.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraInfo.distortion_model = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraInfo.distortion_model = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sd'%length
      start = end
      s = struct.Struct(pattern)
      end += s.size
      self.CameraInfo.D = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=length)
      start = end
      end += 72
      self.CameraInfo.K = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=9)
      start = end
      end += 72
      self.CameraInfo.R = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=9)
      start = end
      end += 96
      self.CameraInfo.P = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=12)
      _x = self
      start = end
      end += 37
      (_x.CameraInfo.binning_x, _x.CameraInfo.binning_y, _x.CameraInfo.roi.x_offset, _x.CameraInfo.roi.y_offset, _x.CameraInfo.roi.height, _x.CameraInfo.roi.width, _x.CameraInfo.roi.do_rectify, _x.CameraPose.header.seq, _x.CameraPose.header.stamp.secs, _x.CameraPose.header.stamp.nsecs,) = _get_struct_6IB3I().unpack(str[start:end])
      self.CameraInfo.roi.do_rectify = bool(self.CameraInfo.roi.do_rectify)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraPose.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraPose.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.CameraPose.child_frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.CameraPose.child_frame_id = str[start:end]
      _x = self
      start = end
      end += 68
      (_x.CameraPose.transform.translation.x, _x.CameraPose.transform.translation.y, _x.CameraPose.transform.translation.z, _x.CameraPose.transform.rotation.x, _x.CameraPose.transform.rotation.y, _x.CameraPose.transform.rotation.z, _x.CameraPose.transform.rotation.w, _x.Local_Map.header.seq, _x.Local_Map.header.stamp.secs, _x.Local_Map.header.stamp.nsecs,) = _get_struct_7d3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.Local_Map.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.Local_Map.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 8
      (_x.Local_Map.height, _x.Local_Map.width,) = _get_struct_2I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.Local_Map.fields = []
      for i in range(0, length):
        val1 = sensor_msgs.msg.PointField()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.name = str[start:end].decode('utf-8', 'rosmsg')
        else:
          val1.name = str[start:end]
        _x = val1
        start = end
        end += 9
        (_x.offset, _x.datatype, _x.count,) = _get_struct_IBI().unpack(str[start:end])
        self.Local_Map.fields.append(val1)
      _x = self
      start = end
      end += 9
      (_x.Local_Map.is_bigendian, _x.Local_Map.point_step, _x.Local_Map.row_step,) = _get_struct_B2I().unpack(str[start:end])
      self.Local_Map.is_bigendian = bool(self.Local_Map.is_bigendian)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      self.Local_Map.data = str[start:end]
      start = end
      end += 1
      (self.Local_Map.is_dense,) = _get_struct_B().unpack(str[start:end])
      self.Local_Map.is_dense = bool(self.Local_Map.is_dense)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_12d = None
def _get_struct_12d():
    global _struct_12d
    if _struct_12d is None:
        _struct_12d = struct.Struct("<12d")
    return _struct_12d
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_6IB3I = None
def _get_struct_6IB3I():
    global _struct_6IB3I
    if _struct_6IB3I is None:
        _struct_6IB3I = struct.Struct("<6IB3I")
    return _struct_6IB3I
_struct_7d3I = None
def _get_struct_7d3I():
    global _struct_7d3I
    if _struct_7d3I is None:
        _struct_7d3I = struct.Struct("<7d3I")
    return _struct_7d3I
_struct_9d = None
def _get_struct_9d():
    global _struct_9d
    if _struct_9d is None:
        _struct_9d = struct.Struct("<9d")
    return _struct_9d
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B
_struct_B2I = None
def _get_struct_B2I():
    global _struct_B2I
    if _struct_B2I is None:
        _struct_B2I = struct.Struct("<B2I")
    return _struct_B2I
_struct_BI = None
def _get_struct_BI():
    global _struct_BI
    if _struct_BI is None:
        _struct_BI = struct.Struct("<BI")
    return _struct_BI
_struct_IBI = None
def _get_struct_IBI():
    global _struct_IBI
    if _struct_IBI is None:
        _struct_IBI = struct.Struct("<IBI")
    return _struct_IBI
