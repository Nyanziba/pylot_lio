// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include "pylot_lio/livox_conversion.hpp"

namespace pylot_lio
{

PointCloud convertLivoxRawPoints(const std::vector<LivoxRawPoint> & raw_points)
{
  PointCloud output_cloud;
  output_cloud.points.reserve(raw_points.size());
  for (const LivoxRawPoint & raw_point : raw_points) {
    Point converted_point;
    converted_point.x = raw_point.x;
    converted_point.y = raw_point.y;
    converted_point.z = raw_point.z;
    // reflectivity (0..255) を float に格上げ。 PCL の intensity は float なので
    // 後段は浮動小数として扱う。
    converted_point.intensity = static_cast<float>(raw_point.reflectivity);
    output_cloud.points.push_back(converted_point);
  }
  output_cloud.width = static_cast<uint32_t>(output_cloud.points.size());
  output_cloud.height = 1;
  output_cloud.is_dense = true;
  return output_cloud;
}

}  // namespace pylot_lio
