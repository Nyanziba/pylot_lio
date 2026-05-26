// Copyright 2026 PyLoT Robotics. Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>

#include "pylot_lio/factory.hpp"

namespace pylot_lio
{

TEST(Factory, DefaultConfigBuildsValidBackends)
{
  LioBackendConfig config;
  // small_gicp が無い環境では plain_gicp に切り替え。
  config.registration_name = "plain_gicp";
  const auto backends = createBackendsFromConfig(config);
  EXPECT_NE(backends.preprocessor, nullptr);
  EXPECT_NE(backends.point_cloud_map, nullptr);
  EXPECT_NE(backends.registration, nullptr);
  EXPECT_NE(backends.state_estimator, nullptr);
  EXPECT_FALSE(backends.summary.empty());
}

TEST(Factory, UnknownPreprocessorThrows)
{
  LioBackendConfig config;
  config.preprocessor_name = "definitely_not_a_real_name";
  EXPECT_THROW(createBackendsFromConfig(config), std::invalid_argument);
}

TEST(Factory, UnknownMapThrows)
{
  LioBackendConfig config;
  config.map_name = "non_existent";
  EXPECT_THROW(createBackendsFromConfig(config), std::invalid_argument);
}

TEST(Factory, UnknownStateEstimatorThrows)
{
  LioBackendConfig config;
  config.registration_name = "plain_gicp";
  config.state_estimator_name = "non_existent";
  EXPECT_THROW(createBackendsFromConfig(config), std::invalid_argument);
}

TEST(Factory, AlternativeBackendsAlsoBuild)
{
  LioBackendConfig config;
  config.preprocessor_name = "random_sampling";
  config.map_name = "normal_map";
  config.registration_name = "plain_gicp";
  config.state_estimator_name = "hgo";
  const auto backends = createBackendsFromConfig(config);
  EXPECT_NE(backends.state_estimator, nullptr);
}

}  // namespace pylot_lio

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
