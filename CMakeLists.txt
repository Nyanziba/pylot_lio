cmake_minimum_required(VERSION 3.16)
project(pylot_lio)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Dependencies
# ============================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)
find_package(TBB QUIET)
find_package(small_gicp QUIET)
find_package(GTSAM QUIET)
find_package(livox_ros_driver2 QUIET)

# ------------------------------------------------------------
# 3rdparty fallback: small_gicp が見つからない場合は in-tree の
# submodule (3rdparty/small_gicp) を add_subdirectory で取り込む。
# システムインストール (conda-forge / robostack) を優先するが、
# それが無い環境でもビルドできるようにするためのフォールバック。
# ------------------------------------------------------------
if(NOT small_gicp_FOUND)
  set(PYLOT_LIO_SMALL_GICP_SUBMODULE
      "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/small_gicp")
  if(EXISTS "${PYLOT_LIO_SMALL_GICP_SUBMODULE}/CMakeLists.txt")
    message(STATUS "small_gicp not found via find_package; using bundled "
                   "submodule at 3rdparty/small_gicp (pinned to v1.0.0).")
    # small_gicp 自身のテスト・ベンチ・サンプルはビルドしない。
    set(BUILD_HELPER OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${PYLOT_LIO_SMALL_GICP_SUBMODULE}
                     ${CMAKE_CURRENT_BINARY_DIR}/small_gicp
                     EXCLUDE_FROM_ALL)
    set(small_gicp_FOUND TRUE)
    set(PYLOT_LIO_SMALL_GICP_FROM_SUBMODULE TRUE)
  else()
    message(STATUS "small_gicp not found and 3rdparty/small_gicp submodule "
                   "is not initialized; SmallGicpRegistration backend disabled. "
                   "Run 'git submodule update --init --recursive' to enable.")
  endif()
endif()

# IntelLLVM (icpx) を検出した場合のみ SYCL バックエンドをビルドする。
# stock conda-forge clang / gcc では SYCL 不可。
set(PYLOT_LIO_BUILD_SYCL OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(PYLOT_LIO_BUILD_SYCL ON)
endif()

# ------------------------------------------------------------
# sycl_points (fateshelled/sycl_points, Apache-2.0): SYCL ベースの
# header-only 点群処理ユーティリティ。 SYCL コンパイラ (= IntelLLVM)
# のときだけ include path を pylot_lio_core に追加する。 sycl_points 側
# は INTERFACE target を export していないので add_subdirectory ではなく
# include path 直挿しで取り込む。
# ------------------------------------------------------------
set(PYLOT_LIO_HAS_SYCL_POINTS FALSE)
if(PYLOT_LIO_BUILD_SYCL)
  set(PYLOT_LIO_SYCL_POINTS_INCLUDE
      "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sycl_points/cpp/include")
  if(IS_DIRECTORY "${PYLOT_LIO_SYCL_POINTS_INCLUDE}/sycl_points")
    message(STATUS "Found 3rdparty/sycl_points headers at "
                   "${PYLOT_LIO_SYCL_POINTS_INCLUDE}; SYCL backend can use them.")
    set(PYLOT_LIO_HAS_SYCL_POINTS TRUE)
  else()
    message(WARNING "PYLOT_LIO_BUILD_SYCL is ON but 3rdparty/sycl_points is "
                    "not initialized. Run 'git submodule update --init --recursive'.")
  endif()
endif()

set(COMMON_ROS_DEPENDENCIES
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_conversions
  pcl_ros
)

# ============================================================
# Core library: backend-agnostic primitives
# (lie_algebra, voxel_grid_preprocessor, voxel_map, kd_tree_map,
#  random_sampling_preprocessor, normal_map, IESKF, HGO, GICP-only)
# ============================================================
set(PYLOT_LIO_CORE_SOURCES
  src/lie_algebra.cpp
  src/preprocess/voxel_grid_preprocessor.cpp
  src/preprocess/random_sampling_preprocessor.cpp
  src/preprocess/voxel_random_sampling_preprocessor.cpp
  src/map/voxel_map.cpp
  src/map/kd_tree_map.cpp
  src/map/normal_map.cpp
  src/map/voxel_random_map.cpp
  src/map/voxel_keyframe_submap_map.cpp
  src/registration/plain_gicp_registration.cpp
  src/estimator/ieskf_estimator.cpp
  src/estimator/hgo_estimator.cpp
  src/estimator/gicp_only_estimator.cpp
  src/keyframe/distance_keyframe_selector.cpp
  src/livox_conversion.cpp
  src/loop/scan_context.cpp
  src/loop/scan_context_loop_detector.cpp
  src/loop/loop_closure_icp.cpp
  src/factory.cpp
)

add_library(pylot_lio_core SHARED ${PYLOT_LIO_CORE_SOURCES})
target_include_directories(pylot_lio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
ament_target_dependencies(pylot_lio_core ${COMMON_ROS_DEPENDENCIES})
target_link_libraries(pylot_lio_core ${PCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(pylot_lio_core OpenMP::OpenMP_CXX)
endif()

# ============================================================
# Optional parallel backend: Intel TBB (oneTBB)
# 見つかれば PlainGicp / SmallGicp の "tbb" バックエンドが有効になる。
# 無い環境では PYLOT_LIO_HAS_TBB が未定義のまま OpenMP/逐次へフォールバックする。
# ============================================================
if(TBB_FOUND)
  target_link_libraries(pylot_lio_core TBB::tbb)
  target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_TBB=1)
  message(STATUS "TBB found; 'tbb' parallel backend enabled.")
else()
  message(STATUS "TBB not found; 'tbb' backend disabled (falls back to OpenMP).")
endif()

# ============================================================
# Optional GPU backend: Apple Metal (metal-cpp)
# macOS でのみ有効。 metal-cpp は 3rdparty/metal-cpp に vendor 済みの
# ヘッダオンリー配布 (Apache-2.0)。 Metal/Foundation フレームワークをリンクし、
# PYLOT_LIO_HAS_METAL を定義する。 まずは saxpy プローブ (土台検証) のみ。
# 非 Apple ビルドでは metal_compute_probe.cpp はスタブとしてビルドされる。
# ============================================================
set(PYLOT_LIO_METALCPP_INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/metal-cpp")
target_sources(pylot_lio_core PRIVATE
  src/gpu/metal_compute_probe.cpp
  src/gpu/metal_vgicp_linearizer.cpp
)
if(APPLE AND IS_DIRECTORY "${PYLOT_LIO_METALCPP_INCLUDE}/Metal")
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    target_include_directories(pylot_lio_core PRIVATE "${PYLOT_LIO_METALCPP_INCLUDE}")
    target_link_libraries(pylot_lio_core ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_METAL=1)
    # metal-cpp は anonymous struct (GNU 拡張) を使うため、 metal-cpp を含む TU だけ
    # 警告を抑制する (-Wall -Wextra -Wpedantic 下で -Werror 化しても落ちないように)。
    set_source_files_properties(
      src/gpu/metal_compute_probe.cpp
      src/gpu/metal_vgicp_linearizer.cpp
      PROPERTIES
      COMPILE_OPTIONS "-Wno-gnu-anonymous-struct;-Wno-nested-anon-types")
    message(STATUS "Metal found; 'metal' GPU backend probe enabled (metal-cpp).")
  else()
    message(STATUS "Metal/Foundation frameworks not found; Metal probe builds as stub.")
  endif()
else()
  message(STATUS "Not Apple or metal-cpp absent; Metal probe builds as stub.")
endif()

# ============================================================
# Optional backend: small_gicp registration
# ============================================================
if(small_gicp_FOUND)
  target_sources(pylot_lio_core PRIVATE
    src/registration/small_gicp_registration.cpp
  )
  target_link_libraries(pylot_lio_core small_gicp::small_gicp)
  target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_SMALL_GICP=1)
else()
  message(STATUS "small_gicp not found; SmallGicpRegistration backend disabled.")
endif()

# ============================================================
# Optional backend: GTSAM pose graph optimization
# ============================================================
if(GTSAM_FOUND)
  target_sources(pylot_lio_core PRIVATE
    src/loop/pose_graph_optimizer.cpp
  )
  target_link_libraries(pylot_lio_core gtsam)
  target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_GTSAM=1)
else()
  target_sources(pylot_lio_core PRIVATE
    src/loop/pose_graph_optimizer_stub.cpp
  )
  message(STATUS "GTSAM not found; PoseGraphOptimizer falls back to stub (no-op).")
endif()

# ============================================================
# Optional backend: SYCL registration (Intel oneAPI / IntelLLVM only)
# ============================================================
if(PYLOT_LIO_BUILD_SYCL)
  target_sources(pylot_lio_core PRIVATE
    src/registration/sycl_registration.cpp
  )
  target_compile_options(pylot_lio_core PRIVATE -fsycl)
  target_link_options(pylot_lio_core PRIVATE -fsycl)
  target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_SYCL=1)
  # sycl_points (header-only) を include path として渡し、 識別用 macro を立てる。
  # sycl_registration.cpp 内部で #ifdef PYLOT_LIO_HAS_SYCL_POINTS により
  # sycl_points API (DegenerateRegularization 等) を呼べる。
  if(PYLOT_LIO_HAS_SYCL_POINTS)
    target_include_directories(pylot_lio_core SYSTEM PRIVATE
      ${PYLOT_LIO_SYCL_POINTS_INCLUDE})
    target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_SYCL_POINTS=1)
  endif()
else()
  target_sources(pylot_lio_core PRIVATE
    src/registration/sycl_registration_stub.cpp
  )
  message(STATUS "SYCL compiler not detected; SyclRegistration backend uses CPU fallback stub.")
endif()

# ============================================================
# Optional: Livox CustomMsg subscription support
# ============================================================
if(livox_ros_driver2_FOUND)
  message(STATUS "livox_ros_driver2 found; CustomMsg input format enabled.")
  target_compile_definitions(pylot_lio_core PUBLIC PYLOT_LIO_HAS_LIVOX_DRIVER=1)
else()
  message(STATUS "livox_ros_driver2 not found; only PointCloud2 input format will work.")
endif()

# ============================================================
# lio_node executable
# ============================================================
add_executable(lio_node src/lio_node.cpp)
if(livox_ros_driver2_FOUND)
  ament_target_dependencies(lio_node ${COMMON_ROS_DEPENDENCIES} livox_ros_driver2)
else()
  ament_target_dependencies(lio_node ${COMMON_ROS_DEPENDENCIES})
endif()
target_link_libraries(lio_node pylot_lio_core)

# ============================================================
# Install
# ============================================================
install(TARGETS pylot_lio_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS lio_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/run_benchmark.py
  scripts/compute_relative_metrics.py
  scripts/make_dashboard.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(pylot_lio_core)
ament_export_dependencies(${COMMON_ROS_DEPENDENCIES} PCL Eigen3)

# ============================================================
# Tests
# ============================================================
if(BUILD_TESTING)
  # ament_lint_auto はスタイル差分を大量に吐き本パッケージの機能検証ノイズを増やすため意図的に外す。
  # gtest による純ロジック単体テストのみを採用する。
  find_package(ament_cmake_gtest REQUIRED)

  set(PYLOT_LIO_GTESTS
    test_lie_algebra
    test_voxel_grid_preprocessor
    test_random_sampling_preprocessor
    test_voxel_random_sampling_preprocessor
    test_voxel_map
    test_voxel_random_map
    test_kd_tree_map
    test_ieskf_estimator
    test_gicp_only_estimator
    test_factory
    test_plain_gicp_degenerate
    test_distance_keyframe_selector
    test_voxel_keyframe_submap
    test_scan_context
    test_pose_graph_optimizer
    test_livox_conversion
    test_metal_compute_probe
    test_metal_vgicp_linearizer
  )

  foreach(gtest_name IN LISTS PYLOT_LIO_GTESTS)
    ament_add_gtest(${gtest_name} test/${gtest_name}.cpp)
    if(TARGET ${gtest_name})
      target_link_libraries(${gtest_name} pylot_lio_core)
    endif()
  endforeach()

  if(small_gicp_FOUND)
    ament_add_gtest(test_small_gicp_registration test/test_small_gicp_registration.cpp)
    if(TARGET test_small_gicp_registration)
      target_link_libraries(test_small_gicp_registration pylot_lio_core)
    endif()
  endif()
endif()

ament_package()
