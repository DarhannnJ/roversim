cmake_minimum_required(VERSION 3.20)
project(roversim VERSION 0.1.0
        DESCRIPTION "Lightweight deterministic simulator of a differential-drive rover"
        LANGUAGES CXX)

option(ROVERSIM_BUILD_PYTHON "Собирать Python-модуль (pybind11)" ON)
option(ROVERSIM_BUILD_APPS "Собирать roversim_server, roversim_bench и roversim_nav_bench" ON)
option(ROVERSIM_BUILD_TESTS "Собирать тесты" ON)
option(ROVERSIM_NATIVE "Оптимизировать под процессор сборочной машины (-march=native)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Тип сборки" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # статическое ядро линкуется в разделяемый модуль Python
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------ ядро

# Страница веб-просмотрщика встраивается в бинарник как строковый литерал:
# серверу не нужны файлы рядом с исполняемым модулем.
set(ROVERSIM_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/web/viewer.html ROVERSIM_VIEWER_HTML)
configure_file(cmake/viewer_html.hpp.in ${ROVERSIM_GENERATED_DIR}/viewer_html.hpp @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/web/viewer.html)

add_library(roversim_core STATIC
  core/src/world.cpp
  core/src/navgrid.cpp
  core/src/robot.cpp
  core/src/sensors.cpp
  core/src/simulator.cpp
  core/src/vec_simulator.cpp
  core/src/shm.cpp
  core/src/viz_raster.cpp
  core/src/viz_web.cpp
  core/src/nav/occupancy_grid.cpp
  core/src/nav/slam.cpp
  core/src/nav/planner.cpp
  core/src/nav/tracker.cpp
  core/src/nav/dwa.cpp
  core/src/nav/navigator.cpp)
add_library(roversim::core ALIAS roversim_core)
target_include_directories(roversim_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include
  PRIVATE ${ROVERSIM_GENERATED_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Детерминизм вычислений с плавающей точкой:
  #   -ffp-contract=off  запрещает слияние a*b+c в FMA, иначе результат зависел бы
  #                      от набора инструкций целевого процессора;
  #   -ffast-math не используется нигде и никогда.
  # Флаг PUBLIC: inline-функции из заголовков ядра компилируются и в чужих единицах трансляции.
  target_compile_options(roversim_core PUBLIC -ffp-contract=off)
  # -fno-math-errno позволяет встраивать sqrt одной инструкцией; на результат не влияет.
  target_compile_options(roversim_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow -fno-math-errno)
  if(ROVERSIM_NATIVE)
    target_compile_options(roversim_core PUBLIC -march=native)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(roversim_core PUBLIC Threads::Threads)
if(UNIX AND NOT APPLE)
  target_link_libraries(roversim_core PUBLIC rt)  # shm_open в старых glibc
endif()

# ------------------------------------------------------------- приложения

if(ROVERSIM_BUILD_APPS)
  add_executable(roversim_server apps/server.cpp)
  target_link_libraries(roversim_server PRIVATE roversim_core)

  add_executable(roversim_bench apps/bench.cpp)
  target_link_libraries(roversim_bench PRIVATE roversim_core)

  add_executable(roversim_nav_bench apps/nav_bench.cpp)
  target_link_libraries(roversim_nav_bench PRIVATE roversim_core)
endif()

# ----------------------------------------------------------------- Python

if(ROVERSIM_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

  # pybind11 ищется в трёх местах: системная установка → пакет из pip → загрузка исходников.
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE ROVERSIM_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE ROVERSIM_PYBIND11_RC ERROR_QUIET)
    if(ROVERSIM_PYBIND11_RC EQUAL 0)
      find_package(pybind11 CONFIG QUIET PATHS ${ROVERSIM_PYBIND11_DIR} NO_DEFAULT_PATH)
    endif()
  endif()
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6)
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(_core MODULE python/bindings.cpp)
  target_link_libraries(_core PRIVATE roversim_core)

  if(SKBUILD)
    # Сборка колеса через scikit-build-core (pip install .): модуль кладётся в пакет.
    install(TARGETS _core LIBRARY DESTINATION roversim)
  else()
    # Обычная сборка: в build/python получается готовый к импорту пакет
    # (PYTHONPATH=build/python). Python-файлы копируются при каждой сборке.
    set(ROVERSIM_PYTHON_DIR ${CMAKE_BINARY_DIR}/python/roversim)
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ROVERSIM_PYTHON_DIR})
    add_custom_target(roversim_python ALL
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              ${CMAKE_CURRENT_SOURCE_DIR}/python/roversim ${ROVERSIM_PYTHON_DIR}
      COMMENT "Копирование Python-пакета в ${ROVERSIM_PYTHON_DIR}")
  endif()
endif()

# ------------------------------------------------------------------ тесты

if(ROVERSIM_BUILD_TESTS)
  enable_testing()
  add_executable(roversim_tests tests/test_core.cpp)
  target_link_libraries(roversim_tests PRIVATE roversim_core)
  target_include_directories(roversim_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/apps)
  add_test(NAME core COMMAND roversim_tests)

  if(ROVERSIM_BUILD_PYTHON AND ROVERSIM_BUILD_APPS AND NOT SKBUILD)
    add_test(NAME python
      COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py)
    set_tests_properties(python PROPERTIES ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python;ROVERSIM_SERVER=$<TARGET_FILE:roversim_server>")
  endif()
endif()
