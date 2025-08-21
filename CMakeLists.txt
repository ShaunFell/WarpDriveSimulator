cmake_minimum_required(VERSION 3.10)
project(WarpDriveSimulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenGL preference to suppress warning
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

# Find packages
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

# For now, let's use GLEW instead of GLAD (much simpler)
find_package(GLEW REQUIRED)

# Include directories
include_directories(src)

# Source files
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.h")

# Create executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL 
    ${GLFW3_LIBRARIES}
    GLEW::GLEW
)

# Include flags
target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW3_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${GLFW3_CFLAGS_OTHER})

# Copy shaders to build directory
file(COPY shaders DESTINATION ${CMAKE_BINARY_DIR})

# Compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()
