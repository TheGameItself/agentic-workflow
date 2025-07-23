cmake_minimum_required(VERSION 3.20)
project(MCP_Core_System)

# Set Python as the primary language
set(CMAKE_CXX_STANDARD 17)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add Python path
set(PYTHON_PATH "${CMAKE_SOURCE_DIR}/core/src")

# Define source groups for better organization in CLion
file(GLOB_RECURSE CORE_SOURCES "core/src/mcp/*.py")
file(GLOB_RECURSE NEURAL_SOURCES "core/src/mcp/neural_network_models/*.py")
file(GLOB_RECURSE LOBE_SOURCES "core/src/mcp/lobes/*.py")
file(GLOB_RECURSE TEST_SOURCES "core/tests/*.py" "test_*.py")
file(GLOB_RECURSE SCRIPT_SOURCES "core/*.py")
file(GLOB_RECURSE CONFIG_FILES "*.json" "*.yaml" "*.yml" "*.toml" "*.cfg")
file(GLOB_RECURSE DOC_FILES "*.md" "*.rst" "*.txt")

# Create source groups for CLion
source_group("Core\\MCP" FILES ${CORE_SOURCES})
source_group("Core\\Neural Networks" FILES ${NEURAL_SOURCES})
source_group("Core\\Lobes" FILES ${LOBE_SOURCES})
source_group("Tests" FILES ${TEST_SOURCES})
source_group("Scripts" FILES ${SCRIPT_SOURCES})
source_group("Configuration" FILES ${CONFIG_FILES})
source_group("Documentation" FILES ${DOC_FILES})

# Define common environment properties
set(COMMON_ENV "PYTHONPATH=${PYTHON_PATH}")

# Function to create Python targets with common environment
function(add_python_target target_name script_path comment source_files)
    add_custom_target(${target_name}
        COMMAND ${Python3_EXECUTABLE} ${script_path}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "${comment}"
        SOURCES ${source_files}
)
    set_property(TARGET ${target_name} PROPERTY ENVIRONMENT "${COMMON_ENV}")
endfunction()

# Add custom targets for Python scripts
add_python_target(mcp_core_system "core/cli.py interactive" "Running MCP Core System CLI" "${CORE_SOURCES}")
add_python_target(mcp_health_check "core/system_health_check.py" "Running MCP Health Check" "${SCRIPT_SOURCES}")
add_python_target(neural_network_pretraining "core/pretrain_neural_networks.py --models hormone diffusion --synthetic --epochs 1" "Running Neural Network Pretraining" "${SCRIPT_SOURCES}")
add_python_target(test_neural_models "test_neural_models_simple.py" "Testing Neural Models" "${TEST_SOURCES}")
add_python_target(test_hormone_integration "test_hormone_integration.py" "Testing Hormone Neural Integration" "${TEST_SOURCES}")
add_python_target(mcp_stdio_server "core/stdio_server.py" "Running MCP stdio JSON-RPC server" "${SCRIPT_SOURCES}")
add_python_target(mcp_setup "core/setup_core.py" "Setting up MCP Core System" "${SCRIPT_SOURCES}")
add_python_target(neural_perpetual_training "core/pretrain_neural_networks.py --models brain_state genetic --synthetic --epochs 2" "Running Perpetual Neural Network Training" "${SCRIPT_SOURCES}")

# Add cognitive architecture test target (special case with inline Python)
add_custom_target(cognitive_architecture_test
    COMMAND ${Python3_EXECUTABLE} -c "
import sys
sys.path.insert(0, 'core/src')
from mcp.cognitive_architecture import get_cognitive_architecture
arch = get_cognitive_architecture()
arch.create_thought('Test cognitive architecture')
print('Cognitive architecture test completed')
"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Testing Cognitive Architecture"
    SOURCES ${CORE_SOURCES}
)
set_property(TARGET cognitive_architecture_test PROPERTY ENVIRONMENT "${COMMON_ENV}")

# Add all files to a single target for CLion indexing
add_custom_target(all_sources
        ${CORE_SOURCES}
        ${NEURAL_SOURCES}
        ${LOBE_SOURCES}
        ${TEST_SOURCES}
        ${SCRIPT_SOURCES}
        ${CONFIG_FILES}
        ${DOC_FILES}
)

# Add install targets
install(DIRECTORY core/src/mcp DESTINATION lib/python/mcp)
install(FILES requirements.txt DESTINATION .)
install(DIRECTORY data DESTINATION . OPTIONAL)