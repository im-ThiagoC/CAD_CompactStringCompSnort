# =============================================================================
# CMakeLists.txt - Compactação do Algoritmo de Comparação de Strings do Snort
# 
# Autor: Thiago Carvalho
# Data: 27/11/2025
# Curso: TN741 - Computação de Alto Desempenho - UFRRJ
# =============================================================================

cmake_minimum_required(VERSION 3.18)

project(AhoCorasickCUDA VERSION 1.0 LANGUAGES C CUDA)

# Configurações C e CUDA
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Arquitetura da GPU (RTX 4060 Ti = compute capability 8.9)
# Ajuste conforme sua GPU
set(CMAKE_CUDA_ARCHITECTURES 89)

# Encontrar CUDA
find_package(CUDAToolkit REQUIRED)

message(STATUS "╔════════════════════════════════════════════╗")
message(STATUS "║  Configuração CUDA - Projeto AC Snort     ║")
message(STATUS "╚════════════════════════════════════════════╝")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Diretórios de include
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Arquivos fonte
set(SOURCES
    src/main.cu
    src/aho_corasick_serial.c
    src/aho_corasick_gpu.cu
    src/utils.cu
)

# Executável principal
add_executable(aho_corasick ${SOURCES})

# Propriedades CUDA
set_target_properties(aho_corasick PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Linkar bibliotecas CUDA
target_link_libraries(aho_corasick 
    CUDA::cudart
    CUDA::cuda_driver
)

# Flags de compilação para otimização
target_compile_options(aho_corasick PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        --expt-relaxed-constexpr
        -lineinfo
    >
)

# Modo Debug
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(aho_corasick PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -g
            -G
            -lineinfo
        >
    )
    message(STATUS "Debug mode enabled")
endif()

# Criar diretório de resultados
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)

# Informações de compilação
message(STATUS "")
message(STATUS "Arquivos fonte:")
foreach(source ${SOURCES})
    message(STATUS "  - ${source}")
endforeach()
message(STATUS "")
message(STATUS "Para compilar:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j")
message(STATUS "")
message(STATUS "Para executar:")
message(STATUS "  ./aho_corasick")
message(STATUS "════════════════════════════════════════════")