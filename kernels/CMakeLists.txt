enable_language(CUDA)
add_library(kernels
count_alignment.cu
count_alignment.cuh
allocate_stringset.cu
allocate_stringset.cuh
)
target_compile_features(kernels PUBLIC cxx_std_11)
set_target_properties(kernels
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(kernels)
