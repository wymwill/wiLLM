# Kernels are split by backend. ref/ is the naive scalar reference and is never
# deleted: every optimized kernel (neon/, metal/) is tested against it.
add_subdirectory(ref)

# willm::kernels is what model/ links. Today it is just the reference backend.
# NEON (M4) and Metal (M5) will be added here behind the same entry point.
add_library(willm_kernels INTERFACE)
add_library(willm::kernels ALIAS willm_kernels)
target_link_libraries(willm_kernels INTERFACE willm::kernels_ref)
