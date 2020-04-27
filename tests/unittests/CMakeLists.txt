add_library(TestMain
            TestMain.cpp)
target_link_libraries(TestMain
                      PRIVATE
                        LLVMSupport
                        gtest)
add_library(BackendTestUtils
            BackendTestUtils.cpp)
target_link_libraries(BackendTestUtils
                      PUBLIC
                        Backend
                        Backends
                        Base
                        Converter
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        Quantization
                        QuantizationBase
                        LLVMSupport
                        gtest)


# Loop through all backends present in lib/Backends (see FindBackends.cmake)
# and instantiate a backend-parameterized test for each of them.
foreach(backend ${GLOW_BACKENDS})
  add_backend_test(TEST
                   RecommendationSystemTest
                   BACKEND
                   "${backend}"
                   EXPENSIVE
                   PRIVATE
                   Partitioner
                   HostManager)
endforeach()
