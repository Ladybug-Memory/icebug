networkit_add_module(auxiliary
    BloomFilter.cpp
    BucketPQ.cpp
    HierarchicalBucketPQ.cpp
    Log.cpp
    NumericTools.cpp
    Parallelism.cpp
    Random.cpp
    SignalHandling.cpp
    Timer.cpp
    RangeMinimumQuery.cpp
    )

if(NOT WIN32)
    networkit_module_link_modules(auxiliary
            globalstate
        )
endif()

add_subdirectory(test)
