networkit_add_module(coarsening
    ClusteringProjector.cpp
    CoarsenedGraphView.cpp
    GraphCoarsening.cpp
    MatchingCoarsening.cpp
    ParallelPartitionCoarsening.cpp
    ParallelPartitionCoarseningView.cpp
    )

networkit_module_link_modules(coarsening
    auxiliary graph matching structures)

add_subdirectory(test)

