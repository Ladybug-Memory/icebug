networkit_add_module(community
    AdjustedRandMeasure.cpp
    ClusteringGenerator.cpp
    CommunityDetectionAlgorithm.cpp
    Conductance.cpp
    CoverHubDominance.cpp
    CoverF1Similarity.cpp
    OverlappingNMIDistance.cpp
    Coverage.cpp
    CutClustering.cpp
    DissimilarityMeasure.cpp
    DynamicNMIDistance.cpp
    EdgeCut.cpp
    GraphClusteringTools.cpp
    GraphStructuralRandMeasure.cpp
    HubDominance.cpp
    IntrapartitionDensity.cpp
    IsolatedInterpartitionConductance.cpp
    IsolatedInterpartitionExpansion.cpp
    JaccardMeasure.cpp
    LFM.cpp
    LPDegreeOrdered.cpp
    LocalCoverEvaluation.cpp
    LocalPartitionEvaluation.cpp
    Modularity.cpp
    NMIDistance.cpp
    NodeStructuralRandMeasure.cpp
    OverlappingCommunityDetectionAlgorithm.cpp
    PLM.cpp
    PLP.cpp
    ParallelAgglomerativeClusterer.cpp
    PartitionFragmentation.cpp
    PartitionHubDominance.cpp
    PartitionIntersection.cpp
    SampledGraphStructuralRandMeasure.cpp
    SampledNodeStructuralRandMeasure.cpp
    StablePartitionNodes.cpp
    LouvainMapEquation.cpp
    ParallelLeiden.cpp
    ParallelLeidenView.cpp
    )

networkit_module_link_modules(community
    auxiliary coarsening components flow graph matching scd structures)

add_library(networkit_parallel_leiden_modularity_extension SHARED
    ParallelLeidenModularityScoringExtension.cpp)
target_include_directories(networkit_parallel_leiden_modularity_extension BEFORE PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
set_target_properties(networkit_parallel_leiden_modularity_extension PROPERTIES
    OUTPUT_NAME "networkit_parallel_leiden_modularity_extension"
    CXX_STANDARD ${NETWORKIT_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES)

add_library(networkit_parallel_leiden_cpm_extension SHARED
    ParallelLeidenCPMScoringExtension.cpp)
target_include_directories(networkit_parallel_leiden_cpm_extension BEFORE PUBLIC
    "${PROJECT_SOURCE_DIR}/include")
set_target_properties(networkit_parallel_leiden_cpm_extension PROPERTIES
    OUTPUT_NAME "networkit_parallel_leiden_cpm_extension"
    CXX_STANDARD ${NETWORKIT_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES)

add_subdirectory(test)
