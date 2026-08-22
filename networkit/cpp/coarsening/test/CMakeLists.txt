networkit_add_test(coarsening CoarseningGTest
    auxiliary community generators io matching)
networkit_add_test(coarsening CoarsenedGraphViewGTest graph)

networkit_add_benchmark(coarsening CoarseningBenchmark
        auxiliary community generators)
