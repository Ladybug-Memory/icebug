/*
 * ParallelLeidenModularityScoringExtension.cpp
 *
 *  Default modularity scorer exported through the ParallelLeidenView extension ABI.
 */

#include <networkit/community/ParallelLeidenScoringExtension.hpp>

extern "C" double networkitParallelLeidenCommunityScore(double cutWeight, double degree,
                                                        double communityVolume, double gamma,
                                                        double inverseGraphVolume) {
    return cutWeight - gamma * degree * communityVolume * inverseGraphVolume;
}

extern "C" double networkitParallelLeidenCurrentCommunityThreshold(double cutWeight, double degree,
                                                                   double communityVolume,
                                                                   double gamma,
                                                                   double inverseGraphVolume) {
    return cutWeight - gamma * (communityVolume - degree) * degree * inverseGraphVolume;
}
