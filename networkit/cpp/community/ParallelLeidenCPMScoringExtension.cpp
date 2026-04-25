/*
 * ParallelLeidenCPMScoringExtension.cpp
 *
 *  Constant Potts Model scorer exported through the ParallelLeidenView extension ABI.
 */

#include <networkit/community/ParallelLeidenScoringExtension.hpp>

extern "C" double networkitParallelLeidenCommunityScore(double cutWeight, double degree,
                                                        double communityVolume,
                                                        NetworKit::count subsetSize,
                                                        NetworKit::count communitySize,
                                                        double gamma,
                                                        double inverseGraphVolume) {
    (void)degree;
    (void)communityVolume;
    (void)inverseGraphVolume;
    return cutWeight
           - gamma * static_cast<double>(subsetSize) * static_cast<double>(communitySize);
}

extern "C" double networkitParallelLeidenCurrentCommunityThreshold(double cutWeight, double degree,
                                                                   double communityVolume,
                                                                   NetworKit::count subsetSize,
                                                                   NetworKit::count communitySize,
                                                                   double gamma,
                                                                   double inverseGraphVolume) {
    (void)degree;
    (void)communityVolume;
    (void)inverseGraphVolume;
    return cutWeight - gamma * static_cast<double>(subsetSize)
                           * static_cast<double>(communitySize - subsetSize);
}

extern "C" bool networkitParallelLeidenRefineRSetCondition(double cutWeight, double subsetVolume,
                                                           NetworKit::count subsetSize,
                                                           double targetVolume,
                                                           NetworKit::count targetSize,
                                                           double sourceVolume,
                                                           NetworKit::count sourceSize,
                                                           double gamma,
                                                           double inverseGraphVolume) {
    (void)subsetVolume;
    (void)targetVolume;
    (void)sourceVolume;
    (void)sourceSize;
    (void)inverseGraphVolume;
    return cutWeight >= gamma * static_cast<double>(subsetSize) * static_cast<double>(targetSize);
}

extern "C" bool networkitParallelLeidenRefineTSetCondition(double cutWeight, double subsetVolume,
                                                           NetworKit::count subsetSize,
                                                           double targetVolume,
                                                           NetworKit::count targetSize,
                                                           double sourceVolume,
                                                           NetworKit::count sourceSize,
                                                           double gamma,
                                                           double inverseGraphVolume) {
    (void)subsetVolume;
    (void)targetVolume;
    (void)sourceVolume;
    (void)sourceSize;
    (void)inverseGraphVolume;
    return cutWeight >= gamma * static_cast<double>(subsetSize) * static_cast<double>(targetSize);
}
