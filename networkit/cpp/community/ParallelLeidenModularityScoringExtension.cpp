/*
 * ParallelLeidenModularityScoringExtension.cpp
 *
 *  Default modularity scorer exported through the ParallelLeidenView extension ABI.
 */

#include <networkit/community/ParallelLeidenScoringExtension.hpp>

extern "C" double networkitParallelLeidenCommunityScore(double cutWeight, double degree,
                                                        double communityVolume,
                                                        NetworKit::count subsetSize,
                                                        NetworKit::count communitySize,
                                                        double gamma, double inverseGraphVolume) {
    (void)subsetSize;
    (void)communitySize;
    return cutWeight - gamma * degree * communityVolume * inverseGraphVolume;
}

extern "C" double networkitParallelLeidenCurrentCommunityThreshold(
    double cutWeight, double degree, double communityVolume, NetworKit::count subsetSize,
    NetworKit::count communitySize, double gamma, double inverseGraphVolume) {
    (void)subsetSize;
    (void)communitySize;
    return cutWeight - gamma * (communityVolume - degree) * degree * inverseGraphVolume;
}

extern "C" bool networkitParallelLeidenRefineRSetCondition(
    double cutWeight, double subsetVolume, NetworKit::count subsetSize, double targetVolume,
    NetworKit::count targetSize, double sourceVolume, NetworKit::count sourceSize, double gamma,
    double inverseGraphVolume) {
    (void)subsetSize;
    (void)targetSize;
    (void)sourceVolume;
    (void)sourceSize;
    return cutWeight >= gamma * subsetVolume * targetVolume * inverseGraphVolume;
}

extern "C" bool networkitParallelLeidenRefineTSetCondition(
    double cutWeight, double subsetVolume, NetworKit::count subsetSize, double targetVolume,
    NetworKit::count targetSize, double sourceVolume, NetworKit::count sourceSize, double gamma,
    double inverseGraphVolume) {
    (void)subsetSize;
    (void)targetSize;
    (void)sourceVolume;
    (void)sourceSize;
    return cutWeight >= gamma * subsetVolume * targetVolume * inverseGraphVolume;
}
