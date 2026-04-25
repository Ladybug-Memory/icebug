/*
 * ParallelLeidenScoringExtension.hpp
 *
 *  Shared-library ABI for custom ParallelLeidenView move scoring metrics.
 */

#ifndef NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_SCORING_EXTENSION_HPP_
#define NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_SCORING_EXTENSION_HPP_

#include <networkit/Globals.hpp>

namespace NetworKit {

using ParallelLeidenCommunityScoreFunction =
    double (*)(double cutWeight, double degree, double communityVolume, count subsetSize,
               count communitySize, double gamma, double inverseGraphVolume);

using ParallelLeidenRefineSetConditionFunction =
    bool (*)(double cutWeight, double subsetVolume, count subsetSize, double targetVolume,
             count targetSize, double sourceVolume, count sourceSize, double gamma,
               double inverseGraphVolume);

} // namespace NetworKit

extern "C" {

/**
 * Required: score a candidate community during the move phase.
 */
double networkitParallelLeidenCommunityScore(double cutWeight, double degree, double communityVolume,
                                             NetworKit::count subsetSize,
                                             NetworKit::count communitySize, double gamma,
                                             double inverseGraphVolume);

/**
 * Optional: override the current-community stay threshold used to accept or reject the best move.
 * When omitted, ParallelLeidenView falls back to the built-in modularity threshold.
 */
double networkitParallelLeidenCurrentCommunityThreshold(double cutWeight, double degree,
                                                        double communityVolume,
                                                        NetworKit::count subsetSize,
                                                        NetworKit::count communitySize,
                                                        double gamma,
                                                        double inverseGraphVolume);

/**
 * Optional: override the refine-phase R-set condition.
 * When omitted, ParallelLeidenView falls back to the built-in modularity condition.
 */
bool networkitParallelLeidenRefineRSetCondition(double cutWeight, double subsetVolume,
                                                NetworKit::count subsetSize, double targetVolume,
                                                NetworKit::count targetSize, double sourceVolume,
                                                NetworKit::count sourceSize, double gamma,
                                                double inverseGraphVolume);

/**
 * Optional: override the refine-phase T-set condition.
 * When omitted, ParallelLeidenView falls back to the built-in modularity condition.
 */
bool networkitParallelLeidenRefineTSetCondition(double cutWeight, double subsetVolume,
                                                NetworKit::count subsetSize, double targetVolume,
                                                NetworKit::count targetSize, double sourceVolume,
                                                NetworKit::count sourceSize, double gamma,
                                                        double inverseGraphVolume);

} // extern "C"

#endif // NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_SCORING_EXTENSION_HPP_
