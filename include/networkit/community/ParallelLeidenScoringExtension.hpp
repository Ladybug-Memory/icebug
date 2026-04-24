/*
 * ParallelLeidenScoringExtension.hpp
 *
 *  Shared-library ABI for custom ParallelLeidenView move scoring metrics.
 */

#ifndef NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_SCORING_EXTENSION_HPP_
#define NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_SCORING_EXTENSION_HPP_

namespace NetworKit {

using ParallelLeidenCommunityScoreFunction =
    double (*)(double cutWeight, double degree, double communityVolume, double gamma,
               double inverseGraphVolume);

} // namespace NetworKit

extern "C" {

/**
 * Required: score a candidate community during the move phase.
 */
double networkitParallelLeidenCommunityScore(double cutWeight, double degree, double communityVolume,
                                             double gamma, double inverseGraphVolume);

/**
 * Optional: override the current-community stay threshold used to accept or reject the best move.
 * When omitted, ParallelLeidenView falls back to the built-in modularity threshold.
 */
double networkitParallelLeidenCurrentCommunityThreshold(double cutWeight, double degree,
                                                        double communityVolume, double gamma,
                                                        double inverseGraphVolume);

} // extern "C"

#endif // NETWORKIT_COMMUNITY_PARALLEL_LEIDEN_SCORING_EXTENSION_HPP_
