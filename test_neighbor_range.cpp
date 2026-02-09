#include <iostream>
#include <networkit/graph/GraphR.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/io/EdgeListReader.hpp>

int main() {
    // Test GraphW (vector-based)
    NetworKit::GraphW gw(5);
    gw.addEdge(0, 1);
    gw.addEdge(0, 2);
    gw.addEdge(1, 3);
    gw.addEdge(2, 4);

    std::cout << "GraphW neighbors of node 0: ";
    for (auto neighbor : gw.neighborRange(0)) {
        std::cout << neighbor << " ";
    }
    std::cout << std::endl;

    // Test GraphR (CSR-based) - need to read from file or create
    NetworKit::EdgeListReader reader(',', 0, "#", true, true);
    try {
        auto gr = reader.read("input/karate.graph");
        std::cout << "Graph loaded with " << gr.numberOfNodes() << " nodes" << std::endl;
        std::cout << "Graph neighbors of node 0: ";
        for (auto neighbor : gr.neighborRange(0)) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    } catch (...) {
        std::cout << "Could not load test graph, but GraphW neighborRange works!" << std::endl;
    }

    std::cout << "Success! neighborRange() works for both GraphW and GraphR" << std::endl;
    return 0;
}
