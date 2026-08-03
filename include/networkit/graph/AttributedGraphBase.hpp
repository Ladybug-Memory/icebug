/*
 * AttributedGraphBase.hpp
 *
 *  The per-node and per-edge attribute maps, shared by every graph class that carries them.
 */

#ifndef NETWORKIT_GRAPH_ATTRIBUTED_GRAPH_BASE_HPP_
#define NETWORKIT_GRAPH_ATTRIBUTED_GRAPH_BASE_HPP_

#include <string>

#include <networkit/graph/Attributes.hpp>
#include <networkit/graph/ReferenceGraph.hpp>

namespace NetworKit {

/**
 * Attribute storage for a graph class, mixed in by inheritance.
 *
 * @tparam Derived is unconstrained because the base is instantiated while Derived is still
 * incomplete.
 */
template <typename Derived>
class AttributedGraphBase : public SelfHandled<Derived> {
    AttributeMap<PerNode, ReferenceGraph> nodeAttributeMap{this->asGraph()};
    AttributeMap<PerEdge, ReferenceGraph> edgeAttributeMap{this->asGraph()};

public:
    using NodeIntAttribute = Attribute<PerNode, ReferenceGraph, int, false>;
    using NodeDoubleAttribute = Attribute<PerNode, ReferenceGraph, double, false>;
    using NodeStringAttribute = Attribute<PerNode, ReferenceGraph, std::string, false>;
    using EdgeIntAttribute = Attribute<PerEdge, ReferenceGraph, int, false>;
    using EdgeDoubleAttribute = Attribute<PerEdge, ReferenceGraph, double, false>;
    using EdgeStringAttribute = Attribute<PerEdge, ReferenceGraph, std::string, false>;

    AttributeMap<PerNode, ReferenceGraph> &nodeAttributes() noexcept { return nodeAttributeMap; }
    const AttributeMap<PerNode, ReferenceGraph> &nodeAttributes() const noexcept {
        return nodeAttributeMap;
    }
    AttributeMap<PerEdge, ReferenceGraph> &edgeAttributes() noexcept { return edgeAttributeMap; }
    const AttributeMap<PerEdge, ReferenceGraph> &edgeAttributes() const noexcept {
        return edgeAttributeMap;
    }

    auto attachNodeIntAttribute(const std::string &name) {
        return nodeAttributes().template attach<int>(name);
    }
    auto attachEdgeIntAttribute(const std::string &name) {
        return edgeAttributes().template attach<int>(name);
    }
    auto attachNodeDoubleAttribute(const std::string &name) {
        return nodeAttributes().template attach<double>(name);
    }
    auto attachEdgeDoubleAttribute(const std::string &name) {
        return edgeAttributes().template attach<double>(name);
    }
    auto attachNodeStringAttribute(const std::string &name) {
        return nodeAttributes().template attach<std::string>(name);
    }
    auto attachEdgeStringAttribute(const std::string &name) {
        return edgeAttributes().template attach<std::string>(name);
    }

    auto getNodeIntAttribute(const std::string &name) {
        return nodeAttributes().template get<int>(name);
    }
    auto getEdgeIntAttribute(const std::string &name) {
        return edgeAttributes().template get<int>(name);
    }
    auto getNodeDoubleAttribute(const std::string &name) {
        return nodeAttributes().template get<double>(name);
    }
    auto getEdgeDoubleAttribute(const std::string &name) {
        return edgeAttributes().template get<double>(name);
    }
    auto getNodeStringAttribute(const std::string &name) {
        return nodeAttributes().template get<std::string>(name);
    }
    auto getEdgeStringAttribute(const std::string &name) {
        return edgeAttributes().template get<std::string>(name);
    }

    void detachNodeAttribute(const std::string &name) {
        nodeAttributes().detach(name);
    }
    void detachEdgeAttribute(const std::string &name) {
        edgeAttributes().detach(name);
    }

    /// Edge attributes are only copied when @a other has edge ids to key them by.
    template <typename Other>
    void copyAttributesFrom(const Other &other) {
        nodeAttributeMap =
            AttributeMap<PerNode, ReferenceGraph>(other.nodeAttributes(), this->asGraph());
        if (other.hasEdgeIds() && static_cast<const Derived &>(*this).hasEdgeIds()) {
            edgeAttributeMap =
                AttributeMap<PerEdge, ReferenceGraph>(other.edgeAttributes(), this->asGraph());
        }
    }

protected:
    AttributedGraphBase() = default;

    AttributedGraphBase(const AttributedGraphBase &other)
        : nodeAttributeMap(other.nodeAttributeMap, this->asGraph()),
          edgeAttributeMap(other.edgeAttributeMap, this->asGraph()) {}

    AttributedGraphBase(AttributedGraphBase &&other) noexcept
        : nodeAttributeMap(std::move(other.nodeAttributeMap), this->asGraph()),
          edgeAttributeMap(std::move(other.edgeAttributeMap), this->asGraph()) {}

    AttributedGraphBase &operator=(const AttributedGraphBase &other) {
        if (this != &other) {
            nodeAttributeMap =
                AttributeMap<PerNode, ReferenceGraph>(other.nodeAttributeMap, this->asGraph());
            edgeAttributeMap =
                AttributeMap<PerEdge, ReferenceGraph>(other.edgeAttributeMap, this->asGraph());
        }
        return *this;
    }

    AttributedGraphBase &operator=(AttributedGraphBase &&other) noexcept {
        nodeAttributeMap =
            AttributeMap<PerNode, ReferenceGraph>(std::move(other.nodeAttributeMap),
                                                  this->asGraph());
        edgeAttributeMap =
            AttributeMap<PerEdge, ReferenceGraph>(std::move(other.edgeAttributeMap),
                                                  this->asGraph());
        return *this;
    }
};

} // namespace NetworKit

#endif // NETWORKIT_GRAPH_ATTRIBUTED_GRAPH_BASE_HPP_
