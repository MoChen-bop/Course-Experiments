#pragma once
#include <cassert>
#include <stdlib.h>
#include "edge.h"


class Triangle {

public:

    Triangle() {
        edge = NULL;
        id = next_triangle_id;
        next_triangle_id++;
    }

    Vertex* operator[](int i) const {
        assert(edge != NULL);
        if (i == 0) return edge->getStartVertex();
        if (i == 1) return edge->getNext()->getStartVertex();
        if (i == 2) return edge->getNext()->getNext()->getStartVertex();
        assert(0); exit(0);
    }
    Edge* getEdge() {
        assert(edge != NULL);
        return edge;
    }
    void setEdge(Edge* e) {
        assert(edge == NULL);
        edge = e;
    }
    int getID() { return id; }

protected:

    Triangle(const Triangle&/*t*/) { assert(0); exit(0); }
    Triangle& operator= (const Triangle&/*t*/) { assert(0); exit(0); }

    Edge* edge;
    int id;

    static int next_triangle_id;
};
