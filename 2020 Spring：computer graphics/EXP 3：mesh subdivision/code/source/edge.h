#pragma once

#include <cassert>
#include <stdlib.h>

class Vertex;
class Triangle;

class Edge {

public:
    Edge(Vertex* vs, Vertex* ve, Triangle* t);
    ~Edge();

    Vertex* getStartVertex() const { assert(start_vertex != NULL); return start_vertex; }
    Vertex* getEndVertex() const { assert(end_vertex != NULL); return end_vertex; }
    Edge* getNext() const { assert(next != NULL); return next; }
    Triangle* getTriangle() const { assert(triangle != NULL); return triangle; }
    Edge* getOpposite() const {
        // warning!  the opposite edge might be NULL!
        return opposite;
    }
    float getCrease() const { return crease; }
    float Length() const;

    void setOpposite(Edge* e) {
        assert(opposite == NULL);
        assert(e != NULL);
        assert(e->opposite == NULL);
        opposite = e;
        e->opposite = this;
    }
    void clearOpposite() {
        if (opposite == NULL) return;
        assert(opposite->opposite == this);
        opposite->opposite = NULL;
        opposite = NULL;
    }
    void setNext(Edge* e) {
        assert(next == NULL);
        assert(e != NULL);
        assert(triangle == e->triangle);
        next = e;
    }
    void setCrease(float c) { crease = c; }

private:

    Edge(const Edge&) { assert(0); }
    Edge& operator=(const Edge&) { assert(0); exit(0); }

    Vertex* start_vertex;
    Vertex* end_vertex;
    Triangle* triangle;
    Edge* opposite;
    Edge* next;

    float crease;
};
