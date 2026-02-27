#ifndef SOLVER_H
#define SOLVER_H

#include "game.h"

typedef struct {
    int moves;            /* solution length, or -1 if unsolvable */
    int states_explored;  /* total states visited */
} SolveResult;

/* BFS solver. Returns optimal (shortest) solution length. */
SolveResult solve(const Level *lev);

#endif /* SOLVER_H */
