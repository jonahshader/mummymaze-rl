#ifndef GAME_H
#define GAME_H

#include "parse.h"

/*
 * Game state — matches the binary's 0x3c (60-byte) solver state exactly.
 *
 * Binary layout (uint32_t array indices):
 *   [0]  player_row
 *   [1]  player_col
 *   [2]  (padding / unused)
 *   [3]  mummy1_row
 *   [4]  mummy1_col
 *   [5]  (padding)
 *   [6]  mummy1_alive (char at byte 0x18)
 *   [7]  mummy2_row
 *   [8]  mummy2_col
 *   [9]  (padding)
 *   [10] mummy2_alive (char at byte 0x28)
 *   [11] scorpion_row (0x2C)
 *   [12] scorpion_col (0x30)
 *   [13] (padding)
 *   [14] scorpion_alive (char at byte 0x38)
 *   byte 0x39: gate_open
 *
 * The state comparison function FUN_00404f60 compares:
 *   [0],[1],[3],[4],(char)[6],[7],[8],(char)[10],[11],[12],(char)[14], byte 0x39
 */
typedef struct {
    int player_row, player_col;
    int mummy1_row, mummy1_col;
    int mummy1_alive;
    int mummy2_row, mummy2_col;
    int mummy2_alive;
    int scorpion_row, scorpion_col;
    int scorpion_alive;
    int gate_open;  /* 1 = blocking (closed), 0 = open */
} State;

/* Step result codes */
#define STEP_OK    0
#define STEP_DEAD  1
#define STEP_WIN   2

/* Initialize a state from a parsed level */
void state_init(State *s, const Level *lev);

/*
 * Execute one full game turn. Direct port of FUN_00405580.
 * dr,dc is the player's movement delta (-1,0,1).
 * Returns STEP_OK, STEP_DEAD, or STEP_WIN.
 */
int step(const Level *lev, State *s, int dr, int dc);

/*
 * Hash a state for the visited set.
 * Matches the fields compared in FUN_00404f60.
 */
uint64_t state_hash(const State *s);

/* Check if two states are equal (same fields as FUN_00404f60). */
int state_eq(const State *a, const State *b);

#endif /* GAME_H */
