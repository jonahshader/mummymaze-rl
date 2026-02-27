#include "game.h"
#include <string.h>

void state_init(State *s, const Level *lev) {
    s->player_row   = lev->player_row;
    s->player_col   = lev->player_col;
    s->mummy1_row   = lev->mummy1_row;
    s->mummy1_col   = lev->mummy1_col;
    s->mummy1_alive = 1;
    s->mummy2_row   = lev->mummy2_row;
    s->mummy2_col   = lev->mummy2_col;
    s->mummy2_alive = lev->has_mummy2;
    s->scorpion_row   = lev->scorpion_row;
    s->scorpion_col   = lev->scorpion_col;
    s->scorpion_alive = lev->has_scorpion;
    s->gate_open = 1; /* starts closed/blocking */
}

/*
 * can_move_enemy — direct port of movement checks from FUN_00405580.
 *
 * In the binary, movement is checked by testing wall bits directly:
 *   walls[col + row * 10] & bit
 *
 * For vertical movement (row changes, col stays):
 *   Moving south (row increases): check WALL_S(4) at (col, row)
 *   Moving north (row decreases): check WALL_N(8) at (col, row)
 *
 * For horizontal movement (col changes, row stays):
 *   Moving east (col increases): check WALL_E(2) at (col, row)
 *   Moving west (col decreases): check WALL_W(1) at (col, row)
 *
 * Gate check applies only to horizontal movement (W/E directions).
 * The gate blocks based on the gate cell position and gate_open state,
 * with inverted polarity for flip=0 vs flip=1.
 */

/*
 * Move enemy one step toward the player, matching the binary's logic.
 *
 * For flip=0 (white mummies): try vertical first, then horizontal.
 * For flip=1 (red mummies): try horizontal first, then vertical.
 *
 * Returns: updated (row, col) through er, ec pointers.
 *
 * The gate interaction for the enemy movement checks:
 *
 * flip=0: Wall bit blocks UNLESS gate condition overrides.
 *   Moving east (E, bit 2): blocked if wall_E set, AND NOT
 *     (gate_open != 0 AND entity is at gate cell)
 *   Wait, let me re-read the binary more carefully.
 *
 * From FUN_00405580, flip=0 path for mummy horizontal movement east:
 *   if (col < player_col &&
 *       (walls[col + row*10] & 2) == 0 &&
 *       (gate_open == 0 || (row != gate_row || col != gate_col)))
 *     → col++
 *
 * And west:
 *   if (player_col < col &&
 *       (walls[col + row*10] & 1) == 0 &&
 *       (gate_open == 0 || (row != gate_row || (col-1) != gate_col)))
 *     → col--
 *
 * For flip=1 path for mummy horizontal movement east:
 *   if (col < player_col &&
 *       ((walls[col + row*10] & 2) == 0 ||
 *        (gate_open == 0 && row == gate_row && col == gate_col)))
 *     → col++
 *
 * And west:
 *   if (player_col < col &&
 *       ((walls[col + row*10] & 1) == 0 ||
 *        (gate_open == 0 && row == gate_row && col-1 == gate_col)))
 *     → col--
 *
 * So the gate logic is:
 *   flip=0: gate_open(1) means blocking → extra condition blocks move
 *           when entity is at gate cell and gate_open != 0
 *   flip=1: gate_open(1) means blocking → BUT the wall IS present in walls[],
 *           and gate_open(0) OPENS it, allowing through even when wall bit is set
 *
 * Wait, this is subtly different. Let me re-read flip=0 more carefully:
 *
 * flip=0, moving east (col < player_col):
 *   (walls[col+row*10] & WALL_E) == 0    ← no wall
 *   AND (gate_open == 0 || row != gate_row || col != gate_col)
 *   = no wall AND (gate is open OR not at gate cell)
 *   = no wall AND NOT (gate is closed AND at gate cell)
 *
 * flip=0, moving west (player_col < col):
 *   (walls[col+row*10] & WALL_W) == 0    ← no wall
 *   AND (gate_open == 0 || row != gate_row || (col-1) != gate_col)
 *   = no wall AND (gate is open OR not approaching gate cell from east)
 *
 * flip=1, moving east (col < player_col):
 *   (walls[col+row*10] & WALL_E) == 0    ← no wall in walls array
 *   OR (gate_open == 0 && row == gate_row && col == gate_col)
 *   = no wall OR (gate is open AND at gate cell)
 *   So if gate is OPEN, the wall (which IS in the walls array for flip=1)
 *   is overridden.
 *
 * This matches the memory note: .dat files store walls WITHOUT the gate wall
 * present. For flip=0, gate_open adds a virtual wall. For flip=1, the gate wall
 * IS in the walls array and gate_open=0 removes it.
 *
 * Wait, but the MEMORY.md says "Engine must ADD the wall at load time
 * (gate starts closed)". And gate_open starts at 1 (closed/blocking).
 *
 * For flip=0: walls[] does NOT contain the gate wall (dat file has no wall there).
 *   gate_open=1 (blocking) adds virtual blocking via the condition.
 *   gate_open=0 (open) → condition is: (0 == 0 || ...) → always true → movement allowed.
 *
 * For flip=1: Looking at the decompiled wall loading... the wall IS loaded into
 *   walls[] as part of the normal wall data? Or is it separate?
 *   Actually re-reading MEMORY.md: "All gate levels have flip=False"
 *   So flip=1 gate logic may never actually be used!
 *   Let me still implement it correctly from the binary.
 */

static void move_enemy(const Level *lev, State *s,
                       int *er, int *ec, int is_scorpion) {
    int pr = s->player_row, pc = s->player_col;
    int r = *er, c = *ec;
    int N = lev->grid_size;
    const uint32_t *walls = lev->walls;
    int flip = lev->flip;

    /* Bounds check */
    if (r < 0 || r >= N || c < 0 || c >= N) {
        if (is_scorpion) {
            /* scorpion dies if out of bounds — set alive=0 elsewhere */
        }
        return;
    }

    uint32_t w = walls[c + r * 10];

    if (!flip) {
        /* flip=0 (white mummies): try vertical first (row), then horizontal (col) */

        /* Try vertical (south/north) */
        if (r < pr && (w & WALL_S) == 0) {
            *er = r + 1;
            return;
        }
        if (pr < r && (w & WALL_N) == 0) {
            *er = r - 1;
            return;
        }

        /* Try horizontal (east/west) with gate check */
        if (c < pc && (w & WALL_E) == 0) {
            /* Gate check for flip=0 east: blocked if gate closed AND at gate cell */
            if (s->gate_open == 0 ||
                r != lev->gate_row || c != lev->gate_col) {
                *ec = c + 1;
                return;
            }
        }
        if (pc < c && (w & WALL_W) == 0) {
            /* Gate check for flip=0 west: blocked if gate closed AND dest is gate cell */
            if (s->gate_open == 0 ||
                r != lev->gate_row || (c - 1) != lev->gate_col) {
                *ec = c - 1;
                return;
            }
        }
    } else {
        /* flip=1 (red mummies): try horizontal first (col), then vertical (row) */

        /* Try horizontal (east/west) with gate check */
        if (c < pc) {
            if ((w & WALL_E) == 0 ||
                (s->gate_open == 0 && r == lev->gate_row && c == lev->gate_col)) {
                *ec = c + 1;
                return;
            }
        }
        if (pc < c) {
            if ((w & WALL_W) == 0 ||
                (s->gate_open == 0 && r == lev->gate_row && (c - 1) == lev->gate_col)) {
                *ec = c - 1;
                return;
            }
        }

        /* Try vertical (south/north) */
        if (r < pr && (w & WALL_S) == 0) {
            *er = r + 1;
            return;
        }
        if (pr < r && (w & WALL_N) == 0) {
            *er = r - 1;
            return;
        }
    }
}

/*
 * Check and toggle gate when an entity enters the key cell.
 * Only fires on ENTRY (position changed to key cell).
 *
 * Binary behavior: gate is set to !original_gate (absolute assignment from
 * the turn-start state), NOT a relative toggle of current gate.
 * The key_toggled flag ensures at most one entity toggles per turn.
 */
static void check_key_toggle(const Level *lev, State *s,
                              int new_r, int new_c,
                              int old_r, int old_c,
                              int original_gate,
                              int *key_toggled) {
    if (!lev->has_gate) return;
    if (*key_toggled) return;
    if (new_r == lev->key_row && new_c == lev->key_col &&
        (new_r != old_r || new_c != old_c)) {
        s->gate_open = !original_gate;
        *key_toggled = 1;
    }
}

/*
 * FUN_004079c0 — can_move for the player.
 *
 * This is simpler than enemy movement because the player also checks
 * for live enemies at the destination (preventing moves onto them),
 * plus the gate check.
 *
 * Actually, looking at FUN_004079c0 more carefully:
 * - param_5 (char) controls whether to check for enemies at destination
 * - param_6 (char) controls whether to check gate
 *
 * When called from FUN_00405580 for the player move:
 *   FUN_004079c0(this, state[0], state[1], state[0]+dr, state[1]+dc,
 *                '\0', gate_open_byte)
 * So param_5 = '\0' (don't check enemies), param_6 = gate_open_byte.
 *
 * The function checks:
 * 1. Bounds: dest must be in [0, N)
 * 2. If param_5 != 0, check live mummies and scorpion at dest → block
 * 3. If dest == src, return true (no movement = always valid)
 * 4. Row movement (col same):
 *    - North (dest_row = src_row - 1): check !(walls[src_col + src_row*10] & WALL_N)
 *    - South (dest_row = src_row + 1): check !(walls[src_col + src_row*10] & WALL_S)
 * 5. Col movement (row same):
 *    - West (dest_col = src_col - 1): check !(walls[src_col + src_row*10] & WALL_W)
 *      AND (param_6 == 0 || src_row != gate_row || dest_col != gate_col)
 *    - East (dest_col = src_col + 1): check !(walls[src_col + src_row*10] & WALL_E)
 *      AND (param_6 == 0 || src_row != gate_row || src_col != gate_col)
 *
 * So param_6 is passed as the gate_open value. When gate_open=1,
 * the extra check blocks movement through the gate cell.
 * For the player, param_6 = *(state + 0x39) = gate_open byte.
 *
 * For flip=0, the gate blocks: east if at gate cell, west if dest is gate cell.
 * This is the same logic as the enemy movement.
 */
static int can_move_player(const Level *lev, const State *s,
                           int src_r, int src_c, int dst_r, int dst_c) {
    int N = lev->grid_size;

    /* Bounds check */
    if (dst_r < 0 || dst_r >= N || dst_c < 0 || dst_c >= N)
        return 0;

    /* No movement = always valid */
    if (dst_r == src_r && dst_c == src_c)
        return 1;

    uint32_t w = lev->walls[src_c + src_r * 10];

    /* Vertical movement */
    if (dst_c == src_c) {
        if (dst_r == src_r - 1) return (w & WALL_N) == 0;
        if (dst_r == src_r + 1) return (w & WALL_S) == 0;
        return 0; /* more than 1 step */
    }

    /* Horizontal movement */
    if (dst_r == src_r) {
        if (dst_c == src_c - 1) {
            /* West */
            if (w & WALL_W) return 0;
            /* Gate check */
            if (!lev->flip) {
                /* flip=0: gate blocks if gate_open AND dest is gate cell */
                if (s->gate_open && src_r == lev->gate_row &&
                    (src_c - 1) == lev->gate_col)
                    return 0;
            } else {
                /* flip=1: wall is in array, gate_open=0 overrides wall.
                 * But if we got here, wall bit was 0, so no override needed.
                 * Actually for flip=1 player: looking at FUN_004079c0,
                 * param_6 is gate_open. The check is:
                 *   (param_6 == 0 || row != gate_row || col-1 != gate_col)
                 * Same structure as flip=0. */
                if (s->gate_open && src_r == lev->gate_row &&
                    (src_c - 1) == lev->gate_col)
                    return 0;
            }
            return 1;
        }
        if (dst_c == src_c + 1) {
            /* East */
            if (w & WALL_E) return 0;
            /* Gate check */
            if (!lev->flip) {
                if (s->gate_open && src_r == lev->gate_row &&
                    src_c == lev->gate_col)
                    return 0;
            } else {
                if (s->gate_open && src_r == lev->gate_row &&
                    src_c == lev->gate_col)
                    return 0;
            }
            return 1;
        }
    }
    return 0;
}

int step(const Level *lev, State *s, int dr, int dc) {
    int N = lev->grid_size;

    /* Save original turn-start state for "moved" checks and gate toggle.
     * Binary uses param_1 (original state) for these comparisons. */
    int orig_gate = s->gate_open;
    int orig_m1r = s->mummy1_row, orig_m1c = s->mummy1_col;
    int orig_m2r = s->mummy2_row, orig_m2c = s->mummy2_col;

    /* 1. Player movement */
    int old_pr = s->player_row, old_pc = s->player_col;
    int new_pr = old_pr + dr, new_pc = old_pc + dc;

    if (dr == 0 && dc == 0) {
        /* Wait action — no movement */
    } else if (can_move_player(lev, s, old_pr, old_pc, new_pr, new_pc)) {
        s->player_row = new_pr;
        s->player_col = new_pc;
    } else {
        /* Can't move — stay in place */
        new_pr = old_pr;
        new_pc = old_pc;
    }

    /* 2. Key/gate toggle from player entering key cell.
     * key_toggled persists across ALL entities for the entire turn.
     * Binary: at most one entity toggles via key entry per turn. */
    int key_toggled = 0;
    check_key_toggle(lev, s, s->player_row, s->player_col, old_pr, old_pc,
                     orig_gate, &key_toggled);

    /* 3. Trap check */
    if ((s->player_row == lev->trap1_row && s->player_col == lev->trap1_col) ||
        (s->player_row == lev->trap2_row && s->player_col == lev->trap2_col)) {
        return STEP_DEAD;
    }

    /* 4. Scorpion movement (1 step) */
    int old_scorp_r = s->scorpion_row, old_scorp_c = s->scorpion_col;
    if (s->scorpion_alive) {
        int sr = s->scorpion_row, sc = s->scorpion_col;
        if (sr < 0 || sr >= N || sc < 0 || sc >= N) {
            s->scorpion_alive = 0;
        } else {
            move_enemy(lev, s, &s->scorpion_row, &s->scorpion_col, 1);
        }
    }

    /* Scorpion-player death check */
    if (s->scorpion_alive &&
        s->player_row == s->scorpion_row && s->player_col == s->scorpion_col) {
        return STEP_DEAD;
    }

    /* Scorpion-mummy collision: scorpion landing on mummy kills scorpion */
    if (s->scorpion_alive && s->mummy1_alive &&
        s->scorpion_row == s->mummy1_row && s->scorpion_col == s->mummy1_col) {
        s->scorpion_alive = 0;
    }
    if (s->scorpion_alive && s->mummy2_alive &&
        s->scorpion_row == s->mummy2_row && s->scorpion_col == s->mummy2_col) {
        s->scorpion_alive = 0;
    }

    /* Scorpion key toggle — NO reset of key_toggled (binary carries it
     * from the player toggle). Uses original gate for assignment. */
    if (s->scorpion_alive) {
        check_key_toggle(lev, s,
                         s->scorpion_row, s->scorpion_col,
                         old_scorp_r, old_scorp_c,
                         orig_gate, &key_toggled);
    }

    /* 5. Mummy movement — 2 iterations */
    for (int mstep = 0; mstep < 2; mstep++) {
        if (s->mummy1_alive) {
            move_enemy(lev, s, &s->mummy1_row, &s->mummy1_col, 0);
        }
        if (s->mummy2_alive) {
            move_enemy(lev, s, &s->mummy2_row, &s->mummy2_col, 0);
        }

        /* Mummy-mummy collision: if two mummies on same cell, mummy2 dies */
        if (s->mummy1_alive && s->mummy2_alive &&
            s->mummy1_row == s->mummy2_row && s->mummy1_col == s->mummy2_col) {
            s->mummy2_alive = 0;
        }

        /* Mummy-scorpion collision: mummy kills scorpion.
         * Binary: does NOT check scorpion_alive here. Also toggles gate
         * (using !current_gate, no key_toggled guard) if kill happens
         * on the key cell. */
        if (s->mummy1_alive &&
            s->mummy1_row == s->scorpion_row && s->mummy1_col == s->scorpion_col) {
            if (lev->has_gate &&
                s->scorpion_row == lev->key_row && s->scorpion_col == lev->key_col) {
                s->gate_open = !s->gate_open;
            }
            s->scorpion_alive = 0;
        }
        if (s->mummy2_alive &&
            s->mummy2_row == s->scorpion_row && s->mummy2_col == s->scorpion_col) {
            if (lev->has_gate &&
                s->scorpion_row == lev->key_row && s->scorpion_col == lev->key_col) {
                s->gate_open = !s->gate_open;
            }
            s->scorpion_alive = 0;
        }

        /* Mummy-player death check */
        if (s->mummy1_alive &&
            s->player_row == s->mummy1_row && s->player_col == s->mummy1_col) {
            return STEP_DEAD;
        }
        if (s->mummy2_alive &&
            s->player_row == s->mummy2_row && s->player_col == s->mummy2_col) {
            return STEP_DEAD;
        }

        /* Mummy key toggles — mutually exclusive (if/else).
         * key_toggled guard: only one entity toggles via key entry per turn.
         * "Moved" check compares against turn-start positions (orig_m*),
         * not iteration-start positions. */
        if (!key_toggled) {
            if (s->mummy1_alive &&
                s->mummy1_row == lev->key_row && s->mummy1_col == lev->key_col &&
                (s->mummy1_row != orig_m1r || s->mummy1_col != orig_m1c)) {
                s->gate_open = !orig_gate;
                key_toggled = 1;
            } else if (s->mummy2_alive &&
                       s->mummy2_row == lev->key_row && s->mummy2_col == lev->key_col &&
                       (s->mummy2_row != orig_m2r || s->mummy2_col != orig_m2c)) {
                s->gate_open = !orig_gate;
                /* Binary does NOT set key_toggled here (goto skips it).
                 * But since the outer if guards on !key_toggled, and
                 * mummy1 gets priority, in practice this matters only
                 * if mummy2 toggles in iteration 0 and then we check
                 * again in iteration 1. The binary allows that. */
            }
        }
    }

    /* 6. Win check — player reached exit cell and exit flag is set */
    if (s->player_row == lev->exit_row && s->player_col == lev->exit_col) {
        /* Verify the exit flag is actually in the wall */
        uint32_t w = lev->walls[lev->exit_col + lev->exit_row * 10];
        if (w & lev->exit_mask) {
            return STEP_WIN;
        }
    }

    return STEP_OK;
}

uint64_t state_hash(const State *s) {
    /* FNV-1a hash of the fields compared in FUN_00404f60 */
    uint64_t h = 0xcbf29ce484222325ULL;
    #define MIX(v) do { h ^= (uint64_t)(v); h *= 0x100000001b3ULL; } while(0)
    MIX(s->player_row);
    MIX(s->player_col);
    MIX(s->mummy1_row);
    MIX(s->mummy1_col);
    MIX(s->mummy1_alive);
    MIX(s->mummy2_row);
    MIX(s->mummy2_col);
    MIX(s->mummy2_alive);
    MIX(s->scorpion_row);
    MIX(s->scorpion_col);
    MIX(s->scorpion_alive);
    MIX(s->gate_open);
    #undef MIX
    return h;
}

int state_eq(const State *a, const State *b) {
    return a->player_row == b->player_row &&
           a->player_col == b->player_col &&
           a->mummy1_row == b->mummy1_row &&
           a->mummy1_col == b->mummy1_col &&
           a->mummy1_alive == b->mummy1_alive &&
           a->mummy2_row == b->mummy2_row &&
           a->mummy2_col == b->mummy2_col &&
           a->mummy2_alive == b->mummy2_alive &&
           a->scorpion_row == b->scorpion_row &&
           a->scorpion_col == b->scorpion_col &&
           a->scorpion_alive == b->scorpion_alive &&
           a->gate_open == b->gate_open;
}
