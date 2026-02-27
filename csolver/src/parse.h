#ifndef PARSE_H
#define PARSE_H

#include <stdint.h>

/* Maximum grid size in the game (6, 8, or 10) */
#define MAX_GRID 10

/* Wall bit flags — match the binary's encoding at param_1+0x300 */
#define WALL_W 1   /* West wall  (bit 0) */
#define WALL_E 2   /* East wall  (bit 1) */
#define WALL_S 4   /* South wall (bit 2) */
#define WALL_N 8   /* North wall (bit 3) */

/* Exit flags (upper nibble, used for win detection) */
#define EXIT_W  0x10
#define EXIT_E  0x20
#define EXIT_S  0x40
#define EXIT_N  0x80

typedef struct {
    int grid_size;
    int flip;
    int num_sublevels;
    int mummy_count;
    int key_gate;
    int trap_count;
    int scorpion;
    int wall_bytes;   /* total wall bytes per sublevel */
    int bytes_per_sub;
} Header;

typedef struct {
    int grid_size;
    int flip;
    uint32_t walls[MAX_GRID * MAX_GRID]; /* indexed: walls[col + row * 10] */

    int player_row, player_col;
    int mummy1_row, mummy1_col;
    int mummy2_row, mummy2_col;
    int scorpion_row, scorpion_col;
    int trap1_row, trap1_col;
    int trap2_row, trap2_col;
    int trap_count;
    int has_mummy2;
    int has_scorpion;

    int gate_row, gate_col;   /* binary's coordinate system */
    int has_gate;
    int key_row, key_col;

    int exit_row, exit_col;   /* in-grid cell adjacent to exit */
    uint32_t exit_mask;       /* the exit flag bit in the wall */
} Level;

/* Parse the 6-byte header from raw .dat data. */
int parse_header(const uint8_t *data, int len, Header *out);

/* Parse one sublevel at the given byte offset. Returns bytes consumed, or -1. */
int parse_sublevel(const uint8_t *data, int offset, int data_len,
                   const Header *hdr, Level *out);

#endif /* PARSE_H */
