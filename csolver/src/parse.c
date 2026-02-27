#include "parse.h"
#include <string.h>
#include <stdio.h>

int parse_header(const uint8_t *data, int len, Header *out) {
    if (len < 6) return -1;
    out->grid_size    = data[0] & 0x0F;
    out->flip         = (data[0] & 0xF0) != 0;
    out->num_sublevels = data[1];
    out->mummy_count  = data[2];
    out->key_gate     = data[3];
    out->trap_count   = data[4];
    out->scorpion     = data[5];

    int N = out->grid_size;
    out->wall_bytes = N * (N > 8 ? 2 : 1) * 2;
    out->bytes_per_sub = out->wall_bytes + 3 + (out->mummy_count - 1)
                         + 2 * out->key_gate + out->scorpion + out->trap_count;
    return 0;
}

/*
 * Helper: read one or two bytes for wall bits depending on grid size.
 */
static uint16_t read_wall_bits(const uint8_t *data, int *pos, int N) {
    uint16_t bits = data[*pos];
    (*pos)++;
    if (N > 8) {
        bits |= (uint16_t)data[*pos] << 8;
        (*pos)++;
    }
    return bits;
}

/*
 * Decode entity position byte.
 * Returns raw col (low nibble) and row (high nibble).
 */

/*
 * ------------------------------------------------------------------
 * Wall loading — direct port of FUN_0040e1d0
 *
 * The binary stores walls in a flat array:  walls[col + row * 10]
 * at object offset 0x300, with 4 bytes (uint32_t) per cell.
 *
 * Wall bits per cell:
 *   bit 0 (1) = WALL_W — wall on the west side
 *   bit 1 (2) = WALL_E — wall on the east side
 *   bit 2 (4) = WALL_S — wall on the south side
 *   bit 3 (8) = WALL_N — wall on the north side
 *
 * Border walls are set first (N/S/E/W edges of the grid).
 * Then the .dat wall bytes are read: first loop = "h" walls,
 * second loop = "v" walls. The interpretation differs for flip=0
 * vs flip=1.
 *
 * Notation for the binary's offset arithmetic:
 *   walls[i] corresponds to *(param_1 + 0x300 + i*4)
 *   param_1 + 0x2d8 + i*4 = walls[i - 10]  (0x300 - 0x2d8 = 0x28 = 40 = 10*4)
 *   param_1 + 0x2d4 + i*4 = walls[i - 11]
 *   param_1 + 0x2fc + i*4 = walls[i - 1]
 *   param_1 + 0x418 + i*4 = walls[i + 70]  (not used directly, but as pointer base)
 * ------------------------------------------------------------------
 */

static void load_walls_flip1(uint32_t walls[], const uint8_t *data, int *pos, int N) {
    /*
     * flip=1 path (the first branch in the decompiled code, where
     * *(param_1 + 0x4b0) != 0).
     *
     * First wall loop — "h-walls" in dat, become N/S walls in binary:
     *   Outer: iVar10 = 0..N-1 (bytes read per column slot)
     *   Inner: puVar12 starts at puVar5 + -0x50 = &walls[-80+70+col] (adjusted)
     *
     * Actually from the decompiled code for flip=0 path (the NOT flipped / white
     * mummy path), puVar5 is initialized to (param_1 + 0x418) which is
     * walls[70]. Then puVar12 = puVar5 + (-0x50) where -0x50 is in uint*
     * units = -80 ints = -320 bytes, but that can't be right...
     *
     * Let me re-read: puVar5 starts at (param_1 + 0x418). In the inner loop,
     * puVar12 = puVar5 + -0x50. Since puVar5 is uint*, -0x50 hex = -80 decimal
     * in uint units = -320 bytes. But 0x418 - 320 = 0x418 - 0x140 = 0x2D8.
     * So puVar12 starts at param_1+0x2D8 = walls[-10].
     *
     * Inner loop increments puVar12 by 10 (uint units = 40 bytes = one row).
     * So puVar12 walks: walls[-10], walls[0], walls[10], walls[20], ...
     *
     * When bit set: puVar12[10] |= 8 (WALL_N) → walls[-10+10] = walls[0+col]
     * And if iVar11>0: *puVar12 |= 4 (WALL_S) → walls[-10+col] for first iter,
     * which is out of bounds... unless the condition 0 < iVar11 prevents the
     * first iteration.
     *
     * Wait, iVar11 is the row counter starting at 0, so the WALL_S assignment
     * only happens for iVar11 >= 1, meaning puVar12 = walls[-10 + 10*1 + col]
     * = walls[col]. That would mean:
     *   WALL_N at walls[col + iVar11*10] for row iVar11
     *   WALL_S at walls[col + (iVar11-1)*10] for row iVar11-1
     * Which is: a horizontal wall between row (iVar11-1) and row iVar11.
     *
     * Outer counter iVar10 iterates 0..N-1 and puVar5 increments by 1 each
     * time, so col = iVar10.
     *
     * So for flip=0: reading byte for column iVar10, bit iVar11 means
     * a wall between row(iVar11-1) and row(iVar11):
     *   walls[iVar10 + iVar11*10] |= WALL_N
     *   walls[iVar10 + (iVar11-1)*10] |= WALL_S  (if iVar11 > 0)
     *
     * But wait — the border walls already set row=0 to WALL_N, so bit 0
     * would add WALL_N to walls[iVar10 + 0] which is row 0 — that's
     * redundant with border. The .dat format seems to include border bits.
     */

    /* This is the flip=1 path from the decompiled code.
     *
     * From the decompiled flip=1 (else) branch:
     * First wall loop uses offsets 0x2d8 and 0x2d4:
     *   walls[iVar8 + (N - iVar10)*10 - 10] |= WALL_W(1)
     *   walls[iVar8 + (N - iVar10)*10 - 11] |= WALL_E(2)  (if iVar8 > 0)
     *
     * Second wall loop uses offsets 0x2d8 and 0x300:
     *   walls[iVar8 + (N - iVar10)*10 - 10] |= WALL_S(4)
     *   walls[iVar8 + (N - iVar10)*10] |= WALL_N(8)  (if iVar10 > 0)
     */

    /* First wall loop: W/E walls */
    for (int iVar10 = 0; iVar10 < N; iVar10++) {
        uint16_t bits = read_wall_bits(data, pos, N);
        for (int iVar8 = 0; iVar8 < N; iVar8++) {
            if (bits & (1 << iVar8)) {
                int idx = iVar8 + (N - iVar10) * 10;
                walls[idx - 10] |= WALL_W;
                if (iVar8 > 0) {
                    walls[idx - 11] |= WALL_E;
                }
            }
        }
    }

    /* Second wall loop: S/N walls */
    for (int iVar10 = 0; iVar10 < N; iVar10++) {
        uint16_t bits = read_wall_bits(data, pos, N);
        for (int iVar8 = 0; iVar8 < N; iVar8++) {
            if (bits & (1 << iVar8)) {
                int idx = iVar8 + (N - iVar10) * 10;
                walls[idx - 10] |= WALL_S;
                if (iVar10 > 0) {
                    walls[idx] |= WALL_N;
                }
            }
        }
    }
}

static void load_walls_flip0(uint32_t walls[], const uint8_t *data, int *pos, int N) {
    /*
     * flip=0 path from decompiled code.
     *
     * First wall loop:
     *   puVar5 starts at (param_1 + 0x418) = walls + (0x418-0x300)/4 = walls+70
     *   But puVar5 is used to index: puVar12 = puVar5 + (-0x50)
     *   In uint* arithmetic: puVar5 - 80 = walls + 70 - 80 = walls - 10
     *   puVar12 walks rows: puVar12 += 10 per iteration
     *
     *   Outer: iVar10 (reused as counter but puVar5 increments by 1 each iter)
     *     → puVar5 goes from walls+70 to walls+70+N-1, but only used as base
     *       for puVar12, which is puVar5 - 80.
     *     → Effective column = outer_iter (0..N-1)
     *   Inner: iVar11 = 0..N-1 (row), limited to < 8 for first byte
     *     puVar12 = walls[-10 + column], then +10 per row
     *
     *   When bit set:
     *     puVar12[10] |= 8 (WALL_N) → walls[-10 + col + (row+1)*10] = walls[col + row*10]
     *     if iVar11 > 0:
     *       *puVar12 |= 4 (WALL_S) → walls[-10 + col + row*10]
     *         = walls[col + (row-1)*10]
     *
     *   So bit iVar11 in column iVar10's byte means N/S wall:
     *     walls[col + row*10] |= WALL_N  (at row iVar11)
     *     walls[col + (row-1)*10] |= WALL_S  (at row iVar11-1, if row > 0)
     *
     * Second wall loop:
     *   local_1f8 starts at (param_1 + 0x43c) = walls + (0x43c-0x300)/4 = walls+79
     *   puVar5 = local_1f8 + (-0x50) → same as above shifted by 9
     *   Actually local_1f8 is the outer pointer, starting at walls+79? No...
     *   Let me re-read.
     *
     *   local_1f8 = (param_1 + 0x43c) = walls + (0x13c/4) = walls + 79
     *   Inner: puVar5 = local_1f8 + (-0x50) = walls + 79 - 80 = walls - 1
     *     puVar5 walks: walls[-1 + row*10]
     *   When bit set:
     *     puVar5[1] |= 1 (WALL_W) → walls[-1+1 + row*10] = walls[row*10]... no
     *     Wait: puVar5 starts at walls[-1] for first outer iter (col=0).
     *     puVar5[1] = walls[-1+1] = walls[0] for row=0.
     *     Then puVar5 += 10, so puVar5[1] = walls[-1+10+1] = walls[10] for row=1.
     *     And *puVar5 = walls[-1+10] = walls[9] ... hmm that's wrong.
     *
     *   Ah wait, I made an error. local_1f8 is uint*, value (param_1 + 0x43c).
     *   0x43c - 0x300 = 0x13c = 316 bytes = 79 ints.
     *   puVar5 = local_1f8 + (-0x50) where -0x50 is -80 in int units.
     *   So puVar5 = walls + 79 - 80 = walls - 1.
     *
     *   puVar5[1] |= WALL_W(1) → walls[-1 + 1 + row*10] for row=0 first iter
     *                            = walls[row*10 + 0] ... but where's the column?
     *
     *   Outer: iVar10 = 0..N-1, local_1f8 increments by 1 each iter.
     *   So effective: puVar5 base = walls[-1 + iVar10], puVar5 += 10 per row.
     *     puVar5[1] = walls[-1 + iVar10 + 1 + row*10] = walls[iVar10 + row*10]
     *     *puVar5 = walls[-1 + iVar10 + row*10]
     *
     *   When bit set:
     *     puVar5[1] |= WALL_W(1) → walls[col + row*10] |= WALL_W
     *     if iVar10 > 0:
     *       *puVar5 |= WALL_E(2) → walls[col-1 + row*10] |= WALL_E
     *
     *   So bit iVar8 in column iVar10's byte means W/E wall:
     *     walls[col + row*10] |= WALL_W
     *     walls[col-1 + row*10] |= WALL_E  (if col > 0)
     */

    /* First wall loop: N/S walls */
    for (int col = 0; col < N; col++) {
        uint16_t bits = read_wall_bits(data, pos, N);
        for (int row = 0; row < N; row++) {
            if (bits & (1 << row)) {
                walls[col + row * 10] |= WALL_N;
                if (row > 0) {
                    walls[col + (row - 1) * 10] |= WALL_S;
                }
            }
        }
    }

    /* Second wall loop: W/E walls */
    for (int col = 0; col < N; col++) {
        uint16_t bits = read_wall_bits(data, pos, N);
        for (int row = 0; row < N; row++) {
            if (bits & (1 << row)) {
                walls[col + row * 10] |= WALL_W;
                if (col > 0) {
                    walls[col - 1 + row * 10] |= WALL_E;
                }
            }
        }
    }
}

static void load_exit_flip0(uint32_t walls[], const uint8_t *data, int *pos, int N,
                             Level *out) {
    /*
     * flip=0 exit: direct port of switch in FUN_0040e1d0 (first branch).
     * exit_byte: side = byte & 0xF, uVar6 = byte >> 4 (position along that side).
     *
     * case 0: walls[uVar6 + 0*10] → walls[uVar6] at row=0
     *         |= 0x10(EXIT_W) then ^= 1(WALL_W)
     *         → West exit at row 0, position uVar6 along that row
     *         → exit is on the west side, cell = (row=uVar6, col=0)?
     *         Actually walls[uVar6 + 0*10] = walls[uVar6] = col=uVar6, row=0
     *         So this puts EXIT_W on cell (row=0, col=uVar6) and removes WALL_W.
     *         → Means exit is on the NORTH side at position uVar6? No...
     *         Wait: walls[col + row*10]. walls[uVar6] means col=uVar6, row=0.
     *         EXIT_W flag and toggling WALL_W means exit is to the west of this cell.
     *         So the cell at (row=0, col=uVar6) has its west wall opened.
     *         But col=uVar6 at row=0... if uVar6=0, that's the NW corner.
     *         Actually this is the WEST exit: the player walks west out of cell
     *         (row=0, col=uVar6). Hmm, but that would be the north row...
     *
     *         In the binary's system: case 0 opens a west-side exit.
     *         The wall index is walls[uVar6 * 0x28/4]... wait no.
     *         walls[uVar6 + 0*10] = walls[uVar6]. In col+row*10 indexing,
     *         col=uVar6, row=0.
     *
     *         |= EXIT_W(0x10), ^= WALL_W(1)
     *         This marks the west wall of cell (col=uVar6, row=0) as an exit
     *         and opens it. The exit cell is (row=0, col=uVar6) — but you exit
     *         by going west (to col=-1).
     *         Hmm, uVar6 should be 0 for a west exit to make sense.
     *         No: it can be any row — the *position* along the west edge.
     *         walls[uVar6 + 0*10]: this is col=uVar6, row=0.
     *         But for a west exit, the cell should be at col=0.
     *
     *         Wait, I think I misread. Let me look again:
     *         case 0: walls[uVar6 * 0x28] offset?
     *         The decompiled code says:
     *           *(uint *)(param_1 + 0x300 + uVar6 * 0x28) |= 0x10 ^ 1
     *         0x28 = 40 bytes = 10 ints. So walls[uVar6 * 10].
     *         In col+row*10 indexing, that's col=0, row=uVar6.
     *         EXIT_W(0x10) ^= WALL_W(1) at (col=0, row=uVar6).
     *         → West exit at row=uVar6, column 0. Makes sense!
     *
     *         So the exit_cell (in-grid cell adjacent to exit) is (row=uVar6, col=0).
     *
     * case 1: walls[uVar6 * 4] → wait, the code says:
     *         *(uint *)(param_1 + 0x300 + uVar6 * 4) |= 0x80 ^ 8
     *         walls[uVar6]. In col+row*10: col=uVar6, row=0.
     *         EXIT_N(0x80) ^= WALL_N(8).
     *         → North exit at col=uVar6, row 0.
     *         exit_cell = (row=0, col=uVar6).
     *
     * case 2: *(param_1 + 0x2d8 + (uVar6 + N*10)*4)
     *         = walls[uVar6 + N*10 - 10] (since 0x2d8 = 0x300 - 40 = walls[-10])
     *         = walls[uVar6 + (N-1)*10]
     *         = col=uVar6, row=N-1
     *         |= EXIT_S(0x40) ^= WALL_S(4)
     *         → South exit at col=uVar6, row=N-1.
     *         exit_cell = (row=N-1, col=uVar6).
     *
     * case 3: *(param_1 + 0x2fc + (uVar6*10 + N)*4)
     *         = walls[uVar6*10 + N - 1] (since 0x2fc = 0x300 - 4 = walls[-1])
     *         = walls[(N-1) + uVar6*10]
     *         = col=N-1, row=uVar6
     *         |= EXIT_E(0x20) ^= WALL_E(2)
     *         → East exit at row=uVar6, col=N-1.
     *         exit_cell = (row=uVar6, col=N-1).
     */
    uint8_t eb = data[*pos]; (*pos)++;
    int side = eb & 0x0F;
    int p = (eb >> 4) & 0x0F;

    switch (side) {
    case 0: /* West exit */
        walls[0 + p * 10] |= EXIT_W;
        walls[0 + p * 10] ^= WALL_W;
        out->exit_row = p;
        out->exit_col = 0;
        out->exit_mask = EXIT_W;
        break;
    case 1: /* North exit */
        walls[p + 0 * 10] |= EXIT_N;
        walls[p + 0 * 10] ^= WALL_N;
        out->exit_row = 0;
        out->exit_col = p;
        out->exit_mask = EXIT_N;
        break;
    case 2: /* South exit */
        walls[p + (N - 1) * 10] |= EXIT_S;
        walls[p + (N - 1) * 10] ^= WALL_S;
        out->exit_row = N - 1;
        out->exit_col = p;
        out->exit_mask = EXIT_S;
        break;
    case 3: /* East exit */
        walls[(N - 1) + p * 10] |= EXIT_E;
        walls[(N - 1) + p * 10] ^= WALL_E;
        out->exit_row = p;
        out->exit_col = N - 1;
        out->exit_mask = EXIT_E;
        break;
    default:
        out->exit_row = -1;
        out->exit_col = -1;
        out->exit_mask = 0;
        break;
    }
}

static void load_exit_flip1(uint32_t walls[], const uint8_t *data, int *pos, int N,
                             Level *out) {
    /*
     * flip=1 exit: from the else-branch of FUN_0040e1d0.
     * The decompiled switch uses uStack_1f4.
     *
     * case 0: *(param_1 + 0x2d8 + (uVar6 + N*10)*4)
     *         = walls[uVar6 + N*10 - 10] = walls[uVar6 + (N-1)*10]
     *         = col=uVar6, row=N-1
     *         |= EXIT_S(0x40) ^= WALL_S(4)
     *         → South exit at col=uVar6, row=N-1.
     *
     * case 1: *(param_1 + 0x2d8 + (N-uVar6)*0x28)
     *         0x2d8 = walls[-10] base, *0x28 = *10 in int units
     *         = walls[(N-uVar6)*10 - 10] = walls[(N-uVar6-1)*10]
     *         = col=0, row=(N-uVar6-1)
     *         |= EXIT_W(0x10) ^= WALL_W(1)
     *         → West exit at row=(N-uVar6-1), col=0.
     *
     * case 2: *(param_1 + 0x2d4 + (N*0x2c + uVar6*(-0x28))/4... )
     *         This is messy. Let me read the actual decompiled code:
     *         iVar10 = N * 0x2c + uVar6 * -0x28
     *         *(param_1 + 0x2d4 + iVar10) ... but iVar10 is in bytes?
     *         No, 0x2c = 44, 0x28 = 40. These look like byte offsets...
     *         Actually the decompiled code operates on uint* arithmetic.
     *         Let me re-read:
     *           iVar10 = *(int *)(param_1 + 0x50) * 0x2c + uVar6 * -0x28;
     *           *(uint *)(iVar10 + 0x2d4 + param_1) ...
     *         This is pointer arithmetic in bytes:
     *           address = param_1 + 0x2d4 + N*0x2c - uVar6*0x28
     *                   = param_1 + 0x2d4 + N*44 - uVar6*40
     *                   = (param_1+0x300) + (0x2d4-0x300) + N*44 - uVar6*40
     *                   = walls_base + (-44) + N*44 - uVar6*40
     *                   = walls_base + (N-1)*44 - uVar6*40
     *         In uint32 terms: walls[(N-1)*11 - uVar6*10]
     *         For N=6: walls[55 - uVar6*10], for N=8: walls[77 - uVar6*10]
     *         = col = (N-1)*11 - uVar6*10 - row*10... hmm, this doesn't factor
     *         neatly into col+row*10 unless 11 = 10+1, so:
     *         (N-1)*11 - uVar6*10 = (N-1)*10 + (N-1) - uVar6*10
     *                             = (N-1-uVar6)*10 + (N-1)
     *                             = (N-1) + (N-1-uVar6)*10
     *         So col=N-1, row=N-1-uVar6.
     *         |= EXIT_E(0x20) ^= WALL_E(2)
     *         → East exit at row=(N-1-uVar6), col=N-1.
     *
     * case 3: *(param_1 + 0x300 + uVar6*4) = walls[uVar6] = col=uVar6, row=0
     *         |= EXIT_N(0x80) ^= WALL_N(8)
     *         → North exit at col=uVar6, row=0.
     */
    uint8_t eb = data[*pos]; (*pos)++;
    int side = eb & 0x0F;
    int p = (eb >> 4) & 0x0F;

    switch (side) {
    case 0: /* South exit */
        walls[p + (N - 1) * 10] |= EXIT_S;
        walls[p + (N - 1) * 10] ^= WALL_S;
        out->exit_row = N - 1;
        out->exit_col = p;
        out->exit_mask = EXIT_S;
        break;
    case 1: /* West exit */
        walls[0 + (N - p - 1) * 10] |= EXIT_W;
        walls[0 + (N - p - 1) * 10] ^= WALL_W;
        out->exit_row = N - p - 1;
        out->exit_col = 0;
        out->exit_mask = EXIT_W;
        break;
    case 2: /* East exit */
        walls[(N - 1) + (N - 1 - p) * 10] |= EXIT_E;
        walls[(N - 1) + (N - 1 - p) * 10] ^= WALL_E;
        out->exit_row = N - 1 - p;
        out->exit_col = N - 1;
        out->exit_mask = EXIT_E;
        break;
    case 3: /* North exit */
        walls[p + 0 * 10] |= EXIT_N;
        walls[p + 0 * 10] ^= WALL_N;
        out->exit_row = 0;
        out->exit_col = p;
        out->exit_mask = EXIT_N;
        break;
    default:
        out->exit_row = -1;
        out->exit_col = -1;
        out->exit_mask = 0;
        break;
    }
}

int parse_sublevel(const uint8_t *data, int offset, int data_len,
                   const Header *hdr, Level *out) {
    int N = hdr->grid_size;
    if (offset + hdr->bytes_per_sub > data_len) return -1;

    memset(out, 0, sizeof(*out));
    out->grid_size = N;
    out->flip = hdr->flip;

    /* Initialize walls to zero */
    memset(out->walls, 0, sizeof(out->walls));

    /* Set border walls */
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < N; row++) {
            uint32_t *w = &out->walls[col + row * 10];
            if (row == 0)     *w |= WALL_N;
            if (row == N - 1) *w |= WALL_S;
            if (col == 0)     *w |= WALL_W;
            if (col == N - 1) *w |= WALL_E;
        }
    }

    int pos = offset;

    /* Load walls depending on flip */
    if (hdr->flip) {
        load_walls_flip1(out->walls, data, &pos, N);
    } else {
        load_walls_flip0(out->walls, data, &pos, N);
    }

    /* Load exit */
    if (hdr->flip) {
        load_exit_flip1(out->walls, data, &pos, N, out);
    } else {
        load_exit_flip0(out->walls, data, &pos, N, out);
    }

    /* Entity positions — the exit byte consumed 1 byte above, now
     * the decompiled code reads another byte (from FUN_00451acb) which
     * seems to be the player construction parameter. Let me check:
     * After the exit switch, the code reads another byte via FUN_00451acb
     * into local_1fc. Then it calls FUN_00426e70 (player constructor).
     *
     * For flip=0: player gets (col = local_1fc & 0xf, row = (local_1fc >> 4) & 0xf)
     *   The constructor call: FUN_00426e70(..., local_1fc & 0xf, (local_1fc>>4) & 0xf, N)
     *   But wait, the exit code already consumed the byte. Looking more carefully,
     *   after the exit switch there's:
     *     FUN_00451acb(local_1d4,(undefined1 *)&local_1fc);
     *   That reads the NEXT byte (player position).
     *   Then FUN_00426e70 is called with (local_1fc & 0xf, (local_1fc & 0xff) >> 4, N)
     *   → (col, row).
     *   In flip=0 these are stored directly.
     *
     * For flip=1: the player constructor gets:
     *   (N - ((uStack_1f4 & 0xff) >> 4) - 1, uStack_1f4 & 0xf)
     *   → (N - row_raw - 1, col_raw)
     *   So player_row = N - row_raw - 1, player_col = col_raw.
     *   Wait actually looking at the code more carefully:
     *   FUN_00426e70(local_1f8, *(param_1+0x58),
     *                (*(param_1+0x50) - ((uStack_1f4 & 0xff) >> 4)) + -1,
     *                uStack_1f4 & 0xf, N)
     *   The first positional param after the game obj = N - row_raw - 1
     *   The second = col_raw
     *   In the player object, these get stored at offsets that become
     *   player_row and player_col respectively.
     *   But what ARE they in the binary's coordinate system?
     *   Looking at FUN_00426e70 / the player sprite constructor, the first
     *   param likely maps to the x/col and second to y/row... or vice versa.
     *   Actually in the step function FUN_00405580, the player state is:
     *   state[0] = player_row, state[1] = player_col
     *   And state[0] is added to param_3 (which is the row delta passed from
     *   the DFS), state[1] is added to param_4 (col delta).
     *
     *   The FUN_00426e70 constructor's first coord param becomes what's stored
     *   at state[0] (player_row) and second at state[1] (player_col).
     *
     *   For flip=0: state[0] = col_raw, state[1] = row_raw  (from the raw byte)
     *   Wait that seems wrong. Let me re-check.
     *
     *   For flip=0, the call is:
     *   FUN_00426e70(local_1f8, game_obj, local_1fc & 0xf, (local_1fc>>4), N)
     *   → first coord = byte & 0xf = col_raw
     *   → second coord = byte >> 4 = row_raw
     *   These become player state[0] and state[1] in FUN_00405580's 0x3c struct.
     *
     *   In the step function, walls are accessed as walls[state[1] + state[0] * 10].
     *   So: walls[col + row * 10] → state[0] is used as "row" (multiplied by 10)
     *   and state[1] as "col" (added directly).
     *
     *   For flip=0: state[0] = col_raw used as binary-row
     *               state[1] = row_raw used as binary-col
     *
     *   This means the binary's internal "row" = dat-file's col,
     *   and binary's internal "col" = dat-file's row.
     *   The NW-SE transpose that the Python parser does is essentially
     *   compensating for this!
     *
     *   For flip=1: state[0] = N - row_raw - 1, state[1] = col_raw
     *   (no transposition, but vertical flip of row_raw)
     */

    /* Read player position byte */
    {
        uint8_t pb = data[pos]; pos++;
        int col_raw = pb & 0x0F;
        int row_raw = (pb >> 4) & 0x0F;
        if (hdr->flip) {
            out->player_row = N - row_raw - 1;
            out->player_col = col_raw;
        } else {
            out->player_row = col_raw;
            out->player_col = row_raw;
        }
    }

    /* Read mummy 1 */
    {
        uint8_t mb = data[pos]; pos++;
        int col_raw = mb & 0x0F;
        int row_raw = (mb >> 4) & 0x0F;
        if (hdr->flip) {
            out->mummy1_row = N - row_raw - 1;
            out->mummy1_col = col_raw;
        } else {
            out->mummy1_row = col_raw;
            out->mummy1_col = row_raw;
        }
    }

    /* Read mummy 2 (if present) */
    out->has_mummy2 = (hdr->mummy_count >= 2);
    if (hdr->mummy_count >= 2) {
        uint8_t mb = data[pos]; pos++;
        int col_raw = mb & 0x0F;
        int row_raw = (mb >> 4) & 0x0F;
        if (hdr->flip) {
            out->mummy2_row = N - row_raw - 1;
            out->mummy2_col = col_raw;
        } else {
            out->mummy2_row = col_raw;
            out->mummy2_col = row_raw;
        }
    } else {
        out->mummy2_row = 99;
        out->mummy2_col = 99;
    }

    /*
     * Entity byte order in the .dat file (verified from FUN_0040e1d0):
     *   player, mummy1, [mummy2], [scorpion], [trap1], [trap2], [gate, key]
     *
     * The binary reads scorpion before traps, and gate/key LAST.
     */

    /* Read scorpion (if present) */
    out->has_scorpion = (hdr->scorpion > 0);
    if (hdr->scorpion > 0) {
        uint8_t sb = data[pos]; pos++;
        int col_raw = sb & 0x0F;
        int row_raw = (sb >> 4) & 0x0F;
        if (hdr->flip) {
            out->scorpion_row = N - row_raw - 1;
            out->scorpion_col = col_raw;
        } else {
            out->scorpion_row = col_raw;
            out->scorpion_col = row_raw;
        }
    } else {
        out->scorpion_row = 99;
        out->scorpion_col = 99;
    }

    /* Read traps */
    out->trap_count = hdr->trap_count;
    out->trap1_row = 99; out->trap1_col = 99;
    out->trap2_row = 99; out->trap2_col = 99;
    if (hdr->trap_count >= 1) {
        uint8_t tb = data[pos]; pos++;
        int col_raw = tb & 0x0F;
        int row_raw = (tb >> 4) & 0x0F;
        if (hdr->flip) {
            out->trap1_row = N - row_raw - 1;
            out->trap1_col = col_raw;
        } else {
            out->trap1_row = col_raw;
            out->trap1_col = row_raw;
        }
    }
    if (hdr->trap_count >= 2) {
        uint8_t tb = data[pos]; pos++;
        int col_raw = tb & 0x0F;
        int row_raw = (tb >> 4) & 0x0F;
        if (hdr->flip) {
            out->trap2_row = N - row_raw - 1;
            out->trap2_col = col_raw;
        } else {
            out->trap2_row = col_raw;
            out->trap2_col = row_raw;
        }
    }

    /* Read gate + key (if present) — comes LAST after scorpion and traps */
    out->has_gate = (hdr->key_gate > 0);
    if (hdr->key_gate > 0) {
        /* Gate position */
        uint8_t gb = data[pos]; pos++;
        int col_raw = gb & 0x0F;
        int row_raw = (gb >> 4) & 0x0F;
        if (hdr->flip) {
            out->gate_row = N - row_raw - 1;
            out->gate_col = col_raw;
        } else {
            out->gate_row = col_raw;
            out->gate_col = row_raw;
        }

        /* Key position */
        uint8_t kb = data[pos]; pos++;
        col_raw = kb & 0x0F;
        row_raw = (kb >> 4) & 0x0F;
        if (hdr->flip) {
            out->key_row = N - row_raw - 1;
            out->key_col = col_raw;
        } else {
            out->key_row = col_raw;
            out->key_col = row_raw;
        }
    } else {
        out->gate_row = 99;
        out->gate_col = 99;
        out->key_row = 99;
        out->key_col = 99;
    }

    return hdr->bytes_per_sub;
}
