/* Standalone trace tool: apply a sequence of moves and print state after each */
#include "parse.h"
#include "game.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_file(const char *path, int *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(len);
    if (!buf) { fclose(f); return NULL; }
    *out_len = (int)fread(buf, 1, len, f);
    fclose(f);
    return buf;
}

static void print_state(const State *s) {
    printf("  player=(%d,%d) m1=(%d,%d)%s m2=(%d,%d)%s scorp=(%d,%d)%s gate_open=%d\n",
           s->player_row, s->player_col,
           s->mummy1_row, s->mummy1_col, s->mummy1_alive ? "" : "(dead)",
           s->mummy2_row, s->mummy2_col, s->mummy2_alive ? "" : "(dead)",
           s->scorpion_row, s->scorpion_col, s->scorpion_alive ? "" : "(dead)",
           s->gate_open);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: trace <dat_dir> <file_stem> <sub_idx> <moves>\n");
        fprintf(stderr, "  moves: string of N/S/E/W/. characters\n");
        return 1;
    }
    const char *dat_dir = argv[1];
    const char *file_stem = argv[2];
    int sub_idx = atoi(argv[3]);
    const char *moves = argv[4];

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.dat", dat_dir, file_stem);

    int data_len;
    uint8_t *data = read_file(path, &data_len);
    if (!data) { fprintf(stderr, "Cannot read %s\n", path); return 1; }

    Header hdr;
    parse_header(data, data_len, &hdr);

    int offset = 6 + sub_idx * hdr.bytes_per_sub;
    Level lev;
    parse_sublevel(data, offset, data_len, &hdr, &lev);
    free(data);

    State s;
    state_init(&s, &lev);

    printf("Initial:");
    print_state(&s);
    printf("Gate: (%d,%d)  Key: (%d,%d)  Exit: (%d,%d)\n",
           lev.gate_row, lev.gate_col, lev.key_row, lev.key_col,
           lev.exit_row, lev.exit_col);

    for (int i = 0; moves[i]; i++) {
        int dr = 0, dc = 0;
        char c = moves[i];
        if (c == 'N') dr = -1;
        else if (c == 'S') dr = 1;
        else if (c == 'E') dc = 1;
        else if (c == 'W') dc = -1;
        else if (c == '.') { /* wait */ }
        else continue;

        int result = step(&lev, &s, dr, dc);
        printf("Turn %d (%c): result=%s", i + 1, c,
               result == STEP_OK ? "ok" : result == STEP_DEAD ? "DEAD" : "WIN");
        print_state(&s);
        if (result != STEP_OK) break;
    }

    return 0;
}
