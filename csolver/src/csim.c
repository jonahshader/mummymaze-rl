/*
 * csim — dump level + BFS solution as JSON for the visualizer.
 *
 * Modes:
 *   csim <dat_dir> <file_stem> <sub_idx>
 *     → solve with BFS, output JSON with level info + step-by-step states
 *   csim <dat_dir> <file_stem> <sub_idx> <moves>
 *     → replay given moves (N/S/E/W/. in C coords), output JSON trace
 */
#include "parse.h"
#include "game.h"
#include "solver.h"
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

static void json_state(const State *s, int result) {
    printf("{\"pr\":%d,\"pc\":%d,"
           "\"m1r\":%d,\"m1c\":%d,\"m1a\":%d,"
           "\"m2r\":%d,\"m2c\":%d,\"m2a\":%d,"
           "\"sr\":%d,\"sc\":%d,\"sa\":%d,"
           "\"gate\":%d,\"result\":%d}",
           s->player_row, s->player_col,
           s->mummy1_row, s->mummy1_col, s->mummy1_alive,
           s->mummy2_row, s->mummy2_col, s->mummy2_alive,
           s->scorpion_row, s->scorpion_col, s->scorpion_alive,
           s->gate_open, result);
}

static void json_level(const Level *lev) {
    int N = lev->grid_size;
    printf("{\"grid_size\":%d,\"flip\":%d,", N, lev->flip);

    /* Walls as flat array */
    printf("\"walls\":[");
    for (int i = 0; i < N * N; i++) {
        /* Store in row-major: walls[col + row*10] for col=i%N, row=i/N */
        int row = i / N, col = i % N;
        if (i > 0) printf(",");
        printf("%u", lev->walls[col + row * 10]);
    }
    printf("],");

    /* Entities */
    printf("\"player\":[%d,%d],", lev->player_row, lev->player_col);
    printf("\"mummy1\":[%d,%d],", lev->mummy1_row, lev->mummy1_col);
    printf("\"mummy2\":[%d,%d],\"has_mummy2\":%d,",
           lev->mummy2_row, lev->mummy2_col, lev->has_mummy2);
    printf("\"scorpion\":[%d,%d],\"has_scorpion\":%d,",
           lev->scorpion_row, lev->scorpion_col, lev->has_scorpion);
    printf("\"gate\":[%d,%d],\"has_gate\":%d,",
           lev->gate_row, lev->gate_col, lev->has_gate);
    printf("\"key\":[%d,%d],", lev->key_row, lev->key_col);
    printf("\"trap1\":[%d,%d],\"trap2\":[%d,%d],\"trap_count\":%d,",
           lev->trap1_row, lev->trap1_col,
           lev->trap2_row, lev->trap2_col, lev->trap_count);
    printf("\"exit\":[%d,%d],\"exit_mask\":%u",
           lev->exit_row, lev->exit_col, lev->exit_mask);
    printf("}");
}

/* BFS that records the full solution path (action sequence) */
typedef struct { int depth; int action; int parent; } BFSNode;

static const int DELTAS[5][2] = {
    {-1, 0}, {1, 0}, {0, 1}, {0, -1}, {0, 0}
};
static const char *ACTION_NAMES[] = {"N","S","E","W","."};

static int bfs_solve(const Level *lev, int *out_actions, int max_actions) {
    /* Simple BFS with parent tracking */
    #define MAX_STATES (1 << 20)

    typedef struct { State s; int parent; int action; } QEntry;
    QEntry *q = calloc(MAX_STATES, sizeof(QEntry));
    char *visited_occ = calloc(1 << 18, 1);
    State *visited_states = calloc(1 << 18, sizeof(State));
    int visited_cap = 1 << 18;
    int visited_count = 0;

    State init;
    state_init(&init, lev);

    /* Hash table helper - uses the outer scope variables via macro */
    #define HT_INSERT(s_ptr, result_var) do { \
        uint64_t _h = state_hash(s_ptr); \
        int _mask = visited_cap - 1; \
        int _idx = (int)(_h & (uint64_t)_mask); \
        result_var = 1; \
        while (visited_occ[_idx]) { \
            if (state_eq(&visited_states[_idx], s_ptr)) { result_var = 0; break; } \
            _idx = (_idx + 1) & _mask; \
        } \
        if (result_var) { \
            visited_states[_idx] = *(s_ptr); \
            visited_occ[_idx] = 1; \
            visited_count++; \
        } \
    } while(0)

    int _dummy;
    HT_INSERT(&init, _dummy);
    (void)_dummy;
    q[0].s = init;
    q[0].parent = -1;
    q[0].action = -1;
    int head = 0, tail = 1;

    int found = -1;
    while (head < tail && tail < MAX_STATES) {
        QEntry *cur = &q[head];

        for (int a = 0; a < 5; a++) {
            State next = cur->s;
            int result = step(lev, &next, DELTAS[a][0], DELTAS[a][1]);

            if (result == STEP_WIN) {
                q[tail].s = next;
                q[tail].parent = head;
                q[tail].action = a;
                found = tail;
                tail++;
                goto done;
            }
            if (result == STEP_DEAD) continue;

            int ins;
            HT_INSERT(&next, ins);
            if (ins == 1) {
                q[tail].s = next;
                q[tail].parent = head;
                q[tail].action = a;
                tail++;
            }
        }
        head++;
    }
done:
    ;
    int n_moves = 0;
    if (found >= 0) {
        /* Trace back to find action sequence */
        int path[1024];
        int plen = 0;
        int idx = found;
        while (q[idx].parent >= 0) {
            path[plen++] = q[idx].action;
            idx = q[idx].parent;
        }
        /* Reverse */
        for (int i = 0; i < plen && i < max_actions; i++) {
            out_actions[i] = path[plen - 1 - i];
        }
        n_moves = plen;
    }

    free(q);
    free(visited_occ);
    free(visited_states);
    return found >= 0 ? n_moves : -1;
    #undef MAX_STATES
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: csim <dat_dir> <file_stem> <sub_idx> [moves]\n");
        return 1;
    }
    const char *dat_dir = argv[1];
    const char *file_stem = argv[2];
    int sub_idx = atoi(argv[3]);

    char path[512];
    snprintf(path, sizeof(path), "%s/%s.dat", dat_dir, file_stem);

    int data_len;
    uint8_t *data = read_file(path, &data_len);
    if (!data) { fprintf(stderr, "Cannot read %s\n", path); return 1; }

    Header hdr;
    if (parse_header(data, data_len, &hdr) < 0) {
        fprintf(stderr, "Bad header\n"); return 1;
    }

    int offset = 6 + sub_idx * hdr.bytes_per_sub;
    Level lev;
    if (parse_sublevel(data, offset, data_len, &hdr, &lev) < 0) {
        fprintf(stderr, "Parse error\n"); return 1;
    }
    free(data);

    /* Determine actions */
    int actions[1024];
    int n_actions = 0;

    if (argc >= 5) {
        /* Replay given moves */
        const char *moves = argv[4];
        for (int i = 0; moves[i]; i++) {
            char c = moves[i];
            if (c == 'N') actions[n_actions++] = 0;
            else if (c == 'S') actions[n_actions++] = 1;
            else if (c == 'E') actions[n_actions++] = 2;
            else if (c == 'W') actions[n_actions++] = 3;
            else if (c == '.') actions[n_actions++] = 4;
        }
    } else {
        /* BFS solve */
        n_actions = bfs_solve(&lev, actions, 1024);
    }

    /* Output JSON */
    printf("{\"file\":\"%s\",\"sub\":%d,", file_stem, sub_idx);
    printf("\"level\":");
    json_level(&lev);
    printf(",\"solved\":%s,", n_actions >= 0 ? "true" : "false");
    printf("\"n_moves\":%d,", n_actions >= 0 ? n_actions : 0);

    /* Action names */
    printf("\"actions\":[");
    for (int i = 0; i < n_actions; i++) {
        if (i > 0) printf(",");
        printf("\"%s\"", ACTION_NAMES[actions[i]]);
    }
    printf("],");

    /* Step-by-step states */
    printf("\"states\":[");
    State s;
    state_init(&s, &lev);
    json_state(&s, STEP_OK);

    for (int i = 0; i < n_actions; i++) {
        printf(",");
        int result = step(&lev, &s, DELTAS[actions[i]][0], DELTAS[actions[i]][1]);
        json_state(&s, result);
        if (result != STEP_OK) break;
    }
    printf("]}\n");

    return 0;
}
