#include "solver.h"
#include <stdlib.h>
#include <string.h>

/*
 * Open-addressing hash table for visited states.
 * Uses power-of-2 sizing with linear probing.
 */

#define HT_INITIAL_CAP (1 << 16)  /* 65536 */
#define HT_LOAD_FACTOR 0.7

typedef struct {
    State *entries;    /* NULL-initialized; we use a sentinel to mark occupied */
    char  *occupied;   /* 1 if slot is used, 0 if empty */
    int    cap;
    int    count;
} HashTable;

static void ht_init(HashTable *ht, int cap) {
    ht->cap = cap;
    ht->count = 0;
    ht->entries = calloc(cap, sizeof(State));
    ht->occupied = calloc(cap, 1);
}

static void ht_free(HashTable *ht) {
    free(ht->entries);
    free(ht->occupied);
    ht->entries = NULL;
    ht->occupied = NULL;
}

static void ht_grow(HashTable *ht);

static int ht_insert(HashTable *ht, const State *s) {
    if (ht->count >= (int)(ht->cap * HT_LOAD_FACTOR)) {
        ht_grow(ht);
    }
    uint64_t h = state_hash(s);
    int mask = ht->cap - 1;
    int idx = (int)(h & (uint64_t)mask);
    while (ht->occupied[idx]) {
        if (state_eq(&ht->entries[idx], s)) return 0; /* already present */
        idx = (idx + 1) & mask;
    }
    ht->entries[idx] = *s;
    ht->occupied[idx] = 1;
    ht->count++;
    return 1; /* newly inserted */
}

static void ht_grow(HashTable *ht) {
    int old_cap = ht->cap;
    State *old_entries = ht->entries;
    char *old_occ = ht->occupied;

    ht->cap = old_cap * 2;
    ht->count = 0;
    ht->entries = calloc(ht->cap, sizeof(State));
    ht->occupied = calloc(ht->cap, 1);

    for (int i = 0; i < old_cap; i++) {
        if (old_occ[i]) {
            ht_insert(ht, &old_entries[i]);
        }
    }
    free(old_entries);
    free(old_occ);
}

/*
 * BFS queue using a growable ring buffer.
 */
typedef struct {
    State *state;
    int    depth;
} QueueEntry;

typedef struct {
    QueueEntry *buf;
    int cap;
    int head;
    int tail;
    int count;
} Queue;

static void q_init(Queue *q, int cap) {
    q->cap = cap;
    q->buf = malloc(cap * sizeof(QueueEntry));
    q->head = 0;
    q->tail = 0;
    q->count = 0;
}

static void q_free(Queue *q) {
    free(q->buf);
    q->buf = NULL;
}

static void q_grow(Queue *q) {
    int new_cap = q->cap * 2;
    QueueEntry *new_buf = malloc(new_cap * sizeof(QueueEntry));
    /* Copy contiguous from head to tail */
    for (int i = 0; i < q->count; i++) {
        new_buf[i] = q->buf[(q->head + i) % q->cap];
    }
    free(q->buf);
    q->buf = new_buf;
    q->head = 0;
    q->tail = q->count;
    q->cap = new_cap;
}

static void q_push(Queue *q, const State *s, int depth) {
    if (q->count >= q->cap) q_grow(q);
    q->buf[q->tail].state = malloc(sizeof(State));
    *q->buf[q->tail].state = *s;
    q->buf[q->tail].depth = depth;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
}

static QueueEntry q_pop(Queue *q) {
    QueueEntry e = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    q->count--;
    return e;
}

/*
 * 5 actions: North, South, East, West, Wait
 * Represented as (dr, dc) pairs.
 */
static const int DELTAS[5][2] = {
    {-1,  0},  /* North */
    { 1,  0},  /* South */
    { 0,  1},  /* East  */
    { 0, -1},  /* West  */
    { 0,  0},  /* Wait  */
};

SolveResult solve(const Level *lev) {
    SolveResult res = { .moves = -1, .states_explored = 0 };

    State init;
    state_init(&init, lev);

    HashTable ht;
    ht_init(&ht, HT_INITIAL_CAP);

    Queue q;
    q_init(&q, HT_INITIAL_CAP);

    ht_insert(&ht, &init);
    q_push(&q, &init, 0);

    while (q.count > 0) {
        QueueEntry e = q_pop(&q);
        State *cur = e.state;
        int depth = e.depth;

        for (int a = 0; a < 5; a++) {
            State next = *cur;
            int result = step(lev, &next, DELTAS[a][0], DELTAS[a][1]);

            if (result == STEP_WIN) {
                res.moves = depth + 1;
                res.states_explored = ht.count;
                free(cur);
                /* Drain remaining queue */
                while (q.count > 0) {
                    QueueEntry rem = q_pop(&q);
                    free(rem.state);
                }
                q_free(&q);
                ht_free(&ht);
                return res;
            }

            if (result == STEP_DEAD) continue;

            if (ht_insert(&ht, &next)) {
                q_push(&q, &next, depth + 1);
            }
        }
        free(cur);
    }

    res.states_explored = ht.count;
    q_free(&q);
    ht_free(&ht);
    return res;
}
