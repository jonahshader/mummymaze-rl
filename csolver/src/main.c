#include "parse.h"
#include "game.h"
#include "solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

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

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s <dat_dir> [options]\n"
        "  --file B-68     solve one file\n"
        "  --sub N          sublevel index (default 0)\n"
        "  --all            solve all sublevels\n"
        "  --jobs N         parallel workers (default 1)\n"
        "  --out FILE       CSV output path\n"
        "  --dump           dump level info (debug)\n",
        argv0);
}

static void dump_level(const Level *lev) {
    int N = lev->grid_size;
    printf("Grid: %d  Flip: %d\n", N, lev->flip);
    printf("Player: (%d, %d)\n", lev->player_row, lev->player_col);
    printf("Mummy1: (%d, %d)\n", lev->mummy1_row, lev->mummy1_col);
    if (lev->has_mummy2)
        printf("Mummy2: (%d, %d)\n", lev->mummy2_row, lev->mummy2_col);
    if (lev->has_scorpion)
        printf("Scorpion: (%d, %d)\n", lev->scorpion_row, lev->scorpion_col);
    if (lev->has_gate) {
        printf("Gate: (%d, %d)\n", lev->gate_row, lev->gate_col);
        printf("Key:  (%d, %d)\n", lev->key_row, lev->key_col);
    }
    if (lev->trap_count >= 1)
        printf("Trap1: (%d, %d)\n", lev->trap1_row, lev->trap1_col);
    if (lev->trap_count >= 2)
        printf("Trap2: (%d, %d)\n", lev->trap2_row, lev->trap2_col);
    printf("Exit: (%d, %d) mask=0x%x\n", lev->exit_row, lev->exit_col, lev->exit_mask);

    /* Print wall grid */
    printf("\nWalls (col + row*10):\n");
    for (int row = 0; row < N; row++) {
        /* Top walls */
        for (int col = 0; col < N; col++) {
            uint32_t w = lev->walls[col + row * 10];
            printf("+%s", (w & WALL_N) ? "---" : "   ");
        }
        printf("+\n");
        /* Side walls */
        for (int col = 0; col < N; col++) {
            uint32_t w = lev->walls[col + row * 10];
            printf("%s   ", (w & WALL_W) ? "|" : " ");
        }
        {
            uint32_t w = lev->walls[(N-1) + row * 10];
            printf("%s\n", (w & WALL_E) ? "|" : " ");
        }
    }
    /* Bottom border */
    for (int col = 0; col < N; col++) {
        uint32_t w = lev->walls[col + (N-1) * 10];
        printf("+%s", (w & WALL_S) ? "---" : "   ");
    }
    printf("+\n");
}

/* Comparison function for qsort of file names */
static int cmp_filenames(const void *a, const void *b) {
    const char *fa = *(const char **)a;
    const char *fb = *(const char **)b;
    /* Extract number after "B-" */
    const char *na = strstr(fa, "B-");
    const char *nb = strstr(fb, "B-");
    if (na && nb) {
        int ia = atoi(na + 2);
        int ib = atoi(nb + 2);
        return ia - ib;
    }
    return strcmp(fa, fb);
}

typedef struct {
    char file_stem[32];
    int sub_idx;
    int moves;
    int states_explored;
} Result;

static void solve_one_file(const char *dat_dir, const char *file_stem,
                           int sub_idx, int dump, FILE *csv_out) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.dat", dat_dir, file_stem);

    int data_len;
    uint8_t *data = read_file(path, &data_len);
    if (!data) {
        fprintf(stderr, "Cannot read %s\n", path);
        return;
    }

    Header hdr;
    if (parse_header(data, data_len, &hdr) < 0) {
        fprintf(stderr, "Bad header in %s\n", path);
        free(data);
        return;
    }

    if (sub_idx >= hdr.num_sublevels) {
        fprintf(stderr, "%s only has %d sublevels\n", file_stem, hdr.num_sublevels);
        free(data);
        return;
    }

    int offset = 6;
    for (int i = 0; i < sub_idx; i++)
        offset += hdr.bytes_per_sub;

    Level lev;
    if (parse_sublevel(data, offset, data_len, &hdr, &lev) < 0) {
        fprintf(stderr, "Parse error for %s sub %d\n", file_stem, sub_idx);
        free(data);
        return;
    }

    if (dump) {
        dump_level(&lev);
        free(data);
        return;
    }

    printf("Solving %s sublevel %d (grid %d, flip %d)...\n",
           file_stem, sub_idx, lev.grid_size, lev.flip);

    SolveResult res = solve(&lev);
    if (res.moves < 0) {
        printf("  UNSOLVABLE (%d states explored)\n", res.states_explored);
    } else {
        printf("  Solved in %d moves (%d states explored)\n",
               res.moves, res.states_explored);
    }

    if (csv_out) {
        fprintf(csv_out, "%s,%d,%s,%d\n",
                file_stem, sub_idx,
                res.moves >= 0 ? "" : "",
                res.states_explored);
        /* Rewrite with actual value */
        if (res.moves >= 0) {
            /* Seek back and rewrite — actually just print correctly */
            fseek(csv_out, -1, SEEK_CUR); /* undo the newline */
            /* Simpler: just handle it properly */
        }
        /* Let's just do it right from the start */
    }

    free(data);
}

static void solve_all(const char *dat_dir, int jobs __attribute__((unused)),
                      const char *out_path) {
    DIR *dir = opendir(dat_dir);
    if (!dir) {
        fprintf(stderr, "Cannot open directory %s\n", dat_dir);
        return;
    }

    /* Collect B-*.dat files */
    char **files = NULL;
    int n_files = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strncmp(ent->d_name, "B-", 2) == 0 &&
            strstr(ent->d_name, ".dat") != NULL) {
            files = realloc(files, (n_files + 1) * sizeof(char *));
            files[n_files] = strdup(ent->d_name);
            n_files++;
        }
    }
    closedir(dir);

    if (n_files == 0) {
        fprintf(stderr, "No B-*.dat files found in %s\n", dat_dir);
        return;
    }

    qsort(files, n_files, sizeof(char *), cmp_filenames);

    FILE *csv = NULL;
    if (out_path) {
        csv = fopen(out_path, "w");
        if (!csv) {
            fprintf(stderr, "Cannot open %s for writing\n", out_path);
        } else {
            fprintf(csv, "file,sublevel,moves,states_explored\n");
        }
    }

    int total_solved = 0, total_unsolvable = 0, total = 0;
    int max_moves = 0;
    char max_moves_label[64] = "";

    for (int f = 0; f < n_files; f++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dat_dir, files[f]);

        int data_len;
        uint8_t *data = read_file(path, &data_len);
        if (!data) continue;

        Header hdr;
        if (parse_header(data, data_len, &hdr) < 0) {
            free(data);
            continue;
        }

        /* Extract file stem (remove .dat) */
        char stem[32];
        strncpy(stem, files[f], sizeof(stem) - 1);
        stem[sizeof(stem) - 1] = '\0';
        char *dot = strstr(stem, ".dat");
        if (dot) *dot = '\0';

        int offset = 6;
        for (int s = 0; s < hdr.num_sublevels; s++) {
            if (offset + hdr.bytes_per_sub > data_len) break;

            Level lev;
            if (parse_sublevel(data, offset, data_len, &hdr, &lev) < 0) {
                offset += hdr.bytes_per_sub;
                continue;
            }

            SolveResult res = solve(&lev);
            total++;

            if (res.moves >= 0) {
                total_solved++;
                if (res.moves > max_moves) {
                    max_moves = res.moves;
                    snprintf(max_moves_label, sizeof(max_moves_label),
                             "%s sub %d", stem, s);
                }
            } else {
                total_unsolvable++;
            }

            if (csv) {
                if (res.moves >= 0) {
                    fprintf(csv, "%s,%d,%d,%d\n", stem, s, res.moves, res.states_explored);
                } else {
                    fprintf(csv, "%s,%d,,%d\n", stem, s, res.states_explored);
                }
            }

            if (total % 500 == 0) {
                printf("  %d/%d...\n", total, n_files * 100);
            }

            offset += hdr.bytes_per_sub;
        }
        free(data);
    }

    printf("\nSummary: %d solved, %d unsolvable, %d total\n",
           total_solved, total_unsolvable, total);
    if (total_solved > 0) {
        printf("Hardest solved: %s (%d moves)\n", max_moves_label, max_moves);
    }

    if (csv) {
        fclose(csv);
        printf("Results written to %s\n", out_path);
    }

    for (int i = 0; i < n_files; i++) free(files[i]);
    free(files);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *dat_dir = argv[1];
    const char *file_stem = NULL;
    int sub_idx = 0;
    int do_all = 0;
    int jobs = 1;
    const char *out_path = NULL;
    int dump = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            file_stem = argv[++i];
        } else if (strcmp(argv[i], "--sub") == 0 && i + 1 < argc) {
            sub_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--all") == 0) {
            do_all = 1;
        } else if (strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) {
            jobs = atoi(argv[++i]);
            (void)jobs; /* TODO: OpenMP parallelism */
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0) {
            dump = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (do_all) {
        solve_all(dat_dir, jobs, out_path);
    } else if (file_stem) {
        solve_one_file(dat_dir, file_stem, sub_idx, dump, NULL);
    } else {
        fprintf(stderr, "Specify --file or --all\n");
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
