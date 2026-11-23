/* Host-side harness: compiles rf_model.h and replays every dataset row
 * through the REAL compiled C, comparing against sklearn's prediction.
 * This is the check that actually matters - it exercises the compiler,
 * the float literals and the control flow, not a Python imitation of them.
 *
 * build & run:
 *   gcc -O2 -std=c99 -I outputs test_rf_model.c -o /tmp/test_rf
 *   /tmp/test_rf outputs/_vectors.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include "rf_model.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s vectors.csv\n", argv[0]); return 2; }
    FILE *fp = fopen(argv[1], "r");
    if (!fp) { perror("open"); return 2; }

    char line[2048];
    long n = 0, ok = 0;
    int first_bad = -1;

    while (fgets(line, sizeof line, fp)) {
        float x[RF_N_FEATURES];
        int expected;
        char *p = line;
        for (int i = 0; i < RF_N_FEATURES; i++) {
            x[i] = strtof(p, &p);
            if (*p == ',') p++;
        }
        expected = (int)strtof(p, &p);

        uint8_t got = rf_predict(x);
        if (got == expected) ok++;
        else if (first_bad < 0) first_bad = (int)n;
        n++;
    }
    fclose(fp);

    printf("rows replayed through compiled C : %ld\n", n);
    printf("matching sklearn                 : %ld  (%.4f%%)\n",
           ok, 100.0 * (double)ok / (double)n);
    if (ok != n) {
        printf("FAIL - first mismatch at row %d. Do NOT flash.\n", first_bad);
        return 1;
    }
    printf("PASS - compiled firmware logic == validated sklearn model.\n");
    return 0;
}
