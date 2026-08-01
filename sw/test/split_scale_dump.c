/* =====================================================================
 * split_scale_dump.c -- print mirage_split_scale's answer for each
 * multiplier in scale_vectors.txt
 *
 * check_scale.py runs model_ref/mirage_int8.split_scale over the same
 * file and diffs. The vectors are hex float literals so both sides see
 * bit-identical doubles rather than whatever their decimal parsers
 * happen to produce.
 *
 * This exists because the ops trace cannot test the requantise
 * arithmetic: every multiplier in test_ops.py is a power of two, so
 * the scale lands on an exact integer and the rounding mode never
 * matters. Changing nearbyint() to floor(0.5 + x) passes the whole ops
 * check untouched -- and would be wrong on the first multiplier that
 * is not a power of two.
 * ===================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mirage.h"

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "scale_vectors.txt";
    char line[256];
    FILE *f = fopen(path, "r");

    if (!f) {
        fprintf(stderr, "split_scale_dump: cannot read %s\n", path);
        return 1;
    }

    while (fgets(line, sizeof line, f)) {
        char *hash;
        double mult;
        uint32_t scale, shift;

        hash = strchr(line, '#');
        if (hash)
            *hash = '\0';
        if (line[strspn(line, " \t\r\n")] == '\0')
            continue;

        mult = strtod(line, NULL);

        if (mirage_split_scale(mult, &scale, &shift) != MIRAGE_OK)
            printf("%s RANGE\n", line[0] ? strtok(line, " \t\r\n") : "?");
        else
            printf("%s %u %u\n", strtok(line, " \t\r\n"), scale, shift);
    }

    fclose(f);
    return 0;
}
