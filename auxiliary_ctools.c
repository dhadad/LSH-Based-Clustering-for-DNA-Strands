#include <math.h>
#include <string.h>
#include <stdlib.h>

long* single_numset(const char* seq, int q) {
    long* numset = malloc((strlen(seq) - q + 1) * sizeof(long));
    for (unsigned int i = 0; i < strlen(seq) - q + 1; ++i) {
        long tot = 0;
        for (unsigned int pos = i; pos < i + q; ++pos) {
            switch (seq[pos]) {
                case 'C':
                    tot += pow(4, pos - i);
                    break;
                case 'G':
                    tot += 2 * pow(4, pos - i);
                    break;
                case 'T':
                    tot += 3 * pow(4, pos - i);
                    break;
                default:
                    break;
            }
        }
        numset[i] = tot;
    }
    return numset;
}