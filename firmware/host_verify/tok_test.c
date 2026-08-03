// Host harness for the on-device BPE encoder: reads one line of text per line
// of stdin, writes the token ids it produces. Pair it with the Python tokenizer
// over real corpus text (src/tok_check.py) to prove the C encoder and the
// training tokenizer agree before any of it is flashed.
//
//   cc -O3 -o /tmp/tok_test firmware/host_verify/tok_test.c
//   uv run python src/tok_check.py
#include <stdio.h>
#include <string.h>
#include "../esp32_llm/bpe.h"
#include "../common/tokenizer.h"

#define MAX_LINE 4096
#define MAX_IDS 4096

int main(void) {
  Bpe bpe = {BPE_BYTE_TOK, BPE_PAIR_KEY, BPE_PAIR_RANK, BPE_PAIR_NEW, BPE_N_PAIRS};
  static char line[MAX_LINE];
  static int ids[MAX_IDS];

  while (fgets(line, sizeof(line), stdin)) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;

    int count = tok_encode(&bpe, line, ids, MAX_IDS);
    for (int i = 0; i < count; i++)
      printf(i ? " %d" : "%d", ids[i]);
    putchar('\n');
    fflush(stdout);
  }
  return 0;
}
