// Portable BPE encoder for the PLE TinyLM: text -> token ids.
//
// vocab.h only ever went the other way (id -> bytes), which is why the sketch
// had to bake its prompt in as literal ids. This is the missing direction, and
// it must reproduce `tokenizers`' ByteLevel+BPE exactly -- a prompt tokenized
// differently from training is silently out of distribution, with no error to
// notice. firmware/host_verify/tok_test.c diffs this against the Python
// tokenizer over a corpus sample for that reason.
//
// Two stages, matching the training config
// (ByteLevel{add_prefix_space:false, use_regex:true} + BPE):
//   1. Pre-tokenize with GPT-2's split pattern. Merges never cross a chunk
//      boundary, so getting this wrong changes the output even when the merge
//      table is perfect.
//   2. Within each chunk, seed one token per byte and repeatedly apply the
//      lowest-rank adjacent merge. Real BPE, not greedy longest-match: greedy
//      matching agrees with BPE often enough to look correct in spot checks and
//      then disagrees on exactly the words that matter.
//
// ASCII limitation: GPT-2's pattern classifies with Unicode properties, this
// classifies with ASCII. Bytes >= 0x80 fall into the "other" class instead of
// being letters. TinyStories is ASCII so the training corpus is unaffected, but
// typed non-ASCII input may tokenize differently than Python would.
#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <stdint.h>
#include <string.h>

// Merge table view. Entries are sorted by `keys` so lookup is a binary search;
// `rank` is BPE merge priority (lower wins), `newtok` the resulting token.
typedef struct {
  const uint16_t *byte_tok;  // [256] byte -> seed token id
  const uint32_t *keys;      // [n] sorted (id_a << 16) | id_b
  const uint16_t *rank;      // [n]
  const uint16_t *newtok;    // [n]
  int n;
} Bpe;

#define TOK_NO_RANK 0xFFFF

static inline int tok_is_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
static inline int tok_is_alpha(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static inline int tok_is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

// Rank of the pair (a, b), or TOK_NO_RANK if it is not a merge. Sets *out to
// the merged token when found.
static int tok_pair_rank(const Bpe *bpe, uint16_t a, uint16_t b, uint16_t *out) {
  uint32_t key = ((uint32_t)a << 16) | (uint32_t)b;
  int lo = 0, hi = bpe->n - 1;
  while (lo <= hi) {
    int mid = lo + ((hi - lo) >> 1);
    uint32_t k = bpe->keys[mid];
    if (k == key) { *out = bpe->newtok[mid]; return bpe->rank[mid]; }
    if (k < key) lo = mid + 1; else hi = mid - 1;
  }
  return TOK_NO_RANK;
}

// GPT-2 contractions ('s 't 're 've 'm 'll 'd), matched before everything else.
// Returns the chunk length, or 0 if none applies.
static int tok_contraction(const char *s, int n, int i) {
  if (s[i] != '\'' || i + 1 >= n) return 0;
  char a = s[i + 1];
  char b = (i + 2 < n) ? s[i + 2] : 0;
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
  if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  return 0;
}

// Length of the pre-token starting at i, following GPT-2's alternation order:
//   's|'t|'re|'ve|'m|'ll|'d | ' ?\p{L}+' | ' ?\p{N}+' | ' ?[^\s\p{L}\p{N}]+'
//   | '\s+(?!\S)' | '\s+'
// The optional leading space is a literal space, not any whitespace, which is
// why a chunk can absorb " word" but never "\nword".
static int tok_chunk_len(const char *s, int n, int i) {
  int c = tok_contraction(s, n, i);
  if (c) return c;

  unsigned char here = (unsigned char)s[i];
  int j = i + (here == ' ' ? 1 : 0);

  if (j < n) {
    unsigned char cj = (unsigned char)s[j];
    if (tok_is_alpha(cj)) {
      int k = j;
      while (k < n && tok_is_alpha((unsigned char)s[k])) k++;
      return k - i;
    }
    if (tok_is_digit(cj)) {
      int k = j;
      while (k < n && tok_is_digit((unsigned char)s[k])) k++;
      return k - i;
    }
    if (!tok_is_space(cj)) {
      int k = j;
      while (k < n && !tok_is_space((unsigned char)s[k])
             && !tok_is_alpha((unsigned char)s[k])
             && !tok_is_digit((unsigned char)s[k])) k++;
      if (k > j) return k - i;
    }
  }

  // Whitespace run. '\s+(?!\S)' means a run followed by a non-space keeps its
  // last space for the following chunk's optional leading space.
  int k = i;
  while (k < n && tok_is_space((unsigned char)s[k])) k++;
  if (k < n && k - i > 1) k--;
  return k > i ? k - i : 1;
}

// Encode one pre-token chunk, appending ids to out. Returns tokens written.
static int tok_encode_chunk(const Bpe *bpe, const char *s, int len,
                            int *out, int max_out) {
  if (len <= 0 || max_out <= 0) return 0;

  // Seed one token per byte, capped by the caller's buffer.
  int count = len < max_out ? len : max_out;
  for (int i = 0; i < count; i++)
    out[i] = bpe->byte_tok[(unsigned char)s[i]];

  // Repeatedly apply the single lowest-rank adjacent merge.
  while (count > 1) {
    int best_rank = TOK_NO_RANK, best_at = -1;
    uint16_t best_new = 0;
    for (int p = 0; p + 1 < count; p++) {
      uint16_t merged;
      int r = tok_pair_rank(bpe, (uint16_t)out[p], (uint16_t)out[p + 1], &merged);
      if (r < best_rank) { best_rank = r; best_at = p; best_new = merged; }
    }
    if (best_at < 0) break;
    out[best_at] = best_new;
    memmove(&out[best_at + 1], &out[best_at + 2],
            (size_t)(count - best_at - 2) * sizeof(int));
    count--;
  }
  return count;
}

// Encode NUL-terminated text. Returns the number of ids written (<= max_out).
static int tok_encode(const Bpe *bpe, const char *text, int *out, int max_out) {
  int n = (int)strlen(text), i = 0, total = 0;
  while (i < n && total < max_out) {
    int len = tok_chunk_len(text, n, i);
    total += tok_encode_chunk(bpe, text + i, len, out + total, max_out - total);
    i += len;
  }
  return total;
}

#endif
