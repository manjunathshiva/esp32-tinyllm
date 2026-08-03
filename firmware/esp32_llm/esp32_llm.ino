// PLE TinyLM inference on the ESP32-S3.
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped so the 25M table is read a row at a time from flash; the hot
// tied head plus scratch and KV cache sit in PSRAM. Same llm.h that was verified
// against PyTorch on the host -- only the platform hooks differ here.

#include <stdarg.h>

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_random.h"
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"
#include "../common/tokenizer.h"
#include "vocab.h"
#include "bpe.h"

// Set to 1 once a GMT020-02-7P (2.0" 240x320 ST7789) is wired up — see display.h.
// Leave 0 to run serial-only (no panel needed).
#define USE_DISPLAY 0
#if USE_DISPLAY
#include "display.h"
#endif

// The model continues text; it does not answer questions (TinyStories domain).
// You type an opening line, it writes the rest of the story.
static const int EOT_ID = 0;            // <|endoftext|>
static const int MAX_PROMPT_TOKENS = 96;
static int   g_max_new  = 200;
static float g_temp     = 0.8f;         // matches src/model.py generate()
static int   g_top_k    = 40;
static bool  g_stop_at_eot = true;      // /eot 0 keeps going into a new story
static uint32_t g_rng;

static Bpe g_bpe = {BPE_BYTE_TOK, BPE_PAIR_KEY, BPE_PAIR_RANK, BPE_PAIR_NEW,
                    BPE_N_PAIRS};

// All serial output goes through these two so the device is correct on any
// terminal. A raw-mode terminal (screen, minicom, picocom) does not translate
// LF to CRLF: a bare '\n' drops the cursor one line without returning it to
// column 0, so text walks diagonally down the screen. Terminals whose stdout is
// in cooked mode translate for us and hide the bug, which is exactly why it is
// worth fixing at the source instead of picking a forgiving terminal.
static void out_bytes(const unsigned char *b, int n) {
  for (int i = 0; i < n; i++) {
    if (b[i] == '\n') Serial.write('\r');
    Serial.write(b[i]);
  }
}

static void outf(const char *fmt, ...) {
  static char buf[256];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n > (int)sizeof(buf) - 1) n = (int)sizeof(buf) - 1;
  out_bytes((const unsigned char *)buf, n);
}

// Emit one token to every active output (serial always; TFT when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills. +2 leaves room for a CR.
  if ((int)Serial.availableForWrite() >= len + 2) out_bytes(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
}

Model model;
Scratch s;

// ---- int8 output head (SIMD-friendly) --------------------------------------
// The head is scanned in full every token and dominates runtime. We stage it as
// int8 in PSRAM at boot (int4 nibbles unpacked ONCE), so per token there is no
// nibble unpacking and no float conversion of weights -- just int8 x int8 ->
// int32 dot per row. Its input dim (D=96) is a single group, so one scale per
// row. int8-activation quality was validated on host (val perplexity delta ~0,
// see firmware/host_verify/ppl.c). Output rows split across both LX7 cores.
static int8_t *head_w8 = NULL;      // [rows * cols] unpacked int8 weights (-7..7)
static float  *head_scale8 = NULL;  // [rows] per-row dequant scale
static int head_rows, head_cols;

static int8_t head_actq[128];       // quantized activation, shared by both cores
static float  head_acts;            // its scale

// int8 dot -> int32. Tight and branch-free so the S3 int SIMD / -O3 unrolls it.
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

static void head_rows_range(float *y, int r0, int r1) {
  for (int r = r0; r < r1; r++)
    y[r] = (float)dot_i8(head_actq, head_w8 + (size_t)r * head_cols, head_cols)
           * head_scale8[r] * head_acts;
}

// dual-core plumbing (worker does the first half of the rows on core 0)
static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

// Matches Model.head_matvec (QT*, float*, float*); QT unused (weights staged).
static void head_matvec_int8(const QT *t, const float *x, float *y) {
  (void)t;
  quantize_act(x, head_cols, head_actq, &head_acts);  // once; both cores read it
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { outf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

// Unpack the (row-capped) head from int4 to int8 in PSRAM, once at boot.
static void stage_head_int8(QT *t) {
  head_rows = t->rows; head_cols = t->cols;
  head_w8 = (int8_t *)ps((size_t)head_rows * head_cols);
  head_scale8 = (float *)ps((size_t)head_rows * sizeof(float));
  for (int r = 0; r < head_rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = head_w8 + (size_t)r * head_cols;
    for (int j = 0; j < head_cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    head_scale8[r] = half2float(t->scales[(size_t)r * t->n_groups]);  // n_groups==1
  }
  outf("head staged int8: %.2f MB\n",
                ((size_t)head_rows * head_cols + (size_t)head_rows * 4) / 1e6);
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// Says plainly what this model does. It was trained only on TinyStories, so it
// continues text and cannot answer questions -- better to set that expectation
// in the banner than to let every first-time user ask it one.
static void banner() {
  outf("\nType an opening line and it writes the rest of the story." "\n");
  outf("It continues text; it cannot answer questions or follow instructions." "\n");
  outf("  /temp <f>    sampling temperature (now %.2f, 0 = greedy)\n", g_temp);
  outf("  /topk <n>    top-k cutoff        (now %d)\n", g_top_k);
  outf("  /tokens <n>  max new tokens      (now %d, ctx %d)\n",
                g_max_new, model.c.seq_len);
  outf("  /eot <0|1>   stop at end of story (now %s)\n",
       g_stop_at_eot ? "on" : "off");
  outf("  /help        show this again" "\n");
}

static void prompt_line() { outf("\n> "); }

void setup() {
  Serial.begin(115200);
  delay(1500);
  outf("\n=== ESP32-S3 PLE TinyLM ===" "\n");

  // Map the model partition.
  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { outf("model partition not found" "\n"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { outf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { outf("bad model magic" "\n"); return; }
  Cfg *c = &model.c;
  outf("model: V=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
                part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#endif

  // Cap head rows to the trained vocab BEFORE staging: the tokenizer learned
  // 25,353 entries; the padded rows above that can never be emitted (and have no
  // decode entry), so we neither stage nor score them.
  model.tok_emb.rows = VOCAB_N;
  stage_head_int8(&model.tok_emb);  // int8-staged head; input embedding still uses mmap
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                             &head_worker, 0) != pdPASS) {
    outf("head worker creation failed" "\n");
    return;
  }
  model.head_matvec = head_matvec_int8;

  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, V = c->vocab, S = c->seq_len;
  s.x = (float *)ps(D * 4);
  s.h = (float *)ps((F > D ? F : D) * 4);
  s.qkv = (float *)ps(3 * D * 4);
  s.att = (float *)ps(D * 4);
  s.g1 = (float *)ps(F * 4);
  s.g2 = (float *)ps((P > F ? P : F) * 4);
  s.ple = (float *)ps(L * P * 4);
  s.tmpP = (float *)ps(L * P * 4);
  s.trow = (float *)ps(L * P * 4);
  s.logits = (float *)ps(V * 4);
  s.scores = (float *)ps(S * 4);
  s.kcache = (float *)ps((size_t)L * S * D * 4);
  s.vcache = (float *)ps((size_t)L * S * D * 4);
  outf("PSRAM free after alloc: %u KB\n",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

  g_rng = esp_random() | 1u;
  banner();
  prompt_line();
}

// Continue `text` as a story. The KV cache is rebuilt from position 0 every
// time: seq_len is 256 positions total, which is not enough to carry earlier
// turns as context, so each line is treated as a fresh opening.
static void generate_from(const char *text) {
  static int ids[MAX_PROMPT_TOKENS];
  int n_prompt = tok_encode(&g_bpe, text, ids, MAX_PROMPT_TOKENS);
  if (n_prompt <= 0) { outf("(nothing to tokenize)" "\n"); return; }

  outf("[%d prompt tokens]\n\n", n_prompt);
#if USE_DISPLAY
  display_home();
#endif

  int pos = 0;
  for (int i = 0; i < n_prompt && pos < model.c.seq_len; i++) {
    emit(ids[i]);
    llm_forward(&model, ids[i], pos++, &s);
  }

  llm_profile_reset(&s);
  int64_t t_start = esp_timer_get_time();
  int64_t decode_us = 0;
  int decoded = 0;
  bool hit_eot = false;

  int slides = 0, stories = 0;
  for (int step = 0; step < g_max_new; step++) {
    // Make room before the context fills. Dropping half keeps a rolling window
    // of recent text rather than restarting from nothing.
    if (pos >= model.c.seq_len) {
      int drop = model.c.seq_len / 2;
      llm_kv_slide(&model, &s, drop, pos);
      pos -= drop;
      slides++;
    }

    int tok = llm_sample(s.logits, VOCAB_N, g_temp, g_top_k, &g_rng);
    if (tok == EOT_ID) {
      // In practice this, not the context window, is what ends most runs:
      // TinyStories are short, so the model calls "The end." well before 256
      // positions. Feeding the token through instead of stopping starts a
      // fresh story, which is how you get a long continuous stream.
      if (g_stop_at_eot) { hit_eot = true; break; }
      outf("\n\n");
      stories++;
    }
    emit(tok);  // EOT itself decodes to zero bytes, so this prints nothing
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens, near-free
  }
  int64_t total_us = esp_timer_get_time() - t_start;
  blink(0);

  if (!decoded) { outf("\n(no tokens generated)" "\n"); return; }
  outf("\n\n--- %d tokens in %.2f s%s ---\n", decoded, total_us / 1e6,
                hit_eot ? ", end of story" : "");
  if (slides)
    outf("context slid %d x (rolling %d-token window)\n", slides,
         model.c.seq_len);
  if (stories)
    outf("rolled into %d new stor%s past <|endoftext|>\n", stories,
         stories == 1 ? "y" : "ies");
  outf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    outf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#if USE_DISPLAY
  display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
}

// Slash commands, so temperature and length can be explored without reflashing.
// Returns true if the line was a command.
static bool run_command(const char *line) {
  if (line[0] != '/') return false;
  if (!strncmp(line, "/temp", 5)) {
    g_temp = atof(line + 5);
    outf("temperature = %.2f%s\n", g_temp,
                  g_temp <= 0.f ? " (greedy)" : "");
  } else if (!strncmp(line, "/topk", 5)) {
    g_top_k = atoi(line + 5);
    outf("top_k = %d\n", g_top_k);
  } else if (!strncmp(line, "/tokens", 7)) {
    g_max_new = atoi(line + 7);
    if (g_max_new < 1) g_max_new = 1;
    if (g_max_new > 4000) g_max_new = 4000;   // sanity bound, not a model limit
    outf("max new tokens = %d\n", g_max_new);
    if (g_max_new > model.c.seq_len)
      outf("(past %d the context slides; the model keeps a rolling window "
           "and forgets the start)\n", model.c.seq_len);
  } else if (!strncmp(line, "/eot", 4)) {
    g_stop_at_eot = atoi(line + 4) != 0;
    outf("stop at end of story = %s\n", g_stop_at_eot ? "on" : "off");
  } else if (!strncmp(line, "/help", 5)) {
    banner();
  } else {
    outf("unknown command; try /help" "\n");
  }
  return true;
}

void loop() {
  static char line[256];
  static int len = 0;

  while (Serial.available()) {
    int c = Serial.read();
    if (c == '\n' || c == '\r') {
      outf("\n");
      line[len] = 0;
      if (len > 0 && !run_command(line)) generate_from(line);
      len = 0;
      prompt_line();
    } else if (c == 8 || c == 127) {          // backspace / delete
      if (len > 0) { len--; outf("\b \b"); }
    } else if (c >= 32 && c < 127 && len < (int)sizeof(line) - 1) {
      line[len++] = (char)c;
      Serial.write((char)c);                   // echo, so typing is visible
    }
  }
  delay(2);
}
