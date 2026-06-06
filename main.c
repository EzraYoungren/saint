#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define MAX_LINES 32050
#define N_EMBD 16
#define N_LAYER 4
#define N_HEAD 4
#define BLOCK_SIZE 16
#define NUM_STEPS 1 
#define MIN(a, b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#define HEAD_DIM N_EMBD/N_HEAD

static uint64_t s[4] = {0x1123, 0x5621, 0xe4bc, 0xdfee};  // seed these properly

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256ss() {
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1];
    s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

double uniform01() {
    return (xoshiro256ss() >> 11) * 0x1.0p-53;  // 53-bit precision
}

double randn() {
    double u1 = uniform01(), u2 = uniform01();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int m = 8;
int n = 8;
int k = 8;

typedef struct  {
  float *data;
  int x;
  int y;
} Vector;

typedef struct {
  Vector attn_wq;
  Vector attn_wk;
  Vector attn_wv;
  Vector attn_wo;
  Vector attn_fc1;
  Vector attn_fc2;
} Layer;

void randomize_vector(Vector *vector) {
  for (int i = 0; i < vector->x * vector->y; i++) {
    vector->data[i] = randn() * 0.1;
  }
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

Vector create_vector(int x, int y) {
  float *data = calloc(x * y, sizeof(float));
  Vector vector = {data, x, y};
  randomize_vector(&vector);
  return vector;
}

Vector linear(float *x, int xsize, Vector *vector, bool freex) {
  Vector C = create_vector(vector->x, 1);
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    1, vector->x, xsize,
    1.0,        // alpha
    x, xsize,       // A and its leading dimension
    vector->data, vector->x,       // B and its leading dimension
    0.0,        // beta
    C.data, vector->x        // C and its leading dimension
  );
  if (freex) free(x);

  return C;
}

void rnsnorm(float *x, int size) {
    float ms = 0;
    for (int i = 0; i < size; i++)
        ms += x[i] * x[i];
    ms /= size;
    float scale = powf(ms + 1e-5, -0.5);
    printf("scale: %f\n", scale);
    for (int i = 0; i < size; i++)
        x[i] *= scale;
}

void softmax(float *logits, int size) {
    float max_val = logits[0];
    float sum = 0;
    for (int i = 0; i < size; i++)
        if (logits[i] > max_val) 
            max_val = logits[i];
    for (int i = 0; i < size; i++) {
        float ex = expf(logits[i] - max_val);
        logits[i] = ex;
        sum += ex;
    }
    for (int i = 0; i < size; i++)
        logits[i] /= sum;
}

int main() {
    char **docs = malloc(MAX_LINES * sizeof(char *));
    int *doc_lengths = malloc(MAX_LINES * sizeof(int));
    int num_docs = 0;

    FILE *f = fopen("input.txt", "r");
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        // strip newline
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) > 0) {
            doc_lengths[num_docs] = strlen(line);
            docs[num_docs++] = strdup(line);
        }
    }
    fclose(f);
    printf("num docs: %d\n", num_docs);

    int seen[256] = {0};
    char uchars[256];
    int vocab_size = 0;

    for (int i = 0; i < num_docs; i++)
        for (char *c = docs[i]; *c; c++)
            if (!seen[(unsigned char)*c]) {
                seen[(unsigned char)*c] = 1;
                uchars[vocab_size++] = *c;
            }

    // // sort it to match Python's sorted()
    // // simple insertion sort for small vocab
    // for (int i = 1; i < vocab_size; i++) {
    //     char key = uchars[i];
    //     int j = i - 1;
    //     while (j >= 0 && uchars[j] > key)
    //         uchars[j+1] = uchars[j--];
    //     uchars[j+1] = key;
    // }

    int BOS = vocab_size;       // BOS token id
    int total_vocab = vocab_size + 1; // +1 for BOS 
    int char_to_id[256];
    memset(char_to_id, -1, sizeof(char_to_id));  // -1 = "not in vocab"

    for (int i = 0; i < vocab_size; i++) {
        char_to_id[(unsigned char)uchars[i]] = i;
    }

    // Then tokenizing a string is O(1) per character:
    char *word = "hello";
    for (char *c = word; *c; c++) {
        int token = char_to_id[(unsigned char)*c];
        if (token == -1) { /* unknown char */ }
    }

    Layer *layers = malloc(N_LAYER * sizeof(Layer));
    for (int i = 0; i < N_LAYER; i++) {
        Vector attn_wq = create_vector(N_EMBD, N_EMBD);
        Vector attn_wk = create_vector(N_EMBD, N_EMBD);
        Vector attn_wv = create_vector(N_EMBD, N_EMBD);
        Vector attn_wo = create_vector(N_EMBD, N_EMBD);
        Vector attn_fc1 = create_vector(4 * N_EMBD, N_EMBD);
        Vector attn_fc2 = create_vector(N_EMBD, 4 * N_EMBD);
        Layer layer = { attn_wq, attn_wk, attn_wv, attn_wo, attn_fc1, attn_fc2 };
        layers[i] = layer;
    }

    Vector wte = create_vector(total_vocab, N_EMBD);
    Vector wpe = create_vector(BLOCK_SIZE, N_EMBD);
    Vector lm_head = create_vector(total_vocab, N_EMBD);

    for (int step = 0; step < NUM_STEPS; step++) {
        char *doc = docs[step % num_docs];
        int doc_length = doc_lengths[step % num_docs];
        int *tokens = malloc((doc_length + 2) * sizeof(int));

        tokens[0] = BOS;
        for (int i = 0; i < doc_length; i++) {
          tokens[i + 1] = char_to_id[(unsigned char)doc[i]];
        }
        tokens[doc_length + 1] = BOS;
        
        int n = MIN(BLOCK_SIZE, doc_length + 1);
        Vector *keys = malloc(N_LAYER * n * sizeof(Vector));
        Vector *values = malloc(N_LAYER * n * sizeof(Vector));
        int *kv_sizes = calloc(N_LAYER, sizeof(int));
        
        float *losses = malloc(n * sizeof(Vector));
        // GPT
        for (int pos_id = 0; pos_id < n; pos_id++) {
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            float *x = malloc(N_EMBD * sizeof(float));
            for (int i = 0; i < N_EMBD; i++) {
                x[i] = wte.data[token_id + (wte.x*i)] + wpe.data[pos_id + wpe.x*i];
            }
            rnsnorm(x, N_EMBD);

            for (int li = 0; li < N_LAYER; li++) {
                float *x_residual = malloc(N_EMBD * sizeof(float));
                for (int i = 0; i < N_EMBD; i++)
                    x_residual[i] = x[i];

                rnsnorm(x, N_EMBD);

                // Attention
                Vector q = linear(x, N_EMBD, &layers[li].attn_wq, false);
                Vector k = linear(x, N_EMBD, &layers[li].attn_wk, false);
                Vector v = linear(x, N_EMBD, &layers[li].attn_wv, false);
                keys[li*n + pos_id] = k;
                values[li*n + pos_id] = v;
                kv_sizes[li]++;
                float *x_attn = calloc(N_EMBD, sizeof(float));
          
                for (int h = 0; h < N_HEAD; h++) {
                    int hs = h * HEAD_DIM;
                    // k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
                    // v_h = [vi[hs:hs+head_dim] for vi in values[li]]
                    // attn_logits = [sum(q_h[j] * k_h[t][j] 
                    //                  for j in range(head_dim)) / head_dim**0.5 
                    //                    for t in range(len(k_h))]
                    float *attn_logits = calloc(kv_sizes[li], sizeof(float));
                    for (int t = 0; t < kv_sizes[li]; t++) {
                        for (int j = 0; j < HEAD_DIM; j++) {
                            // printf("q.data[j + hs]: %f\n", q.data[j + hs]);
                            // printf("keys[li*n + t].data[j + hs]: %f\n", keys[li*n + t].data[j + hs]);
                            attn_logits[t] += 
                                q.data[j + hs] * 
                                keys[li*n + t].data[j + hs];
                        }
                        attn_logits[t] /= pow(HEAD_DIM, 0.5);
                    }
                    softmax(attn_logits, kv_sizes[li]);
                    // head_out = [sum(attn_weights[t] * v_h[t][j] 
                    //    for t in range(len(v_h))) 
                    //      for j in range(head_dim)]
                    // float *head_out = calloc(kv_sizes[li], sizeof(float));

                    for (int j = 0; j < HEAD_DIM; j++) {
                        for (int t = 0; t < kv_sizes[li]; t++) {
                            x_attn[j + hs] += 
                                attn_logits[t] + 
                                values[li*n + t].data[j + hs];
                        }
                    }
                    free(attn_logits);
                    // free(head_out);
                }
                free(x);
                x = linear(x_attn, N_EMBD, &layers[li].attn_wo, true).data;

                // MLP
                for (int i = 0; i < N_EMBD; i++)
                    x_residual[i] = x[i];
                rnsnorm(x, N_EMBD);
                x = linear(x, N_EMBD, &layers[li].attn_fc1, true).data;
                for (int i = 0; i < N_EMBD * 4; i++)
                    x[i] = relu(x[i]);
                x = linear(x, N_EMBD * 4, &layers[li].attn_fc2, true).data;
                for (int i = 0; i < N_EMBD; i++)
                    x[i] += x_residual[i];
                for (int i = 0; i < N_EMBD; i++) {
                    x[i] = relu(x[i]) + x_residual[i];
                    printf("x[%i]: %f\n", i, x[i]);
                }

                free(x_residual);
                free(q.data);
            }

            Vector logits = linear(x, N_EMBD, &lm_head, false);
            softmax(logits.data, N_EMBD);
            float loss = -log(logits.data[target_id]);
            losses[pos_id] = loss;

            free(logits.data);
            free(x);
        }

        float loss = 0;
        for (int i = 0; i < n; i++)
            loss += losses[i];
        loss /= n;
        printf("loss: %f\n", loss);

        // Backwards pass
        
        for (int i = 0; i < N_LAYER * n; i++) {
            free(keys[i].data);
            free(values[i].data);
        }
        free(losses);
        free(keys);
        free(values);
        free(tokens);
        free(kv_sizes);
    }

    for (int i = 0; i < N_LAYER; i++) {
        free(layers[i].attn_wq.data);
        free(layers[i].attn_wk.data);
        free(layers[i].attn_wv.data);
        free(layers[i].attn_wo.data);
        free(layers[i].attn_fc1.data);
        free(layers[i].attn_fc2.data);
    }
    free(layers);

    free(wte.data);
    free(wpe.data);
    free(lm_head.data);

    for (int i = 0; i < num_docs; i++)
        free(docs[i]);
    free(docs);
    free(doc_lengths);
    return 0;
}
