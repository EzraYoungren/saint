#include "saint.h"

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

void vector_print(const Vector *v) {
    printf("Vector(%d x %d): [", v->x, v->y);
    int len = v->x * v->y;
    for (int i = 0; i < len; i++) {
        printf("%.4f", v->data[i]);
        if (i < len - 1) printf(", ");
    }
    printf("]\n");
}

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

Vector create_vector_from_float(int x, int y, float *data) {
  Vector vector = {data, x, y};
  return vector;
}

Vector create_vector_zeros(int x, int y) {
  float *data = calloc(x * y, sizeof(float));
  Vector vector = {data, x, y};
  return vector;
}

void linear(float *x, int xsize, Vector *y, Vector *out) {
  out->x = y->x;
  out->y = 1;
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    1, y->x, xsize,
    1.0,        // alpha
    x, xsize,       // A and its leading dimension
    y->data, y->x,       // B and its leading dimension
    0.0,        // beta
    out->data, y->x        // C and its leading dimension
  );
}

void linearf(float *x, int xsize, Vector *y, float *out) {
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans,
    1, y->x, xsize,
    1.0,        // alpha
    x, xsize,       // A and its leading dimension
    y->data, y->x,       // B and its leading dimension
    0.0,        // beta
    out, y->x        // C and its leading dimension
  );
}
// mank + ind
// M N K
// A=M*K
// B=N*K
// C=M*N
void linear_t1(float *x, int xsize, float *y, int ysize, Vector *out) {
  out->x = ysize;
  out->y = xsize;
  cblas_sgemm(
    CblasRowMajor, CblasTrans, CblasNoTrans,
    xsize, ysize, 1,
    1.0,        // alpha
    x, xsize,       // A and its leading dimension
    y, ysize, // B and its leading dimension
    // because this function is predominately used for the weights, we want to accumulate the gradients
    1.0,        // beta 
    out->data, ysize // C and its leading dimension
  );
}

void linear_t2(float *x, int xsize, Vector *vector, float *out) {
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasTrans,
    1, vector->x, xsize,
    1.0,        // alpha
    x, xsize,       // A and its leading dimension
    vector->data, vector->x,       // B and its leading dimension
    0.0,        // beta
    out, vector->x        // C and its leading dimension
  );
}

float rnsnorm(float *x, int size, float *out) {
    float ms = 0;
    for (int i = 0; i < size; i++)
        ms += x[i] * x[i];
    ms /= size;
    float scale = powf(ms + 1e-5, -0.5);
    for (int i = 0; i < size; i++)
        out[i] = x[i] * scale;
    return ms;
}

void rnsnorm_backward(
    float *dx,       // output: dL/dx (accumulate into this)
    const float *x,  // original input
    const float *dy, // upstream gradient dL/dy
    int size
) {
    // Recompute forward pass values
    float ms = 0;
    for (int i = 0; i < size; i++)
        ms += x[i] * x[i];
    ms /= size;

    float scale  = powf(ms + 1e-5f, -0.5f);
    float scale3 = scale * scale * scale;  // (ms + ε)^(-3/2)

    // Compute dot product: Σ_j (dy_j * x_j)
    float dot = 0;
    for (int j = 0; j < size; j++)
        dot += dy[j] * x[j];

    // Accumulate gradient
    for (int i = 0; i < size; i++)
        dx[i] += scale * dy[i]
               - (x[i] / size) * scale3 * dot;
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

// s:  softmax output from the forward pass (length n)
// g:  upstream gradient dL/ds (length n)
// dz: result dL/dz (length n)
void softmax_backward(const float *s, const float *g, float *dz, int n) {
    float d = cblas_sdot(n, g, 1, s, 1);   // d = g · s
    for (int i = 0; i < n; i++)
        dz[i] = s[i] * (g[i] - d);
}

void init_fp(FP *fp, int pos_id, int total_vocab) {
    fp->x1 = malloc(N_EMBD * sizeof(float));
    fp->x2 = malloc(N_EMBD * sizeof(float));
    for (int li = 0; li < N_LAYER; li++) {
        fp->layers[li].fx1 = malloc(N_EMBD * sizeof(float));
        fp->layers[li].fx2 = calloc(N_EMBD, sizeof(float));
        for (int i = 0; i < N_HEAD; i++)
            fp->layers[li].attn_logits[i] = calloc(pos_id, sizeof(float));
        fp->layers[li].fx3 = malloc(N_EMBD * sizeof(float));
        fp->layers[li].fx4 = malloc(N_EMBD * sizeof(float));
        fp->layers[li].fx5 = malloc(N_EMBD * 4 * sizeof(float));
        fp->layers[li].fx6 = malloc(N_EMBD * 4 * sizeof(float));
        fp->layers[li].fx7 = malloc(N_EMBD * sizeof(float));
        fp->layers[li].fx8 = malloc(N_EMBD * sizeof(float));
        fp->layers[li].q = create_vector_zeros(N_EMBD, 1);
        fp->layers[li].k = create_vector_zeros(N_EMBD, 1);
        fp->layers[li].v = create_vector_zeros(N_EMBD, 1);
    }
    fp->logits = malloc(total_vocab * sizeof(float));
}

void init_layers_zeros(Layer *layers) {
    for (int i = 0; i < N_LAYER; i++) {
        Vector attn_wq = create_vector_zeros(N_EMBD, N_EMBD);
        Vector attn_wk = create_vector_zeros(N_EMBD, N_EMBD);
        Vector attn_wv = create_vector_zeros(N_EMBD, N_EMBD);
        Vector attn_wo = create_vector_zeros(N_EMBD, N_EMBD);
        Vector attn_fc1 = create_vector_zeros(4 * N_EMBD, N_EMBD);
        Vector attn_fc2 = create_vector_zeros(N_EMBD, 4 * N_EMBD);
        Layer layer = { attn_wq, attn_wk, attn_wv, attn_wo, attn_fc1, attn_fc2 };
        layers[i] = layer;
    }
}

void free_layers(Layer *layers) {
    for (int i = 0; i < N_LAYER; i++) {
        free(layers[i].attn_wq.data);
        free(layers[i].attn_wk.data);
        free(layers[i].attn_wv.data);
        free(layers[i].attn_wo.data);
        free(layers[i].attn_fc1.data);
        free(layers[i].attn_fc2.data);
    }
    free(layers);
}

void free_fp(FP *fp) {
    // TODO: figure out why we can't free all these
    free(fp->x1);
    free(fp->x2);
    for (int li = 0; li < N_LAYER; li++) {
        free(fp->layers[li].fx1);
        free(fp->layers[li].fx2);
        for (int i = 0; i < N_HEAD; i++)
            free(fp->layers[li].attn_logits[i]);
        // free(fp->layers[li].fx3);
        // free(fp->layers[li].fx4);
        // free(fp->layers[li].fx5);
        free(fp->layers[li].fx6);
        // free(fp->layers[li].fx7);
        free(fp->layers[li].fx8);
        free(fp->layers[li].q.data);
        free(fp->layers[li].k.data);
        free(fp->layers[li].v.data);
    }
    // free(fp->logits);
}

void adjust_param(Vector *p, Vector *dp, Vector *mp, Vector *vp, float lr_t, int step) {
    for (int i = 0; i < (p->x * p->y); i++) {
        mp->data[i] = BETA1 * mp->data[i] + (1 - BETA2) * dp->data[i];
        vp->data[i] = BETA2 * vp->data[i] + (1 - BETA2) * powf(dp->data[i], 2);
        float m_hat = mp->data[i] / (1 - powf(BETA1, step + 1));
        float v_hat = vp->data[i] / (1 - powf(BETA1, step + 1));
        p->data[i] -= lr_t * m_hat / (pow(v_hat, 0.5) + EPS_ADAM);
        dp->data[i] = 0;
    }
}

void log_vector(Vector *vector, string name) {
  printf("%s: %i, %i\n", name, vector->x, vector->y)
  printf("[\n")
  for (int y = 0; y < vector->y; y++) {
      printf("  [")
      for (int x = 0; x < vector->x; x++) {
          printf("%f, ", vector->data[vector->x * y + x]);
      }
      printf("],\n")
  }
  printf("]")
}

void log_floats(float *floats, int size, string name) {
    printf("%s: %i\n", name, size);
    printf("  [");
    for (int i = 0; i < size; i++) {
        printf("%f, ", floats[i])
    }
    printf("]");
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

    int BOS = vocab_size;       // BOS token id
    int total_vocab = vocab_size + 1; // +1 for BOS 
    int char_to_id[256];
    memset(char_to_id, -1, sizeof(char_to_id));  // -1 = "not in vocab"

    for (int i = 0; i < vocab_size; i++) {
        char_to_id[(unsigned char)uchars[i]] = i;
    }

    // Init Weights
    Layer *layers = malloc(N_LAYER * sizeof(Layer));
    Layer *dlayers = malloc(N_LAYER * sizeof(Layer));
    Layer *mlayers = malloc(N_LAYER * sizeof(Layer));
    Layer *vlayers = malloc(N_LAYER * sizeof(Layer));
    for (int i = 0; i < N_LAYER; i++) {
        { 
            Vector attn_wq = create_vector(N_EMBD, N_EMBD);
            Vector attn_wk = create_vector(N_EMBD, N_EMBD);
            Vector attn_wv = create_vector(N_EMBD, N_EMBD);
            Vector attn_wo = create_vector(N_EMBD, N_EMBD);
            Vector attn_fc1 = create_vector(4 * N_EMBD, N_EMBD);
            Vector attn_fc2 = create_vector(N_EMBD, 4 * N_EMBD);
            Layer layer = { attn_wq, attn_wk, attn_wv, attn_wo, attn_fc1, attn_fc2 };
            layers[i] = layer;
        }
    }
    init_layers_zeros(dlayers);
    init_layers_zeros(mlayers);
    init_layers_zeros(vlayers);

    Vector wte = create_vector(total_vocab, N_EMBD);
    Vector wpe = create_vector(BLOCK_SIZE, N_EMBD);
    Vector lm_head = create_vector(total_vocab, N_EMBD);
    Vector dwte = create_vector_zeros(total_vocab, N_EMBD);
    Vector dwpe = create_vector_zeros(BLOCK_SIZE, N_EMBD);
    Vector dlm_head = create_vector_zeros(total_vocab, N_EMBD);
    Vector mwte = create_vector_zeros(total_vocab, N_EMBD);
    Vector mwpe = create_vector_zeros(BLOCK_SIZE, N_EMBD);
    Vector mlm_head = create_vector_zeros(total_vocab, N_EMBD);
    Vector vwte = create_vector_zeros(total_vocab, N_EMBD);
    Vector vwpe = create_vector_zeros(BLOCK_SIZE, N_EMBD);
    Vector vlm_head = create_vector_zeros(total_vocab, N_EMBD);

    for (int s = 0; s < NUM_STEPS; s++) {
        char *doc = docs[s % num_docs];
        int doc_length = doc_lengths[s % num_docs];
        int *tokens = malloc((doc_length + 2) * sizeof(int));

        tokens[0] = BOS;
        for (int i = 0; i < doc_length; i++) {
          tokens[i + 1] = char_to_id[(unsigned char)doc[i]];
        }
        tokens[doc_length + 1] = BOS;
        
        int n = MIN(BLOCK_SIZE, doc_length + 1);
        
        FP *fp = malloc(n * sizeof(FP));
        FP *dfp = malloc(n * sizeof(FP));
        // GPT
        for (int pos_id = 0; pos_id < n; pos_id++) {
            init_fp(&fp[pos_id], pos_id, total_vocab);
            init_fp(&dfp[pos_id], pos_id, total_vocab);
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            for (int i = 0; i < N_EMBD; i++) {
                fp[pos_id].x1[i] = wte.data[token_id + (wte.x*i)] + wpe.data[pos_id + wpe.x*i];
            }
            rnsnorm(fp[pos_id].x1, N_EMBD, fp[pos_id].x2);

            for (int li = 0; li < N_LAYER; li++) {
                float *x = li == 0 ? fp[pos_id].x2 : fp[pos_id].layers[li - 1].fx8;
                rnsnorm(x, N_EMBD, fp[pos_id].layers[li].fx1);

                // Attention
                linear(fp[pos_id].layers[li].fx1, N_EMBD, &layers[li].attn_wq, &fp[pos_id].layers[li].q);
                linear(fp[pos_id].layers[li].fx1, N_EMBD, &layers[li].attn_wk, &fp[pos_id].layers[li].k);
                linear(fp[pos_id].layers[li].fx1, N_EMBD, &layers[li].attn_wv, &fp[pos_id].layers[li].v);
          
                for (int h = 0; h < N_HEAD; h++) {
                    int hs = h * HEAD_DIM;

                    float *attn_logits = fp[pos_id].layers[li].attn_logits[h];
                    for (int t = 0; t < pos_id; t++) {
                        for (int j = 0; j < HEAD_DIM; j++) {
                            attn_logits[t] += 
                                fp[pos_id].layers[li].q.data[j + hs] * 
                                fp[t].layers[li].k.data[j + hs];
                        }
                        attn_logits[t] /= pow(HEAD_DIM, 0.5);
                    }
                    softmax(attn_logits, pos_id);

                    for (int j = 0; j < HEAD_DIM; j++) {
                        for (int t = 0; t < pos_id; t++) {
                            fp[pos_id].layers[li].fx2[j + hs] += 
                                attn_logits[t] + 
                                fp[t].layers[li].v.data[j + hs];
                        }
                    }
                }
                linearf(fp[pos_id].layers[li].fx2, N_EMBD, &layers[li].attn_wo, fp[pos_id].layers[li].fx3);

                // MLP
                rnsnorm(fp[pos_id].layers[li].fx3, N_EMBD, fp[pos_id].layers[li].fx4);
                linearf(fp[pos_id].layers[li].fx4, N_EMBD, &layers[li].attn_fc1, fp[pos_id].layers[li].fx5);
                for (int i = 0; i < N_EMBD * 4; i++)
                    fp[pos_id].layers[li].fx6[i] = relu(fp[pos_id].layers[li].fx5[i]);

                linearf(fp[pos_id].layers[li].fx6, N_EMBD * 4, &layers[li].attn_fc2, fp[pos_id].layers[li].fx7);
                for (int i = 0; i < N_EMBD; i++)
                    fp[pos_id].layers[li].fx8[i] = fp[pos_id].layers[li].fx7[i] + fp[pos_id].layers[li].fx3[i];
            }

            Vector logits_vec = create_vector_from_float(N_EMBD, 1, fp[pos_id].logits);
            linear(fp[pos_id].layers[N_LAYER - 1].fx8, N_EMBD, &lm_head, &logits_vec);
            softmax(fp[pos_id].logits, total_vocab);
            fp[pos_id].loss = -log(fp[pos_id].logits[target_id]);
        }

        float loss = 0;
        for (int i = 0; i < n; i++) {
            printf("%f\n", fp[i].loss);
            loss += fp[i].loss;
        }
        loss /= n;
        printf("loss: %f\n", loss);

        // Backward Pass
        for (int pos_id = 0; pos_id < n; pos_id++) {
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            dfp[pos_id].logits = fp[pos_id].logits;
            dfp[pos_id].logits[target_id] -= 1;
            linear_t1(fp[pos_id].layers[N_LAYER - 1].fx8, N_EMBD, dfp[pos_id].logits, vocab_size, &dlm_head);

            dfp[pos_id].layers[N_LAYER - 1].fx7 = dfp[pos_id].logits;
            for (int li = N_LAYER - 1; li >= 0; li--) {
                linear_t1(
                    fp[pos_id].layers[li].fx6, 
                    N_EMBD * 4, 
                    dfp[pos_id].layers[li].fx7,
                    N_EMBD, 
                    &dlayers[li].attn_fc2
                );
                linear_t2(dfp[pos_id].layers[li].fx7, N_EMBD, &layers[li].attn_fc2, dfp[pos_id].layers[li].fx6);

                // Relu
                for (int i = 0; i < N_EMBD * 4; i++) {
                    dfp[pos_id].layers[li].fx5[i] = dfp[pos_id].layers[li].fx6[i] * fp[pos_id].layers[li].fx5[i] > 0.0f ? 1 : 0;
                }

                linear_t1(
                    fp[pos_id].layers[li].fx4,
                    N_EMBD, 
                    dfp[pos_id].layers[li].fx5,
                    N_EMBD * 4,
                    &dlayers[li].attn_fc1
                );
                linear_t2(dfp[pos_id].layers[li].fx5, N_EMBD * 4, &layers[li].attn_fc1, dfp[pos_id].layers[li].fx4);

                rnsnorm_backward(dfp[pos_id].layers[li].fx3, fp[pos_id].layers[li].fx3, dfp[pos_id].layers[li].fx4, N_EMBD);

                for (int i = 0; i < N_EMBD; i++) {
                    dfp[pos_id].layers[li].fx3[i] += dfp[pos_id].layers[li].fx7[i];
                }

                linear_t1(fp[pos_id].layers[li].fx2, N_EMBD, dfp[pos_id].layers[li].fx3, N_EMBD, &dlayers[li].attn_wo);
                linear_t2(dfp[pos_id].layers[li].fx3, N_EMBD, &layers[li].attn_wo, dfp[pos_id].layers[li].fx2);

                for (int h = 0; h < N_HEAD; h++) {
                    float *dattn_logits = dfp[pos_id].layers[li].attn_logits[h];
                    int hs = h * HEAD_DIM;
                    for (int j = 0; j < HEAD_DIM; j++) {
                        for (int t = 0; t < pos_id; t++) {
                            dattn_logits[t] += dfp[pos_id].layers[li].fx2[j + hs];
                            dfp[t].layers[li].v.data[j + hs] += dfp[pos_id].layers[li].fx2[j + hs];// d_attn[j + hs];
                        }
                    }
                    softmax_backward(fp[pos_id].layers[li].attn_logits[h], dattn_logits, dattn_logits, pos_id);
                    for (int t = 0; t < pos_id; t++) {
                        for (int j = 0; j < HEAD_DIM; j++) {
                            dfp[pos_id].layers[li].q.data[j + hs] += dattn_logits[t];
                            dfp[t].layers[li].v.data[j + hs] += dattn_logits[t];// d_attn[j + hs];
                        }
                    }
                }
                
                linear_t1(fp[pos_id].layers[li].fx1, N_EMBD, dfp[pos_id].layers[li].v.data, N_EMBD, &dlayers[li].attn_wv);
                linear_t1(fp[pos_id].layers[li].fx1, N_EMBD, dfp[pos_id].layers[li].k.data, N_EMBD, &dlayers[li].attn_wk);
                linear_t1(fp[pos_id].layers[li].fx1, N_EMBD, dfp[pos_id].layers[li].q.data, N_EMBD, &dlayers[li].attn_wq);
                float *fx1_v = malloc(N_EMBD * sizeof(float));
                float *fx1_k = malloc(N_EMBD * sizeof(float));
                float *fx1_q = malloc(N_EMBD * sizeof(float));
                linear_t2(dfp[pos_id].layers[li].v.data, N_EMBD, &layers[li].attn_wv, fx1_v);
                linear_t2(dfp[pos_id].layers[li].k.data, N_EMBD, &layers[li].attn_wk, fx1_k);
                linear_t2(dfp[pos_id].layers[li].q.data, N_EMBD, &layers[li].attn_wq, fx1_q);
                for (int i = 0; i < N_EMBD; i++)
                    dfp[pos_id].layers[li].fx1[i] = fx1_v[i] + fx1_k[i] + fx1_q[i];
                float *x = li == 0 ? dfp[pos_id].x2 : dfp[pos_id].layers[li - 1].fx7;
                rnsnorm_backward(x, fp[pos_id].layers[li].fx1, dfp[pos_id].layers[li].fx1, N_EMBD);
            }
            rnsnorm_backward(dfp[pos_id].x1, fp[pos_id].x2, dfp[pos_id].x2, N_EMBD);
            for (int i = 0; i < N_EMBD; i++) {
                dwte.data[token_id + (wte.x*i)] += fp[pos_id].x1[i];
                dwpe.data[pos_id + (wte.x*i)] += fp[pos_id].x1[i];
            }
        }

        // Adam Optimizer
        float lr_t = LEARNING_RATE * (1 - s / NUM_STEPS);
        for (int li = 0; li < N_LAYER; li++) {
            adjust_param(&layers[li].attn_wq, &dlayers[li].attn_wq, &mlayers[li].attn_wq, &vlayers[li].attn_wq, lr_t, s);
            adjust_param(&layers[li].attn_wk, &dlayers[li].attn_wk, &mlayers[li].attn_wk, &vlayers[li].attn_wk, lr_t, s);
            adjust_param(&layers[li].attn_wv, &dlayers[li].attn_wv, &mlayers[li].attn_wv, &vlayers[li].attn_wv, lr_t, s);
            adjust_param(&layers[li].attn_wo, &dlayers[li].attn_wo, &mlayers[li].attn_wo, &vlayers[li].attn_wo, lr_t, s);
            adjust_param(&layers[li].attn_fc1, &dlayers[li].attn_fc1, &mlayers[li].attn_fc1, &vlayers[li].attn_fc1, lr_t, s);
            adjust_param(&layers[li].attn_fc2, &dlayers[li].attn_fc2, &mlayers[li].attn_fc2, &vlayers[li].attn_fc2, lr_t, s);
        }
        adjust_param(&wte, &dwte, &mwte, &vwte, lr_t, s);
        adjust_param(&wpe, &dwpe, &mwpe, &vwpe, lr_t, s);
        adjust_param(&lm_head, &dlm_head, &mlm_head, &vlm_head, lr_t, s);

        free(tokens);
        // for (int pos_id = 0; pos_id < n; pos_id++) {
        //     free_fp(&fp[pos_id]);
        //     free_fp(&dfp[pos_id]);
        // }
        // free(fp);
        // free(dfp);
    }

    free_layers(layers);
    free_layers(dlayers);
    free_layers(mlayers);
    free_layers(vlayers);

    free(wte.data);
    free(wpe.data);
    free(lm_head.data);

    for (int i = 0; i < num_docs; i++)
        free(docs[i]);
    free(docs);
    free(doc_lengths);
    printf("Finished %i steps", NUM_STEPS);
    return 0;
}
