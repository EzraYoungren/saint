#pragma once

#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#define LOGGING 0 
#define MAX_LINES 32050
#define N_EMBD 16
#define N_LAYER 4
#define N_HEAD 4
#define BLOCK_SIZE 16
#define NUM_STEPS 100000
#define MIN(a, b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#define HEAD_DIM N_EMBD/N_HEAD
#define LEARNING_RATE 0.01
#define BETA1 0.85
#define BETA2 0.99
#define EPS_ADAM 1e-8

typedef struct {
    float *data;
    int x;
    int y;
} Vector;

typedef struct {
    float *fx1;
    float *fx2;
    float *fx3;
    float *fx4;
    float *fx5;
    float *fx6;
    float *fx7;
    float *fx8;
    Vector q;
    Vector k;
    Vector v;
    float *attn_logits[N_HEAD];
} FPLayer;

typedef struct {
  float *x1;
  float *x2;
  FPLayer layers[N_LAYER];
  float *logits;
  float loss;
} FP;

typedef struct {
    Vector attn_wq;
    Vector attn_wk;
    Vector attn_wv;
    Vector attn_wo;
    Vector attn_fc1;
    Vector attn_fc2;
} Layer;

typedef struct {
    float *u;
    float *w;
    float *h1;
    float *h2;
} LayerInfo;

uint64_t xoshiro256ss(void);
double uniform01(void);
double randn(void);

void vector_print(const Vector *v);
void randomize_vector(Vector *vector);
float relu(float x);
Vector create_vector(int x, int y);
Vector create_vector_from_float(int x, int y, float *data);
Vector create_vector_zeros(int x, int y);
void linear(float *x, int xsize, Vector *y, Vector *out);
void linearf(float *x, int xsize, Vector *y, float *out);
void linear_t1(float *x, int xsize, float *y, int ysize, Vector *out);
void linear_t2(float *x, int xsize, Vector *y, float *out);
float rnsnorm(float *x, int size, float *out);
void rnsnorm_backward(float *dx, const float *x, const float *dy, int size);
void softmax(float *logits, int size);
