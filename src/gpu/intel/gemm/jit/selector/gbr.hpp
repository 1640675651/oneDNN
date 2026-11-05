/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GEMMSTONE_SELECTOR_GBR_HPP
#define GEMMSTONE_SELECTOR_GBR_HPP

// The GBR ranker of the G catalog rows (gbr.db): scores come from the row's
// strategy string and the problem. gbr_data.cpp is generated, together with a
// check that this engine matches the trained model bit-exact.

#include <cstdint>

#include "gemmstone/config.hpp"

GEMMSTONE_NAMESPACE_START

namespace gbr {

// The numeric features, then f_TOKEN (the token in the flags).
constexpr const char *featureNames[] = {"lm", "ln", "lk", "c_wgK", "lchunkA",
        "lchunkB", "copiesA", "copiesB", "lpfA", "lpfB", "ldistA", "ldistB",
        "um", "un", "larea", "tai", "wgM", "wgN", "threads", "wgeff", "lwgs",
        "wgfill", "lkpt", "lkit", "ltraf", "ltcomp", "ltmem", "lslack", "wfrac",
        "gM", "ctzk", "kt_chunk", "f_xaf", "f_st", "f_dm", "f_fx", "f_rr",
        "f_sm", "f_sn", "f_bk0", "f_of", "f_pfaux", "f_pt", "f_bo", "f_hi",
        "f_li"};
constexpr int nFeatures = sizeof(featureNames) / sizeof(*featureNames);

constexpr bool same(const char *a, const char *b) {
    return *a == *b && (!*a || same(a + 1, b + 1));
}

// The engine's limits (64-bit leaf masks); the data static_asserts its trees.
constexpr int maxTrees = 1024, maxLeaves = 64;

struct Scorer {
    uint16_t nTrees;
    uint8_t thrBits, leafBits;
    const uint16_t *thrCount; // per feature: its thresholds in thr
    const double *thr; // sorted by (feature, value)
    const double *scale; // per tree: leaf scale
    const uint64_t *blob; // the packed trees back to back, LSB-first bits
};

// The rows of a precision letter, with or without the K tag, and their scorer.
struct RowClass {
    char precision;
    bool scale2D;
    int scorer;
};

extern const int nDeviceIds;
extern const uint32_t deviceIds[]; // CL_DEVICE_ID_INTEL
extern const int nkMult; // trained on N, K multiples of it (elements)
extern const int mxGroup; // the e8m0 k-group
extern const int nScorers, nRowClasses;
extern const Scorer scorers[];
extern const RowClass rowClasses[];

// The problem as a matmul: src M x K, wei K x N.
struct Shape {
    int64_t m, n, k, batch, eus;
};

struct Row {
    const char *strategy; // the entry's: "aX.. aY.. aB wg MxN[xK] flags.."
    int um, un;
    int threadsPerEU;
    double esizeA, esizeB; // bytes per element in memory
};

// The rows' scorer (rows of one class share it); -1 when there is none.
int scorerOf(char precision, bool scale2D);

// out[i]: rows[i]'s score, lower = better.
void score(int scorer, const Row *rows, int n, const Shape &shape, double *out);

} // namespace gbr

GEMMSTONE_NAMESPACE_END

#endif
