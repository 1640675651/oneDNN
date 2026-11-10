/*******************************************************************************
 * Case 5.6: keyed BLAKE3 b3_compress64 on 16 wrap-GRF slot words per logical tile.
 * Message: the 16 dwords in tileXorWrapGrf[fi*2 .. fi*2+1] (same as 5.5 staging).
 *
 * One tile per thread, so the hash runs on one SIMD lane: ~800 SIMD1 instructions per
 * thread (the message permutation is applied statically, so no words are moved). This is
 * the structural cost of fusing the hash into a gemmstone kernel; a separate SIMD16 kernel
 * hashes 16 tiles per instruction. See note.txt for the measurements and the variants tried.
 ******************************************************************************/

#include "alloc_utils.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "gemmstone/generator.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;

namespace {

constexpr int kBlake3MsgWords = 16;
constexpr int kBlake3DigestWords = 8;

constexpr uint32_t kBlake3IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                   0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
constexpr uint8_t kBlake3Perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
// Block flags: CHUNK_START | CHUNK_END | ROOT | KEYED_HASH; block length 64.
constexpr uint32_t kBlake3Flags = 0x1Bu;
constexpr uint32_t kBlake3BlockLen = 64;

inline int grfDwords(HW hw) {
    return GRF::bytes(hw) / int(sizeof(uint32_t));
}

template <HW hw>
Subregister grfUd(GRF &grf, int dword) {
    return grf.ud(dword % grfDwords(hw));
}

// The 7 rounds. v(i) / m(i) give state and message word i; tmp(k) is the temporary of
// G chain k. The message schedule is applied statically: round r reads word perm_r[i].
// The four G functions of a half-round are independent and are emitted in lockstep so
// the in-order EU has 4 chains to overlap.
template <typename GeneratorT, typename OpV, typename OpM, typename OpT>
void emitBlake3Rounds(GeneratorT &g, OpV v, OpM m, OpT tmp, int esize = 1) {
    int perm[16];
    for (int i = 0; i < 16; i++) perm[i] = i;
    auto G4 = [&](const int (&a)[4], const int (&b)[4], const int (&c)[4], const int (&d)[4],
                  const int (&mx)[4], const int (&my)[4]) {
        for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), v(b[k]));
        for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), m(mx[k]));
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(d[k]), v(a[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(d[k]), tmp(k), uint16_t(16));
        for (int k = 0; k < 4; k++) g.add(esize, v(c[k]), v(c[k]), v(d[k]));
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(b[k]), v(c[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(b[k]), tmp(k), uint16_t(12));
        for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), v(b[k]));
        for (int k = 0; k < 4; k++) g.add(esize, v(a[k]), v(a[k]), m(my[k]));
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(d[k]), v(a[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(d[k]), tmp(k), uint16_t(8));
        for (int k = 0; k < 4; k++) g.add(esize, v(c[k]), v(c[k]), v(d[k]));
        for (int k = 0; k < 4; k++) g.xor_(esize, tmp(k), v(b[k]), v(c[k]));
        for (int k = 0; k < 4; k++) g.ror(esize, v(b[k]), tmp(k), uint16_t(7));
    };
    const int ca[4] = {0, 1, 2, 3}, cb[4] = {4, 5, 6, 7}, cc[4] = {8, 9, 10, 11}, cd[4] = {12, 13, 14, 15};
    const int da[4] = {0, 1, 2, 3}, db[4] = {5, 6, 7, 4}, dc[4] = {10, 11, 8, 9}, dd[4] = {15, 12, 13, 14};
    for (int round = 0; round < 7; ++round) {
        const int mx0[4] = {perm[0], perm[2], perm[4], perm[6]},   my0[4] = {perm[1], perm[3], perm[5], perm[7]};
        const int mx1[4] = {perm[8], perm[10], perm[12], perm[14]}, my1[4] = {perm[9], perm[11], perm[13], perm[15]};
        G4(ca, cb, cc, cd, mx0, my0);    // columns
        G4(da, db, dc, dd, mx1, my1);    // diagonals
        int next[16];
        for (int i = 0; i < 16; i++) next[i] = perm[kBlake3Perm[i]];
        for (int i = 0; i < 16; i++) perm[i] = next[i];
    }
}

// One tile's compression: key + IV + block parameters in vGrf, message in mGrf, digest in outGrf.
template <HW hw, typename GeneratorT>
void emitBlake3Compress64(GeneratorT &g, GEMMState &state, const GRF &keyGrf, const GRF mGrfIn[2],
                          GRF &outGrf) {
    GRF vGrf[2] = {state.ra.alloc(), state.ra.alloc()};
    GRF tmp = state.ra.alloc();              // 4 scalar temporaries, one per G chain

    // State word i is dword i of the vGrf pair: words 8..15 sit in vGrf[1] on 32-byte GRFs but in
    // vGrf[0] on 64-byte GRFs (Xe2), so every access goes through v(i) rather than a fixed GRF.
    auto v = [&](int i) { return grfUd<hw>(vGrf[i / grfDwords(hw)], i); };
    for (int i = 0; i < 8; ++i)
        g.mov(1, v(i), keyGrf.ud(i));
    for (int i = 0; i < 4; ++i)
        g.mov(1, v(8 + i), uint32_t(kBlake3IV[i]));
    g.mov(1, v(12), uint32_t(0));
    g.mov(1, v(13), uint32_t(0));
    g.mov(1, v(14), uint32_t(kBlake3BlockLen));
    g.mov(1, v(15), uint32_t(kBlake3Flags));

    GRF mGrf[2] = {mGrfIn[0], mGrfIn[1]};
    emitBlake3Rounds(g, v,
            [&](int i) { return grfUd<hw>(mGrf[i / grfDwords(hw)], i); },
            [&](int k) { return tmp.ud(k); });
    for (int i = 0; i < 8; ++i)
        g.xor_(1, outGrf.ud(i), v(i), v(8 + i));

    state.ra.safeRelease(tmp);
    state.ra.safeRelease(vGrf[1]);
    state.ra.safeRelease(vGrf[0]);
}

} // namespace

template <HW hw>
void Generator<hw>::gemmCase5ReleaseAccForBlake3(const GEMMStrategy &strategy, GEMMState &state) {
    (void)strategy;
    safeReleaseRanges(state.C_regs, state);
    safeReleaseRanges(state.Cr_regs, state);
    safeReleaseRanges(state.C_addrs[0], state);
    safeReleaseRanges(state.C_addrs[1], state);
    safeReleaseRanges(state.Cp_regs, state);
    safeReleaseRanges(state.Cp_addrs, state);
    safeReleaseRanges(state.A_regs, state);
    safeReleaseRanges(state.B_regs, state);
    safeReleaseRanges(state.Ar_regs, state);
    safeReleaseRanges(state.Br_regs, state);
    safeReleaseRanges(state.Ap_regs, state);
    safeReleaseRanges(state.Bp_regs, state);
    safeReleaseRanges(state.tempMul_regs, state);
    safeReleaseRanges(state.tileXorSlmAddrs, state);
    for (auto &f : state.tileXorFolds) {
        if (f.isValid()) {
            state.ra.safeRelease(f);
            f.invalidate();
        }
    }
    if (state.tileXorFold.isValid()) {
        state.ra.safeRelease(state.tileXorFold);
        state.tileXorFold.invalidate();
    }
}

template <HW hw>
void Generator<hw>::gemmCase5TileXorBlake3FromGrf(const GEMMProblem &problem,
                                                  const GEMMStrategy &strategy, GEMMState &state) {
    if (!problem.case5TileXor || !problem.case5TileXorBlake3 || !problem.case5TileXorWrapGrf) {
        return;
    }

    const int subGridM = (problem.case5XorSubGridM > 0) ? problem.case5XorSubGridM : 1;
    const int subGridN = (problem.case5XorSubGridN > 0) ? problem.case5XorSubGridN : 1;
    const bool split = (subGridM > 1) || (subGridN > 1);
    const int nFolds = split ? subGridM * subGridN : 1;
    const int dpg = grfDwords(hw);

    // Case 5.7: the message blocks were staged in SLM; fold them into tileXorFoldAcc first.
    if (problem.case5TileXorAccSlm)
        case5TileXorLoadAccFromSlm(problem, strategy, state);

    for (auto &g : state.tileXorWrapGrf) {
        if (g.isValid()) {
            state.ra.claim(g);
        }
    }

    // Spatial id of fold fi (the thread's own for the unsplit case).
    auto spatialOf = [&](int fi, const Subregister &out) {
        if (!split) {
            mov(1, out, state.spatialId);
            return;
        }
        const int sr = fi / subGridN, sc = fi % subGridN;
        auto logicalRow = state.ra.alloc_sub<uint32_t>();
        auto logicalCol = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, logicalRow, state.tileXorRow, subGridM);
        add(1, logicalRow, logicalRow, uint16_t(sr));
        mulConstant(1, logicalCol, state.tileXorCol, subGridN);
        add(1, logicalCol, logicalCol, uint16_t(sc));
        emad(1, out, logicalCol, logicalRow, state.inputs.tileCols, strategy, state);
        state.ra.safeRelease(logicalCol);
        state.ra.safeRelease(logicalRow);
    };

    // Slot word w of tile fi: the wrap-GRF staging, or, on the interleaved path, the per-sub-tile
    // accumulator of the 16-milestone line blocks (same 16 dwords, laid out like the lines).
    const int accGrfs = std::max(1, 16 * 4 / GRF::bytes(hw));
    auto slotWord = [&](int fi, int w) -> Subregister {
        if (state.tileXorFoldAcc.isValid())
            return state.tileXorFoldAcc[fi * accGrfs + w / dpg].ud(w % dpg);
        return state.tileXorWrapGrf[fi * 2 + w / dpg].ud(w % dpg);
    };

    GRF stagedMsg[2] = {state.ra.alloc(getHint(HintType::LongTerm, strategy)),
                        state.ra.alloc(getHint(HintType::LongTerm, strategy))};
    auto stageWrapMsg = [&](int fi) {
        // Slot words are at fixed positions: read the subregisters directly.
        for (int w = 0; w < kBlake3MsgWords; ++w)
            mov(1, stagedMsg[w / dpg].ud(w % dpg), slotWord(fi, w));
    };
    stageWrapMsg(0);

    gemmCase5ReleaseAccForBlake3(strategy, state);

    GRF keyGrf = state.ra.alloc();
    // problem.case5JackpotArgsInBuf: the key is the first 32 bytes of blake3_out (64-byte header) and the
    // digests follow at byte 64; otherwise it comes from the blake3_k* kernel arguments.
    const uint32_t digestBase = problem.case5JackpotArgsInBuf ? 64u : 0u;
    if (problem.case5JackpotArgsInBuf) {
        auto khdr = state.ra.alloc_range(1);
        emov(1, khdr[0].uq(0), state.inputs.blake3Out, strategy, state);
        if (hw >= HW::XeHPG)
            load(1, GRFRange(keyGrf.getBase(), 1), block(D32, 8), A64, khdr);
        else
            load(16, keyGrf, block_oword(2), A64, khdr[0]);
        state.ra.safeRelease(khdr);
    }
    for (int i = 0; i < kBlake3DigestWords && !problem.case5JackpotArgsInBuf; ++i) {
        if (state.inputs.blake3KeyWords[i].isValid()) {
            mov(1, keyGrf.ud(i), state.inputs.blake3KeyWords[i]);
        } else {
            mov(1, keyGrf.ud(i), uint32_t(0xA5A5A5A5u ^ static_cast<uint32_t>(i * 0x9E3779B9u)));
        }
    }

    // 8 contiguous digest words at blake3_out + spatial*32 (+64 behind the key header): one SIMD8
    // scattered store, or on Xe2 (no legacy data port) one LSC block store of the 8 dwords.
    auto storeDigestAtSpatial = [&](const Subregister &spatial, const GRF &digest) {
        if (hw >= HW::Xe2) {
            auto hdr = state.ra.alloc_range(2);
            auto base = state.ra.alloc_sub<uint32_t>();
            mulConstant(1, base, spatial, kBlake3DigestWords * 4);
            if (digestBase) add(1, base, base, digestBase);
            eadd(1, hdr[0].uq(0), state.inputs.blake3Out, base, strategy, state);
            store(1, block(D32, kBlake3DigestWords), A64, hdr, digest);
            state.ra.safeRelease(base);
            state.ra.safeRelease(hdr);
            return;
        }
        auto hdr = state.ra.alloc_range(2);
        auto off = state.ra.alloc();
        auto t = state.ra.alloc();
        mov(8, t.uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));   // 4-bit vector: word dst
        mov(8, off.ud(), t.uw(0)(1));
        state.ra.safeRelease(t);
        shl(8, off.ud(), off.ud(), uint16_t(2));
        auto base = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, base, spatial, kBlake3DigestWords * 4);
        if (digestBase) add(1, base, base, digestBase);
        add(8, off.ud(), off.ud(), base);
        eadd(8, hdr[0].uq(), state.inputs.blake3Out, off.ud(), strategy, state);
        store(8, scattered_dword(), A64, hdr, digest);
        state.ra.safeRelease(base);
        state.ra.safeRelease(off);
        state.ra.safeRelease(hdr);
    };

    // SIMD-N (default for N = subGridM x subGridN logical tiles per thread, 2..8, the same design as
    // CPPminer's fused judge): the N tiles are hashed together, one per lane, in a single compression of
    // the next power-of-two width (6 tiles: SIMD8, two idle lanes) instead of N one-lane compressions,
    // then each lane's digest is written as below. CASE5_JACKPOT_SIMDN=0 keeps one compression per tile.
    {
        const char *simdEnv = std::getenv("CASE5_JACKPOT_SIMDN");
        const int N = nFolds;
        if (split && !(simdEnv && simdEnv[0] == '0') && N >= 2 && N <= 8 && N <= dpg
                && state.inputs.blake3Out.isValid()) {
            int Nx = 2;
            while (Nx < N) Nx *= 2;
            GRFRange m = state.ra.alloc_range(kBlake3MsgWords), v = state.ra.alloc_range(16);
            GRFRange t = state.ra.alloc_range(4);
            // Wrap slot w of tile fi -> lane fi of message register w.
            for (int fi = 0; fi < N; fi++)
                for (int w = 0; w < kBlake3MsgWords; w++)
                    mov(1, m[w].ud(fi), slotWord(fi, w));
            auto V = [&](int i) { return v[i].ud(); };
            auto Mw = [&](int w) { return m[w].ud(); };
            auto T = [&](int k) { return t[k].ud(); };
            for (int i = 0; i < kBlake3DigestWords; ++i) mov(Nx, V(i), keyGrf.ud(i));
            for (int i = 0; i < 4; i++) mov(Nx, V(8 + i), uint32_t(kBlake3IV[i]));
            mov(Nx, V(12), uint32_t(0));
            mov(Nx, V(13), uint32_t(0));
            mov(Nx, V(14), uint32_t(64));      // block length
            mov(Nx, V(15), uint32_t(0x1Bu));   // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
            emitBlake3Rounds(*this, V, Mw, T, Nx);
            for (int i = 0; i < kBlake3DigestWords; i++) xor_(Nx, V(i), V(i), V(8 + i));   // digest in V(0..7)
            state.ra.safeRelease(m);
            GRF digest = state.ra.alloc();
            auto spatial = state.ra.alloc_sub<uint32_t>();
            for (int fi = 0; fi < N; fi++) {
                for (int w = 0; w < kBlake3DigestWords; w++) mov(1, digest.ud(w), v[w].ud(fi));
                spatialOf(fi, spatial);
                storeDigestAtSpatial(spatial, digest);
            }
            state.ra.safeRelease(spatial);
            state.ra.safeRelease(digest);
            state.ra.safeRelease(t);
            state.ra.safeRelease(v);
            state.ra.safeRelease(keyGrf);
            state.ra.safeRelease(stagedMsg[1]);
            state.ra.safeRelease(stagedMsg[0]);
            return;
        }
    }

    auto hashFold = [&](int fi) {
        if (fi != 0) stageWrapMsg(fi);
        GRF digest = state.ra.alloc();
        auto spatial = state.ra.alloc_sub<uint32_t>();
        spatialOf(fi, spatial);
        emitBlake3Compress64<hw>(*this, state, keyGrf, stagedMsg, digest);
        storeDigestAtSpatial(spatial, digest);
        state.ra.safeRelease(spatial);
        state.ra.safeRelease(digest);
    };

    for (int fi = 0; fi < nFolds; fi++)
        hashFold(fi);

    state.ra.safeRelease(keyGrf);
    state.ra.safeRelease(stagedMsg[1]);
    state.ra.safeRelease(stagedMsg[0]);
}

GEMMSTONE_NAMESPACE_END
