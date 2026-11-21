#pragma once

// Minimal 8-lane SIMD types for the CPU terrain port. With AVX2 (+FMA) they wrap __m256 / __m256i; otherwise each is
// a plain array and every operation is a scalar loop, so the same terrain code builds on any x86 or ARM target.

#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#define ENGINE_SIMD_AVX2 1
#include <immintrin.h>
#else
#define ENGINE_SIMD_AVX2 0
#endif

namespace simd {

inline constexpr int kLanes = 8;

#if ENGINE_SIMD_AVX2

struct F8;
struct I8;

// Lane mask in float form (all bits set = true).
struct M8 {
  __m256 v;
};

struct F8 {
  __m256 v;
  F8() = default;
  F8(__m256 x) : v(x) {}
  F8(float x) : v(_mm256_set1_ps(x)) {}
  static F8 load(const float *p) { return _mm256_loadu_ps(p); }
  void store(float *p) const { _mm256_storeu_ps(p, v); }
};

struct I8 {
  __m256i v;
  I8() = default;
  I8(__m256i x) : v(x) {}
  I8(int32_t x) : v(_mm256_set1_epi32(x)) {}
  static I8 load(const int32_t *p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)); }
  static I8 iota() { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
  void store(int32_t *p) const { _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v); }
};

inline F8 operator+(F8 a, F8 b) { return _mm256_add_ps(a.v, b.v); }
inline F8 operator-(F8 a, F8 b) { return _mm256_sub_ps(a.v, b.v); }
inline F8 operator*(F8 a, F8 b) { return _mm256_mul_ps(a.v, b.v); }
inline F8 operator/(F8 a, F8 b) { return _mm256_div_ps(a.v, b.v); }
inline F8 operator-(F8 a) { return _mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f)); }
inline F8 &operator+=(F8 &a, F8 b) { return a = a + b; }
inline F8 &operator*=(F8 &a, F8 b) { return a = a * b; }
inline F8 fmadd(F8 a, F8 b, F8 c) { return _mm256_fmadd_ps(a.v, b.v, c.v); }
inline F8 min(F8 a, F8 b) { return _mm256_min_ps(a.v, b.v); }
inline F8 max(F8 a, F8 b) { return _mm256_max_ps(a.v, b.v); }
inline F8 abs(F8 a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a.v); }
inline F8 floor(F8 a) { return _mm256_round_ps(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
inline F8 sqrt(F8 a) { return _mm256_sqrt_ps(a.v); }
inline F8 clamp(F8 x, F8 lo, F8 hi) { return min(max(x, lo), hi); }
inline F8 saturate(F8 x) { return clamp(x, 0.0f, 1.0f); }

inline M8 operator<(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)}; }
inline M8 operator<=(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)}; }
inline M8 operator>(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)}; }
inline M8 operator>=(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)}; }
inline M8 operator==(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)}; }
inline M8 operator&(M8 a, M8 b) { return {_mm256_and_ps(a.v, b.v)}; }
inline M8 operator|(M8 a, M8 b) { return {_mm256_or_ps(a.v, b.v)}; }
inline M8 operator!(M8 a) { return {_mm256_xor_ps(a.v, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))}; }
inline bool any(M8 m) { return _mm256_movemask_ps(m.v) != 0; }
inline bool all(M8 m) { return _mm256_movemask_ps(m.v) == 0xff; }
// select(m, a, b): a where m, else b.
inline F8 select(M8 m, F8 a, F8 b) { return _mm256_blendv_ps(b.v, a.v, m.v); }
inline I8 select(M8 m, I8 a, I8 b) {
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v), m.v));
}

inline I8 operator+(I8 a, I8 b) { return _mm256_add_epi32(a.v, b.v); }
inline I8 operator-(I8 a, I8 b) { return _mm256_sub_epi32(a.v, b.v); }
inline I8 operator*(I8 a, I8 b) { return _mm256_mullo_epi32(a.v, b.v); }
inline I8 operator^(I8 a, I8 b) { return _mm256_xor_si256(a.v, b.v); }
inline I8 operator&(I8 a, I8 b) { return _mm256_and_si256(a.v, b.v); }
inline I8 operator|(I8 a, I8 b) { return _mm256_or_si256(a.v, b.v); }
template <int N> inline I8 srl(I8 a) { return _mm256_srli_epi32(a.v, N); }
template <int N> inline I8 sll(I8 a) { return _mm256_slli_epi32(a.v, N); }
inline I8 min(I8 a, I8 b) { return _mm256_min_epi32(a.v, b.v); }
inline I8 max(I8 a, I8 b) { return _mm256_max_epi32(a.v, b.v); }
inline M8 operator==(I8 a, I8 b) { return {_mm256_castsi256_ps(_mm256_cmpeq_epi32(a.v, b.v))}; }
inline M8 operator>(I8 a, I8 b) { return {_mm256_castsi256_ps(_mm256_cmpgt_epi32(a.v, b.v))}; }

// Truncating / exact conversions (inputs are already integral where it matters).
inline I8 toInt(F8 a) { return _mm256_cvttps_epi32(a.v); }
inline F8 toFloat(I8 a) { return _mm256_cvtepi32_ps(a.v); }
// Unsigned 32-bit to float, as HLSL float(uint): split to keep exactness for values >= 2^31.
inline F8 toFloatUnsigned(I8 a) {
  const __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(a.v, 16));
  const __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(a.v, _mm256_set1_epi32(0xffff)));
  return _mm256_fmadd_ps(hi, _mm256_set1_ps(65536.0f), lo);
}
inline F8 bitsToFloat(I8 a) { return _mm256_castsi256_ps(a.v); }
inline M8 toMask(I8 a) { return {_mm256_castsi256_ps(a.v)}; } // lanes must be 0 or -1
inline I8 maskBits(M8 m) { return _mm256_castps_si256(m.v); } // true -> -1, false -> 0
// a << count per lane; counts >= 32 give 0.
inline I8 sllv(I8 a, I8 count) { return _mm256_sllv_epi32(a.v, count.v); }
// a & ~b
inline I8 andnot(I8 a, I8 b) { return _mm256_andnot_si256(b.v, a.v); }

#else

struct M8 {
  bool v[kLanes];
};

struct F8 {
  float v[kLanes];
  F8() = default;
  F8(float x) {
    for (float &l : v)
      l = x;
  }
  static F8 load(const float *p) {
    F8 r;
    std::memcpy(r.v, p, sizeof(r.v));
    return r;
  }
  void store(float *p) const { std::memcpy(p, v, sizeof(v)); }
};

struct I8 {
  int32_t v[kLanes];
  I8() = default;
  I8(int32_t x) {
    for (int32_t &l : v)
      l = x;
  }
  static I8 load(const int32_t *p) {
    I8 r;
    std::memcpy(r.v, p, sizeof(r.v));
    return r;
  }
  static I8 iota() {
    I8 r;
    for (int i = 0; i < kLanes; ++i)
      r.v[i] = i;
    return r;
  }
  void store(int32_t *p) const { std::memcpy(p, v, sizeof(v)); }
};

#define SIMD_LANEWISE_F(op, expr)                                                                                      \
  inline F8 op {                                                                                                       \
    F8 r;                                                                                                              \
    for (int i = 0; i < kLanes; ++i)                                                                                   \
      r.v[i] = expr;                                                                                                   \
    return r;                                                                                                          \
  }
SIMD_LANEWISE_F(operator+(F8 a, F8 b), a.v[i] + b.v[i])
SIMD_LANEWISE_F(operator-(F8 a, F8 b), a.v[i] - b.v[i])
SIMD_LANEWISE_F(operator*(F8 a, F8 b), a.v[i] * b.v[i])
SIMD_LANEWISE_F(operator/(F8 a, F8 b), a.v[i] / b.v[i])
SIMD_LANEWISE_F(operator-(F8 a), -a.v[i])
SIMD_LANEWISE_F(fmadd(F8 a, F8 b, F8 c), std::fma(a.v[i], b.v[i], c.v[i]))
SIMD_LANEWISE_F(min(F8 a, F8 b), b.v[i] < a.v[i] ? b.v[i] : a.v[i])
SIMD_LANEWISE_F(max(F8 a, F8 b), b.v[i] > a.v[i] ? b.v[i] : a.v[i])
SIMD_LANEWISE_F(abs(F8 a), std::fabs(a.v[i]))
SIMD_LANEWISE_F(floor(F8 a), std::floor(a.v[i]))
SIMD_LANEWISE_F(sqrt(F8 a), std::sqrt(a.v[i]))
SIMD_LANEWISE_F(select(M8 m, F8 a, F8 b), m.v[i] ? a.v[i] : b.v[i])
SIMD_LANEWISE_F(toFloat(I8 a), static_cast<float>(a.v[i]))
SIMD_LANEWISE_F(toFloatUnsigned(I8 a), static_cast<float>(static_cast<uint32_t>(a.v[i])))
#undef SIMD_LANEWISE_F
inline F8 bitsToFloat(I8 a) {
  F8 r;
  std::memcpy(r.v, a.v, sizeof(r.v));
  return r;
}
inline F8 &operator+=(F8 &a, F8 b) { return a = a + b; }
inline F8 &operator*=(F8 &a, F8 b) { return a = a * b; }
inline F8 clamp(F8 x, F8 lo, F8 hi) { return min(max(x, lo), hi); }
inline F8 saturate(F8 x) { return clamp(x, 0.0f, 1.0f); }

#define SIMD_LANEWISE_M(op, expr)                                                                                      \
  inline M8 op {                                                                                                       \
    M8 r;                                                                                                              \
    for (int i = 0; i < kLanes; ++i)                                                                                   \
      r.v[i] = expr;                                                                                                   \
    return r;                                                                                                          \
  }
SIMD_LANEWISE_M(operator<(F8 a, F8 b), a.v[i] < b.v[i])
SIMD_LANEWISE_M(operator<=(F8 a, F8 b), a.v[i] <= b.v[i])
SIMD_LANEWISE_M(operator>(F8 a, F8 b), a.v[i] > b.v[i])
SIMD_LANEWISE_M(operator>=(F8 a, F8 b), a.v[i] >= b.v[i])
SIMD_LANEWISE_M(operator==(F8 a, F8 b), a.v[i] == b.v[i])
SIMD_LANEWISE_M(operator&(M8 a, M8 b), a.v[i] && b.v[i])
SIMD_LANEWISE_M(operator|(M8 a, M8 b), a.v[i] || b.v[i])
SIMD_LANEWISE_M(operator!(M8 a), !a.v[i])
SIMD_LANEWISE_M(operator==(I8 a, I8 b), a.v[i] == b.v[i])
SIMD_LANEWISE_M(operator>(I8 a, I8 b), a.v[i] > b.v[i])
SIMD_LANEWISE_M(toMask(I8 a), a.v[i] != 0)
#undef SIMD_LANEWISE_M
inline bool any(M8 m) {
  for (bool b : m.v)
    if (b)
      return true;
  return false;
}
inline bool all(M8 m) {
  for (bool b : m.v)
    if (!b)
      return false;
  return true;
}

#define SIMD_LANEWISE_I(op, expr)                                                                                      \
  inline I8 op {                                                                                                       \
    I8 r;                                                                                                              \
    for (int i = 0; i < kLanes; ++i)                                                                                   \
      r.v[i] = static_cast<int32_t>(expr);                                                                             \
    return r;                                                                                                          \
  }
#define U32(x) static_cast<uint32_t>(x)
SIMD_LANEWISE_I(operator+(I8 a, I8 b), U32(a.v[i]) + U32(b.v[i]))
SIMD_LANEWISE_I(operator-(I8 a, I8 b), U32(a.v[i]) - U32(b.v[i]))
SIMD_LANEWISE_I(operator*(I8 a, I8 b), U32(a.v[i]) * U32(b.v[i]))
SIMD_LANEWISE_I(operator^(I8 a, I8 b), a.v[i] ^ b.v[i])
SIMD_LANEWISE_I(operator&(I8 a, I8 b), a.v[i] & b.v[i])
SIMD_LANEWISE_I(operator|(I8 a, I8 b), a.v[i] | b.v[i])
SIMD_LANEWISE_I(min(I8 a, I8 b), b.v[i] < a.v[i] ? b.v[i] : a.v[i])
SIMD_LANEWISE_I(max(I8 a, I8 b), b.v[i] > a.v[i] ? b.v[i] : a.v[i])
SIMD_LANEWISE_I(toInt(F8 a), a.v[i])
SIMD_LANEWISE_I(select(M8 m, I8 a, I8 b), m.v[i] ? a.v[i] : b.v[i])
#undef SIMD_LANEWISE_I
inline I8 maskBits(M8 m) {
  I8 r;
  for (int i = 0; i < kLanes; ++i)
    r.v[i] = m.v[i] ? -1 : 0;
  return r;
}
inline I8 sllv(I8 a, I8 count) {
  I8 r;
  for (int i = 0; i < kLanes; ++i)
    r.v[i] = U32(count.v[i]) >= 32u ? 0 : static_cast<int32_t>(U32(a.v[i]) << count.v[i]);
  return r;
}
inline I8 andnot(I8 a, I8 b) {
  I8 r;
  for (int i = 0; i < kLanes; ++i)
    r.v[i] = a.v[i] & ~b.v[i];
  return r;
}
template <int N> inline I8 srl(I8 a) {
  I8 r;
  for (int i = 0; i < kLanes; ++i)
    r.v[i] = static_cast<int32_t>(U32(a.v[i]) >> N);
  return r;
}
template <int N> inline I8 sll(I8 a) {
  I8 r;
  for (int i = 0; i < kLanes; ++i)
    r.v[i] = static_cast<int32_t>(U32(a.v[i]) << N);
  return r;
}
#undef U32

#endif

} // namespace simd
