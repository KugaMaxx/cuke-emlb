#include "denoisor.hpp"
#include "kore.hpp"

#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <cblas.h>
#include <dv-sdk/module.hpp>

namespace edn {

    class ReclusiveEventDenoisor : public EventDenoisor {
    public:
        float_t sigmaS;
        int16_t sigmaT;
        float_t samplarT;
        float_t threshold;

        static const size_t _POLES_  = 4;
        static const size_t _THREAD_ = 8;

        float_t *Xt;
        float_t *Yt;
        float_t *Ut;

        int64_t lastTimestamp{INT64_MAX};
        int64_t sampleTimestamp;

        float_t A[_POLES_ * _POLES_]     = {0}; // _POLES_ * _POLES_
        float_t B[_POLES_ * 1]           = {0}; // _POLES_ * 1
        float_t C[1 * _POLES_]           = {0}; // 1       * _POLES_
        float_t expmA[_POLES_ * _POLES_] = {0}; // _POLES_ * _POLES_
        float_t expmAB[1 * _POLES_]      = {0}; // 1       * _POLES_

        float_t a0 = 1.6800, a1 = -0.6803;
        float_t b0 = 3.7350, b1 = -0.2598;
        float_t w0 = 0.6319, w1 = 1.99700;
        float_t k0 = -1.783, k1 = -1.7230;

        float_t n0, n1, n2, n3;
        float_t d1, d2, d3, d4;
        float_t m1, m2, m3, m4;

        void regenerateParam() {
            // initialize state space filter parameters
            sampleTimestamp = 10000. * pow(10, samplarT);

            std::vector<float_t> sA, sB, sC, sExpmA, sExpmAB;
            if (sigmaT <= 1) {
                sA      = {-0.34814793501919405, -0.10174852241382108, -0.01328291668701808, -0.00084187090564424416, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
                sB      = {1., 0., 0., 0.};
                sC      = {-0.034766724767231791, 0.036125652985477069, -0.016670622089231497, 0.0053166209859268057};
                sExpmA  = {0.66399241926716235, -0.09044021202041419, -0.011394763641191423, -0.00069856717581890022, 0.82977944852996133, 0.952878420794234, -0.0060113792031353148, -0.00037287235776813543, 0.44290918627576731, 0.9839773671329014, 0.99794377606130125, -0.00012825338191932437, 0.15234328809733375, 0.49594718744088817, 0.99947807159646807, 0.99996733926492454};
                sExpmAB = {0.663969918056621, 0.829767541116738, 0.442905109514069, 0.152342252959110};
            } else if (sigmaT > 1 && sigmaT <= 2) {
                sA      = {-0.17098972046164529, -0.024875838219134382, -0.0016052950028390444, -5.0551175578653814e-5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
                sB      = {1., 0., 0., 0.};
                sC      = {-0.01705326369239276, 0.0089085623181199489, -0.0020745048610063166, 0.00031901210274821139};
                sExpmA  = {0.83149935216980708, -0.023534376268960908, -0.0014931868449235045, -4.6269975624019622e-5, 0.91530958665890993, 0.98800788252847815, -0.00076528307081109447, -2.3844939409284761e-5, 0.47169900870423548, 0.99596526829928278, 0.99974179075713077, -8.0670092940545049e-6, 0.15958104241320459, 0.49898572653744755, 0.99993498049319429, 0.99999796540706454};
                sExpmAB = {0.83149935216980708, 0.91530958665890993, 0.47169900870423548, 0.15958104241320459};
            } else if (sigmaT > 2 && sigmaT <= 3) {
                sA      = {-0.11339782076274316, -0.010983598156890194, -0.00047093677600788629, -9.8679937793303037e-6, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
                sB      = {1., 0., 0., 0.};
                sC      = {-0.011384704088379596, 0.003957547550206518, -0.00061538737315228783, 6.226034304359035e-5};
                sExpmA  = {0.88763185414683976, -0.010592731779083792, -0.0004491415555926903, -9.3118014670244656e-6, 0.94363673865797826, 0.99463820390231672, -0.00022820503558614892, -4.7483121665055019e-6, 0.48118313333875529, 0.99820185736638167, 0.99992332607878287, -1.598202102222661e-6, 0.16195815866547131, 0.49954883558616642, 0.99998074069939313, 0.999999598131873};
                sExpmAB = {0.88763185414683976, 0.94363673865797826, 0.48118313333875529, 0.16195815866547131};
            } else if (sigmaT > 3 && sigmaT <= 4) {
                sA      = {-0.084836901067780066, -0.0061589870760125195, -0.00019773696550050819, -3.1047247197182629e-6, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
                sB      = {1., 0., 0., 0.};
                sC      = {-0.0085565041875550821, 0.0022272572448679313, -0.00025988898439293121, 1.9586699556576541e-5};
                sExpmA  = {0.915721166783626, -0.0059954463251460956, -0.00019089402103170859, -2.9735946728021672e-6, 0.95776435634267953, 0.996974926728916, -9.6588032566083786e-5, -1.5086035439597596e-6, 0.48590573405060439, 0.99898709303059741, 0.999967613865094, -5.06507195620297e-7, 0.16314077457606604, 0.49974609180343482, 0.99999187495278241, 0.999999872826808};
                sExpmAB = {0.915721166783626, 0.95776435634267953, 0.48590573405060439, 0.16314077457606604};
            } else if (sigmaT > 4) {
                sA      = {-0.0677710754102731, -0.00393456153400374, -0.000100961390987156, -1.26750756435765e-06, 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.};
                sB      = {1., 0., 0., 0.};
                sC      = {-0.00685692081153018, 0.00142621971013860, -0.000133166630507876, 7.99581545048427e-06};
                sExpmA  = {0.932578063204146, -0.00385123907019436, -9.81713160747396e-05, -1.22470280652005e-06, 0.966229189441337, 0.998060454465382, -4.95508683868674e-05, -6.19473096349716e-07, 0.488733253961807, 0.999351167651091, 0.999983405526808, -2.07679245204314e-07, 0.163848525282422, 0.499837444724584, 0.999995839756070, 0.999999947901832};
                sExpmAB = {0.932578063204146, 0.966229189441337, 0.488733253961807, 0.163848525282422};
            }
            std::copy(sA.begin(), sA.end(), A);
            std::copy(sB.begin(), sB.end(), B);
            std::copy(sC.begin(), sC.end(), C);
            std::copy(sExpmA.begin(), sExpmA.end(), expmA);
            std::copy(sExpmAB.begin(), sExpmAB.end(), expmAB);

            // initialize deriche blur filter parameters
            float_t scale = 1 / (sqrt(2 * M_PI) * sigmaS);

            n0 = a1 + a0;
            n1 = exp(k1 / sigmaS) * (b1 * sin(w1 / sigmaS) - (a1 + 2 * a0) * cos(w1 / sigmaS)) + exp(k0 / sigmaS) * (b0 * sin(w0 / sigmaS) - (a0 + 2 * a1) * cos(w0 / sigmaS));
            n2 = 2 * exp((k0 + k1) / sigmaS) * ((a0 + a1) * cos(w1 / sigmaS) * cos(w0 / sigmaS) - b0 * cos(w1 / sigmaS) * sin(w0 / sigmaS) - b1 * cos(w0 / sigmaS) * sin(w1 / sigmaS)) + a1 * exp(2 * k0 / sigmaS) + a0 * exp(2 * k1 / sigmaS);
            n3 = exp((k1 + 2 * k0) / sigmaS) * (b1 * sin(w1 / sigmaS) - a1 * cos(w1 / sigmaS)) + exp((k0 + 2 * k1) / sigmaS) * (b0 * sin(w0 / sigmaS) - a0 * cos(w0 / sigmaS));

            n0 *= scale, n1 *= scale, n2 *= scale, n3 *= scale;

            d1 = -2 * exp(k1 / sigmaS) * cos(w1 / sigmaS) - 2 * exp(k0 / sigmaS) * cos(w0 / sigmaS);
            d2 = 4 * exp((k0 + k1) / sigmaS) * cos(w1 / sigmaS) * cos(w0 / sigmaS) + exp(2 * k1 / sigmaS) + exp(2 * k0 / sigmaS);
            d3 = -2 * exp((k0 + 2 * k1) / sigmaS) * cos(w1 / sigmaS) - 2 * exp((k1 + 2 * k0) / sigmaS) * cos(w1 / sigmaS);
            d4 = 1 * exp(2 * (k0 + k1) / sigmaS);

            m1 = n1 - d1 * n0;
            m2 = n2 - d2 * n0;
            m3 = n3 - d3 * n0;
            m4 = -d4 * n0;
        };

        void updateStateSpace(float *Xt, float *Yt, float *Ut) {
            float *expmABU = (float *)calloc(_POLES_ * _LENGTH_, sizeof(float));

            // Yt = C * Xt
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, _LENGTH_, _POLES_, 1., C, _POLES_, Xt, _LENGTH_, 0., Yt, _LENGTH_);

            // expmABU = expm(A) * B * Ut
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        _POLES_, _LENGTH_, 1, 1., expmAB, 1, Ut, _LENGTH_, 0., expmABU, _LENGTH_);

            // expmABU = expm(A) * Xt + expmABU
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        _POLES_, _LENGTH_, _POLES_, 1., expmA, _POLES_, Xt, _LENGTH_, 1., expmABU, _LENGTH_);

            // after-processing
            cblas_scopy(_POLES_ * _LENGTH_, expmABU, 1, Xt, 1);
            memset(Ut, 0, _LENGTH_ * sizeof(*Ut));
            free(expmABU);

            return;
        };

        void fastDericheBlur(float *Yt) {
            int32_t *index = (int32_t *)calloc(_THREAD_, sizeof(int32_t));
            float_t *tmpRe = (float_t *)calloc(_THREAD_, sizeof(float_t));
            float_t *tmpYt = (float_t *)calloc(_LENGTH_, sizeof(float_t));

            __m256i mIndex;
            __m256 mPrevIn1, mPrevIn2, mPrevIn3, mPrevIn4;
            __m256 mPrevOut1, mPrevOut2, mPrevOut3, mPrevOut4;

            __m256 mCurIn, mSumIn, mCurOut, mSumOut;
            __m256 mSumN0, mSumN1, mSumN2, mSumN3;
            __m256 mSumD1, mSumD2, mSumD3, mSumD4;

            // from left to right
            for (size_t idx = 0; idx < sizeX / _THREAD_; idx++) {
                for (size_t th = 0; th < _THREAD_; th++) {
                    index[th] = (idx * _THREAD_ + th) * sizeY;
                }
                mIndex = _mm256_loadu_si256((__m256i *)index);

                mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                for (size_t idy = 0; idy < sizeY; idy++) {
                    // In = image
                    mCurIn = _mm256_i32gather_ps(Yt, mIndex, sizeof(float));
                    // PreIn = n0 * In[0] + n1 * In[1] + n2 * In[2] + n3 * In[3]
                    mSumN0 = _mm256_mul_ps(mCurIn, _mm256_set1_ps(n0));
                    mSumN1 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(n1));
                    mSumN2 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(n2));
                    mSumN3 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(n3));
                    mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                    // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4]
                    mSumD1  = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                    mSumD2  = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                    mSumD3  = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                    mSumD4  = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                    mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                    // Out = PreIn - PreOut
                    mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                    _mm256_storeu_ps((float *)tmpRe, mCurOut);
                    for (size_t k = 0; k < _THREAD_; k++) {
                        *(tmpYt + index[k] + idy) += tmpRe[k];
                    }

                    // step
                    mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                    mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                    mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(1));
                }
            }

            // from right to left
            for (size_t idx = 0; idx < sizeX / _THREAD_; idx++) {
                for (size_t th = 0; th < _THREAD_; th++) {
                    index[th] = (idx * _THREAD_ + th + 1) * sizeY - 1;
                }
                mIndex = _mm256_loadu_si256((__m256i *)index);

                mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                for (size_t idy = 0; idy < sizeY; idy++) {
                    // In = image
                    mCurIn = _mm256_i32gather_ps(Yt, mIndex, sizeof(float));
                    // PreIn = m1 * In[1] + m2 * In[2] + m3 * In[3] + m4 * In[4]
                    mSumN0 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(m1));
                    mSumN1 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(m2));
                    mSumN2 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(m3));
                    mSumN3 = _mm256_mul_ps(mPrevIn4, _mm256_set1_ps(m4));
                    mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                    // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4]
                    mSumD1  = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                    mSumD2  = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                    mSumD3  = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                    mSumD4  = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                    mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                    // Out = PreIn - PreOut
                    mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                    _mm256_storeu_ps((float *)tmpRe, mCurOut);
                    for (size_t k = 0; k < _THREAD_; k++) {
                        *(tmpYt + index[k] - idy) += tmpRe[k];
                    }

                    // step
                    mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                    mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                    mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-1));
                }
            }

            // from top to bottom
            for (size_t idy = 0; idy < sizeY / _THREAD_; idy++) {
                for (size_t th = 0; th < _THREAD_; th++) {
                    index[th] = idy * _THREAD_ + th;
                }
                mIndex = _mm256_loadu_si256((__m256i *)index);

                mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                for (size_t idx = 0; idx < sizeX; idx++) {
                    // In = image
                    mCurIn = _mm256_i32gather_ps(tmpYt, mIndex, sizeof(float));
                    // PreIn = n0 * In[0] + n1 * In[1] + n2 * In[2] + n3 * In[3]
                    mSumN0 = _mm256_mul_ps(mCurIn, _mm256_set1_ps(n0));
                    mSumN1 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(n1));
                    mSumN2 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(n2));
                    mSumN3 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(n3));
                    mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                    // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4]
                    mSumD1  = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                    mSumD2  = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                    mSumD3  = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                    mSumD4  = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                    mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                    // Out = PreIn - PreOut
                    mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                    _mm256_storeu_ps((float *)tmpRe, mCurOut);
                    for (size_t k = 0; k < _THREAD_; k++) {
                        *(Yt + index[k] + idx * sizeY) = tmpRe[k];
                    }

                    // step
                    mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                    mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                    mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(sizeY));
                }
            }

            // from bottom to top
            for (size_t idy = 0; idy < sizeY / _THREAD_; idy++) {
                for (size_t th = 0; th < _THREAD_; th++) {
                    index[th] = (sizeX - 1) * sizeY + idy * _THREAD_ + th;
                }
                mIndex = _mm256_loadu_si256((__m256i *)index);

                mPrevIn1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevIn4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                mPrevOut1 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut2 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
                mPrevOut4 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

                for (size_t idx = 0; idx < sizeX; idx++) {
                    // In = image
                    mCurIn = _mm256_i32gather_ps(tmpYt, mIndex, sizeof(float));
                    // PreIn = m1 * In[1] + m2 * In[2] + m3 * In[3] + m4 * In[4]
                    mSumN0 = _mm256_mul_ps(mPrevIn1, _mm256_set1_ps(m1));
                    mSumN1 = _mm256_mul_ps(mPrevIn2, _mm256_set1_ps(m2));
                    mSumN2 = _mm256_mul_ps(mPrevIn3, _mm256_set1_ps(m3));
                    mSumN3 = _mm256_mul_ps(mPrevIn4, _mm256_set1_ps(m4));
                    mSumIn = _mm256_add_ps(_mm256_add_ps(mSumN0, mSumN1), _mm256_add_ps(mSumN2, mSumN3));
                    // PreOut = d1 * Out[1] + d2 * Out[2] + d3 * Out[3] + d4 * Out[4]
                    mSumD1  = _mm256_mul_ps(mPrevOut1, _mm256_set1_ps(d1));
                    mSumD2  = _mm256_mul_ps(mPrevOut2, _mm256_set1_ps(d2));
                    mSumD3  = _mm256_mul_ps(mPrevOut3, _mm256_set1_ps(d3));
                    mSumD4  = _mm256_mul_ps(mPrevOut4, _mm256_set1_ps(d4));
                    mSumOut = _mm256_add_ps(_mm256_add_ps(mSumD1, mSumD2), _mm256_add_ps(mSumD3, mSumD4));
                    // Out = PreIn - PreOut
                    mCurOut = _mm256_sub_ps(mSumIn, mSumOut);

                    _mm256_storeu_ps((float *)tmpRe, mCurOut);
                    for (size_t k = 0; k < _THREAD_; k++) {
                        *(Yt + index[k] - idx * sizeY) += tmpRe[k];
                    }

                    // step
                    mPrevIn4 = mPrevIn3, mPrevIn3 = mPrevIn2, mPrevIn2 = mPrevIn1, mPrevIn1 = mCurIn;
                    mPrevOut3 = mPrevOut2, mPrevOut3 = mPrevOut2, mPrevOut2 = mPrevOut1, mPrevOut1 = mCurOut;
                    mIndex = _mm256_add_epi32(mIndex, _mm256_set1_epi32(-sizeY));
                }
            }

            free(index);
            free(tmpRe);
            free(tmpYt);

            return;
        };

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {
            uint32_t index = evtX * sizeY + evtY;
            if (evtTimestamp - lastTimestamp < 0) {
                lastTimestamp = evtTimestamp;
            } else if (evtTimestamp - lastTimestamp >= sampleTimestamp) {
                updateStateSpace(Xt, Yt, Ut);
                fastDericheBlur(Yt);
                lastTimestamp = evtTimestamp;
            }

            Ut[index] += 1;
            if (Yt[index] + Ut[index] > exp(threshold)) {
                return true;
            }

            return false;
        };
    };

}
