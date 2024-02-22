/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <cmath>
#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

int main() {
  constexpr size_t dataSize = 1024;

  std::array<float, dataSize> a, r_seq, r_sycl;
  for (int i = 0; i < dataSize; ++i) {
    a[i] = i;
    r_seq[i] = r_sycl[i] = 0.0f;
  }

  // Sequential version
  {
    for (int i = 0; i < dataSize; ++i) {
      r_seq[i] = std::sqrt(a[i]);
    }
  }

  // SYCL version
  {
    auto defaultQueue = sycl::queue{};

    sycl::buffer bufA{a.data(), sycl::range<1>{dataSize}};
    sycl::buffer bufR{r_sycl.data(), sycl::range<1>{dataSize}};

    defaultQueue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor accA{bufA, cgh, sycl::read_only};
          sycl::accessor accR{bufR, cgh, sycl::write_only};

          cgh.parallel_for<class vector_sqrt>(
              sycl::range<1>{dataSize},
              [=](sycl::id<1> idx) { accR[idx] = sycl::sqrt(accA[idx]); });
        })
        .wait();
  }

  for (int i = 0; i < dataSize; ++i) {
    if (std::abs(r_seq[i] - r_sycl[i]) > 0.001f) {
      std::cerr << "Output mismatch at r[" << i << "]: Expected: " << r_seq[i]
                << " Got: " << r_sycl[i] << '\n';
      return 1;
    }
  }

  return 0;
}
