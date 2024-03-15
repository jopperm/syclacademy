/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <algorithm>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sycl/sycl.hpp>

#include <benchmark.h>
#include <image_conv.h>

using namespace sycl;

class image_convolution;

static constexpr util::filter_type filterType = util::filter_type::blur;
static constexpr int filterWidth = 21;

TEST_CASE("image_convolution", "image_convolution") {
  const char *inputImageFile = "../Code_Exercises/Images/tawharanui_2048.png";
  const char *outputImageFile =
      "../Code_Exercises/Images/blurred_tawharanui.png";

  auto inputImage = util::read_image(inputImageFile, 0);

  auto outputImage = util::allocate_image(
      inputImage.width(), inputImage.height(), inputImage.channels());

  auto filter = util::generate_filter(util::filter_type::blur, filterWidth);

  try {
    queue myQueue{};

    std::cout << "Running on "
              << myQueue.get_device().get_info<info::device::name>() << "\n";

    auto width = inputImage.width();
    auto height = inputImage.height();
    auto channels = inputImage.channels();
    auto filterWidth = filter.width();

    auto bufferRange = range<3>(height, width, channels);
    auto filterRange = range<3>(filterWidth, filterWidth, channels);

    {
      auto inputBuf = buffer{inputImage.data(), bufferRange};
      auto outputBuf = buffer<float, 3>{bufferRange};
      auto filterBuf = buffer{filter.data(), filterRange};
      outputBuf.set_final_data(outputImage.data());

      util::benchmark(
          [&]() {
            myQueue.submit([&](handler &cgh) {
              accessor inputAcc{inputBuf, cgh, read_only};
              accessor outputAcc{outputBuf, cgh, write_only};
              accessor filterAcc{filterBuf, cgh, read_only};

              cgh.parallel_for<image_convolution>(
                  nd_range<2>(range<2>(height, width), range<2>(16, 16)),
                  [=](id<2> idx) {
                    auto i = idx[0]; // row
                    auto j = idx[1]; // column

                    auto k = filterWidth / 2;

                    float sum[4] = {0.f, 0.f, 0.f, 0.f};

                    if (!(i < k || i >= height - k || j < k ||
                          j >= width - k)) {
                      constexpr bool coalesceChannels = true;
                      if constexpr (!coalesceChannels) {
                        for (int c = 0; c < 4; ++c)
                          for (int u = -k; u <= k; ++u)
                            for (int v = -k; v <= k; ++v)
                              sum[c] += inputAcc[i + u][j + v][c] *
                                        filterAcc[u + k][v + k][c];
                      } else {
                        for (int u = -k; u <= k; ++u)
                          for (int v = -k; v <= k; ++v)
                            for (int c = 0; c < 4; ++c)
                              sum[c] += inputAcc[i + u][j + v][c] *
                                        filterAcc[u + k][v + k][c];
                      }
                    }

                    for (int c = 0; c < 4; ++c)
                      outputAcc[i][j][c] = sum[c];
                  });
            });

            myQueue.wait_and_throw();
          },
          10, "image convolution");
    }
  } catch (exception e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outputImage, outputImageFile);

  REQUIRE(true);
}
