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
static constexpr int filterWidth = 11;

TEST_CASE("image_convolution_local", "image_convolution_local") {
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

    auto bufferRange = range(height, width, channels);
    auto filterRange = range(filterWidth, filterWidth, channels);

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

              local_accessor<float, 3> filterTile{filterRange, cgh};

              cgh.parallel_for<image_convolution>(
                  nd_range<2>(range<2>(height, width), range<2>(16, 16)),
                  [=](nd_item<2> item) {
                    auto gi = item.get_global_id(0); // global row
                    auto gj = item.get_global_id(1); // global column
                    auto li = item.get_local_id(0);  // local row
                    auto lj = item.get_local_id(1);  // local column

                    auto k = filterWidth / 2;

                    if (li < filterWidth && lj == 0) {
                      // Copy filter
                      for (int j = 0; j < filterWidth; ++j) {
                        for (int c = 0; c < 4; ++c) {
                          id<3> idx(li, j, c);
                          filterTile[idx] = filterAcc[idx];
                        }
                      }
                    }

                    group_barrier(item.get_group());

                    float sum[4] = {0.f, 0.f, 0.f, 0.f};

                    if (!(gi < k || gi >= height - k || gj < k ||
                          gj >= width - k)) {
                      for (int u = -k; u <= k; ++u)
                        for (int v = -k; v <= k; ++v)
                          for (int c = 0; c < 4; ++c)
                            sum[c] += inputAcc[gi + u][gj + v][c] *
                                      filterTile[u + k][v + k][c];
                    }

                    for (int c = 0; c < 4; ++c)
                      outputAcc[gi][gj][c] = sum[c];
                  });
            });

            myQueue.wait_and_throw();
          },
          10, "image convolution (local memory)");
    }
  } catch (exception e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outputImage, outputImageFile);

  REQUIRE(true);
}
