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

class transpose;

TEST_CASE("transpose", "transpose") {
  const char *inputImageFile = "../Code_Exercises/Images/tawharanui_2048.png";
  const char *outputImageFile =
      "../Code_Exercises/Images/transposed_tawharanui.png";

  auto inputImage = util::read_image(inputImageFile, 0);

  auto outputImage = util::allocate_image(
      inputImage.width(), inputImage.height(), inputImage.channels());

  try {
    queue myQueue{};

    std::cout << "Running on "
              << myQueue.get_device().get_info<info::device::name>() << "\n";

    auto width = inputImage.width();
    auto height = inputImage.height();
    auto channels = inputImage.channels();
    assert(channels == 4);

    auto bufferRange = range<3>(height, width, channels);
    auto globalRange = range<2>(height, width);

    {
      auto inputBuf = buffer{inputImage.data(), bufferRange};
      auto outputBuf = buffer<float, 3>{bufferRange};
      outputBuf.set_final_data(outputImage.data());

      auto inputBufVec = inputBuf.reinterpret<sycl::float4>(globalRange);
      auto outputBufVec = outputBuf.reinterpret<sycl::float4>(globalRange);

      util::benchmark(
          [&]() {
            myQueue.submit([&](handler &cgh) {
              accessor inputAcc{inputBufVec, cgh, read_only};
              accessor outputAcc{outputBufVec, cgh, write_only};

              cgh.parallel_for<transpose>(nd_range<2>{globalRange, range<2>{16, 16}},
                  [=](id<2> idx) {
                    auto i = idx[0]; // row
                    auto j = idx[1]; // column

                    outputAcc[i][j] = inputAcc[j][i];
                  });
            });

            myQueue.wait_and_throw();
          },
          1000, "transpose");
    }
  } catch (exception e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outputImage, outputImageFile);

  REQUIRE(true);
}
