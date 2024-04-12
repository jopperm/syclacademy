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

TEST_CASE("tiled_transpose", "tiled_transpose") {
  const char *inputImageFile = "../Code_Exercises/Images/tawharanui_4096.png";
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

    auto bufferRange = range<3>(height, width, channels);

    {
      auto inputBuf = buffer{inputImage.data(), bufferRange};
      auto outputBuf = buffer<float, 3>{bufferRange};
      outputBuf.set_final_data(outputImage.data());

      util::benchmark(
          [&]() {
            myQueue.submit([&](handler &cgh) {
              accessor inputAcc{inputBuf, cgh, read_only};
              accessor outputAcc{outputBuf, cgh, write_only};
              local_accessor<float, 3> tileAcc{range<3>(16, 16, 4), cgh};

              cgh.parallel_for<transpose>(
                  nd_range<2>(range<2>(height, width), range<2>(16, 16)),
                  [=](nd_item<2> item) {
                     auto gi = item.get_global_id(0); // global row
                     auto gj = item.get_global_id(1); // global column
                     auto li = item.get_local_id(0);  // local row
                     auto lj = item.get_local_id(1);  // local column 

                    for (int c = 0; c < 4; ++c)
                      tileAcc[lj][li][c] = inputAcc[gi][gj][c];

                    group_barrier(item.get_group());

                    auto base_i = item.get_group()[1] * 16;
                    auto base_j = item.get_group()[0] * 16;

                    for (int c = 0; c < 4; ++c)
                      outputAcc[base_i + li][base_j + lj][c] = tileAcc[li][lj][c];
                  });
            });

            myQueue.wait_and_throw();
          },
          10, "transpose (tiled)");
    }
  } catch (exception e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  util::write_image(outputImage, outputImageFile);

  REQUIRE(true);
}
