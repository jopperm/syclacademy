#[[
  SYCL Academy (c)

  SYCL Academy is licensed under a Creative Commons Attribution-ShareAlike 4.0
  International License.

  You should have received a copy of the license along with this work.  If not,
  see <http://creativecommons.org/licenses/by-sa/4.0/>.
]]

# Exercise exercise_12_temporary_data
add_sycl_executable(exercise_12 source)
if(SYCL_ACADEMY_ENABLE_SOLUTIONS)
  add_sycl_executable(exercise_12 solution)
endif()
