file(GLOB_RECURSE test_py_files "${CMAKE_SOURCE_DIR}/systemtest/rimea_tests/*runtest_*.py")
foreach (file ${test_py_files})
    get_filename_component(test ${file} NAME_WE)
    string(REGEX MATCH "[0-9]+" test_number ${test})
    add_test(
            NAME rimea_test-${test_number}
            COMMAND ${PYTHON_EXECUTABLE} ${file} ${jpscore_exe}
    )
endforeach ()
