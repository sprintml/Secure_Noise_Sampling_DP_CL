# emp-aby
![arm](https://github.com/emp-toolkit/emp-aby/workflows/arm/badge.svg)
![x86](https://github.com/emp-toolkit/emp-aby/workflows/x86/badge.svg)

<img src="https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/art/logo-full.jpg" width=300px/>

# Installation
1. `wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py`
2. `python install.py -install -tool -ot`
    1. By default it will build for Release. `-DCMAKE_BUILD_TYPE=[Release|Debug]` option is also available.
    2. No sudo? Change [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/v2.8.8/cmake.html#variable%3aCMAKE_INSTALL_PREFIX).
    3. On Mac [homebrew](https://brew.sh/) is needed for installation. 



