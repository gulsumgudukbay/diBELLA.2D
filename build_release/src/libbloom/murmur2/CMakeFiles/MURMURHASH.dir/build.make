# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/mdl/gxg5138/local-software/installs/cmake-3.16.3/bin/cmake

# The command to remove a file.
RM = /home/mdl/gxg5138/local-software/installs/cmake-3.16.3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mdl/gxg5138/bioinf/diBELLA.2D

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release

# Include any dependencies generated for this target.
include src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/depend.make

# Include the progress variables for this target.
include src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/progress.make

# Include the compile flags for this target's objects.
include src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/flags.make

src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/MurmurHash2.c.o: src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/flags.make
src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/MurmurHash2.c.o: ../src/libbloom/murmur2/MurmurHash2.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/MurmurHash2.c.o"
	cd /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/src/libbloom/murmur2 && /home/mdl/gxg5138/local-software/installs/gcc_5.4.0/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/MURMURHASH.dir/MurmurHash2.c.o   -c /home/mdl/gxg5138/bioinf/diBELLA.2D/src/libbloom/murmur2/MurmurHash2.c

src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/MurmurHash2.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/MURMURHASH.dir/MurmurHash2.c.i"
	cd /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/src/libbloom/murmur2 && /home/mdl/gxg5138/local-software/installs/gcc_5.4.0/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mdl/gxg5138/bioinf/diBELLA.2D/src/libbloom/murmur2/MurmurHash2.c > CMakeFiles/MURMURHASH.dir/MurmurHash2.c.i

src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/MurmurHash2.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/MURMURHASH.dir/MurmurHash2.c.s"
	cd /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/src/libbloom/murmur2 && /home/mdl/gxg5138/local-software/installs/gcc_5.4.0/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mdl/gxg5138/bioinf/diBELLA.2D/src/libbloom/murmur2/MurmurHash2.c -o CMakeFiles/MURMURHASH.dir/MurmurHash2.c.s

MURMURHASH: src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/MurmurHash2.c.o
MURMURHASH: src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/build.make

.PHONY : MURMURHASH

# Rule to build all files generated by this target.
src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/build: MURMURHASH

.PHONY : src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/build

src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/clean:
	cd /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/src/libbloom/murmur2 && $(CMAKE_COMMAND) -P CMakeFiles/MURMURHASH.dir/cmake_clean.cmake
.PHONY : src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/clean

src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/depend:
	cd /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mdl/gxg5138/bioinf/diBELLA.2D /home/mdl/gxg5138/bioinf/diBELLA.2D/src/libbloom/murmur2 /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/src/libbloom/murmur2 /home/mdl/gxg5138/bioinf/diBELLA.2D/build_release/src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/libbloom/murmur2/CMakeFiles/MURMURHASH.dir/depend

