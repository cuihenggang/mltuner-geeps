# Copyright (C) 2013 by Carnegie Mellon University.

import os
import re
import fnmatch
import filecmp

#### Decisions are made here.  The rest of the file are functions defintions.
def main():
    build_all('build')

def build_all(variant):
    env = DefaultEnvironment().Clone()
    env['CXX'] = 'g++'
    # Scons has no idea that nvcc compiles the code using '-fPIC' flag, so we need this option to get rid of the link error
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1
    
    # set nvcc builder
    nvcc = Builder(action = "/usr/local/cuda/bin/nvcc --compiler-options '-fPIC' --shared -Iinclude -Isrc -c $SOURCE -o $TARGET -g -O3",
                   suffix = '.o',
                   src_suffix = '.cu')

    env.Append(CPPPATH = ['src'])
    env.Append(CPPPATH = ['include'])
    env.Append(LIBS = [
      'zmq', 'boost_system', 'boost_thread', 'tbb', 'glog', 'gflags', 'numa'])
    # The -fPIC flag is necessary to build a shared library
    #env.Append(CCFLAGS = '-Wall -Werror -g -fPIC')
    env.Append(CCFLAGS = '-Wall -Wno-sign-compare -g -fPIC')
    if (env['build_debug'] == '1'):
        env.Append(CCFLAGS = '-ggdb')
    else:
        env.Append(CCFLAGS = '-O3')

    env.Append(CPPPATH = ['/usr/local/cuda/include'])
    env.Append(LIBPATH = ['/usr/local/cuda/lib64'])
    env.Append(LIBS = ['cblas', 'cudart', 'cublas'])
    env.Append(BUILDERS = {'NvccBuilder' : nvcc})

    # build client library
    src_files = ['src/client/geeps.cpp']
    src_files.append('src/client/clientlib.cpp')
    src_files.append('src/client/clientlib-viter.cpp')
    src_files.append('src/client/encoder-decoder.cpp')
    src_files.append('src/client/client-tuner-encoder-decoder.cpp')
    src_files.append('src/common/background-worker.cpp')
    src_files.append('src/common/work-puller.cpp')
    src_files.append('src/common/work-pusher.cpp')
    src_files.append('src/common/router-handler.cpp')
    src_files.append('src/common/gpu-util/math_functions.cpp')
    src_files.append(env.NvccBuilder('src/common/gpu-util/math_functions_cuda.cu'))
    src_files.append(env.NvccBuilder('src/common/row-op-util.cu'))
    src_files.append('src/server/tablet-server.cpp')
    src_files.append('src/server/tablet-server-update-functions.cpp')
    src_files.append('src/server/metadata-server.cpp')
    src_files.append('src/server/server-encoder-decoder.cpp')
    src_files.append('src/server/server-entry.cpp')
    src_files.append('src/mltuner/mltuner.cpp')
    src_files.append('src/mltuner/mltuner-encoder-decoder.cpp')
    src_files.append('src/mltuner/mltuner-entry.cpp')
    src_files.append('src/mltuner/mltuner-utils.cpp')
    src_files.append('src/mltuner/mltuner-impl.cpp')
    src_files.append('src/mltuner/mltuner-math.cpp')
    src_files.append('src/mltuner/tuner-logics/mltuner-logic.cpp')
    src_files.append('src/mltuner/tuner-logics/spearmint-logic.cpp')
    src_files.append('src/mltuner/tuner-logics/tupaq-logic.cpp')
    src_files.append('src/mltuner/tuner-logics/hyperband-logic.cpp')
    src_files.append('src/mltuner/tunable-searchers/spearmint-searcher.cpp')
    src_files.append('src/mltuner/tunable-searchers/hyperopt-searcher.cpp')
    src_files.append('src/mltuner/tunable-searchers/grid-searcher.cpp')
    src_files.append('src/mltuner/tunable-searchers/random-searcher.cpp')
    src_files.append('src/mltuner/tunable-searchers/marginal-grid-searcher.cpp')
    src_files.append('src/mltuner/tunable-searchers/marginal-hyperopt-searcher.cpp')
    clientlib = env.Library('geeps', src_files)
    clientsharedlib = env.SharedLibrary('geeps', src_files)
    env.Install('lib', clientlib)

# end of build_all()

main()
