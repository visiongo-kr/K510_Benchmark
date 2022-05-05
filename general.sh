#!/bin/sh

# make dir
releasePath="./benchmark"
if [ -d "$releasePath" ]; then
rm -rf $releasePath
fi
mkdir $releasePath

# copy kmodel
cp -r ./model ./$releasePath
# copy src
cp -r ./src ./$releasePath
# copy cmakelists
cp CMakeLists.txt ./$releasePath
# copy benchmark.mk
cp benchmark.mk ./$releasePath
# copy config.in
cp Config.in ./$releasePath
# copy shell
cp benchmark.sh ./$releasePath