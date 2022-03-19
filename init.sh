### conda environment
conda create --name USRN --file requirements.txt

### Balanced K-means
git clone https://github.com/zhu-he/regularized-k-means.git
cd regularized-k-means
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cd ..