/*
 Copyright (c) 2020 CNRS
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIEDi
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm> //clamp
#include <cassert>
//Command-line parsing
#include "CLI11.hpp"

//Image filtering and I/O
#define cimg_display 0
#include "CImg.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif

//Global flag to silent verbose messages
bool silent;
//Global images dimensions.
int width, height, nbChannels;
const unsigned int DIMENSIONS = 3u;
//Global threads number.
unsigned int nbThreads;
using Vector = std::array<double, DIMENSIONS>;
//using Sample = std::tuple<double, unsigned int, Vector, Vector>; //(projection, index, color, offset)
using Sample = Vector; //(color)

unsigned int pixelIndex(const unsigned int i, const unsigned int j)
{
  return width*j + i;
}
unsigned int pixelIndex(const unsigned int cIndex)
{
  return cIndex / nbChannels;
}
unsigned int componentIndex(const unsigned int pIndex)
{
  return nbChannels * pIndex;
}
unsigned int componentIndex(const unsigned int i, const unsigned int j)
{
  return componentIndex(pixelIndex(i, j));
}
std::pair<unsigned int, unsigned int> indexPixel(const unsigned int pIndex)
{
  return {pIndex % width, pIndex / width};
}
std::ostream& operator<<(std::ostream& os, const Vector& v)
{
  os << "(" << v[0];
  for(auto i = 1; i < DIMENSIONS; i++)
    os << ", " << v[i];
  os << ")";
  return os;
}
double squaredLength(const Vector& v)
{
  double squaredLength(0.);
  for(const auto x : v)
    squaredLength += x*x;
  return squaredLength;
}
double length(const Vector& v)
{
  return std::sqrt(squaredLength(v));
}
Vector normalized(const Vector& v)
{
  const double l(length(v));
  Vector normalizedVector;
  for(auto i = 0; i < DIMENSIONS; i++)
    normalizedVector[i] = v[i]/l;
  return normalizedVector;
}
double dot(const Vector& v1, const Vector& v2)
{
  double product(0.);
  for(auto i = 0; i < DIMENSIONS; i++)
    product += v1[i] * v2[i];
  return product;
}
Vector randomDirection()
{
  //static std::random_device randomDevice;
  //static std::mt19937 generator(randomDevice());
  static std::mt19937 generator;
  static const double mean(0.);
  static const double standardDeviation(1.);
  static std::normal_distribution<double> normalDistribution(mean, standardDeviation);

  Vector randomVector;
  for(auto i = 0; i < DIMENSIONS; i++)
    randomVector[i] = normalDistribution(generator);
  return normalized(randomVector);
}

double projection(const Vector& d, const Vector& v)
{
  return dot(d, v);
}
void projectSamples(const Vector& d, const std::vector<Sample>& samples, std::vector<std::pair<double, unsigned int>>& projectionsSamples)
{
  for(unsigned int i = 0u; i < samples.size(); i++)
    projectionsSamples[i] = {projection(d, samples[i]), i};
}
void sortSamples(const Vector& direction, const std::vector<Sample>& samples, std::vector<std::pair<double, unsigned int>>& projectionsSamples)
{
  projectSamples(direction, samples, projectionsSamples);
  std::sort(projectionsSamples.begin(), projectionsSamples.end());
  //std::sort(projectionsSamples.begin(), projectionsSamples.end(), [](const Sample& s1, const Sample& s2){return s1.first < s2.first;});
}
void checkNormalizedColor(const Sample& sample)
{
  bool normalized(true);
  const Vector& color(sample);
  for(const auto x : color)
    if(x < 0. || x > 1.)
    {
      normalized = false;
      break;
    }
  if(!normalized)
    std::cout << "WARNING: Color " << color << " is not normalized!" << std::endl;
}
double computeOffsetsAndCost(
    const Vector& direction,
    const std::vector<Sample>& targetSamples, const std::vector<std::pair<double, unsigned int>>& targetProjectionsSamples,
    const std::vector<Sample>& sourceSamples, const std::vector<std::pair<double, unsigned int>>& sourceProjectionsSamples,
    std::vector<Vector>& offsets
)
{
  double cost(0.);
  for(auto i = 0; i < sourceProjectionsSamples.size(); i++)
  {
    //Offset.
    const double delta = targetProjectionsSamples[i].first - sourceProjectionsSamples[i].first;
    const unsigned int index = sourceProjectionsSamples[i].second;
    for(unsigned int d = 0; d < DIMENSIONS; d++)
    {
      const double offset = delta * direction[d];
      offsets[index][d] = offset;
    }
    //checkNormalizedColor(sourceSamples[index]);
    //Cost.
    cost += delta*delta;
  }
  return cost;
}
void advectSamples(const double normalizationFactor, const std::vector<Vector>& offsets, std::vector<Sample>& samples)
{
  #pragma omp parallel for
  for(auto i = 0; i < samples.size(); i++)
    for(auto d = 0; d < DIMENSIONS; d++)
    {
      double& component = samples[i][d];
      component = std::clamp(component + offsets[i][d]*normalizationFactor, 0., 1.);
    }
}
std::vector<Sample> imageSamples(const unsigned char * image)
{
  std::vector<Sample> samples(width * height);
  for(auto i = 0 ; i < width ; ++i) for(auto j = 0; j < height; ++j)
  {
    const int pIndex = pixelIndex(i, j);
    const int cIndex = componentIndex(pIndex);
    Vector& color = samples[pIndex];
    for(auto d = 0; d < DIMENSIONS; d++)
      color[d] = image[cIndex + d]/255.;
  }
  return samples;
}

void outputSamples(const std::vector<Sample>& samples, std::vector<unsigned char>& output)
{
  for(unsigned int pIndex = 0u; pIndex < samples.size(); pIndex++)
  {
    const Vector& color = samples[pIndex];
    const auto cIndex = componentIndex(pIndex);
    for(auto d = 0; d < DIMENSIONS; d++)
      //output[cIndex + d] = color[d]*255.;
      output[cIndex + d] = std::clamp(
        static_cast<unsigned char>(color[d]*255.),
        static_cast<unsigned char>(0),
        static_cast<unsigned char>(255)
      );
  }
}

struct IterationData {
    Vector direction;
    std::vector<std::pair<double, unsigned int>> targetProjectionsSamples;
    std::vector<std::pair<double, unsigned int>> sourceProjectionsSamples;
    std::vector<Vector> offsets;
    double cost;
};
void computation(const unsigned int nbBatchs, const unsigned int batchSize, const unsigned char * source, const unsigned char * target, std::vector<unsigned char>& output)
{
  /*
  //As an example, we just scan the pixels of the source image and swap the color channels.
  for(auto i = 0 ; i < width ; ++i)
  {
    for(auto j = 0; j < height; ++j)
    {
      const int cIndex = componentIndex(i, j);
      const unsigned char r = source[cIndex];
      const unsigned char g = source[cIndex + 1];
      const unsigned char b = source[cIndex + 2];
      //Swapping the channels
      output[cIndex] = b;
      output[cIndex + 1] = g;
      output[cIndex + 2] = r;
      if(nbChannels == 4) //just copying the alpha value if any
        output[cIndex + 3] = source[cIndex + 3];
    }
  }
  //*/

  auto sourceSamples = imageSamples(source);
  auto targetSamples = imageSamples(target);

  //Iterations data.
  std::vector<IterationData> iterationsData(batchSize);
  for(auto& it : iterationsData)
  {
    it.targetProjectionsSamples.resize(targetSamples.size());
    it.sourceProjectionsSamples.resize(sourceSamples.size());
    it.offsets.resize(sourceSamples.size());
  }
  std::vector<Vector> mergedOffsets(sourceSamples.size());

  //SOT loop:
  if(!silent) std::cout << "SOT batchs:" << std::endl;
  for(unsigned int batch = 0u; batch < nbBatchs; batch++)
  {
    if(!silent) std::cout << "  " << batch + 1 << "/" << nbBatchs << ":" << std::endl;

    //Draw batch random directions.
    for(auto& it : iterationsData)
      it.direction = randomDirection();

    #pragma omp parallel for
    for(unsigned int iteration = 0u; iteration < batchSize; iteration++)
    {
      auto& it = iterationsData[iteration];

      //Sort samples projections on the random direction.
      sortSamples(it.direction, targetSamples, it.targetProjectionsSamples);
      sortSamples(it.direction, sourceSamples, it.sourceProjectionsSamples);

      it.cost = computeOffsetsAndCost(it.direction, targetSamples, it.targetProjectionsSamples, sourceSamples, it.sourceProjectionsSamples, it.offsets);
    }

    //Compute batch offset.
    #pragma omp parallel for
    for(auto i = 0u; i < sourceSamples.size(); i++)
      for(auto d = 0u; d < DIMENSIONS; d++)
      {
        mergedOffsets[i][d] = 0.;
        for(auto& it : iterationsData)
          mergedOffsets[i][d] += it.offsets[i][d];
      }

    //Advect source samples.
    advectSamples(1./batchSize, mergedOffsets, sourceSamples);

    double cost(0.);
    for(auto& it : iterationsData)
      cost += it.cost;
    cost /= batchSize;

    if(!silent) std::cout << "    SOT cost:" << cost << std::endl;
  }

  //Save result.
  outputSamples(sourceSamples, output);
}
int main(int argc, char **argv)
{
  CLI::App app{"colorTransfer"};
  std::string sourceImage;
  app.add_option("-s,--source", sourceImage, "Source image")->required()->check(CLI::ExistingFile);;
  std::string targetImage;
  app.add_option("-t,--target", targetImage, "Target image")->required()->check(CLI::ExistingFile);;
  std::string outputImage= "output.png";
  app.add_option("-o,--output", outputImage, "Output image")->required();
  unsigned int nbSteps = 8;
  app.add_option("-n,--nbsteps", nbSteps, "Number of sliced steps (8)");
  unsigned int nbBatchs = 1;
  app.add_option("-b,--nbbatchs", nbBatchs, "Number of batchs (1)");
  nbThreads = 1;
  app.add_option("-j,--nbthreadss", nbThreads, "Number of threads (1)");
  silent = false;
  app.add_flag("--silent", silent, "No verbose messages");
  CLI11_PARSE(app, argc, argv);

  //OPENMP threading.
  omp_set_num_threads(nbThreads);

  //Image loading
  //int width, height, nbChannels;
  unsigned char *source = stbi_load(sourceImage.c_str(), &width, &height, &nbChannels, 0);
  if (!silent) std::cout<< "Source image: "<<width<<"x"<<height<<"   ("<<nbChannels<<")"<< std::endl;
  int width_target, height_target, nbChannels_target;
  unsigned char *target = stbi_load(targetImage.c_str(), &width_target, &height_target, &nbChannels_target, 0);
  if (!silent) std::cout<< "Target image: "<<width_target<<"x"<<height_target<<"   ("<<nbChannels_target<<")"<< std::endl;
  if ((width*height) != (width_target*height_target))
  {
    std::cout<< "Image sizes do not match. "<<std::endl;
    exit(1);
  }
  if (nbChannels < 3)
  {
    std::cout<< "Input images must be RGB images."<<std::endl;
    exit(1);
  }

  //Main computation
  std::vector<unsigned char> output(width*height*nbChannels);
  computation(nbBatchs, nbSteps, source, target, output);
  //just copying the alpha value if any
  if(nbChannels == 4) for(auto i = 0; i < width*height; i++) output[i*nbChannels + 3] = source[i*nbChannels + 3];

  //Final export
  if (!silent) std::cout<<"Exporting.."<<std::endl;
  int errcode = stbi_write_png(outputImage.c_str(), width, height, nbChannels, output.data(), nbChannels*width);
  if (!errcode)
  {
    std::cout<<"Error while exporting the resulting image."<<std::endl;
    exit(errcode);
  }

  stbi_image_free(source);
  stbi_image_free(target);
  exit(0);
}



