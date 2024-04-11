#include <stdio.h>
#include "cs1300bmp.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "Filter.h"
#include <immintrin.h>
#include <omp.h>

using namespace std;

#include "rdtsc.h"

//
// Forward declare the functions
//
Filter * readFilter(string filename);
double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output);

int
main(int argc, char **argv)
{

  if ( argc < 2) {
    fprintf(stderr,"Usage: %s filter inputfile1 inputfile2 .... \n", argv[0]);
  }

  //
  // Convert to C++ strings to simplify manipulation
  //
  string filtername = argv[1];

  //
  // remove any ".filter" in the filtername
  //
  string filterOutputName = filtername;
  string::size_type loc = filterOutputName.find(".filter");
  if (loc != string::npos) {
    //
    // Remove the ".filter" name, which should occur on all the provided filters
    //
    filterOutputName = filtername.substr(0, loc);
  }

  Filter *filter = readFilter(filtername);

  double sum = 0.0;
  int samples = 0;

  string inputFilename; //my change start
  string outputFilename;
  struct cs1300bmp *input;
  struct cs1300bmp *output;
  int ok;
  double sample; //my change end

  for (int inNum = 2; inNum < argc; inNum++) {
    inputFilename = argv[inNum];
    outputFilename = "filtered-" + filterOutputName + "-" + inputFilename;
    input = new struct cs1300bmp;
    output = new struct cs1300bmp;
    ok = cs1300bmp_readfile( (char *) inputFilename.c_str(), input);

    if ( ok ) {
      sample = applyFilter(filter, input, output);
      sum += sample;
      samples++;
      cs1300bmp_writefile((char *) outputFilename.c_str(), output);
    }
    delete input;
    delete output;
  }

  fprintf(stdout, "Average cycles per sample is %f\n", sum / samples);

}

class Filter *
readFilter(string filename)
{
  ifstream input(filename.c_str());

  if ( ! input.bad() ) {
    int size = 0;
    input >> size;
    Filter *filter = new Filter(size);
    int div;
    input >> div;
    filter -> setDivisor(div);

    int value; //my change

    for (int i=0; i < size; i++) {
      for (int j=0; j < size; j++) {
        input >> value;
        filter -> set(i,j,value);
      }
    }
    return filter;

  } else {
    cerr << "Bad input in readFilter:" << filename << endl;
    exit(-1);
  }
}


double
applyFilter(class Filter *filter, cs1300bmp *input, cs1300bmp *output)
{

  long long cycStart, cycStop;

  cycStart = rdtscll();

  output -> width = input -> width;
  output -> height = input -> height;

  int width = input -> width - 1; //start my change
  int height = input -> height - 1;
  double filterDivisor = 1 / (float) filter -> getDivisor();
  int* filterData = filter -> getdata();
  int temp;
  __m256i multiplier;
  __m128i sum4;
  __m128i sum3;
  __m256i sum2;
  __m256i sum1;
  __m256i result;
  __m256i vec;


omp_set_num_threads(15);
#pragma omp parallel for private (temp, sum4, sum3, sum2, sum1, result, vec)
  for(int row = 1; row < height; row = row + 1) { //swapped row and col for loop order. unroll these loops as well
    for(int col = 1; col < width; col = col + 1) {
        int* valueArray = new int[8];
        valueArray[0] = input -> red[row-1][col-1];
        valueArray[1] = input -> red[row-1][col];
        valueArray[2] = input -> red[row-1][col+1];
        valueArray[3] = input -> red[row][col-1];
        valueArray[4] = input -> red[row][col];
        valueArray[5] = input -> red[row][col+1];
        valueArray[6] = input -> red[row+1][col-1];
        valueArray[7] = input -> red[row+1][col];
        temp = input -> red[row+1][col+1]* filterData[8];
        
        vec = _mm256_loadu_si256((__m256i*)valueArray);

        valueArray[0] = filterData[0];
        valueArray[1] = filterData[1];
        valueArray[2] = valueArray[2];
        valueArray[3] = filterData[3];
        valueArray[4] = filterData[4];
        valueArray[5] = valueArray[5];
        valueArray[6] = filterData[6];
        valueArray[7] = filterData[7];

        multiplier = _mm256_loadu_si256((__m256i*)valueArray);

        result = _mm256_mullo_epi32(vec, multiplier);
        sum1 = _mm256_hadd_epi32(result, result); // Add quads
        sum2 = _mm256_hadd_epi32(sum1, sum1);// Extract the lower and upper 128 bits and add them together
        sum3 = _mm256_extracti128_si256(sum2, 1);
        sum4 = _mm_add_epi32(_mm256_castsi256_si128(sum2), sum3);
        temp = temp + _mm_extract_epi32(sum4, 0) + _mm_extract_epi32(sum4, 1);
        temp = temp*filterDivisor;

        if(temp < 0){
          temp = 0;
        }
        if(temp > 255){
          temp = 255;
        }

        output -> red[row][col] = temp;

        valueArray[0] = input -> green[row-1][col-1];
        valueArray[1] = input -> green[row-1][col];
        valueArray[2] = input -> green[row-1][col+1];
        valueArray[3] = input -> green[row][col-1];
        valueArray[4] = input -> green[row][col];
        valueArray[5] = input -> green[row][col+1];
        valueArray[6] = input -> green[row+1][col-1];
        valueArray[7] = input -> green[row+1][col];

        temp = input -> green[row+1][col+1] * filterData[8];
        
        vec = _mm256_loadu_si256((__m256i*)valueArray);

        valueArray[0] = filterData[0];
        valueArray[1] = filterData[1];
        valueArray[2] = valueArray[2];
        valueArray[3] = filterData[3];
        valueArray[4] = filterData[4];
        valueArray[5] = valueArray[5];
        valueArray[6] = filterData[6];
        valueArray[7] = filterData[7];

        multiplier = _mm256_loadu_si256((__m256i*)valueArray);

        result = _mm256_mullo_epi32(vec, multiplier); // Multiply each by 'multiplier'
        sum1 = _mm256_hadd_epi32(result, result); // Horizontal add pairs of integers
        sum2 = _mm256_hadd_epi32(sum1, sum1); // Horizontal add quads of integers
        sum3 = _mm256_extracti128_si256(sum2, 1); // Extract upper 128 bits
        sum4 = _mm_add_epi32(_mm256_castsi256_si128(sum2), sum3); // Add the lower and upper halves
        temp = temp + _mm_extract_epi32(sum4, 0) + _mm_extract_epi32(sum4, 1) + valueArray[8] * filterDivisor; // Add the last value after multiplying by filterDivisor
        temp = temp*filterDivisor;
        // Clamp the result to the range [0, 255]
        if(temp < 0){
          temp = 0;
        }
        if(temp > 255){
          temp = 255;
        }

        output -> green[row][col] = temp;

        valueArray[0] = input -> blue[row-1][col-1];
        valueArray[1] = input -> blue[row-1][col];
        valueArray[2] = input -> blue[row-1][col+1];
        valueArray[3] = input -> blue[row][col-1];
        valueArray[4] = input -> blue[row][col];
        valueArray[5] = input -> blue[row][col+1];
        valueArray[6] = input -> blue[row+1][col-1];
        valueArray[7] = input -> blue[row+1][col];

        vec = _mm256_loadu_si256((__m256i*)valueArray); // Load the first 8 integers from valueArray

        temp = input -> blue[row+1][col+1]* filterData[8];
        
        vec = _mm256_loadu_si256((__m256i*)valueArray);

        valueArray[0] = filterData[0];
        valueArray[1] = filterData[1];
        valueArray[2] = valueArray[2];
        valueArray[3] = filterData[3];
        valueArray[4] = filterData[4];
        valueArray[5] = valueArray[5];
        valueArray[6] = filterData[6];
        valueArray[7] = filterData[7];

        multiplier = _mm256_loadu_si256((__m256i*)valueArray);
        result = _mm256_mullo_epi32(vec, multiplier); // Multiply each by 'multiplier'
        sum1 = _mm256_hadd_epi32(result, result); // Horizontal add pairs of integers
        sum2 = _mm256_hadd_epi32(sum1, sum1); // Horizontal add quads of integers
        sum3 = _mm256_extracti128_si256(sum2, 1); // Extract upper 128 bits
        sum4 = _mm_add_epi32(_mm256_castsi256_si128(sum2), sum3); // Add the lower and upper halves
        temp = temp + _mm_extract_epi32(sum4, 0) + _mm_extract_epi32(sum4, 1) + valueArray[8] * filterDivisor; // Add the last value after multiplying by filterDivisor
        temp = temp*filterDivisor;
        // Clamp the result to the range [0, 255]
        if(temp < 0){
          temp = 0;
        }
        if(temp > 255){
          temp = 255;
        }

        output -> blue[row][col] = temp;
    }
  }

  cycStop = rdtscll();
  double diff = cycStop - cycStart;
  double diffPerPixel = diff / (output -> width * output -> height);

  fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n", diff, diff / (output -> width * output -> height));
  return diffPerPixel;
}
