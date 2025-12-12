// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

// AOC kernel demonstrating the saxpy computation 
// using one work group

__kernel void saxpy(__global const int *x,
		    __global const int *y,
		    __global int *z,
		    const int alpha) {

  // Get index of the work item
  int index = get_global_id(0);

 if(index>0) return; //using only one work group

  //compute saxpy for specific index
  for(int i = 0; i<N; i+=1){
    z[i] = alpha * x[i] + y[i];
  }
}
