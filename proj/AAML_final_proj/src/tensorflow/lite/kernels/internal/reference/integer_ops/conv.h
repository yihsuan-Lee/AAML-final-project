/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "perf.h"
#include "playground_util/print_params.h"
#include "cfu.h"
namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  perf_enable_counter(0);
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  // const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int im2col_row = output_height*output_width;
  const int im2col_col = filter_input_depth*filter_height*filter_width;


  int8_t im2col[576][4];

  int row_ = im2col_row/4;
  int ch_ = output_depth/4;


// ////////////////////////////////////-- im2col --/////////////////////////////////////////////

  cfu_op0(1,input_offset,im2col_col);
  int image_cnt = 0;
  int mm = 0;
  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int in_y_origin = (out_y * stride_height) - pad_height;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        const int in_y = in_y_origin + dilation_height_factor * filter_y;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int in_x = in_x_origin + dilation_width_factor * filter_x;
          for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
            const bool is_point_inside_image =
              (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
              (in_y < input_height);
            if (!is_point_inside_image) [[unlikely]] {

              // im2col[filter_y*filter_width*filter_input_depth + filter_x*filter_input_depth + in_channel][out_y*output_width + out_x] = -input_offset;
              im2col[filter_y*filter_width*filter_input_depth + filter_x*filter_input_depth + in_channel][image_cnt] = -input_offset;
              // continue;
            }
            else{
              // im2col[filter_y*filter_width*filter_input_depth + filter_x*filter_input_depth + in_channel][out_y*output_width + out_x]
              //   = input_data[Offset(input_shape, 0, in_y, in_x, in_channel)];
              im2col[filter_y*filter_width*filter_input_depth + filter_x*filter_input_depth + in_channel][image_cnt]
                = input_data[Offset(input_shape, 0, in_y, in_x, in_channel)];

            }
          }
        }
      }
      image_cnt++;
      if(image_cnt == 4) {
        for(int k=0;k<im2col_col;k++){
          int32_t b = *(uint32_t*)(*im2col+k*4);
          cfu_op5(1, b, k+(mm*im2col_col));
        }
        image_cnt = 0;
        mm++;
      }
    }
  }

  if(image_cnt != 0) {
    for(int k=0;k<im2col_col;k++){
      int32_t b = *(uint32_t*)(*im2col+k*4);
      cfu_op5(1, b, k+(mm*im2col_col));
    }
    // image_cnt = 0;
    // mm++;
  }


  ////////////////////////////////////-- kernel --/////////////////////////////////////////////

  int filter_depth_offset = 0;
  int8_t kernel_value[4] = {0};
  int num_cnt = 0;
  for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
    for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
      for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
        int n = 0;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          kernel_value[num_cnt++] = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
          if(num_cnt == 4) {
            int32_t b = *(uint32_t*)(kernel_value);
            cfu_op6(1, b, filter_depth_offset+(n*im2col_col));
            kernel_value[0] = 0;
            kernel_value[1] = 0;
            kernel_value[2] = 0;
            kernel_value[3] = 0;
            n++;
            num_cnt = 0;
          }
        }
        if(num_cnt != 0) {
          int32_t b = *(uint32_t*)(kernel_value);
          cfu_op6(1, b, filter_depth_offset+(n*im2col_col));
          kernel_value[0] = 0;
          kernel_value[1] = 0;
          kernel_value[2] = 0;
          kernel_value[3] = 0;
          n++;
          num_cnt = 0;
        }
        filter_depth_offset ++;
      }
    }
  }


  //////////////////////////////////-- output --/////////////////////////////////////////////


    // if((im2col_row%4) != 0){
    //   for(int m=4*row_ ; m<im2col_row; m++){
    //     for (int out_channel = 0; out_channel < output_depth; out_channel++) {
    //       int32_t acc = 0;
    //       for(int k = 0; k<im2col_col; ++k){
    //         acc+=kernel[k][out_channel]*(im2col[k][m]+input_offset);
    //       }
    //         if (bias_data) {
    //           acc += bias_data[out_channel];
    //         }
    //         acc = MultiplyByQuantizedMultiplier(
    //             acc, output_multiplier[out_channel], output_shift[out_channel]);
    //         acc += output_offset;
    //         acc = std::max(acc, output_activation_min);
    //         acc = std::min(acc, output_activation_max);
    //         int out_x = m%output_width;
    //         int out_y = m/output_width;
    //         output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] =
    //             static_cast<int8_t>(acc);
    //     }
    //   }
    // }

    // if(( output_depth%4) != 0){
    //   for(int m=0 ; m<im2col_row; m++){
    //     for (int out_channel = ch_*4; out_channel < output_depth; out_channel++) {
    //       int32_t acc = 0;
    //       for(int k = 0; k<im2col_col; ++k){
    //         acc+=kernel[k][out_channel]*(im2col[k][m]+input_offset);
    //       }
    //         if (bias_data) {
    //           acc += bias_data[out_channel];
    //         }
    //         acc = MultiplyByQuantizedMultiplier(
    //             acc, output_multiplier[out_channel], output_shift[out_channel]);
    //         acc += output_offset;
    //         acc = std::max(acc, output_activation_min);
    //         acc = std::min(acc, output_activation_max);
    //         int out_x = m%output_width;
    //         int out_y = m/output_width;
    //         output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] =
    //             static_cast<int8_t>(acc);
    //     }
    //   }
    // }

//////////////////////////////////////////////////////////////////////////////////////////////////////


      // for(int m=0 ; m<im2col_row; m++){
      //   for (int out_channel = 0; out_channel < output_depth; out_channel++) {
      //     int32_t acc = 0;
      //     for(int k = 0; k<im2col_col; ++k){
      //       acc+=kernel[k][out_channel]*(im2col[k][m]+input_offset);
      //     }
      //     res[m][out_channel]=acc;
      //   }
      // }




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // int temp = 0;

    for(int m=0; m<row_; m++){
      for (int out_channel = 0; out_channel < ch_; out_channel++) {

          cfu_op1(1,row_,ch_);

        for(int i=0;i<4;i++){
          for(int j=0;j<4;j++){
            // printf("%d %d\n",im2col[0][row_*4-1], kernel[0][ch_*4-1]);
            int32_t acc = cfu_op7(1,0,0);
            // if(m==0 && out_channel==0 && i==0 && j==0) printf("acc  %ld\n",acc);
            int out_ch = out_channel*4+j;

            if (bias_data) {
              acc += bias_data[out_ch];
            }
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[out_ch], output_shift[out_ch]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            int out_x = (m*4+i)%output_width;
            int out_y = (m*4+i)/output_width;
            output_data[Offset(output_shape, 0, out_y, out_x, out_ch)] =
                static_cast<int8_t>(acc);

          }
        }
      }
    }



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




  // for (int batch = 0; batch < batches; ++batch) {
    // for (int out_y = 0; out_y < output_height; ++out_y) {
    //   const int in_y_origin = (out_y * stride_height) - pad_height;
    //   for (int out_x = 0; out_x < output_width; ++out_x) {
    //     const int in_x_origin = (out_x * stride_width) - pad_width;
    //     for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    //       // auto group = out_channel / filters_per_group;
    //       int32_t acc = 0;
    //       for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
    //         const int in_y = in_y_origin + dilation_height_factor * filter_y;
    //         for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
    //           const int in_x = in_x_origin + dilation_width_factor * filter_x;

    //           // Zero padding by omitting the areas outside the image.
    //           const bool is_point_inside_image =
    //               (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
    //               (in_y < input_height);

    //           if (!is_point_inside_image) {
    //             continue;
    //           }

    //           for (int in_channel = 0; in_channel < filter_input_depth;
    //                ++in_channel) {
    //             // int32_t input_val =
    //             //     input_data[Offset(input_shape, batch, in_y, in_x,
    //             //                       in_channel + group * filter_input_depth)];
    //             int32_t input_val =
    //                 input_data[Offset(input_shape, 0, in_y, in_x,
    //                                   in_channel)];
    //             int32_t filter_val = filter_data[Offset(
    //                 filter_shape, out_channel, filter_y, filter_x, in_channel)];
    //             // Accumulate with 32 bits accumulator.
    //             // In the nudging process during model quantization, we force
    //             // real value of 0.0 be represented by a quantized value. This
    //             // guarantees that the input_offset is a int8_t, even though
    //             // it is represented using int32_t. int32_t += int8_t *
    //             // (int8_t - int8_t) so the highest value we can get from each
    //             // accumulation is [-127, 127] * ([-128, 127] -
    //             // [-128, 127]), which is [-32512, 32512]. log2(32512)
    //             // = 14.98, which means we can accumulate at least 2^16
    //             // multiplications without overflow. The accumulator is
    //             // applied to a filter so the accumulation logic will hold as
    //             // long as the filter size (filter_y * filter_x * in_channel)
    //             // does not exceed 2^16, which is the case in all the models
    //             // we have seen so far.
    //             // TODO(b/174275578): Add a check to make sure the
    //             // accumulator depth is smaller than 2^16.
    //             acc += filter_val * (input_val + input_offset);
    //           }
    //         }
    //       }

    //       if (bias_data) {
    //         acc += bias_data[out_channel];
    //       }
    //       acc = MultiplyByQuantizedMultiplier(
    //           acc, output_multiplier[out_channel], output_shift[out_channel]);
    //       acc += output_offset;
    //       acc = std::max(acc, output_activation_min);
    //       acc = std::min(acc, output_activation_max);
    //       // output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
    //       //     static_cast<int8_t>(acc);
    //       output_data[Offset(output_shape, 0, out_y, out_x, out_channel)] =
    //           static_cast<int8_t>(acc);
    //     }
    //   }
    // }
  // }
  perf_disable_counter(0);
  // print_conv_params(params, input_shape, filter_shape, output_shape);
}



inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_