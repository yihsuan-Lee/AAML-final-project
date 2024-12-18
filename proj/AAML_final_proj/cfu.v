// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



// module Cfu (
//   input               cmd_valid,
//   output              cmd_ready,
//   input      [9:0]    cmd_payload_function_id,
//   input      [31:0]   cmd_payload_inputs_0,
//   input      [31:0]   cmd_payload_inputs_1,
//   output              rsp_valid,
//   input               rsp_ready,
//   output     [31:0]   rsp_payload_outputs_0,
//   input               reset,
//   input               clk
// );

//   // Trivial handshaking for a combinational CFU
//   assign rsp_valid = cmd_valid;
//   assign cmd_ready = rsp_ready;

//   //
//   // select output -- note that we're not fully decoding the 3 function_id bits
//   //
//   assign rsp_payload_outputs_0 = cmd_payload_function_id[0] ? 
//                                            cmd_payload_inputs_1 :
//                                            cmd_payload_inputs_0 ;


// endmodule

module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output reg          rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
);

    reg [8:0]  FilterOffset;
    localparam InputOffset = $signed(9'd128);


// add by ps
reg flag_a_defer;
reg [15:0] cnt_A_addr;
reg [31:0] w_bram_second_one;

reg [15:0]temp_addr;
reg [31:0]w_bram;
////////////////////////////////////-- BRAM A--////////////////////////////////////////////
reg          A_wr_en;
reg [15:0]    A_index;
// reg [31:0]    A_data_in;
wire [31:0]    A_data_in;
wire [31:0]    A_data_out;
global_buffer_bram #(
    .ADDR_BITS(16), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
  );


// always @(*) begin
//   if(flag_a) begin   
//     A_index = temp_addr;
//     A_data_in = w_bram;
//     A_wr_en = 1;
//     // B_index = 0;
//     // B_data_in = 0;
//     // B_wr_en = 0;
//   end
//   else begin
//     A_index = img_size;
//     A_data_in = 0;
//     A_wr_en = 0;
//   end
// end


assign A_data_in = (flag_a)? w_bram : w_bram_second_one;

always @(*) begin
  if(flag_a || flag_a_defer) begin   
    A_index = cnt_A_addr;
    A_wr_en = 1;
  end
  else begin
    A_index = img_size;
    A_wr_en = 0;
  end
end
////////////////////////////////////-- BRAM B--////////////////////////////////////////////
reg          B_wr_en;
reg [13:0]    B_index;
reg [31:0]    B_data_in;
wire [31:0]    B_data_out;
global_buffer_bram #(
    .ADDR_BITS(14), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_B(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out)
  );



always @(*) begin
  if(flag_b) begin   
    B_index = temp_addr;
    B_data_in = w_bram;
    B_wr_en = 1;
    // B_index = 0;
    // B_data_in = 0;
    // B_wr_en = 0;
  end
  else begin
    B_index = ker_size;
    B_data_in = 0;
    B_wr_en = 0;
  end
end
/////////////////////////--systoloic array--///////////////////////////////////////
reg signed[31:0]C_temp[15:0];
reg signed[31:0]A_temp[15:0];
reg signed[31:0]B_temp[15:0];
// reg [511:0]C_temp;
// reg [127:0]A_temp;
// reg [127:0]B_temp;
// reg [31:0]A_temp_1, B_temp_1;
reg [23:0]A_temp_2, B_temp_2;
reg [15:0]A_temp_3, B_temp_3;
reg [7:0]A_temp_4, B_temp_4;
reg has_in_valid;
reg [15:0] size_K,size_M, size_N,start_row;
reg [7:0] A_dim_counter,B_dim_counter;
reg get_AorB;
reg [6:0] A_run,A_run_counter,B_run,B_run_counter;
reg [4:0] counter;
reg output_flag, count_flag,flag_a,flag_b,start_count, get_val_or_cal;
reg [15:0]img_size;
reg [13:0]ker_size;
//////////////////////////////////////////////////////////////////////////////////////

    // SIMD multiply step:
    
    reg signed[31:0]offset;

    // Only not ready for a command when we have a response.
    assign cmd_ready = ~rsp_valid;
    integer j,k;
    
    // flag_a_defer
    always @(posedge clk) begin
      if(reset) begin
        flag_a_defer <= 0;
      end
      else begin
        flag_a_defer <= flag_a;
      end
    end

    // cnt_A_addr
    always @(posedge clk) begin
      if(reset) begin
        cnt_A_addr <= 0;
      end
      else begin
        if(cmd_payload_function_id[2:0]==6) cnt_A_addr <= 0;
        else if(flag_a || flag_a_defer) cnt_A_addr <= cnt_A_addr + 1;
        
      end
    end


    always @(posedge clk) begin
      if (reset) begin
        rsp_payload_outputs_0 <= 32'b0;
        rsp_valid <= 1'b0;
        output_flag <= 0;
        counter <= 0;
        count_flag <= 0;
        flag_a <= 0;
        flag_b <= 0;
        img_size <= 0;
        ker_size <= 0;
        start_row <= 0;
        get_val_or_cal <= 0;
        for (j = 0; j<16; j=j+1) begin
            C_temp[j] <= 0;
            A_temp[j] <= 0;
            B_temp[j] <= 0;
        end
        A_temp_2 <= 0;
        A_temp_3 <= 0;
        A_temp_4 <= 0;
        B_temp_2 <= 0;
        B_temp_3 <= 0;
        B_temp_4 <= 0;
        start_count <= 0;

        w_bram_second_one <= 0;
      end 

      else if (rsp_valid) begin
        // Waiting to hand off response to CPU.
        rsp_valid <= ~rsp_ready;
        if(count_flag) begin
          for (k = 0; k<16; k=k+1) begin
              C_temp[k] <= C_temp[k] + (A_temp[k]*(B_temp[k]));
          end  
          count_flag <= 0;
        end
      end 

      else if (cmd_valid) begin
        // Accumulate step:
        if(cmd_payload_function_id[2:0]==0)begin
          offset <= cmd_payload_inputs_0;
          size_K <= cmd_payload_inputs_1;
          rsp_payload_outputs_0 <= 0;
          rsp_valid <= 1'b1;
          start_row <= 0;
          img_size <= 0;
          ker_size <= 0;
        end
        if(cmd_payload_function_id[2:0]==1)begin
          size_M <= cmd_payload_inputs_0;
          size_N <= cmd_payload_inputs_1;
          start_count <= 1;
          get_val_or_cal <= 0;
          counter <= 0;
        end
        else if(cmd_payload_function_id[2:0]==5)begin
          flag_a <= 1;
          // temp_addr <= cmd_payload_inputs_1;
          w_bram <= cmd_payload_inputs_0;
          w_bram_second_one <= cmd_payload_inputs_1;
        end
        else if(cmd_payload_function_id[2:0]==6)begin
          flag_b <= 1;
          temp_addr <= cmd_payload_inputs_1;
          w_bram <= cmd_payload_inputs_0;
        end
        else if(cmd_payload_function_id[2:0]==7) begin
          count_flag <= 1;
          rsp_valid <= 1'b1;
          rsp_payload_outputs_0 <= C_temp[counter];
          if(counter==15) begin
            for (j = 0; j<16; j=j+1) begin
              C_temp[j] <= 0;
            end
          end
          counter <= counter+1;
          if(counter == 0) begin
            B_temp[0] <= $signed(cmd_payload_inputs_0[7:0]);
            B_temp[4] <= $signed(B_temp_2[7:0])+ offset;
            B_temp[8] <= $signed(B_temp_3[7:0])+ offset;
            B_temp[12] <= $signed(B_temp_4[7:0])+ offset;
          end
          else if(counter == 1)begin
            B_temp[0] <= $signed(cmd_payload_inputs_0[7:0]);
            B_temp[4] <= $signed(B_temp_2[7:0]);
            B_temp[8] <= $signed(B_temp_3[7:0])+ offset;
            B_temp[12] <= $signed(B_temp_4[7:0])+ offset;
          end
          else if(counter == 2)begin
            B_temp[0] <= $signed(cmd_payload_inputs_0[7:0]);
            B_temp[4] <= $signed(B_temp_2[7:0]);
            B_temp[8] <= $signed(B_temp_3[7:0]);
            B_temp[12] <= $signed(B_temp_4[7:0])+ offset;
          end
          else begin
            B_temp[0] <= $signed(cmd_payload_inputs_0[7:0]);
            B_temp[4] <= $signed(B_temp_2[7:0]);
            B_temp[8] <= $signed(B_temp_3[7:0]);
            B_temp[12] <= $signed(B_temp_4[7:0]);
          end
          A_temp_2 <= cmd_payload_inputs_0[31:8];
          A_temp_3 <= A_temp_2[23:8];
          A_temp_4 <= A_temp_3[15:8];
          
        
          B_temp_2 <= cmd_payload_inputs_0[31:8];
          B_temp_3 <= B_temp_2[23:8];
          B_temp_4 <= B_temp_3[15:8];

          A_temp[0] <= $signed(cmd_payload_inputs_0[7:0]);
          A_temp[1] <= $signed(A_temp_2[7:0]);
          A_temp[2] <= $signed(A_temp_3[7:0]);
          A_temp[3] <= $signed(A_temp_4[7:0]);

        
          //PE col 1
          A_temp[4] <= A_temp[0];
          A_temp[5] <= A_temp[1];
          A_temp[6] <= A_temp[2];
          A_temp[7] <= A_temp[3];

          B_temp[1] <= B_temp[0];
          B_temp[5] <= B_temp[4];
          B_temp[9] <= B_temp[8];
          B_temp[13] <= B_temp[12];
          //PE col 0
          A_temp[8] <= A_temp[4];
          A_temp[9] <= A_temp[5];
          A_temp[10] <= A_temp[6];
          A_temp[11] <= A_temp[7];

          B_temp[2] <= B_temp[1];
          B_temp[6] <= B_temp[5];
          B_temp[10] <= B_temp[9];
          B_temp[14] <= B_temp[13];
          //PE col 0
          A_temp[12] <= A_temp[8];
          A_temp[13] <= A_temp[9];
          A_temp[14] <= A_temp[10];
          A_temp[15] <= A_temp[11];

          B_temp[3] <= B_temp[2];
          B_temp[7] <= B_temp[6];
          B_temp[11] <= B_temp[10];
          B_temp[15] <= B_temp[14];
        end

      end
      else if(flag_a) begin
        flag_a <= 0;
        rsp_valid <= 1'b1;
        rsp_payload_outputs_0 <= 0;
      end
      else if(flag_b)begin
        flag_b <= 0;
        rsp_valid <= 1'b1;
        rsp_payload_outputs_0 <= 0;
      end
      else if(start_count) begin
        get_val_or_cal <= ~get_val_or_cal;
        if(~get_val_or_cal) begin
          count_flag <= 1;
          if((img_size%size_K)==size_K-1)begin
            start_count <= 0;
            if(ker_size  == (size_K*size_N)-1) begin
              if(start_row == size_M-1)begin
                rsp_valid <= 1'b1;
                rsp_payload_outputs_0 <= C_temp[0];
              end
              else begin
                rsp_valid <= 1'b1;
                rsp_payload_outputs_0 <= C_temp[0];
              end 
              ker_size <= 0;
              start_row <= start_row+1;
              img_size <= (start_row+1)*size_K;

            end
            else begin
              img_size <= start_row*size_K;
              ker_size <= ker_size+1;
              rsp_valid <= 1'b1;
                rsp_payload_outputs_0 <= C_temp[0];
            end
          end 
          else begin
            img_size <= img_size+1;
            ker_size <= ker_size+1;
          end
          
          
          B_temp[0] <= $signed(A_data_out[7:0]) + offset;
          B_temp[4] <= $signed(B_temp_2[7:0])+ offset;
          B_temp[8] <= $signed(B_temp_3[7:0])+ offset;
          B_temp[12] <= $signed(B_temp_4[7:0])+ offset;
        
          // end
          
          A_temp_2 <= B_data_out[31:8];
          A_temp_3 <= A_temp_2[23:8];
          A_temp_4 <= A_temp_3[15:8];
          
          
          B_temp_2 <= A_data_out[31:8];
          B_temp_3 <= B_temp_2[23:8];
          B_temp_4 <= B_temp_3[15:8];

          A_temp[0] <= $signed(B_data_out[7:0]);
          A_temp[1] <= $signed(A_temp_2[7:0]);
          A_temp[2] <= $signed(A_temp_3[7:0]);
          A_temp[3] <= $signed(A_temp_4[7:0]);

        
          //PE col 1
          A_temp[4] <= A_temp[0];
          A_temp[5] <= A_temp[1];
          A_temp[6] <= A_temp[2];
          A_temp[7] <= A_temp[3];

          B_temp[1] <= B_temp[0];
          B_temp[5] <= B_temp[4];
          B_temp[9] <= B_temp[8];
          B_temp[13] <= B_temp[12];
          //PE col 0
          A_temp[8] <= A_temp[4];
          A_temp[9] <= A_temp[5];
          A_temp[10] <= A_temp[6];
          A_temp[11] <= A_temp[7];

          B_temp[2] <= B_temp[1];
          B_temp[6] <= B_temp[5];
          B_temp[10] <= B_temp[9];
          B_temp[14] <= B_temp[13];
          //PE col 0
          A_temp[12] <= A_temp[8];
          A_temp[13] <= A_temp[9];
          A_temp[14] <= A_temp[10];
          A_temp[15] <= A_temp[11];

          B_temp[3] <= B_temp[2];
          B_temp[7] <= B_temp[6];
          B_temp[11] <= B_temp[10];
          B_temp[15] <= B_temp[14];
            
            // rsp_payload_outputs_0 <= ($signed(cmd_payload_inputs_1[7 : 0]) + offset) * $signed(cmd_payload_inputs_0[7 : 0]);
        end
        else begin
          for (k = 0; k<16; k=k+1) begin
              C_temp[k] <= C_temp[k] + (A_temp[k]*B_temp[k]);
          end 
        end
      end
    end
    
endmodule

module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
  input                      clk,
  input                      rst_n,
  input                      ram_en,
  input                      wr_en,
  input      [ADDR_BITS-1:0] index,
  input      [DATA_BITS-1:0] data_in,
  output reg [DATA_BITS-1:0] data_out
  );

  parameter DEPTH = 2**ADDR_BITS;

  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (negedge clk) begin
    if (ram_en) begin
      if(wr_en) begin
        gbuff[index] <= data_in;
      end
      else begin
        data_out <= gbuff[index];
      end
    end
  end

endmodule