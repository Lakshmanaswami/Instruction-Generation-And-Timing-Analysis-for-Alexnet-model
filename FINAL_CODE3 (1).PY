import onnx
import os
BYTES=4
def load_onnx_model(model_path):
    return onnx.load(model_path)

def extract_conv_params(node, prev_out_channels=None, prev_height=None, prev_width=None):  #fucntion to extract conv attribute values
    params = {
        "name": node.name, "kernel_h": None, "kernel_w": None, "stride_h": None, "stride_w": None,
        "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0, "in_channels": None, "out_channels": None,
        "input_width": prev_width, "input_height": prev_height
    }

    for attr in node.attribute:
        if attr.name == "kernel_shape":
            params["kernel_h"], params["kernel_w"] = attr.ints
        elif attr.name == "strides":
            params["stride_h"], params["stride_w"] = attr.ints
        elif attr.name == "pads":
            params["pad_top"], params["pad_left"], params["pad_bottom"], params["pad_right"] = attr.ints

    weights_tensor_name = node.input[1]
    weights_tensor = None
    for initializer in model.graph.initializer:
        if initializer.name == weights_tensor_name:
            weights_tensor = initializer
            break

    if weights_tensor is not None:
        weights_shape = weights_tensor.dims
        params['height'] = weights_shape[2]
        params['width'] = weights_shape[3]

        if prev_out_channels is None:
            params["in_channels"] = weights_shape[1]
        else:
            params["in_channels"] = prev_out_channels

        params["out_channels"] = weights_shape[0]

        params["weights"] = weights_tensor.float_data
    else:
        raise ValueError(f"Weights tensor not found for node '{node.name}'")

    return params

def extract_pool_params(node):     #fucntion to extract maxpool attribute values
    params = {"name": node.name, "kernel_h": None, "kernel_w": None, "stride_h": None, "stride_w": None, "pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}

    for attr in node.attribute:
        if attr.name == "kernel_shape":
            params["kernel_h"], params["kernel_w"] = attr.ints
        elif attr.name == "strides":
            params["stride_h"], params["stride_w"] = attr.ints
        elif attr.name == "pads":
            params["pad_top"], params["pad_left"], params["pad_bottom"], params["pad_right"] = attr.ints

    return params

def extract_lrn_params(node):      #fucntion to extract LRN attribute values
    params = {"name": node.name, "alpha": None, "beta": None, "bias": None, "size": None}

    for attr in node.attribute:
        if attr.name == "alpha":
            params["alpha"] = attr.f
        elif attr.name == "beta":
            params["beta"] = attr.f
        elif attr.name == "bias":
            params["bias"] = attr.f
        elif attr.name == "size":
            params["size"] = attr.i

    return params

def extract_input_dimensions(model):  #fucntion to extract input dimensions from the model first time
    input_node = model.graph.input[0]
    input_shape = input_node.type.tensor_type.shape.dim
    height = input_shape[2].dim_value
    width = input_shape[3].dim_value
    return height, width


def save_layer_instructions(instruction, folder_name, file_name):
   
    
    os.makedirs(folder_name, exist_ok=True)
    
    #  file path based on given file name
    file_path = os.path.join(folder_name, f"{file_name}.txt")
    
    # instruction set  saving in a file immediately after generation
    with open(file_path, "w") as f:
        for instr in instruction:
            f.write(instr + "\n")
    

def save_addition_instruction(file_name, instructions):  #function to save addition instructions
    """ Save each instruction set into a separate file. """
    with open(file_name, "w") as f:
        for instr in instructions:
            f.write("\n".join(instr) + "\n")

def generate_addition_instructions(prev_output, params, output_size, layer_counters, node_name):
    """ Generate correct addition instructions ensuring proper channel-filter alignment and efficient memory usage. """
    node_name="addition"
    layer_counters['conv'] += 1
    folder_name = f"addition{layer_counters['conv']}"  #1 addition  folder per convolution 
    os.makedirs(folder_name, exist_ok=True)

    out_channels = params["out_channels"]
    in_channels = params["in_channels"]
    output_width = prev_output["width"]
    output_height = prev_output["height"]

    
    channel_offset = out_channels * output_width * output_height * BYTES

    #  memory address  starts where addition results will be stored
    combine_address_start = (
        prev_output["address1"]
        + in_channels * output_width * output_height * out_channels * BYTES
    )

    instruction_counter = 0  
    next_available_memory = combine_address_start  #  next memory location
    final_sum_addresses = []  # Store final sum addresses for each filter to give next layers as a input address

    for filter_idx in range(out_channels):  # Iterate over all output filters
        # taking output address of the channels with respect to kernels
        input_addresses = [
            prev_output["address1"] + (j * channel_offset) + (filter_idx * output_size * BYTES)
            for j in range(in_channels)
        ]#storing all channels of the same kernels in a input_adress

        temp_results = input_addresses[:]  # Copy input addresses for reduction
        step = 0  #  binary tree reduction steps

        while len(temp_results) > 1: #tep_results >1 then only we can add else that is the final sum
            next_temp_results = []

            # Pairwise summation for this filter
            for j in range(0, len(temp_results) - 1, 2):
                addr1, addr2 = temp_results[j], temp_results[j + 1]

                # Store temporary results **continuously** to avoid memory gaps
                temp_addr = next_available_memory
                next_available_memory += output_size * BYTES  # Move to the next free address

                instruction = [
                    #f"DWORD0: add_{instruction_counter}",  # Unique name
                    f"DWORD00: {generate_opcode(node_name)}",  
                    f"DWORD1 input addre1: {addr1}",
                    f"DWORD2 input addre 2: {addr2}",
                   # f"DWORD4: {output_width}",
                    f"DWORD3 INPUT0_H :{prev_output['height']}",
                    f"DWORD4 INPUT0_W :{prev_output['width']}",
                    f"DWORD5 INPUT0_SIZE :{prev_output['height']}*{prev_output['width']}",
                    f"DWORD10 KERNEL_H :{params['kernel_h']}",
                    f"DWORD11 KERNEL_W :{params['kernel_w']}",
                    f"DWORD12 STRIDE_H :{params['stride_h']}",
                    f"DWORD13 STRIDE_W :{params['stride_w']}",
                    f"DWORD14 PAD_TOP :{params['pad_top']}",
                    f"DWORD15 PAD_BOTTOM :{params['pad_bottom']}",
                    f"DWORD16 PAD_LEFT :{params['pad_left']}",
                    f"DWORD17 PAD_RIGHT :{params['pad_right']}",
                    f"DWORD18 KERNEL_SIZE :{params['kernel_h']}*{params['kernel_w']}",
                    f"DWORD5 output addr: {temp_addr}"
                ]

                file_name = os.path.join(folder_name, f"add_{instruction_counter}.txt")
                save_addition_instruction(file_name, [instruction])

                next_temp_results.append(temp_addr)  # Store temporary result
                instruction_counter += 1

            # If we have odd number of channels then add last channel to the temp_results for the addition
            if len(temp_results) % 2 == 1:
                next_temp_results.append(temp_results[-1])

            temp_results = next_temp_results  # Move to the next reduction level
            step += 1

        # Store the last remaining value as the final sum for this filter
        final_sum_addresses.append(temp_results[0])

    return final_sum_addresses  # Return addresses to be used as input for the next layer
def generate_opcode(node_type):
    opcode = 0  # starting all bits are 0

    # 8 fixed bit positions for each operation
    node_bit_positions = {
        "Conv": 0,     
        "Relu": 1,    
        "LRN": 2,      
        "MaxPool": 3,   
        "addition": 4,
        "DMA_wr":5,
        "DMA_re":6,
        "quantization":7  
    }

    if node_type in node_bit_positions:
        # Enable the node bits based on the node and opeartion required  and remaining  0
        opcode = 1 << node_bit_positions[node_type]

    return f"{opcode:08b}  (0x{opcode:08X})"




def generate_dword_instructions(model):
    input_height, input_width = extract_input_dimensions(model)  #extracting input dimensions for the first time
    prev_output_channels = None
    layer_counters = {"conv": 0, "ReLU": 0, "LRN": 0, "MaxPool": 0, "Addition": 0} # To track which operation excuted  how many times before the current node

    first_conv = True
    prev_output = {"address1": 0x000000, "height": input_height, "width": input_width, "address2": 0}

    for node in model.graph.node:
        node_name = node.name
        opcode = generate_opcode(node.op_type)

        if node.op_type == "Conv":
            params = extract_conv_params(node, prev_output_channels, prev_output["height"], prev_output["width"])

            output_height = (params['input_height'] - params['kernel_h'] + params['pad_top'] + params['pad_bottom']) // params['stride_h'] + 1
            output_width = (params['input_width'] - params['kernel_w'] + params['pad_left'] + params['pad_right']) // params['stride_w'] + 1
            folder_name = node.output[0]

            if first_conv:
                prev_output = {"address1": 0x000000, "height": input_height, "width": input_width, "address2": 0}
                first_conv = False

                input_size = params['input_width'] * params['input_height']
                kernel_size = params['kernel_h'] * params['kernel_w']
                output_size = output_width * output_height

                input_addr = [prev_output["address1"] + i * input_size * BYTES for i in range(params['in_channels'])]
                kernel_start = input_addr[-1] + input_size * BYTES  # Start kernel storage after the last input
                kernel_addr = [kernel_start + i * kernel_size * BYTES for i in range(params['out_channels'])]

                k_s = params['in_channels'] * input_size * BYTES
                k_ad = [k_s + i * kernel_size * BYTES for i in range(params['out_channels'])]

                output_start_address = k_s + params['out_channels'] * kernel_size * BYTES
                output_chunk_size = output_size * BYTES

            else:
                  input_end_addr = prev_output["address2"]  # Last stored address in the previous layer
                  input_size = params['input_width'] * params['input_height']
                  kernel_size = params['kernel_h'] * params['kernel_w']
                  output_size = output_width * output_height

                  # if input_addr is a list
                  if isinstance(prev_output["address1"], list):
                      input_addr = [addr + i * input_size * BYTES for i, addr in enumerate(prev_output["address1"])]
                  else:
                      input_addr = [prev_output["address1"] + i * input_size * BYTES for i in range(params['in_channels'])]

                  k_s = input_end_addr  # Kernel starts after the last stored address
                  #k_s=prev_output['address2']
                  kernel_addr = [k_s + i * kernel_size * BYTES for i in range(params['out_channels'])]  

                  output_start_address = k_s + params['out_channels'] * kernel_size * BYTES
                  output_chunk_size = output_size * BYTES


           
            for in_channel in range(params['in_channels']):
                for out_channel in range(params['out_channels']):
                    #folder_name=node.output[0]
                    #file_name={folder_name}"_c"{in_channel+1}"_k{out_channel+1}
                    file_name = f"{folder_name}_c{in_channel+1}_k{out_channel+1}" 
                    dword_instr = [
                        #f"DWORD0:{in_channel+1}_f{out_channel+1}",
                        f"DWORD0: {opcode}",  
                        f"DWORD1 INPUT0_ADDRESS :{input_addr[in_channel]}",
                        f"DWORD2 INPUT0_H :{prev_output['height']}",
                        f"DWORD3 INPUT0_W :{prev_output['width']}",
                        f"DWORD4 INPUT0_SIZE :{prev_output['height']}*{prev_output['width']}",
                        f"DWORD10 KERNEL_H :{params['kernel_h']}",
                        f"DWORD11 KERNEL_W :{params['kernel_w']}",
                        f"DWORD12 STRIDE_H :{params['stride_h']}",
                        f"DWORD13 STRIDE_W :{params['stride_w']}",
                        f"DWORD14 PAD_TOP :{params['pad_top']}",
                        f"DWORD16 PAD_LEFT :{params['pad_left']}",
                        f"DWORD17 PAD_RIGHT :{params['pad_right']}",
                        f"DWORD18 KERNEL_SIZE :{params['kernel_h']}*{params['kernel_w']}",
                        f"DWORD20 KERNEL_ADDRESS :{kernel_addr[out_channel]}",
                        f"DWORD28 OUTPUT_ADDRESS :{output_start_address + (in_channel * params['out_channels'] + out_channel) * output_chunk_size}",
                        f"DWORD29 OUTPUT_H :{output_height}",
                        f"DWORD30 OUTPUT_W :{output_width}",
                        f"DWORD31 OUTPUT_SIZE: {prev_output['height']}*{prev_output['width']}"
                    ]
                    # conv_instr.append(dword_instr)

                    save_layer_instructions(dword_instr,folder_name,file_name)

            prev_output = {"address1": output_start_address, "height": output_height, "width": output_width, "address2": 0}
            prev_output_channels = params['out_channels']

            final_adr = generate_addition_instructions(prev_output, params, output_size,layer_counters, node_name)
            prev_output = {"address1": final_adr, "height": output_height, "width": output_width, "address2": 0}
            prev_output_channels = params['out_channels']
            output_end = final_adr[-1] + output_size * BYTES
            output_s = final_adr
            # here output_s is a variable to store final address after addition which is coming from the adiition instruction fucntion
        elif node.op_type == "Relu":
       
            folder_name=node.output[0]
            relu_instr = []
            for i in range(prev_output_channels):
                file_name=f"{folder_name}_c{i}"
                dword_inst=[
                    #f"DWORD0 :Rl_OP {i+1}",
                    f"DWORD0: {opcode}",  
                    # f"DWORD1 INPUT0_ADDRESS: {prev_output['address'] + 2*i * output_size * BYTES}",
                    f"DWORD1 INPUT0_ADDRESS: {output_s[i]}",
                    f"DWORD2 INPUT0_H: {prev_output['height']}",
                    f"DWORD3 INPUT0_W: {prev_output['width']}",
                    f"DWORD4 INPUT0_SIZE: {prev_output['height']}*{prev_output['width']}",
                    f"DWORD28 OUTPUT_ADDRESS: {output_s[i]}",
                    # f"DWORD28 OUTPUT_ADDRESS: {prev_output['address'] + 2*i * output_size * BYTES}",
                    f"DWORD29 OUTPUT_H: {prev_output['height']}",
                    f"DWORD30 OUTPUT_W: {prev_output['width']}",
                    f"DWORD31 OUTPUT_SIZE: {prev_output['height']}*{prev_output['width']}"
                ]
                save_layer_instructions(dword_instr,folder_name,file_name)
            prev_output = {"address1": output_s, "height": output_height, "width": output_width,"address2":output_end}

        elif node.op_type == "LRN":
            folder_name=node.output[0]
            params = extract_lrn_params(node)
            lrn_instr = []
            for i in range(prev_output_channels):
                file_name=f"{folder_name}_c{i}"
                dword_instr=[
                    #f"DWORD0 :lrn_OP{i+1}",
                    f"DWORD0: {opcode}",  
                    f"DWORD1 INPUT0_ADDRESS: {output_s[i]}",
                    # f"DWORD1 INPUT0_ADDRESS: {prev_output['address'] + 2*i * output_size * BYTES}",
                    f"DWORD2 INPUT0_H: {prev_output['height']}",
                    f"DWORD3 INPUT0_W: {prev_output['width']}",
                    f"DWORD4 INPUT0_SIZE: {prev_output['height']}*{prev_output['width']}",
                    f"DWORD5 BNORM_ALPHA: {params['alpha']}",
                    f"DWORD6 BNORM_BETA: {params['beta']}",
                    f"DWORD8 BNORM_bias: {params['bias']}",
                    f"DWORD9 BNORM_size: {params['size']}",
                    f"DWORD28 OUTPUT_ADDRESS: {output_s[i]}",
                    # f"DWORD28 OUTPUT_ADDRESS: {prev_output['address'] + 2*i * output_size * BYTES}",
                    f"DWORD29 OUTPUT_H: {prev_output['height']}",
                    f"DWORD30 OUTPUT_W: {prev_output['width']}",
                    f"DWORD31 OUTPUT_SIZE: {prev_output['height']}*{prev_output['width']}"
                ]
                save_layer_instructions(dword_instr,folder_name,file_name)



        elif node.op_type=="Gemm":
          break

        elif node.op_type == "MaxPool":
            folder_name=node.output[0]
            params = extract_pool_params(node)
            output_ss=final_adr
            output_end1=final_adr[-1]+output_size*BYTES
            output_height = (prev_output['height'] - params['kernel_h'] + params['pad_top'] + params['pad_left']) // params['stride_h'] + 1
            output_width = (prev_output['width'] - params['kernel_w'] + params['pad_top'] + params['pad_left']) // params['stride_w'] + 1
            #output_size = output_height * output_width
            output_end=output_end1+output_width*output_height*BYTES*prev_output_channels
            # output_s = output_end1+i * output_width*output_height * BYTES
            output_s = [output_end1 + i * output_width * output_height * BYTES for i in range(prev_output_channels)]
            maxpool_instr = []
            for i in range(prev_output_channels):
                file_name=f"{folder_name}_c{i}"
                dword_instr=[
                    #f"DWORD0 :MAX_OP{i+1}",
                    f"DWORD0: {opcode}",  
                    f"DWORD1 INPUT0_ADDRESS: {output_ss[i]}",
                    # f"DWORD1 INPUT0_ADDRESS: {prev_output['address'] + 2*i * output_size * BYTES}",
                    f"DWORD2 INPUT0_H: {prev_output['height']}",
                    f"DWORD3 INPUT0_W: {prev_output['width']}",
                    f"DWORD4 INPUT0_SIZE: {prev_output['height']}*{prev_output['width']}",
                    f"DWORD5 MAXPOOL_KERNEL_H: {params['kernel_h']}",
                    f"DWORD6 MAXPOOL_KERNEL_W: {params['kernel_w']}",
                    f"DWORD7 MAXPOOL_STRIDE_H: {params['stride_h']}",
                    f"DWORD8 MAXPOOL_STRIDE_W: {params['stride_w']}",
                    f"DWORD9 MAXPOOL_PAD_TOP:{params['pad_top']}",
                    f"DWORD10 MAXPOOL_PAD_bottom:{params['pad_bottom']}",
                    f"DWORD13 OUTPUT_H: {output_height}",     
                    f"DWORD14 OUTPUT_W: {output_width}",
                    f"DWORD15 OUTPUT_SIZE: {output_height}*{output_width}",
                    f"DWORD16 OUTPUT_ADDRESS: {output_s[i]}"
                ]

                save_layer_instructions(dword_instr,folder_name,file_name)
            prev_output = {"address1": output_s, "height": output_height, "width": output_width,"address2":output_end}



if __name__ == "__main__":
    model_path = "bvlcalexnet-6.onnx" # Taking onnx model
    model = load_onnx_model(model_path) #calling the function to load an onnx model
    generate_dword_instructions(model)   #here is the fucntion to Generate Instructions
    print("DWORD-formatted instructions generated successfully!")
