from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer
from torch import Tensor
from typing import Dict, List, Tuple, Callable, Any

def get_input_length(input: Dict[str, Tensor]) -> int:
    if isinstance(input, Tensor):
        return len(input[0])
    return len(input['input_ids'].squeeze())


def get_output_length(output: GenerateDecoderOnlyOutput) -> int:
    if isinstance(output, Tensor):
        return len(output[0])
    return len(output['sequences'].squeeze())

class ForwardPassProfile:

    def __init__(self, input_token: int, output_token: int):
        self.input_token = input_token
        self.output_token = output_token
        self.hidden_states = []
    
    def add(self, hidden_state: Tensor) -> None:
        self.hidden_states.append(hidden_state)

class GenerationProfile:

    def __init__(self, input: BatchEncoding, output: GenerateDecoderOnlyOutput,
                       tokenizer: AutoTokenizer = None):
        self.input_length = get_input_length(input)
        self.output_length = get_output_length(output)
        self.output = output
        self.forward_passes : List[ForwardPassProfile] = []
        self.tokenizer = tokenizer

        p = 0
        for i in range(self.output_length - self.input_length):
            # tuple of each layers' hidden state
            hidden_states : Tuple[Tensor] = self.output['hidden_states'][i]

            # retrieve length of the first hidden layer for token(s)
            _, length, _ = hidden_states[0].shape 

            for j in range(length):
                input_token = self.output['sequences'][0, p]
                output_token = self.output['sequences'][0, p+1]

                # represents input_token -> [list of hidden states] -> output_token
                fp_profile = ForwardPassProfile(input_token, output_token)
                for hidden_state in hidden_states:
                    fp_profile.add(hidden_state[:,j,:])
                
                # List of forward pass profiles
                self.forward_passes.append(fp_profile)
                p += 1

    def print(self, enable_string:bool = True) -> None:
        for fp in self.forward_passes:
            inp, outp = fp.input_token, fp.output_token
            if self.tokenizer and enable_string:
                inp, outp = self.tokenizer.decode(inp), self.tokenizer.decode(outp)
                
            print(f"{inp} -> {outp}")
    
    def print_summary(self):
        print("Input length:", self.input_length)
        print("Output length:", self.output_length)
    
    def query_single(self, funct: Callable[[Tensor], Any]) -> None:
        for fp in self.forward_passes:
            inp, outp = fp.input_token, fp.output_token
            state = fp.hidden_states
            if self.tokenizer:
                inp, outp = self.tokenizer.decode(inp), self.tokenizer.decode(outp)
            
            print(f"{inp} -> {outp}")
            for i in range(len(state)):
                print(funct(state[i]))
            
    def query_consecutive(self, funct: Callable[[Tensor, Tensor], Any]) -> None:
        for fp in self.forward_passes:
            inp, outp = fp.input_token, fp.output_token
            state = fp.hidden_states
            if self.tokenizer:
                inp, outp = self.tokenizer.decode(inp), self.tokenizer.decode(outp)
            
            print(f"{inp} -> {outp}")
            for i in range(len(state)-1):
                print(funct(state[i], state[i+1]))
    
    def query_pairwise(self, funct: Callable[[Tensor, Tensor], Any]) -> None:
        for fp in self.forward_passes:
            inp, outp = fp.input_token, fp.output_token
            state = fp.hidden_states
            if self.tokenizer:
                inp, outp = self.tokenizer.decode(inp), self.tokenizer.decode(outp)
            
            print(f"{inp} -> {outp}")
            for i in range(len(state)):
                for j in range(i, len(state)):
                    print(i, j, funct(state[i], state[j]))
                print()

    def query_custom(self, funct: Callable[[List[Tensor]], None]) -> None:
        for fp in self.forward_passes:
            inp, outp = fp.input_token, fp.output_token
            state = fp.hidden_states
            if self.tokenizer:
                inp, outp = self.tokenizer.decode(inp), self.tokenizer.decode(outp)
            
            print(f"{inp} -> {outp}")
            funct(state)