from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer
from torch import Tensor

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import Dict, List, Tuple, Callable, Optional, Any
from enum import Enum


class QueryMode(Enum):
    SINGLE = 1
    CONSECUTIVE = 2
    PAIRWISE = 3
    CUSTOM = 4

class OutputMode(Enum):
    PRINT = 1
    PLOT = 2

def get_input_length(input: Dict[str, Tensor]) -> int:
    if isinstance(input, Tensor):
        return len(input[0])
    return len(input['input_ids'].squeeze())


def get_output_length(output: GenerateDecoderOnlyOutput) -> int:
    if isinstance(output, Tensor):
        return len(output[0])
    return len(output['sequences'].squeeze())

class ProfileContext(Enum):
    HIDDEN = 1
    MODIFIED = 2
    CONSECUTIVE = 3
    PAIRWISE = 4

class ForwardPassProfile:
    """
    Keeps track of information about a single forward pass, including the input
    token, output token, and hidden states (or processed information).

    [self.input_token] -> [list of tensors, or other information] -> [self.output_token]
    """

    def __init__(self, input_token: int,
                       output_token: int,
                       context: ProfileContext = ProfileContext.HIDDEN):
        self.input_token = input_token
        self.output_token = output_token
        self.hidden_states : List[Tensor | Any] = []
        self.context : ProfileContext = ProfileContext.HIDDEN
    
    def add(self, hidden_state: Tensor) -> None:
        self.hidden_states.append(hidden_state)
    
    def apply_function(self, funct:Callable[[Tensor, Optional[Tensor]], Any],
                              query_mode:QueryMode) -> "ForwardPassProfile":
        
        new_profile = ForwardPassProfile(self.input_token, self.output_token)
        
        match query_mode:
            case QueryMode.SINGLE:
                new_profile.context = ProfileContext.MODIFIED
                for i in range(len(self.hidden_states)):
                    new_profile.add(funct(self.hidden_states[i]))
            
            case QueryMode.CONSECUTIVE:
                new_profile.context = ProfileContext.CONSECUTIVE
                for i in range(len(self.hidden_states)-1):
                    new_profile.add(funct(self.hidden_states[i], self.hidden_states[i+1]))
            
            case QueryMode.PAIRWISE:
                new_profile.context = ProfileContext.PAIRWISE
                n = len(self.hidden_states)
                new_profile.hidden_states = [[None]*n for _ in range(n)]
                for i in range(n):
                    for j in range(i, n):
                        new_profile.hidden_states[i][j] = funct(self.hidden_states[i], self.hidden_states[j])
            
            case QueryMode.CUSTOM:
                new_profile.context = ProfileContext.PAIRWISE
                pass # TODO: SOME IMPLEMENTATION
        
        return new_profile

    def print(self, tokenizer:Optional[AutoTokenizer] = None):
        i, o = self.input_token, self.output_token
        if tokenizer:
            i, o = f"\"{tokenizer.decode(i)}\"", f"\"{tokenizer.decode(o)}\""
        print(f"{i} ----> {o}")

        match self.context:
            case ProfileContext.HIDDEN | ProfileContext.MODIFIED | ProfileContext.CONSECUTIVE:
                for i, state in enumerate(self.hidden_states):
                    print(f"{i}:", state)
                print()
            case ProfileContext.PAIRWISE:
                for i, state in enumerate(self.hidden_states):
                    for j, s in enumerate(state):
                        print(f"{i}, {j}:", s)
                    print()
    
    def plot(self, tokenizer:Optional[AutoTokenizer] = None):
        i, o = self.input_token, self.output_token
        if tokenizer:
            i, o = f"\"{tokenizer.decode(i)}\"", f"\"{tokenizer.decode(o)}\""
        
        matrix = np.array([[float(x) if x is not None else np.nan for x in row] for row in self.hidden_states])
        cmap = sns.color_palette("RdYlGn", as_cmap=True)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matrix, cmap=cmap, annot=True, linewidths=0.5, linecolor='gray',
                    cbar=True, square=True, mask=np.isnan(matrix), fmt=".2f")
        plt.title(f"{i} ----> {o}")
        plt.show()

class GenerationProfile:
    """
    Generation profile keeps track of multiple ForwardPassProfile.
    """

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


    def print(self, forward_passes:Optional[List[ForwardPassProfile]] = None,
                    enable_string: bool = True) -> None:
        if not forward_passes:
            forward_passes = self.forward_passes
        for fp in forward_passes:
            fp.print(None if not enable_string else self.tokenizer)
        
    def plot(self, forward_passes:Optional[List[ForwardPassProfile]] = None,
                    enable_string: bool = True) -> None:
            for fp in forward_passes:
                fp.plot(None if not enable_string else self.tokenizer)
    
    def print_summary(self):
        print("Input length:", self.input_length)
        print("Output length:", self.output_length)
    

    def query(self, funct:Callable[[Tensor, Optional[Tensor]], Any],
                    query_mode: QueryMode,
                    output_mode: OutputMode = OutputMode.PRINT) -> None:
        
        res : List[ForwardPassProfile] = []
        
        for fp in self.forward_passes:
            res.append(fp.apply_function(funct, query_mode))
        
        match output_mode:
            
            case OutputMode.PRINT:
                self.print(res)
            
            case OutputMode.PLOT:
                self.plot(res)