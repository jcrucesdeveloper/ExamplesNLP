from operator import Operator

class Operator:
    def __init__(self, name, parameters, return_type):
        self.name = name
        self.parameters = parameters
        self.return_type = return_type
    
    def __str__(self):
        # Format the parameters as a comma-separated string
        params_str = ", ".join(str(p) for p in self.parameters)
        # Pad the name to a fixed width (e.g., 15 characters)
        return f"{self.name:<15} | [{params_str}] → {self.return_type}"

# Documentation: https://pytorch.org/docs/stable/torch.html#creation-ops
CREATION_OPERATIONS = [
    # tensor | https://pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor
    Operator("tensor", ["data", "*","dtype=None", "device=None", "requires_grad=False", "pin_memory=False"], "Tensor"),
]

# Documentation: https://pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops
INDEXING_SLICING_JOINING_MUTATING_OPERATIONS = [
    # adjoint | https://pytorch.org/docs/stable/generated/torch.adjoint.html#torch.adjoint
    Operator("adjoint", ["Tensor"], "Tensor"),  
    # argwhere | https://pytorch.org/docs/stable/generated/torch.argwhere.html#torch.argwhere
    Operator("argwhere", ["Tensor"], "Tensor"),  
    # cat | https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
    Operator("cat", ["Sequence of Tensor", "int | None ", "*", "out=None"], "Tensor"),  
    # conj | https://pytorch.org/docs/stable/generated/torch.conj.html#torch.conj
    Operator("conj", ["Tensor"], "Tensor"),  
    # chunk | https://pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk
    Operator("chunk", ["Tensor", "int", "int"], "List of Tensors"),  
    # dsplit | https://pytorch.org/docs/stable/generated/torch.dsplit.html#torch.dsplit
    Operator("dsplit", ["int", "indices or section"], "List of Tensors"),  
    # column_stack | https://pytorch.org/docs/stable/generated/torch.column_stack.html#torch.column_stack
    Operator("column_stack", ["Sequence of Tensors"], "Tensor"),  
    # dstack | https://pytorch.org/docs/stable/generated/torch.dstack.html#torch.dstack
    Operator("dstack", ["Sequence of Tensors"], "Tensor"),  
    # gather | https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather
    Operator("gather", ["Tensor", "int", "int", "LongTensor"], "Tensor"),  
    # hsplit | https://pytorch.org/docs/stable/generated/torch.hsplit.html#torch.hsplit
    Operator("hsplit", ["Tensor", "int | list | tuple"], "List of Tensors"),  
    # hstack | https://pytorch.org/docs/stable/generated/torch.hstack.html#torch.hstack 
    Operator("hstack", ["Sequence of Tensors"], "Tensor"),  
]

# Documentation: https://pytorch.org/docs/stable/torch.html#accelerators
ACCELERATOR_OPERATIONS = [
    
]

# Documentation: https://pytorch.org/docs/stable/torch.html#random-sampling
RANDOM_SAMPLING_OPERATIONS = [
    
]

# Documentationhttps://pytorch.org/docs/stable/torch.html#serialization
SERIALIZATION_OPERATIONS = [
    
]

# Documentation: https://pytorch.org/docs/stable/torch.html#parallelism
PARALLELISM_OPERATIONS = [
    
]

# Documentation: https://pytorch.org/docs/stable/torch.html#locally-disabling-gradient-computation
LOCALLY_DISABLING_GRADIENT_COMPUTATION_OPERATIONS = [
    
]

# Documentation: https://pytorch.org/docs/stable/torch.html#math-operations
MATH_OPERATIONS = [
    
]

# Documentation: https://pytorch.org/docs/stable/torch.html#utilities
UTILITIES_OPERATIONS = [
    
]

PYTORCH_OPERATIONS = [
    CREATION_OPERATIONS,
    INDEXING_SLICING_JOINING_MUTATING_OPERATIONS,
    ACCELERATOR_OPERATIONS,
    RANDOM_SAMPLING_OPERATIONS,
    SERIALIZATION_OPERATIONS,
    PARALLELISM_OPERATIONS,
    LOCALLY_DISABLING_GRADIENT_COMPUTATION_OPERATIONS,
    MATH_OPERATIONS,
    UTILITIES_OPERATIONS,
    MATH_OPERATIONS,
    UTILITIES_OPERATIONS,
]

def main():
    # operation_categories = {
    #     "Creation Operations": CREATION_OPERATIONS,
    #     "Indexing, Slicing, Joining, Mutating Operations": INDEXING_SLICING_JOINING_MUTATING_OPERATIONS,
    #     "Accelerator Operations": ACCELERATOR_OPERATIONS,
    #     "Random Sampling Operations": RANDOM_SAMPLING_OPERATIONS,
    #     "Serialization Operations": SERIALIZATION_OPERATIONS,
    #     "Parallelism Operations": PARALLELISM_OPERATIONS,
    #     "Locally Disabling Gradient Computation Operations": LOCALLY_DISABLING_GRADIENT_COMPUTATION_OPERATIONS,
    #     "Math Operations": MATH_OPERATIONS,
    #     "Utilities Operations": UTILITIES_OPERATIONS
    # }

    for operator in INDEXING_SLICING_JOINING_MUTATING_OPERATIONS:
        print(operator)

if __name__ == "__main__":
    main()
