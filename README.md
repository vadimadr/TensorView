TensorView
========

This library aims to simplify working with tensors (multi-dimensional data) (e.g. outputs or intermediate results of neural networks), via providing
**TensorView** - non-owning (hence copy-free) pointers to an existing data, that allows to use indexing and numpy-style high-level operations on this data.

The library is header-only and framework agnostic.

For usage examples please see tests.

Example:
```c++
float* buffer = .... // ptr to some buffer with known size
auto view = make_view(buffer, {16, 3, 256, 256});

float& x = view.at(0, 1, 4, 4); // index value
auto subview  = view.at(0, 1); // creating sub-views

std::cout << subview; // print data in sub-view
float max_val = subview.max(); 
subview.map_([](float x) { return x * x; }); // transforming data
float product = subview.reduce([](float x, float y) { x * y; }, /*initial=*/ 1.);

view.assign_(0); // filling data
```


**Current status**
- ~~Initializing tensor view with a pointer to an existing data.~~
- ~~Indexing to a specific element~~
- ~~Indexing to a subview~~
- ~~Assigning data to view from constant / other views~~
- ~~Element-wise map (unary / binary)~~
- ~~Reduction operation (reduce all / reduce over axis)~~
- ~~Printing data from view to the output stream~~
- ~~Broadcast semantics~~
- ~~Customizing broadcast semantics (Prohibit broadcast / Explicit broadcast (only axes with size 1 will be extended) / Implicit broadcast)~~
- Basic operations arithmetic operations - in progress
- Common functions - in progress 
- Owning tensor container - TBD.