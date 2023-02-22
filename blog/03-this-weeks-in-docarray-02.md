# This week(s) in DocArray

It’s already been a month since the [last alpha release](https://github.com/docarray/docarray/releases/tag/2023.01.18.alpha) of DocArray v2. Since then a lot has happened: we’ve merged features that we’re really proud of and keep crying tears of joy and misery trying to coerce Python into doing what we want. If you want to learn about interesting Python edge cases or follow the advancement of DocArray v2 development then you’re at the right place in this dev blog!

For those who don’t know, DocArray is a library for **representing, sending, and storing multi-modal data**, with a focus on applications in **ML** and **Neural Search.** The project just moved to the Linux foundation AI and Data and to celebrate its first birthday we decided to rewrite it from scratch, mainly because of a design shift and a will to solidify the codebase from the ground up.

## MultiModalDataset

As part of our goal to make DocArray the go-to library for representing, sending, and storing multi-modal data, we‘ve added a `MultiModalDataset` class to easily convert DocumentArrays into PyTorch Dataset compliant datasets that can be used in the PyTorch DataLoader.

All you need is a DocumentArray and a dictionary of preprocessing functions and you’re up and running!

```python
from docarray import BaseDocument, DocumentArray
from docarray.data import MultiModalDataset
from docarray.documents import Text
from torch.utils.data import DataLoader

class Thesis(BaseDocument):
    title: Text

class Student(BaseDocument):
    thesis: Thesis

da: DocumentArray[Student] = get_students()
ds: MultiModalDataset[Student] = MultiModalDataset[Student](
    da,
    preprocessing={'thesis.title': embed_title, 'thesis': normalize_embedding},
)
loader: DataLoader = DataLoader(
    ds, batch_size=4, collate_fn=MultiModalDataset[Student].collate_fn
)

# Use your loader just like any other dataloader for awesome DL training
```

If you’re interested in using DocArray for training, check out our [example notebook](https://github.com/docarray/docarray/blob/feat-rewrite-v2/docs/tutorials/multimodal_training_and_serving.md), or take a peek at [implementation details of MultiModalDataset](https://github.com/docarray/docarray/pull/1049).

## TensorFlow support

After recently adding PyTorch support, we’ve now gone on to add TensorFlow support to DocArray v2. Like with PyTorch, we planned on subclassing the `tensorflow.Tensor` class with our `TensorFlowTensor` class. By doing so we want to allow DocArray to run operations on it while also being able to hand over our `TensorFlowTensor` instance to ML models or TensorFlow functions without TensorFlow being confused about this instance’s class but instead recognizing it as its own. Since we implemented this for PyTorch already, this should be easy, right?

But stop, not so fast. At first glance, TensorFlow tensors seem to be of class `tf.Tensor`, right?

```python
import tensorflow as tf

tensor = tf.zeros((5,))
tensor
```

```python
<tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 0., 0., 0., 0.], dtype=float32)>
```

When trying to subclass `tf.Tensor` though, we notice that this does not seem to work:

```python
from typing import Any, Type, Union, cast

import tensorflow as tf
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from pydantic.tools import parse_obj_as

class TensorFlowTensor(AbstractTensor, tf.Tensor):
    @classmethod
    def validate(cls, value, field, config) -> 'TensorFlowTensor':
        if isinstance(value, tf.Tensor):
            value.__class__ = cls
            return cast(TensorFlowTensor, value)
        else:
            raise ValueError(f'Expected a tf.Tensor, got {type(value)}')

our_tensor = parse_obj_as(TensorFlowTensor, tf.zeros((5,)))  # will fail
```

Parsing a `tf.Tensor` as `TensorFlowTensor` will fail:

```python
pydantic.error_wrappers.ValidationError: 1 validation error for ParsingModel[TensorFlowTensor]
__root__
  __class__ assignment: 'TensorFlowTensor' object layout differs from 'tensorflow.python.framework.ops.EagerTensor' (type=type_error)
```

But wait, here they talk about an `EagerTensor`, not `tf.Tensor`. This is because TensorFlow actually supports eager execution and as well as graph execution. It defaults to eager execution, where operations are evaluated immediately. In graph execution, a computational graph is constructed for later evaluation.

So maybe we just need to extend TensorFlow’s `EagerTensor` then!

This, however, doesn’t work either, because the class `EagerTensor` is created on the fly, which is why trying to extend this class will fail with:

`TypeError: type 'tensorflow.python.framework.ops.EagerTensor' is not an acceptable base type`.

With all that being said, we’ve decided to go with the following solution for now:

Instead of extending TensorFlow’s tensor, we store a `tf.Tensor` instance as an attribute of our `TensorFlowTensor` class. Therefore if you want to perform operations on the tensor data or hand it over to your ML model, you have to explicitly access the `.tensor` attribute:

```python
import tensorflow as tf
from docarray.typing import TensorFlowTensor

t = TensorFlowTensor(tensor=tf.zeros((224, 224)))

# tensorflow functions
broadcasted = tf.broadcast_to(t.tensor, (3, 224, 224))
broadcasted = tf.broadcast_to(t.unwrap(), (3, 224, 224))
broadcasted = tf.broadcast_to(t, (3, 224, 224))  # this will fail
```

In future we plan to take a closer look and find a solution that enables handling `TensorFlowTensor`s just like our `TorchTensor`s. In particular, we plan to investigate if there’s an equivalent in TensorFlow to Torch’s `__torch_function__()`, which we told you about in the [previous blog post](https://jina.ai/news/this-week-in-docarray-1). With such an equivalent and some tricks here and there we hope to enable smooth usage or our `TensorFlowTensor` class and make it feel like it’s a subclass of TensorFlow’s tensor, without it actually being one.

## Nested class and multiprocessing

As part of our goal to make DocArray the go-to library for representing, sending, and storing multi-modal data, it’s important that DocumentArrays support multiprocessing, namely processing on multi-CPU cores.

In particular, we recently implemented a `MultiModalDataset` class to easily convert a DocumentArray into a dataset that can be used in the PyTorch DataLoader. The PyTorch DataLoader wraps the Python multiprocessing module to implement preprocessing with multiple CPUs.

**The problem**

One of the well-known issues with multiprocessing is that it doesn’t support classes that are declared inside a function:

```python
def get_class():
    class B:
        ...

    return B

MyClass = get_class()

def foo(*args):
    return MyClass()

import multiprocessing as mp

with mp.get_context('fork').Pool(2) as p:
    print(p.map(foo, range(2)))
```

```bash
Traceback (most recent call last):
  File "/Users/jackmin/Jina/docarray/meow.py", line 13, in <module>
    print(p.map(foo, range(2)))
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 367, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 774, in get
    raise self._value
multiprocessing.pool.MaybeEncodingError: Error sending result: '[<__main__.get_class.<locals>.B object at 0x10152e950>]'. Reason: 'AttributeError("Can't pickle local object 'get_class.<locals>.B'")'
```

**Pickling**

This is because multiprocessing uses pickle to share objects with workers. Pickling only saves the qualified class name of an object and unpickling requires re-importing the class by its qualified class name. For that to work, the class needs a global qualified name. Classes defined by functions are local and thus cannot be pickled:

```python
def get_class():
    class B:
        ...

    return B

MyClass = get_class()

import pickle

pickle.dump(MyClass(), open('meow.pkl', 'wb'))
```

```bash
Traceback (most recent call last):
  File "/Users/jackmin/Jina/docarray/meow.py", line 10, in <module>
    pickle.dump(MyClass(), open("meow.pkl", "wb"))
AttributeError: Can't pickle local object 'get_class.<locals>.B'
```

In order to get around this, we need to make the declared class global:

```python
def get_class():
    global B

    class B:
        ...

    return B

MyClass = get_class()

import pickle

pickle.dump(MyClass(), open('meow.pkl', 'wb'))
```

We can now load the pickles in a separate process as long as the process has a declaration of our class:

```python
def get_class():
    global B

    class B:
        ...

    return B

MyClass = get_class()

import pickle

print(pickle.load(open('meow.pkl', 'rb')))
```

It doesn’t really matter how it ends up in the global scope. We can even do this:

```python
class B:
    ...

import pickle

print(pickle.load(open('meow.pkl', 'rb')))
```

**The fix?**

Ok. It just wants it to be global. Simple enough right? Let’s just plop `global` in front of our declaration and be done with it.

```python
def get_class():
    global B

    class B:
        ...

    return B

MyClass = get_class()

def foo(*args):
    return MyClass()

import multiprocessing as mp

with mp.get_context('fork').Pool(2) as p:
    print(p.map(foo, range(2)))
```

Yay this runs fine. But, what if our function returns a different class depending on the input arguments? I mean, why else would I want to return a class from a function?

```python
def get_class(version: int):
    global B

    class B:
        VERSION: int = version

    return B

C1 = get_class(1)
C2 = get_class(2)

def get_version(cls):
    print(cls)
    return cls.VERSION

import multiprocessing as mp

with mp.get_context('fork').Pool(2) as p:
    print(p.map(get_version, [C1, C2]))
```

```bash
<class '__main__.B'>
Traceback (most recent call last):
  File "/Users/jackmin/Jina/docarray/meow.py", line 19, in <module>
    print(p.map(get_version, [C1, C2]))
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 367, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 774, in get
    raise self._value
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 540, in _handle_tasks
    put(task)
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/connection.py", line 211, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
_pickle.PicklingError: Can't pickle <class '__main__.B'>: it's not the same object as __main__.B
```

`Can't pickle <class '__main__.B'>: it's not the same object as __main__.B`. What does that mean?

**Double declaration**

Well, our little trick has some caveats. By performing a global declaration, we’re essentially taking the class declaration out into the top-level scope. This means we’re essentially doing this:

```python
class B:
    VERSION: int = 1

C1 = B

class B:
    VERSION: int = 2

C2 = B

def get_version(cls):
    print(cls)
    return cls.VERSION

import multiprocessing as mp

with mp.get_context('fork').Pool(2) as p:
    print(p.map(get_version, [C1, C2]))
```

If we run this code, we get the exact same error we got before:

```bash
<class '__main__.B'>
Traceback (most recent call last):
  File "/Users/jackmin/Jina/docarray/wow.py", line 15, in <module>
    print(p.map(get_version, [C1, C2]))
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 367, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 774, in get
    raise self._value
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/pool.py", line 540, in _handle_tasks
    put(task)
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/connection.py", line 211, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "/Users/jackmin/miniconda3/envs/docarray/lib/python3.10/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
_pickle.PicklingError: Can't pickle <class '__main__.B'>: it's not the same object as __main__.B
```

What happened here? By declaring the class twice, we’ve overwritten our first `Class B` with a second `Class B` in the global scope. Pickle is aware of this when it tries to serialize `C1`. It will notice that the `Class B` `C1` refers to is no longer the top-level one and raises an exception.

**Qualified names must be unique**

The issue here is that both `Class B`s have the same qualified name. Thus, both definitions are fighting over who gets to be the one the global dictionary knows about.

We can resolve this conflict and allow our two classes to live together peacefully by moving them to different qualified names and thus, different keys in the global scope:

```python
def get_class(version: int):
    global B

    class B:
        VERSION: int = version

    B.__qualname__ = B.__qualname__ + str(version)
    globals()[f'B{version}'] = B
    return B

C1 = get_class(1)
C2 = get_class(2)

def get_version(cls):
    print('Class Name:', cls.__name__)
    print('Class Qualified Name:', cls.__qualname__)
    print('Type repr', cls)
    return cls.VERSION

import multiprocessing as mp

with mp.get_context('fork').Pool(2) as p:
    print(p.map(get_version, [C1, C2]))
```

```bash
Class Name: B
Class Qualified Name: B1
Type repr <class '__main__.B1'>
Class Name: B
Class Qualified Name: B2
Type repr <class '__main__.B2'>
[1, 2]
```

Notice that although the two classes have different qualified names, they can still share the same name with no issues. Printing the type does however show the qualified name.

**Implementation example**

If you’d like to see how we used this pattern to implement DocumentArrays that work with multiprocessing, check out [this PR](https://github.com/docarray/docarray/pull/1049).

## Support Proto 3 and 4

[Protobuf](https://protobuf.dev/) introduced a [breaking change](https://github.com/tensorflow/tensorflow/issues/56077) in their 4.21 release. This has had a big impact on the Python ecosystem, and a lot of libraries have not yet been updated to use version 4.x. Perhaps the biggest pain for the ML ecosystem is TensorFlow’s lack of support for Protobuf, as it’s a widely used library and many packages, including DocArray, depend on it.

At the same time, DocArray can be used without TensorFlow — It’s just one of several available backends. To better support all users, we’ve decided to support both versions of protobuf.

This is actually easier than it may sound. We simply generated two Python files with Protoc, one for each of the Protobuf versions we want to support (3.x and 4.x).

So, depending on the protobuf version you have installed, we either load the first or the second version of the proto file. It’s as straightforward as that. [Here](https://github.com/docarray/docarray/pull/1078) is the PR for the curious.

## Join the conversation

Want to keep up to date or just have a chat with us? **[Join our Discord](https://discord.gg/WaMp6PVPgR)** and say hi!
