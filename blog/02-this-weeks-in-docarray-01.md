# This week(s) in DocArray

It's already been two weeks since the [last alpha release of DocArray v2](https://github.com/docarray/docarray/releases/tag/2023.01.18.alpha). And since then a lot has happened — we've merged features we're really proud of, and we've cried tears of joy and misery trying to coerce Python into doing what we want. If you want to learn about interesting Python edge cases or follow the advancement of DocArray v2 development then you’ve come to the right place in this blog post!
For those who don’t know, DocArray is a library for **representing, sending, and storing multi-modal data, with a focus on applications in ML and Neural Search**.

The project just moved to the [Linux foundation AI and Data](https://lfaidata.foundation/), and to celebrate its first birthday we decided to rewrite it from scratch, mainly because of a design shift and a will to solidify the codebase from the ground up. Also because it can’t eat cake and we had to give it something.

So, what's been happening in the past two weeks?

## Less verbose API

One of DocArray's goals is to give our users powerful abstractions to represent nested data. To do this in v2 we allow nesting of BaseDocument. (Well, this is actually just a feature of [pydantic](https://docs.pydantic.dev/) and one of the reasons its design seduces us to use it as a backend).

```python
from docarray import BaseDocument
from docarray.documents import Image, Text

class MyBanner(BaseDocument):
    title: Text
    image: Image

class MyPoster(BaseDocument):
    left: MyBanner
    right: MyBanner
```

This is a powerful design pattern, but the API is a bit too verbose when using our predefined Document class:

```python
banner_1 = MyBanner(title=Text(text='hello'), image=Image(url='myimage.png'))
banner_2 = MyBanner(title=Text(text='bye bye'), image=Image(url='myimage2.png'))

poster = MyPoster(left=banner_1, right=banner_2)
```

The new API looks like this:

```python
banner_1 = MyBanner(title='hello', image='myimage.png')
banner_2 = MyBanner(title='bye bye', image='myimage2.png')

poster = MyPoster(left=banner_1, right=banner_2)
```

It's waaay less verbose. We basically override pydantic's predefined document validator to let us do this smart casting. But we didn't make this automatic, in the sense that if you create a Document you still need to use the verbose API. This is because this casting isn't always obvious. For instance, look at this Document:

```python
class MyDoc(BaseDocument):
   title: str
   description: str

doc = MyDoc('hello') # won't work
```

In this case, where should 'hello' be assigned? Title or description? There's no obvious way to do it so we'd rather let the user define it, at least until we find a better way.

We're thinking about either:

- Referring to the order and make the first string in the list the “main” one. But this is against one of the core values of this rewrite: “we don’t do things implicitly”.
- Allowing the user to mark a "main" field somehow, either with a Field object or a function.

From the outside, it looks like a minor problem. But we believe the real devil is in the details, so we spent countless hours arguing over such a simple API. Man, that's time we won't get back. 💁‍♂️

Curious? Check out [this PR](https://github.com/docarray/docarray/commit/4311bcc36cd3231fcf03c7befe0a7fe9a8f71f24)

## ``__torch_function__`` , or: How to give PyTorch a little bit more confidence

We had a lot of fun wrapping our heads around the `__torch_function__` concept.
 
Our `TorchTensor` class is a subclass of `torch.Tensor` that injects some useful functionality (mainly the ability to express its shape at the type level: `TorchTensor[3, 224, 224]`, and protobuf serialization), and PyTorch comes with a whole machinery around subclassing, dynamic dispatch and all that jazz.

One part of this machinery is `__torch_function__` , a magic method that allows all kinds of objects to be treated like Torch Tensors. You want instances of your class to be able to be processed by functions like `torch.stack([your_instance, another_instance])`, or be directly added to a `torch.Tensor`? No problem, just implement `__torch_function__` in your class, handle it there, and off you go! No need to even subclass `torch.Tensor`:

```python
import torch

class MyClass:
    def __init__(self, others=None):
        self._others = others or []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.stack or func is torch.Tensor.add:
			# we know how to handle these!
            return cls.combine(args)
        else:
			# ... but are clueless about the rest
            return NotImplemented

    @classmethod
    def combine(cls, *others):
        return cls(others=list(others))

print(torch.stack([MyClass(), MyClass()]))
# outputs:
# <__main__.MyClass object at 0x7fd290c55190>
print(torch.rand(3, 4, 5) + MyClass())
# outputs:
# <__main__.MyClass object at 0x7f363e2ed0d0>
```

Now, the example above isn’t a very useful one, but you get the idea: `__torch_function__` lets you create objects that behave like Torch Tensors without being Torch Tensors.

But hold on. Instances of `TorchTensor` are Torch Tensors, since they directly inherit from `torch.Tensor`! So all the functionality is already there, we inherit `__torch_function__` from `torch.Tensor,` and we don’t need to care about any of this, right?

Well, not quite.

The thing is, we don’t just have one subclass of `torch.Tensor`; we have many: `TorchTensor` is the obvious one, but there's also `TorchTensor[3, 224, 224]`, `TorchTensor[128]` and `TorchTensor['batch', 'c', 'w', 'h']`, etc. All of these are separate classes!
To be a bit more precise, all the parameterized classes (the ones with `[...]` at the end) are direct subclasses of `TorchTensor` and are **siblings of one another** (this becomes important later on).

```
                                    torch.Tensor
                                         ^
                                         |
       ---------------------------> TorchTensor <------
      ^                   ^                            ^
      |                   |           ....             |
TorchTensor[128] TorchTensor[1, 128]  ....   TorchTensor['batch', 'c', 'w', 'h']
```

So where's the problem?

The problem essentially lies in the `types` argument to `__torch_function__`. It contains the types of all the arguments that were passed to the original PyTorch function call, `torch.stack()` in the example above. Again, in the `stack` example above, this would just be the tuple `(MyClass, MyClass)`.

This is meant just as a convenience to the implementer of `__torch_function__`. It lets them quickly decide, based on the type, if they can handle a given input or not.

Let’s take a look at how the default PyTorch (`torch.Tensor`) implementation of `__torch_function__` makes that decision:

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
	# ... some stuff here
    if not all(issubclass(cls, t) for t in types):
        return NotImplemented
	# ... more stuff here
```

Can you already guess where things go wrong?

Let me give you a hint by showing a failure case:

```python
data = torch.rand(128)
print(TorchTensor[128](data) + TorchTensor[1, 128](data))
```

When this call is handled in `__torch_function__` , as inherited from `torch.Tensor`, `cls` will be `TorchTensor[128]` and `types` will contain `TorchTensor[1, 128]`.

That makes sense: those are the two classes involved in this addition.

But what will PyTorch do?

It will throw up its hands and give up!

```python
TypeError: unsupported operand type(s) for +: 'TorchTensor[128]' and 'TorchTensor[1, 128]'
```

`TorchTensor[128]` is not a subclass of `TorchTensor[1, 128]`; they're siblings! So the subclass check above will fail and PyTorch will *announce that it has absolutely no clue* about how to combine instances of these two classes.

But c'mon PyTorch! Both these classes inherit from `torch.Tensor`! Believe in yourself, you do know how to deal with them! Just treat them like normal tensors!

And that’s already the solution to the entire problem: We need to give PyTorch a little confidence boost, by telling it to treat our custom classes just like the `torch.Tensor` class it already knows and loves.

So how do we give it this metaphorical pep talk? It’s actually quite simple:

This is the implementation of of `__torch_function__` that currently powers `TorchTensor`. It does just one thing: For any class that's a subclass of `TorchTensor`, it changes the types argument before passing it along to the default implementation of `__torch_function__`. It substitutes all such types for `torch.Tensor`, telling PyTorch that it's got this! 

Et voilà, it works:

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    # this tells torch to treat all of our custom tensors just like
    # torch.Tensor's. Otherwise, torch will complain that it doesn't
    # know how to handle our custom tensor type.
    docarray_torch_tensors = TorchTensor.__subclasses__()
    types_ = tuple(
        torch.Tensor if t in docarray_torch_tensors else t for t in types
    )
    return super().__torch_function__(func, types_, args, kwargs)
```

[This PR](https://github.com/docarray/docarray/pull/1037/files) demonstrates how we coached PyTorch into having a little more self-esteem and being it's truest, best self:

## Early support for DocArray v2 in Jina

Well, it's not exactly a new feature, but we've been working on early support for DocArray v2 in [Jina](https://github.com/jina-ai/jina/).

DocArray’s relation to Jina is similar to pydantic’s relation to [FastAPI](https://fastapi.tiangolo.com/):

- FastAPI is an HTTP framework that uses pydantic models to define the API schema.
- Jina is a gRPC/HTTP framework that uses DocArray Documents to define the API schema.

There are other conceptual differences of course, but to fully understand the new changes in Jina it's interesting to look at it like this. DocArray is actually built on top of pydantic and adds a hint of multi-modal machine learning on top of that.

Here's an example of the new interface:

```python
from jina import Executor, requests
from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.typing import AnyTensor

import numpy as np

class InputDoc(BaseDocument):
    img: Image

class OutputDoc(BaseDocument):
    embedding: AnyTensor

class MyExec(Executor):
    @requests(on='/bar')
    def bar(
        self, docs: DocumentArray[InputDoc], **kwargs
    ) -> DocumentArray[OutputDoc]:
        docs_return = DocumentArray[OutputDoc](
            [OutputDoc(embedding=np.zeros((100, 1))) for _ in range(len(docs))]
        )
        return docs_return
```

The main difference is that an Executor doesn't necessarily do in-place modification, but can return a different Document type. For instance, we have a toy encoder that takes an image as input and returns embeddings. Similar to FastAPI, we infer the input and output schema of the Executor by inspecting the type hint of the method. You can also use this information as an argument if you don’t want to rely on the type hint. You can check the [v2 docs](https://feat-docarray-v2--jina-docs.netlify.app/concepts/executor/docarray-v2/) for more information.

Here's [the PR](https://github.com/jina-ai/jina/pull/5603).

## Pretty printing

We ported back the pretty printing from DocArray v1 to v2 and tidied it up a bit to reflect the new v2 schema! Under the hood, we're relying on the awesome rich library for everything related to UI.

![](./.github/images/blog/dev-blog-1/pretty_print_0.png)

![](./.github/images/blog/dev-blog-1/pretty_print_1.png)

Check [the PR](https://github.com/docarray/docarray/pull/1043) for more info!

## Document Stores

We’re currently completely rethinking Document Stores. The main points are:

- Every Document Store will have a **schema** assigned, just like a DocumentArray, but with more (backend-dependent) options and configurations.
- First-class support for **hybrid search and multi-vector search**.
- Support **search on nested Documents**.

If you are curious about the **full (preliminary) design** you can check it in detail out [here](https://lightning-scent-57a.notion.site/Document-Stores-v2-design-doc-f11d6fe6ecee43f49ef88e0f1bf80b7f). But here's a small taster:

```python
# define schema
class MyDoc(BaseDocument):
    url: ImageUrl
    tensor: TorchTensor[128]

da = DocumentArray[MyDoc](...)  # data to index

store = DocumentStore[MyDoc](storage='MyFavDB', ...)

# index data
store.index(da)

# search through query builder
with store.query_builder() as q:
    # build complex (composite) query
    q.find(torch.tensor(...), field='image', weight=0.3)
    q.find(torch.tensor(...), field='description')
    q.filter("price < 200")
    q.text_search('jeans', field='title')

results = store.execute_query(q)
```

Beyond the first designs that are just now finding their way into actual code, we're happy to share that we're **closely collaborating with [Weaviate](https://weaviate.io/)** to make our Document Stores as good as they can be!

So far they’ve provided a lot of valuable input for our designs, and we’re looking forward to the collaboration during actual implementation.

Lastly, a word about **Document Store launch plans**: Our current plan is to launch this reincarnation of Document Stores with **three supported backends**: **Weaviate**, **[ElasticSearch](https://www.elastic.co/)**, and one **on-device vector search library** (which one? That's still TBD). 

Unfortunately our capacity doesn't allow for more on launch day, but if you (yes, you!) want to **help us** accelerate development for one of the other vector databases, we would absolutely love that and accelerate our timelines accordingly. If you feel intrigued, **[reach out to us on Discord](https://discord.gg/WaMp6PVPgR)**!


