# DocArray v2: What and Why

DocArray has had a good run so far: Since being spun out of Jina ten months ago, the project has seen 141 releases, integrated six external storage backends, attracted contributors from five companies, and collected 1.4k GitHub stars.

And yet we feel like we have to bring some big changes to the library in order to make it what we want it to be: the go-to solution for modelling, sending, and storing multi-modal data, with a particular soft spot for ML and neural search applications.

The purpose of this post is to outline the technical reasons for this transition, from the perspective of us, the maintainers.
You might also be interested in a slightly different perspective, the one of Han Xiao, CEO of Jina AI and originator of DocArray. You can find his blog post [here](https://jina.ai/news/donate-docarray-lf-for-inclusive-standard-multimodal-data-model/).

If you are interested in the progress of the rewrite itself, you can follow along on our [public roadmap](https://github.com/docarray/docarray/issues/780).

So, without further ado, let's delve into what the plans are for this v2 of DocArray, followed by some explanations of why we think that this is the right move.

# The What

There are two big themes that dominate the new direction of DocArray:
1. We're going all-in on a dataclass-like interface, letting you define your own data models, and
2. The idea of the Document Store becomes more prominent and more clear

The even broader theme could be summarized as follows: Right now, power users often push against the limits of DocArray, and we want to expand these limits into production settings.

## The What: Dataclass-like Interface

After this rewrite, a dataclass-like interface will be the only way to create a `Document`:

```python
from docarray import Document
from docarray.typing import TorchTensor, Embedding, Text

class MyDoc(Document):
  tensor: Optional[TorchTensor]
  embedding: Optional[Embedding]
  txt: Text
  
doc = MyDoc(txt='Hello world')
doc.embedding = MyTextModel(doc.txt)
```

This interface is built on top of [pydantic](https://github.com/pydantic/pydantic), and can be seen as a true extension of the same.

In addition to the bare building blocks outlined above, DocArray will also provide pre-built Documents that can be used directly, extended or composed:

```python

from docarray import Text

# use it directly
txt_doc = Text(url='http://www.jina.ai/')
txt_doc.text = txt_doc.url.load()
model = MyEmbeddingModel()
txt_doc.embedding = model(txt_doc.text)

# extend it
class MyText(Text):
    second_embedding: Optional[Embedding]


txt_doc = MyText(url='http://www.jina.ai/')
txt_doc.text = txt_doc.url.load()
model = MyEmbeddingModel()
txt_doc.embedding = model(txt_doc.text)
txt_doc.second_embedding = model(txt_doc.text)

# compose it
from docarray import Document, Image, Text

class MultiModalDoc(Document):
    image_doc: Image
    text_doc: Text


mmdoc = MultiModalDoc(
    image_doc=Image(url="http://www.jina.ai/image.jpg"),
    text_doc=Text(text="hello world, how are you doing?"),
)
mmdoc.text_doc.text = mmdoc.text_doc.url.load()

```

Ultimately, there is one main theme here: **Documents adapt to your data, so your data doesn't have to adapt to Document.**

### DocumentArray

The changes coming to DocumentArray will be less drastic, it will still be essentially a list of Documents:

```python
from docarray import DocumentArray

da = DocumentArray([MyDoc(txt='hi there!' for _ in range(10)])
```

However, the commitment to a dataclass-like interface allows for DocumentArrays that are typed by a specific schema:

```python
da = DocumentArray[MyDoc]([MyDoc(txt='hi there!' for _ in range(10)])
```

## The What: Document Store

The idea here is simple: Documents can be either in memory or in a database, and this distinction should be clearly reflected in two dedicated objects: `DocumentArray` and `DocumentStore`.

```python
from docarray import DocumentArray, DocumentStore

da = DocumentArray[MyDoc]([MyDoc(txt='hi there!' for _ in range(10)])

store = DocumentStore[MyDoc](storage='annlite')
store.add(da)  # add in-memory Documents into the Document Store
top_k = store.find(MyDoc(txt='hey'))  # perform ANN vector search in the Document Store
```

DocumentArray will be responsible for list-like behaviour and data-in-transit, while DocumentStore is responsible for data in a database and vector search.

# The Why

The overarching theme of this rewrite is making DocArray more universal, more flexible, and more production ready.

Ultimately, there are many reasons and many perspectives that make us believe in this transition.
Below you can find our reasoning from some of these perspectives.

## The Why: Pydantic Perspective

DocArray will build its `Document`s on top of pydantic models.
We believe that this makes sense for the following reasons:

- Pydantic provides a standard and beloved API for data modelling that is an excellent foundation
- DocArray can directly benefit from several pydantic features, such as serialization to JSON/dict, data validation, FastAPI integration, etc

But there is a flip side to this: What does DocArray offer that differentiates it from Pydantic?

- ML focussed types: `Tensor`, `TorchTensor`, `TFTensor`, `Embedding`, ...
- Types that are alive: `ImageUrl` can `.load()` a URL to image tensor, `TextUrl` can load and tokenize text documents, etc.
- Pre-built Documents for different data modalities: `Image`, `Text`, `3DMesh`, `Video`, `Audio`, ... Note that all of these will be valid Pydantic models!
- The concepts of DocumentArray and DocumentStore
- Cloud ready: Serialization to Protobuf for use with microservices and gRPC
- Support for vector search functionalities, such as `find()` and `embed()`

## The Why: Data Modelling Perspective

In the current DocArray, every `Document` has a fixed schema: It as a `text`, an `embedding`, a `tensor`, a `uri`, ...
This setup is fine for simple use cases, but is not flexible enough for advanced scenarios:

- How can you store multiple embeddings for a **hybrid search use case**?
- How do you model deeply nested data?
- How do you store multiple data modalities in one object?

All of these scenarios can only be solved by a solution that gives all the flexibility to the user, and ad dataclass-like API offers just that.

## The Why: ML and Training Perspective

`DocumentArray` is fundamentally a row-based data structure: Every Document is one unit (row), that can be manipulated, shuffled around, etc. This is a great property to have in an information retrieval or neural search setting, where tasks like ranking require row-based data access.

For other use cases like training an ML model, however, a row-based data structure is preferable: When you train your model, you want it to take in all data of a given mini-batch at once, as one big tensor; you don't want to first stack a bunch of tiny tensors before your forward pass.

Therefore, we will be introducing a mode for selectively enabling column-based behaviour on certain fields of your data model ("stacked mode"):

```python
from docarray import Document, DocumentArray
from docarray.typing import TorchTensor, Embedding, Text
import torch

# create some data
class MyDoc(Document):
  tensor: Optional[TorchTensor]
  embedding: Optional[Embedding]
  
da = DocumentArray[MyDoc]([MyDoc(tensor=torch.rand((224, 224, 3)))])

# stack the tensors (row-wise mode)
da.stack('tensor')

# unstack the tensor (back to column-wise mode)
da.unstack('tensor')

# stack und unstack using context manager
with da.stacked('tensor'):
  ...
```

This will make DocArray much more suitable for ML training and inference, and even for use inside of ML models.

## The Why: Document Store / Vector DB perspective

In the current DocArray, every `DocumentArray` can be mapped to a storage backend `da = DocumentArray(storage='annlite', ...)`.
This offers a lot of convenience for simple use cases, but the conflation of the array concept and the DB concept lead to a number of problems:

- It is not always clear what data is on disk, and what data is in memory
- Not all in-place operations on a DocumentArray are automatically reflected in the associated DB, while others are. This is due to the fact that some operations load data into memory before the manipulation happens, and means that a deep understanding of DocArray is necessary to know what is going on
- Supporting list-like operations on a DB-like object carries overhead with little benefit
- It is difficult to expose all the power and flexibility of various vector DBs through the `DocumentArray` API

All of the problems above currently make it difficult to use vector DBs through DocArray in production.
Disentangling the concepts of `DocumentArray` and `DocumentStore` will give more transparency to the user, and more flexibility to the contributors, while directly solving most of the above.

## The Why: Web Application Perspective

Currently it is possible to use DocArray in combination with FastAPI and other web frameworks, as it already provides a translation to Pydantic.
However, this integration is not without friction:

- Since currently every Document follows the same schema, as Document payload cannot be customized
- This means that one is forced to create payload with (potentially many) empty and unused fields
- While at the same time, there is no natural way to add new fields
- Sending requests from programming languages other than Python requires the user to recreated the Document's structure, needlessly

By switching to a dataclass-first approach with Pydantic as a fundamental building block, we are able to ease all of these pains:

- Fields are completely customizable
- Every `Document` is also a Pydantic model, enabling amazing support for FastAPI and other tools
- Creating payloads from other programming languages is as easy as creating a dictionary with the same fields as the dataclass - same workflow as with normal Pydantic

## The Why: Microservices Perspective

In the land of cloud-nativeness and microservices, the concerns from "normal" web development also apply, but are often exacerbated due to the many network calls that occur, and other technologies such as protobuf and gRPC entering the game.

With this in mind, DocArray v2 can offer the following improvements

- Creating valid protobuf definitions from outside of python will be as simple as doing the same for JSON: Just specify a mapping that includes the keys that you defined in the Document dataclass interface
- It is no longer needed to re-create the predefined Document structure in your Protobuf definitions
- For every microservice, the Document schema can function as requirement or contract about the input and output data of that particular microservice

Currently, a DocArray-based microservice architecture will usually rely on `Document` being the unified input and output for all microservices. So there might be concern here: Won't this new, more flexible structure create a huge mess where microservices cannot rely on anything?
We argue the opposite! In complex real-life settings, it is often the case that input and output Documents heavily rely on the `.chunks` field to represent nested data. Therefore, it is already unclear what exact data model can be expected.
The shift to a dataclass-first approach allows you to make all of these (nested) data models explicit instead of implicit, leading to _more_ interoperability between microservices, not less.

