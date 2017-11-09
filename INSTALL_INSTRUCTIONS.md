## DELF installation

### Tensorflow 1.3

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

```bash
# For CPU:
pip install tensorflow
# For GPU:
pip install tensorflow-gpu
```

### Protobuf

The DELF library uses [protobuf](https://github.com/google/protobuf) (the python
version) to configure feature extraction and its format. You will need the
`protoc` compiler, version >= 3.3. The easiest way to get it is to download
directly. For Linux, this can be done as (see
[here](https://github.com/google/protobuf/releases) for other platforms):

```bash
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip
PATH_TO_PROTOC=`pwd`
```

### Python dependencies

Install python library dependencies:

```bash
pip3 install matplotlib numpy scikit-image scipy
```

### `tensorflow/models`

We already gathered required models, which are `object_detection`, `slim`, and `delf` in `tensorflow/models/research/`, here. 

Please add path so `delf` can import these models. 

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export PATH_TO_PROTOC=`pwd`
```

Then, compile DELF's protobufs. Use `PATH_TO_PROTOC` as the directory where you
downloaded the `protoc` compiler.

```bash
# From tensorflow/models/research/delf/
${PATH_TO_PROTOC?}/bin/protoc delf/protos/*.proto --python_out=.
```

Finally, install the DELF package.

```bash
# From tensorflow/models/research/delf/
sudo pip install -e . # Install "delf" package.
```

At this point, running

```bash
python -c 'import delf'
```

should just return without complaints. This indicates that the DELF package is
loaded successfully.

### Troubleshooting

#### Python version

Installation issues may happen if multiple python versions are mixed. The
instructions above assume python2.7 version is used; if using python3.X, be sure
to use `pip3` instead of `pip`, and all should work.

#### `pip install`

Issues might be observed if using `pip install` with `-e` option (editable
mode). You may try out to simply remove the `-e` from the commands above. Also,
depending on your machine setup, you might need to run the `pip install` command
without `sudo` at the beginning.
