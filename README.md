# KerasDropconnect
An implementation of DropConnect Layer in Keras

## Install
  pip3 install .

## Usage
```
from ddrop.layers import DropConnect

x = DropConnect(Dense(64, activation='relu'), prob=0.5)(x)
```
