# KerasDropconnect
An implementation of <a href="http://proceedings.mlr.press/v28/wan13.html" target=”_blank”>DropConnect</a> Layer in Keras

## Install
```
  git clone https://github.com/andry9454/KerasDropconnect.git
  cd KerasDropconnect
  pip3 install .
```

## Usage
```
from ddrop.layers import DropConnect

x = DropConnect(Dense(64, activation='relu'), prob=0.5)(x)
```
## Credits
Original implementation forked from Derek Khu's repository
