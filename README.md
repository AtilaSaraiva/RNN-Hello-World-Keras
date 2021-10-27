# Keras RNN model to identify harmonics of a sin time series

In order for me to learn how to pre process a time series data in order to use RNNs, I developed this simple example.
It tries to find whether a certain number of samples of a time series is the sine wave that represents it, whether it is
sin(1*t), sin(2*t) or sin(n*t). In other words, it tries to find n.

To run this example, either install all of the dependencies using pip or your OS package manager, or use *nix-shell*.
nix-shell is a tool that uses the [nix package manager](https://nixos.org/) to download all of the depedencies for this code.
It is an idea similar to that of the Docker files, but without needing to virtualize a whole OS, just a shell with the dependencies.

To install nix just run:
```
curl -L https://nixos.org/nix/install | sh
```

After installing nix, run:
```
nix-shell python.nix
python rnn.py
```
It will create the database and save the trained model to a file on "/models/modelolegal". Afterwards, it will test the model on a sin(3*t) sequence.
If you with to test it again without training the whole dataset again, just run:
```
nix-shell python.nix
python rnn.py models/modelolegal
```
and it will retrieve the associated model.

Thx for stopping by!

All hail nix!
