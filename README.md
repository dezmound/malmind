# MalMind

This is a [Keras](https://github.com/keras-team/keras) implementation of    
malware signatures of [CFG](https://github.com/dezmound/cfg-worker) base.

# How to install

This packet requires `Python` as backend and some python packets for neuro analyse.

Run the following command to install python dependencies:
```
npm i && npm run pydep
```
or install `Python` dependencies by yourself:

```
sudo pip install tensorflow
sudo pip install keras
sudo pip install TurboGears2
```

# Use

Run the backend:
```
npm start -- --inputLayerSize=20
```
By default:
```
--inputLayerSize=100
```


This program works as neuro backend server.     
To train your model, use this url:

```
http://localhost:3008/train?vector=[5,2,3,4,1,...]&isMalware=0
```

To test malware signature use:   
```
http://localhost:3008/train?test=[5,2,3,4,1,...]
```

Response sent as described format:
```json
{
    "isMalware": true
}
```
