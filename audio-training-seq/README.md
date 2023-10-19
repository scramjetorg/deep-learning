# Deep Learning - Speech to Intent

This repository offers a streamlined solution for converting audio files into spectrograms using Tensorflow generators and subsequently training a Convolutional Neural Network (CNN) Sequential model on the generated spectrogram data packed in a Scramjet Sequence run on <a href= "https://github.com/scramjetorg/transform-hub" target = "_blank">STH</a>. 

Training audio dataset<br/>
['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

Dataset source<br/> 
https://developer.ibm.com/exchanges/data/all/speech-commands/

Audio conversion to PCM_S16LE Mono 16000Hz<br/>
https://convertio.co/opus-wav/

## Key Features

- Audio to Spectrogram Conversion: Utilize powerful Python generators to efficiently transform audio files into spectrogram images.
- CNN Sequential Model Training: Train a Convolutional Neural Network (CNN) Sequential model on the generated spectrograms for classification, regression, or any other suitable task.
- Efficient Data Handling: Handle large audio datasets effectively using generators, ensuring efficient memory usage and seamless training.


## License

This project is licensed under MIT licenses. 

