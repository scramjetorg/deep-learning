# Deep Learning 

Welcome to the Deep Learning repository by Scramjet. This repository hosts a collection of resources, tutorials, and code samples related to Tensorflow Sequential model being run as a Sequence.<br/> 
Using Scramjet's Transform Hub <a href="https://github.com/scramjetorg/transform-hub" target="_blank">(STH)</a> offers a streamlined solution for converting audio files into spectrograms using Tensorflow generators and subsequently training a Convolutional Neural Network (CNN) Sequential model on the generated spectrogram data packed in a Scramjet Sequence.

## Key Features

- Audio to Spectrogram Conversion: Utilize powerful Python generators to efficiently transform audio files into spectrogram images.
- CNN Sequential Model Training: Train a Convolutional Neural Network (CNN) Sequential model on the generated spectrograms for classification, regression, or any other suitable task.
- Efficient Data Handling: Handle large audio datasets effectively using generators, ensuring efficient memory usage and seamless training.

## Prerequisites
- <a href="https://www.npmjs.com/package/@scramjet/cli" target="_blank">STH</a> <br/>
- AWS S3 credentials
- Python (>=3.6)

## Install and Run
Install the Scramjet Transform Hub (STH) locally or use Scramjet's Cloud Platform environment for the Sequence deployment. For more information on the below commands check the 
<a href="https://docs.scramjet.org/platform/cli-reference/" target="_blank">CLI reference</a> section on Scramjet's Website.


## Sequential model Sequence<br/>
This directory demonstrates how to leverage deep learning techniques to recognize audio commands using Tensorflow and Scramjet's STH framework.

**Dataset Core words:** 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'

**Dataset source:** https://developer.ibm.com/exchanges/data/all/speech-commands/

**Audio format specification:** PCM_S16LE Mono 16000Hz<br/>

---

- For this Sequence to run properly on your Linux machine use the following command to start the <a href="https://docs.scramjet.org/platform/self-hosted-installation/" target="_blank">STH</a> on terminal #1.

```bash
$ DEVELOPMENT=true sth --runtime-adapter=process
```

**NOTE:** This Sequence might consume some disk space, clearing out Scramjet's Sequence disk space manually might be required from time to time.

```bash
$ sudo rm -r ~/.scramjet_sequences
```

- To pack and run this Sequence, on terminal #2 of your Linux machine execute the following commands:

```bash
# Create a directory __pypackages__ in the same directory as main.py
~/training-script$ mkdir __pypackages__

# Install dependencies in the __pypackages__ folder. 
~/training-script$ pip3 install -t __pypackages__ -r requirements.txt

# Pack the training-script folder into a gzip format
~$ si sequence pack training-script

# Send the training-script.tar.gz Sequence to the Scramjet's Transform-Hub, with a return <Sequence-id> value
~$ si sequence send training-script.tar.gz --progress

# Start the Sequence with arguments
~$ si seq start - --args=[\"aws_key\","\aws_secret\","\aws_bucket\"] # Without spacing between args

# Send the audio files as input
~$ si instance input <Instance-id> local/path/to/multi-label-audio.wav -e -t application/octet-stream

# Return list of S3 Bucket objects as output
~$ si instance output <Instance-id>
```

## Inference Sequence

This directory contains the code and a pre-trained keras model necessary for running an inference with the ability to send an audio file as `input`.

- For this Sequence to run properly on your Linux machine use the following command to start <a href="https://docs.scramjet.org/platform/self-hosted-installation/" target="_blank">STH</a> on terminal #1.

```bash
$ DEVELOPMENT=true sth --runtime-adapter=process
```

**NOTE:** This Sequence might consume some disk space, clearing out Scramjet's Sequence disk space manually might be required from time to time.

```bash
$ sudo rm -r ~/.scramjet_sequences
```

- To pack and run this Sequence, on terminal #2 of a Linux machine execute the following commands:

```bash
# Create a directory __pypackages__ in the same directory as main.py
~/inference-script$ mkdir __pypackages__

# Install dependencies in the __pypackages__ folder. 
~/inference-script$ pip3 install -t __pypackages__ -r requirements.txt

# Pack the inference-script folder into a gzip format
~$ si sequence pack inference-script

# Send the inference-script.tar.gz Sequence to the Scramjet's Transform-Hub, with a return <Sequence-id> value
~$ si sequence send inference-script.tar.gz --progress

# Start the Sequence
~$ si sequence start <Sequence-id> 

# Send the audio file as input
~$ si instance input <Instance-id> local/path/to/audio.wav -e -t application/octet-stream

# Return classification label of audio .wav file as output
~$ si instance output <Instance-id>
```
## Audio format specification

Audio wave file required to be sent as input must be one of ten labels:<br/> 'right', 'left', 'no', 'stop', 'down', 'go', 'up', 'yes', 'on', 'off'

Dataset source<br/> 
https://developer.ibm.com/exchanges/data/all/speech-commands/

Audio conversion to PCM_S16LE Mono 16000Hz<br/>
https://convertio.co/opus-wav/

### Audio file Details:

Format : Wave<br/>
Duration : <1 second<br/>
Format : PCM<br/>
Format settings : Little / Signed<br/>
Codec ID : 1<br/>
Bit rate mode : Constant<br/>
Bit rate : 256 kb/s<br/>
Channel(s) : 1 channel<br/>
Sampling rate : 16.0 kHz<br/>
Bit depth : 16 bits<br/>

## License

This project is licensed under MIT licenses. 

